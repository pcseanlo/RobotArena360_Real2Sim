"""Step 5 ``surface.trim``: drop floater gaussians over the operation surface.

Inputs:  ``world/gaussian.ply`` (step 1a), ``surface/operation_surface_plane_fit.json``
         (step 4), ``align/aligned_pose_refined.npy`` and ``..._scene_scale_refined.npy``.
Output:  ``world/gaussian_trimmed.ply`` -- binary PLY with the same vertex properties, the
         splat the evaluator renders and step 6 fits the receiver to.

A gaussian is dropped only when it is both a floater candidate (faint: sigmoid opacity
< ``--min-opacity``, or oversized: max(exp(scale_*)) > ``--max-gaussian-scale``) AND inside
the workspace region over the fitted plane (within ``--workspace-radius`` of the plane
centre and ``--plane-band-below``/``--plane-band-above`` of its height). The rest of the
scene is untouched. Ported from the old ``trim_gaussian_ply.py`` (defaults =
``trim_gaussian.sh``); its unused ``--global-trim`` mode is gone. The PLY codec is the old
dependency-free numpy one.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np

from r2s.common.io import prepare_output, read_json
from r2s.common.transforms import splat_world_pose, transform_gaussian_centers
from r2s.layout import SceneLayout
from r2s.steps import Step
from r2s.surface.geometry import selected_plane

# ---------------------------------------------------------------------------
# Minimal binary/ascii PLY codec (no plyfile dependency)
# ---------------------------------------------------------------------------

# PLY scalar type name -> numpy base dtype.
_PLY_TO_NP = {
    "char": "i1", "int8": "i1",
    "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2",
    "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4",
    "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4",
    "double": "f8", "float64": "f8",
}
_NP_TO_PLY = {np.dtype(v).str[1:]: k for k, v in {
    "char": "i1", "uchar": "u1", "short": "i2", "ushort": "u2",
    "int": "i4", "uint": "u4", "float": "f4", "double": "f8",
}.items()}


def read_vertex_ply(path: str | Path) -> tuple[np.ndarray, str]:
    """Return (structured vertex array, byte_order) for the ``vertex`` element.

    Only the ``vertex`` element is parsed; this is sufficient for 3DGS clouds,
    which store every gaussian as a single vertex element.
    """
    with open(path, "rb") as handle:
        if handle.readline().strip() != b"ply":
            raise ValueError(f"Not a PLY file: {path}")
        fmt = ""
        count = 0
        props: list[tuple[str, str]] = []
        in_vertex = False
        while True:
            raw = handle.readline()
            if not raw:
                raise ValueError(f"PLY header is incomplete: {path}")
            parts = raw.decode("ascii").split()
            if not parts:
                continue
            tag = parts[0]
            if tag == "format":
                fmt = parts[1]
            elif tag == "element":
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    count = int(parts[2])
            elif tag == "property" and in_vertex:
                if parts[1] == "list":
                    raise ValueError("list properties are not supported on vertex")
                props.append((parts[-1], parts[1]))
            elif tag == "end_header":
                break

        if not props or count <= 0:
            raise ValueError(f"PLY has no vertex properties: {path}")

        if fmt == "binary_little_endian":
            order = "<"
        elif fmt == "binary_big_endian":
            order = ">"
        elif fmt == "ascii":
            order = "="
        else:
            raise ValueError(f"Unsupported PLY format: {fmt!r}")

        dtype = np.dtype([
            (name, order + _PLY_TO_NP[ply_type]) for name, ply_type in props
        ])
        if order == "=":  # ascii
            flat = np.loadtxt(handle, max_rows=count)
            data = np.empty(count, dtype=dtype)
            for i, (name, _) in enumerate(props):
                data[name] = flat[:, i]
        else:
            data = np.fromfile(handle, dtype=dtype, count=count)
    return data, ("<" if order == "=" else order)


def write_vertex_ply(path: str | Path, data: np.ndarray, byte_order: str = "<") -> None:
    """Write a structured vertex array back out as a binary PLY."""
    fmt = "binary_little_endian" if byte_order == "<" else "binary_big_endian"
    lines = ["ply", f"format {fmt} 1.0", f"element vertex {len(data)}"]
    for name in data.dtype.names:
        base = data.dtype[name].str[1:]
        lines.append(f"property {_NP_TO_PLY[base]} {name}")
    lines.append("end_header\n")
    header = ("\n".join(lines)).encode("ascii")
    contiguous = np.ascontiguousarray(data.astype(data.dtype.newbyteorder(byte_order)))
    with open(prepare_output(path), "wb") as handle:
        handle.write(header)
        handle.write(contiguous.tobytes())


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-values))


def _scale_names(names: tuple[str, ...]) -> list[str]:
    return sorted(
        [n for n in names if n.startswith("scale_")],
        key=lambda v: int(v.split("_")[-1]) if v.split("_")[-1].isdigit() else 0,
    )


def opacity_mask(data: np.ndarray, min_opacity: float) -> np.ndarray:
    if "opacity" not in data.dtype.names:
        raise ValueError("PLY has no 'opacity' property; cannot apply --min-opacity")
    op = np.asarray(data["opacity"], dtype=np.float64)
    if np.nanmin(op) < 0.0 or np.nanmax(op) > 1.0:
        op = _sigmoid(op)
    return op >= float(min_opacity)


def scale_mask(data: np.ndarray, max_gaussian_scale: float) -> np.ndarray:
    names = _scale_names(data.dtype.names)
    if not names:
        raise ValueError("PLY has no 'scale_*' properties; cannot apply --max-gaussian-scale")
    scales = np.column_stack([np.asarray(data[n], dtype=np.float64) for n in names])
    return np.max(np.exp(scales), axis=1) <= float(max_gaussian_scale)


def region_mask(
    data: np.ndarray,
    *,
    plane_json: str | Path,
    result_path: str | Path,
    scene_scale_path: str | Path,
    band_above: float,
    band_below: float,
    workspace_radius: float,
) -> np.ndarray:
    """Boolean mask of gaussians inside the operation-surface workspace region.

    A gaussian is "in region" when it is BOTH (a) horizontally within
    ``workspace_radius`` of the plane centre and (b) inside the world-z band
    ``[plane_top - band_below, plane_top + band_above]``.  This defines *where*
    floater culling is allowed; opacity/scale decide *which* gaussians inside
    that region get dropped.  Everything outside the region is always kept.
    """
    selected = read_json(plane_json).get("selected")
    if not selected:
        raise ValueError(f"plane JSON has no 'selected' box: {plane_json}")

    normal, offset, _u, _v = selected_plane(selected)
    splat_pos, splat_quat, scene_scale = splat_world_pose(result_path, scene_scale_path)

    xyz_local = np.column_stack((data["x"], data["y"], data["z"])).astype(np.float64)
    xyz_world = transform_gaussian_centers(
        xyz_local,
        splat_pos=splat_pos,
        splat_quat_wxyz=splat_quat,
        scene_scale=scene_scale,
    )

    center = np.asarray(selected["center"], dtype=np.float64).reshape(3)
    # Signed distance to the plane; positive points sit above the surface.
    signed = xyz_world @ normal + offset
    in_band = (signed <= float(band_above)) & (signed >= -float(band_below))
    radial = np.linalg.norm(xyz_world[:, :2] - center[None, :2], axis=1)
    in_radius = radial <= float(workspace_radius)
    return in_band & in_radius


# ---------------------------------------------------------------------------
# Plain function + step
# ---------------------------------------------------------------------------

def trim_gaussians(
    input_ply: str | Path,
    output_ply: str | Path,
    *,
    plane_json: str | Path,
    pose_path: str | Path,
    scale_path: str | Path,
    min_opacity: float | None = 0.3,
    max_gaussian_scale: float | None = 0.05,
    workspace_radius: float = 0.70,
    plane_band_above: float = 0.15,
    plane_band_below: float = 0.05,
) -> dict[str, Any]:
    """Write ``output_ply`` = ``input_ply`` minus region-gated floaters; returns the trim stats."""
    if not os.path.exists(input_ply):
        raise FileNotFoundError(f"input PLY not found: {input_ply}")
    if os.path.abspath(output_ply) == os.path.abspath(input_ply):
        raise ValueError("refusing to overwrite the source PLY; choose a different output")

    data, order = read_vertex_ply(input_ply)
    total = len(data)
    stats: dict[str, Any] = {"input_count": total}

    # "bad" = faint OR oversized gaussians (floater candidates).
    bad = np.zeros(total, dtype=bool)
    if min_opacity is not None:
        faint = ~opacity_mask(data, min_opacity)
        bad |= faint
        stats["faint"] = int(faint.sum())
    if max_gaussian_scale is not None:
        oversized = ~scale_mask(data, max_gaussian_scale)
        bad |= oversized
        stats["oversized"] = int(oversized.sum())

    # Region-gated: only cull floater candidates inside the workspace region
    # over the operation surface; the rest of the scene is untouched.
    in_region = region_mask(
        data,
        plane_json=plane_json,
        result_path=pose_path,
        scene_scale_path=scale_path,
        band_above=plane_band_above,
        band_below=plane_band_below,
        workspace_radius=workspace_radius,
    )
    trim = bad & in_region
    stats["scope"] = "region"
    stats["in_region"] = int(in_region.sum())

    mask = ~trim
    kept = int(mask.sum())
    stats["kept_count"] = kept
    stats["dropped_count"] = total - kept

    if kept == total:
        print("[trim] no gaussians removed (no filters set or nothing matched).")
    write_vertex_ply(output_ply, data[mask], byte_order=order)

    pct = 100.0 * (total - kept) / max(total, 1)
    print(f"[trim] input  = {input_ply}")
    print(f"[trim] output = {output_ply}")
    print(f"[trim] stats  = {stats}")
    print(f"[trim] kept {kept}/{total} gaussians (dropped {total - kept}, {pct:.1f}%)")
    return stats


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Trim options. Defaults are the ``trim_gaussian.sh`` production values."""
    parser.add_argument("--min-opacity", type=float, default=0.3,
                        help="Drop gaussians fainter than this (sigmoid-aware).")
    parser.add_argument("--max-gaussian-scale", type=float, default=0.05,
                        help="Drop gaussians whose max(exp(scale_*)) exceeds this (m).")
    parser.add_argument("--plane-band-above", type=float, default=0.15,
                        help="World-z band above the plane where culling is allowed (m).")
    parser.add_argument("--plane-band-below", type=float, default=0.05,
                        help="World-z band below the plane where culling is allowed (m).")
    parser.add_argument("--workspace-radius", type=float, default=0.70,
                        help="Horizontal radius around the plane centre where culling is allowed (m).")


def run(layout: SceneLayout, args: argparse.Namespace) -> dict[str, Any]:
    return trim_gaussians(
        layout.gaussian_ply,
        layout.gaussian_trimmed_ply,
        plane_json=layout.plane_fit_json,
        pose_path=layout.refined_pose,
        scale_path=layout.refined_scale,
        min_opacity=args.min_opacity,
        max_gaussian_scale=args.max_gaussian_scale,
        workspace_radius=args.workspace_radius,
        plane_band_above=args.plane_band_above,
        plane_band_below=args.plane_band_below,
    )


STEP = Step(
    name="surface.trim",
    help="Drop faint/oversized floater gaussians over the operation surface",
    add_arguments=add_arguments,
    run=run,
    inputs=lambda layout, args: [
        layout.gaussian_ply,
        layout.plane_fit_json,
        layout.refined_pose,
        layout.refined_scale,
    ],
    outputs=lambda layout, args: [layout.gaussian_trimmed_ply],
)
