"""Step 3b ``align.check``: how well the aligned scene matches the real sensor depth.

A check, independent of ``mesh_sheet.glb`` (align.scale's scale reference): the real
depth of the anchor frame against the posed collider, exactly as surface.plane and
the sim place it. Nothing downstream reads its outputs.

* ``IN/depth/1.npy`` is unprojected with ``intrinsics.npy`` and ``extrinsics.npy``
  (OpenCV world-to-camera) into world points, one per pixel;
* pixels on or next to objects are left out: the union of ``input_mask/<s>/*_mask.png``
  (objects and the robot/human union), dilated by ``--mask-dilate`` px, as are
  non-finite / non-positive depths;
* the collider (``r2s.common.transforms.collider_world_transform`` +
  ``r2s.common.mesh.mesh_world_geometry`` with ``glb_import_rotation``) is ray-cast
  from the camera through every pixel (Embree); pixels whose ray misses it are left out.

Reported in ``S/align/depth_check.json`` for all kept pixels, for ``workspace`` pixels
(sensor point inside the robot workspace box) and for ``table`` pixels (within
``--table-band`` of a plane fitted to the sensor's dominant horizontal surface in the
workspace):

* ``trimmed_chamfer``: ``0.5 (trim(sensor->collider) + trim(collider->sensor))``, trim =
  mean of the closest ``--trim-q`` (0.8); sensor->collider is the point-to-mesh distance,
  collider->sensor the distance from the collider's hit point to the nearest kept
  sensor point;
* ``sensor_to_collider_median``;
* ``depth_error_median``: signed, collider depth minus sensor depth along the camera
  z axis (> 0: the collider is behind the real surface), and its absolute median;
* ``sensor_optimal_scale``: ``s * median(z_sensor / z_collider)``. align.scale scales the
  scene about the camera centre, so every collider depth scales with ``s`` and this is
  the scale at which the per-pixel depths match (``fitted_scale`` next to it);
* ``table_height``: the sensor's table plane height against the collider's (same
  estimator on the collider hit points), and the scale that would match them.

``S/align/depth_check.png``: the frame (excluded pixels tinted) and the signed depth
error (+-``--error-range`` m; red = collider behind the sensor, blue = in front, grey =
excluded or no hit).

Method of ``runs/mesh_sheet_fill/analysis/depth_vs_collider.py`` (which evaluated the
collider at s = 1 and rescaled; here the collider is posed at the fitted scale), minus
its scale sweep and the mesh-sheet comparison.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from r2s.common.io import prepare_output, write_json
from r2s.common.mesh import mesh_world_geometry
from r2s.common.transforms import collider_world_transform, glb_import_rotation, load_pose, load_scene_scale
from r2s.layout import SceneLayout
from r2s.steps import Step

TRIM_Q = 0.8
TABLE_BAND_M = 0.02
TABLE_HIST_BINS = np.arange(-0.10, 0.60, 0.005)  # candidate table heights (m) for the histogram peak


def trim_mean(x: np.ndarray, q: float = TRIM_Q) -> float:
    """Mean of the smallest ``q`` fraction (at least one value)."""
    x = np.sort(np.asarray(x, dtype=np.float64))
    k = max(1, int(np.floor(q * len(x))))
    return float(x[:k].mean())


def object_mask_union(mask_dir: Path, shape: tuple[int, int], dilate: int) -> np.ndarray:
    """Union of every ``*_mask.png`` (value > 127), dilated ``dilate`` times (4-connected)."""
    from PIL import Image
    from scipy.ndimage import binary_dilation

    files = sorted(Path(mask_dir).glob("*_mask.png"))
    if not files:
        raise FileNotFoundError(f"no *_mask.png in {mask_dir}")
    union = np.zeros(shape, bool)
    for p in files:
        a = np.array(Image.open(p))
        a = a[..., 0] if a.ndim == 3 else a
        if a.shape != shape:
            raise ValueError(f"{p} is {a.shape[::-1]}, the depth is {shape[::-1]}")
        union |= a > 127
    return binary_dilation(union, iterations=dilate) if dilate > 0 else union


def camera_rays(intrinsics: Path, extrinsics: Path, hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(camera centre, R world-to-camera, per-pixel world directions with camera z = 1)``."""
    K = np.load(intrinsics).astype(np.float64)
    E = np.load(extrinsics).astype(np.float64).reshape(4, 4)
    R, t = E[:3, :3], E[:3, 3]
    H, W = hw
    u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    dcam = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
    return -R.T @ t, R, dcam @ R  # dcam @ R == (R.T @ d) per pixel


def raycast_depth(vertices: np.ndarray, faces: np.ndarray, cam: np.ndarray, R: np.ndarray,
                  dworld: np.ndarray) -> np.ndarray:
    """Camera-z depth of the first collider hit per pixel (NaN where the ray misses)."""
    import trimesh
    from trimesh.ray.ray_pyembree import RayMeshIntersector  # Embree (embreex)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    dirs = dworld.reshape(-1, 3)
    origins = np.broadcast_to(cam, dirs.shape)
    _, ray, loc = RayMeshIntersector(mesh).intersects_id(origins, dirs, multiple_hits=False, return_locations=True)
    z = np.full(len(dirs), np.nan)
    z[ray] = (np.asarray(loc) - cam[None]) @ R[2]
    return z.reshape(dworld.shape[:2])


def table_plane(points: np.ndarray, band: float = TABLE_BAND_M) -> dict[str, Any] | None:
    """Dominant horizontal surface: z-histogram peak, then a least-squares plane on +-``band`` of it."""
    if len(points) < 500:
        return None
    z = points[:, 2]
    hist, edges = np.histogram(z, bins=TABLE_HIST_BINS)
    peak = 0.5 * (edges[np.argmax(hist)] + edges[np.argmax(hist) + 1])
    sel = np.abs(z - peak) < band
    Q = points[sel]
    A = np.c_[Q[:, 0], Q[:, 1], np.ones(len(Q))]
    coef, *_ = np.linalg.lstsq(A, Q[:, 2], rcond=None)
    n = np.array([-coef[0], -coef[1], 1.0])
    n /= np.linalg.norm(n)
    centroid = Q[:, :2].mean(axis=0)
    return {
        "peak_z": float(peak),
        "plane_z_at_band_centroid": float(coef[0] * centroid[0] + coef[1] * centroid[1] + coef[2]),
        "band_centroid_xy": centroid.tolist(),
        "tilt_deg": float(np.degrees(np.arccos(abs(n[2])))),
        "band_points": int(sel.sum()),
        "plane_coef": [float(c) for c in coef],
    }


def _metrics(d_s2c: np.ndarray, d_c2s: np.ndarray, zerr: np.ndarray, ratio: np.ndarray, scale: float,
             q: float) -> dict[str, Any]:
    if len(d_s2c) == 0:
        return {"n_pixels": 0}
    return {
        "n_pixels": int(len(d_s2c)),
        "trimmed_chamfer": 0.5 * (trim_mean(d_s2c, q) + trim_mean(d_c2s, q)),
        "sensor_to_collider_median": float(np.median(d_s2c)),
        "sensor_to_collider_trim_mean": trim_mean(d_s2c, q),
        "collider_to_sensor_median": float(np.median(d_c2s)),
        "depth_error_median": float(np.median(zerr)),
        "depth_abs_error_median": float(np.median(np.abs(zerr))),
        "sensor_optimal_scale": float(scale * np.median(ratio)),
    }


def depth_check(
    *,
    depth_npy: Path,
    intrinsics: Path,
    extrinsics: Path,
    mask_dir: Path,
    collider_glb: Path,
    pose_path: Path,
    scale_path: Path,
    workspace: tuple[float, float, float, float, float, float],
    mask_dilate: int = 5,
    trim_q: float = TRIM_Q,
    table_band: float = TABLE_BAND_M,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Metrics (the JSON) and the per-pixel maps used for the review image."""
    import igl
    from scipy.spatial import cKDTree

    depth = np.load(depth_npy).astype(np.float64)
    H, W = depth.shape
    cam, R, dworld = camera_rays(intrinsics, extrinsics, (H, W))
    P = cam[None, None, :] + depth[..., None] * dworld
    finite = np.isfinite(depth) & (depth > 0)
    objmask = object_mask_union(mask_dir, (H, W), mask_dilate)
    bg = finite & ~objmask
    x0, x1, y0, y1, z0, z1 = workspace
    in_ws = (P[..., 0] > x0) & (P[..., 0] < x1) & (P[..., 1] > y0) & (P[..., 1] < y1) & (P[..., 2] > z0) & (P[..., 2] < z1)

    scale = load_scene_scale(scale_path)
    pos, rot, _ = collider_world_transform(pose_path, scale)
    V, F = mesh_world_geometry(collider_glb, scale, pos, rot, glb_import_rotation())
    zc = raycast_depth(V, F, cam, R, dworld)
    valid = bg & np.isfinite(zc)

    sensor_table = table_plane(P[bg & in_ws], table_band)
    table_px = np.zeros((H, W), bool)
    if sensor_table is not None:
        a, b, c = sensor_table["plane_coef"]
        table_px = valid & in_ws & (np.abs(P[..., 2] - (a * P[..., 0] + b * P[..., 1] + c)) < table_band)

    P_real = P[valid]
    d_s2c = np.sqrt(np.maximum(igl.point_mesh_squared_distance(P_real, V, F.astype(np.int64))[0], 0.0))
    Pc_all = cam[None, None, :] + np.where(np.isfinite(zc), zc, 0.0)[..., None] * dworld
    Pc = Pc_all[valid]
    d_c2s, _ = cKDTree(P_real).query(Pc, k=1, workers=-1)
    zerr = zc[valid] - depth[valid]
    ratio = depth[valid] / zc[valid]

    subsets = {"all": np.ones(len(P_real), bool), "workspace": in_ws[valid], "table": table_px[valid]}
    metrics = {k: _metrics(d_s2c[m], d_c2s[m], zerr[m], ratio[m], scale, trim_q) for k, m in subsets.items()}

    collider_ws = valid & (Pc_all[..., 0] > x0) & (Pc_all[..., 0] < x1) & (Pc_all[..., 1] > y0) & \
        (Pc_all[..., 1] < y1) & (Pc_all[..., 2] > z0) & (Pc_all[..., 2] < z1)
    collider_table = table_plane(Pc_all[collider_ws], table_band)
    height: dict[str, Any] = {"sensor": sensor_table, "collider": collider_table}
    if sensor_table is not None and collider_table is not None:
        cx, cy = sensor_table["band_centroid_xy"]
        a, b, c = collider_table["plane_coef"]
        zs, zk = sensor_table["plane_z_at_band_centroid"], a * cx + b * cy + c
        height.update({
            "sensor_z": zs,
            "collider_z": float(zk),  # the collider's plane at the sensor band's centroid
            "collider_minus_sensor": float(zk - zs),
            # collider heights move with the scale about the camera: z(s') = c_z + (s'/s)(z - c_z)
            "matching_scale": float(scale * (zs - cam[2]) / (zk - cam[2])),
        })
    if table_px.any():
        height["collider_z_median_at_table_pixels"] = float(np.median(Pc_all[table_px][:, 2]))

    pose, *_ = load_pose(pose_path)
    report = {
        "fitted_scale": float(scale),
        "pose": np.load(pose_path).astype(np.float64).reshape(-1).tolist(),
        "collider_pos": pose.tolist(),
        "camera_pos": cam.tolist(),
        "image_hw": [H, W],
        "workspace_bounds": list(workspace),
        "mask_dilate_px": int(mask_dilate),
        "trim_q": float(trim_q),
        "table_band_m": float(table_band),
        "object_mask_frac": float(objmask.mean()),
        "valid_frac": float(valid.mean()),
        "workspace_frac": float((valid & in_ws).mean()),
        "table_frac": float(table_px.mean()),
        **metrics,
        "table_height": height,
        "sensor_optimal_scale": {k: metrics[k].get("sensor_optimal_scale") for k in subsets},
    }
    error = np.full((H, W), np.nan)
    error[valid] = zerr
    return report, {"error": error, "excluded": ~valid, "table": table_px}


def review_image(rgb_path: Path, maps: dict[str, np.ndarray], out: Path, title: str, error_range: float) -> Path:
    """Frame (excluded pixels tinted red, table pixels green) | signed depth error."""
    from PIL import Image, ImageDraw

    err = maps["error"]
    rgb = np.array(Image.open(rgb_path).convert("RGB").resize(err.shape[::-1]))
    tinted = rgb.astype(np.float64)
    tinted[maps["excluded"]] = 0.45 * tinted[maps["excluded"]] + 0.55 * np.array([255.0, 0.0, 0.0])
    tinted[maps["table"]] = 0.6 * tinted[maps["table"]] + 0.4 * np.array([0.0, 255.0, 0.0])
    x = np.clip(np.nan_to_num(err / error_range), -1.0, 1.0)
    panel = np.ones(err.shape + (3,))
    pos, neg = x > 0, x < 0
    panel[pos, 1] -= x[pos]
    panel[pos, 2] -= x[pos]  # red: collider behind the sensor surface
    panel[neg, 0] += x[neg]
    panel[neg, 1] += x[neg]  # blue: collider in front
    panel[~np.isfinite(err)] = 0.3
    img = Image.fromarray(np.concatenate([tinted.astype(np.uint8), (np.clip(panel, 0, 1) * 255).astype(np.uint8)], 1))
    draw = ImageDraw.Draw(img)
    W = err.shape[1]
    labels = [(6, title + "  (red: excluded, green: table)"),
              (W + 6, f"collider - sensor depth, +-{100 * error_range:.0f} cm (red: collider behind, blue: in front)")]
    for x0, text in labels:
        draw.rectangle([x0 - 2, 2, x0 + 6 * len(text) + 4, 16], fill=(0, 0, 0))
        draw.text((x0, 4), text, fill=(255, 255, 0))
    img.save(prepare_output(out))
    return out


# --------------------------------------------------------------------------- the step
def output_json(layout: SceneLayout) -> Path:
    return layout.align_dir / "depth_check.json"


def output_png(layout: SceneLayout) -> Path:
    return layout.align_dir / "depth_check.png"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mask-dilate", type=int, default=5, help="px the object-mask union is grown by")
    parser.add_argument("--trim-q", type=float, default=TRIM_Q, help="fraction kept by the trimmed means")
    parser.add_argument("--table-band", type=float, default=TABLE_BAND_M, help="table pixels: within this of the plane (m)")
    parser.add_argument("--workspace-bounds", type=float, nargs=6, default=None, metavar=("X0", "X1", "Y0", "Y1", "Z0", "Z1"),
                        help="workspace box (default: the scene robot's, as surface.plane)")
    parser.add_argument("--error-range", type=float, default=0.25, help="colour range of the review image (m)")


def _workspace(layout: SceneLayout, args: argparse.Namespace) -> tuple[float, ...]:
    from r2s.surface.geometry import DEFAULT_WORKSPACES

    return tuple(args.workspace_bounds) if args.workspace_bounds else DEFAULT_WORKSPACES[layout.config.robot]


def _inputs(layout: SceneLayout, args: argparse.Namespace) -> list[Path]:
    io = layout.inputs
    return [io.depth_dir / "1.npy", io.images_dir / "1.jpg", io.intrinsics, io.extrinsics, io.input_mask_dir,
            layout.collider_glb, layout.refined_pose, layout.refined_scale]


def run(layout: SceneLayout, args: argparse.Namespace) -> dict[str, Any]:
    io = layout.inputs
    report, maps = depth_check(
        depth_npy=io.depth_dir / "1.npy",
        intrinsics=io.intrinsics,
        extrinsics=io.extrinsics,
        mask_dir=io.input_mask_dir,
        collider_glb=layout.collider_glb,
        pose_path=layout.refined_pose,
        scale_path=layout.refined_scale,
        workspace=_workspace(layout, args),
        mask_dilate=args.mask_dilate,
        trim_q=args.trim_q,
        table_band=args.table_band,
    )
    report = {"scene": layout.scene, **report}
    write_json(output_json(layout), report)
    review_image(io.images_dir / "1.jpg", maps, output_png(layout), f"align.check {layout.scene} s={report['fitted_scale']:.4f}",
                 args.error_range)
    a, t, h = report["all"], report["table"], report["table_height"]
    print(f"[align.check] {layout.scene}: s={report['fitted_scale']:.4f}  chamfer={a['trimmed_chamfer']:.4f} m  "
          f"sensor->collider median={a['sensor_to_collider_median']:.4f} m  depth err median={a['depth_error_median']:+.4f} m  "
          f"table depth err={t.get('depth_error_median', float('nan')):+.4f} m  "
          f"table z sensor/collider={h.get('sensor_z', float('nan')):+.4f}/{h.get('collider_z', float('nan')):+.4f} m  "
          f"sensor-optimal s={a['sensor_optimal_scale']:.4f} (table {t.get('sensor_optimal_scale', float('nan')):.4f})")
    return {
        "fitted_scale": report["fitted_scale"],
        "trimmed_chamfer": a["trimmed_chamfer"],
        "sensor_to_collider_median": a["sensor_to_collider_median"],
        "depth_error_median": a["depth_error_median"],
        "table_depth_error_median": t.get("depth_error_median"),
        "table_collider_minus_sensor": h.get("collider_minus_sensor"),
        "sensor_optimal_scale": report["sensor_optimal_scale"],
    }


STEP = Step(
    name="align.check",
    help="check the alignment against the real sensor depth (Chamfer, depth error, table height, sensor-optimal scale)",
    add_arguments=add_arguments,
    run=run,
    inputs=_inputs,
    outputs=lambda layout, args: [output_json(layout), output_png(layout)],
    needs="cpu",
)
