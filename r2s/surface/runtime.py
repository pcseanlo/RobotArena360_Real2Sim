"""Runtime helpers for instantiating precomputed operation-surface colliders (library only).

Used by sim assembly to turn a plane-fit (step 4) or render-receiver (step 6)
descriptor into Eden entity options: the selected box, its ``support_boxes``, or
the triangulated ``physics`` / ``lighting`` footprint polygon extruded into a thin
closed OBJ prism (cached under ``$TMPDIR/eden_operation_surface_meshes``).

Ported verbatim from the old ``operation_surface_runtime.py``; Eden is imported only
inside :func:`load_operation_surface_boxes`, so CPU steps never import it.
"""

from __future__ import annotations

import json
import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any


def _write_footprint_obj(
    footprint: dict[str, Any],
    *,
    z_offset: float,
    minimum_thickness: float | None = None,
) -> str:
    """Materialize a closed thin prism for one triangulated footprint."""
    vertices = footprint.get("vertices_world") or []
    triangles = footprint.get("triangles") or []
    if len(vertices) < 3 or not triangles:
        raise ValueError("Triangulated operation-surface footprint is empty")
    normal = footprint.get("plane_normal") or [0.0, 0.0, 1.0]
    thickness = max(
        float(footprint.get("thickness", 0.01)),
        float(minimum_thickness) if minimum_thickness is not None else 0.0,
        1e-4,
    )
    payload = json.dumps(
        {
            "mesh_schema": 4,
            "vertices": vertices,
            "triangles": triangles,
            "normal": normal,
            "thickness": thickness,
            "z_offset": z_offset,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:20]
    cache_dir = Path(tempfile.gettempdir()) / "eden_operation_surface_meshes"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"operation_surface_{digest}.obj"
    if path.exists():
        return str(path)

    import numpy as np

    top = np.asarray(vertices, dtype=np.float64)
    top[:, 2] += float(z_offset)
    n = np.asarray(normal, dtype=np.float64)
    n /= max(float(np.linalg.norm(n)), 1e-12)
    bottom = top - thickness * n[None, :]
    all_vertices = np.vstack((top, bottom))
    footprint_uv = np.asarray(footprint.get("vertices_uv") or [], dtype=np.float64)
    if footprint_uv.shape != (len(top), 2):
        basis_u = np.asarray(footprint.get("basis_u") or [1.0, 0.0, 0.0], dtype=np.float64)
        basis_v = np.asarray(footprint.get("basis_v") or [0.0, 1.0, 0.0], dtype=np.float64)
        plane_origin = np.asarray(footprint.get("plane_origin") or [0.0, 0.0, 0.0], dtype=np.float64)
        relative = top - plane_origin[None, :]
        footprint_uv = np.column_stack((relative @ basis_u, relative @ basis_v))
    uv_lo = footprint_uv.min(axis=0)
    uv_span = np.maximum(footprint_uv.max(axis=0) - uv_lo, 1e-9)
    texture_uv = (footprint_uv - uv_lo[None, :]) / uv_span[None, :]
    # Rectified table textures store v-min in their first (top) image row,
    # whereas OBJ uses v=0 at the bottom.
    texture_uv[:, 1] = 1.0 - texture_uv[:, 1]
    top_triangles: list[tuple[int, int, int]] = []
    for tri in triangles:
        a, b, c = (int(index) for index in tri)
        signed_area = float(np.dot(np.cross(top[b] - top[a], top[c] - top[a]), n))
        if abs(signed_area) <= 1e-12:
            raise ValueError(f"Degenerate operation-surface triangle: {(a, b, c)}")
        # The fitted contour may be clockwise. Nyx uses mesh normals for PBR,
        # so every visible top face must point along the stabilized plane normal.
        if signed_area < 0.0:
            b, c = c, b
        top_triangles.append((a, b, c))
    vertex_count = len(top)

    edge_counts: dict[tuple[int, int], int] = {}
    for a, b, c in top_triangles:
        for start, end in ((a, b), (b, c), (c, a)):
            key = (min(start, end), max(start, end))
            edge_counts[key] = edge_counts.get(key, 0) + 1

    faces: list[tuple[int, int, int]] = list(top_triangles)
    faces.extend((c + vertex_count, b + vertex_count, a + vertex_count) for a, b, c in top_triangles)
    for (a, b), count in edge_counts.items():
        if count != 1:
            continue
        faces.append((a, a + vertex_count, b + vertex_count))
        faces.append((a, b + vertex_count, b))

    lines = ["# generated operation-surface footprint"]
    lines.extend(f"v {x:.9g} {y:.9g} {z:.9g}" for x, y, z in all_vertices)
    lines.extend(f"vt {u:.9g} {v:.9g}" for u, v in texture_uv)
    lines.extend(
        "f " + " ".join(f"{index + 1}/{index % vertex_count + 1}" for index in face)
        for face in faces
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def load_operation_surface_boxes(
    descriptor_path: str | None,
    *,
    mode: str = "solid",
    name_prefix: str = "operation_surface",
    visualization: bool = False,
    collision: bool = True,
    surface: Any | None = None,
    z_offset: float = 0.0,
    collision_thickness: float | None = None,
    contact_sol_params: dict[str, tuple[float, ...]] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Load operation-surface collider geometry from a fitter JSON descriptor."""
    if descriptor_path is None:
        return {}, []
    if mode not in {"solid", "support_boxes", "physics_polygon", "lighting_polygon", "auto"}:
        raise ValueError(f"Unknown operation-surface mode: {mode}")
    if not os.path.exists(descriptor_path):
        raise FileNotFoundError(f"Operation-surface descriptor not found: {descriptor_path}")
    if collision_thickness is not None and float(collision_thickness) <= 0.0:
        raise ValueError("collision_thickness must be positive when provided")

    with open(descriptor_path, "r", encoding="utf-8") as handle:
        descriptor: dict[str, Any] = json.load(handle)
    selected = descriptor.get("selected")
    if selected is None:
        raise ValueError(f"Operation-surface descriptor has no selected candidate: {descriptor_path}")
    from eden.options.entities import BoxOptions, EntityOptions

    footprints = selected.get("footprints") or {}
    footprint_role = None
    if mode == "physics_polygon":
        footprint_role = "physics"
    elif mode in {"lighting_polygon", "auto"}:
        footprint_role = "lighting"
    footprint = footprints.get(footprint_role) if footprint_role else None
    if footprint is not None:
        try:
            mesh_path = _write_footprint_obj(
                footprint,
                z_offset=float(z_offset),
                minimum_thickness=(
                    float(collision_thickness)
                    if collision and collision_thickness is not None
                    else None
                ),
            )
            kwargs: dict[str, Any] = {}
            if surface is not None:
                kwargs["surface"] = surface
            if collision and contact_sol_params:
                kwargs["default_geoms_sol_params"] = dict(contact_sol_params)
            entity = EntityOptions(
                file=mesh_path,
                up=(0, 0, 1),
                front=(1, 0, 0),
                scale=1.0,
                default_root_pos=(0.0, 0.0, 0.0),
                default_root_quat=(1.0, 0.0, 0.0, 0.0),
                is_fixed_base=True,
                visualization=visualization,
                collision=collision,
                decimate=False,
                convexify=bool(collision),
                decompose_object_error_threshold=0.0,
                coacd_threshold=0.03,
                coacd_preprocess_resolution=150,
                **kwargs,
            )
            return {name_prefix: entity}, [name_prefix]
        except (KeyError, TypeError, ValueError):
            # A malformed v2 footprint must not break legacy descriptors/runs.
            pass

    support_boxes = [selected] if mode == "solid" else selected.get("support_boxes") or [selected]
    entities: dict[str, Any] = {}
    names: list[str] = []
    for index, box in enumerate(support_boxes):
        name = name_prefix if index == 0 else f"{name_prefix}_{index:02d}"
        center = tuple(float(v) for v in box["center"])
        if z_offset:
            center = (center[0], center[1], center[2] + float(z_offset))
        kwargs: dict[str, Any] = {}
        if not collision:
            kwargs["collision"] = False
        if surface is not None:
            kwargs["surface"] = surface
        if collision and contact_sol_params:
            kwargs["default_geoms_sol_params"] = dict(contact_sol_params)
        entities[name] = BoxOptions(
            size=tuple(float(v) for v in box["size"]),
            default_root_pos=center,
            default_root_quat=tuple(float(v) for v in box["quat_wxyz"]),
            is_fixed_base=True,
            visualization=visualization,
            **kwargs,
        )
        names.append(name)
    return entities, names
