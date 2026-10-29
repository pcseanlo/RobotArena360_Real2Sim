"""Step 7 ``collision.crop``: crop the background collider to the robot-reachable workspace.

Port of the old ``collision_mesh_crop.py``. The old repo only ran this inside the
evaluator (~L6495), which cached the crop next
to the pose file (``optimization_results/<s>/cropped_collision/``). Here it is a
step with explicit inputs:

* ``layout.collider_glb`` (``echo2/<s>/collider.glb``)
* ``layout.refined_pose`` / ``layout.refined_scale`` (the evaluator's ``--result`` /
  ``--scene_scale``, i.e. ``aligned_pose_refined.npy`` and
  ``aligned_pose_scene_scale_refined.npy``)

and writes ``layout.cropped_collider_glb`` plus its ``.glb.json`` sidecar.

The collider is placed exactly as the evaluator places it:
``root_pos = (tx, ty, tz)`` (not divided by the scene scale),
``root_quat = q_rel(rz[, rx, ry]) * orig_collider_quat`` (Hamilton product, *not*
re-normalised, as ``apply_relative_transform_to_quat`` in the evaluator; the
``orig_collider_quat`` is Genesis' normalised product identity x ADJUSTMENT, see
:func:`genesis_quat_mul`), GLB axes
``up=(0,1,0)``, ``front=(1,0,0)``, ``scale = scene_scale``. The crop is a vertical
24-sided prism of radius 1.10 m around the robot base at the world origin, from
z = -0.15 to 1.45 m (evaluator defaults ``--collision-crop-*``, never overridden by
the production launchers).

The sim assembly calls :func:`build_reachable_collision_mesh` (or
:func:`load_crop_result` for a crop this step already wrote) to get the mesh path
plus the load pose (baked world space: identity scale/position, the GLB-worldspace
quaternion and z-up axes).
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from r2s.collision.support import (
    DEFAULT_GLB_FRONT,
    DEFAULT_GLB_UP,
    align_up_and_front,
    load_combined_mesh,
    transform_vertices,
)
from r2s.common.io import prepare_output
from r2s.common.quat import axis_quat_np, euler_zyx_to_quat, quat_mul_np
from r2s.common.transforms import SCALE_OPT_ADJUSTMENT, load_pose, load_scene_scale
from r2s.layout import SceneLayout
from r2s.steps import Step

IDENTITY_QUAT_WXYZ = (1.0, 0.0, 0.0, 0.0)
GLB_WORLDSPACE_LOAD_QUAT_WXYZ = (0.7071067811865476, -0.7071067811865475, 0.0, 0.0)
IDENTITY_UP = (0, 0, 1)
IDENTITY_FRONT = (1, 0, 0)

# Production crop (evaluator defaults --collision-crop-radius/-z-min/-z-max/-sides).
DEFAULT_RADIUS = 1.10
DEFAULT_Z_MIN = -0.15
DEFAULT_Z_MAX = 1.45
DEFAULT_SIDES = 24
DEFAULT_MIN_FACES = 4
DEFAULT_SURFACE_MASK_VERTICAL_BAND = 0.05


@dataclass(frozen=True)
class CollisionMeshCropResult:
    mesh_path: str
    original_mesh_path: str
    used_crop: bool
    load_scale: float
    load_pos: tuple[float, float, float]
    load_quat_wxyz: tuple[float, float, float, float]
    load_up: tuple[float, float, float]
    load_front: tuple[float, float, float]
    baked_world_space: bool
    original_face_count: int
    cropped_face_count: int
    bounds_min: tuple[float, float, float] | None
    bounds_max: tuple[float, float, float] | None
    reason: str


def _safe_float_token(value: float) -> str:
    return f"{float(value):.3f}".replace("-", "m").replace(".", "p")


def crop_glb_name(
    radius: float = DEFAULT_RADIUS,
    z_min: float = DEFAULT_Z_MIN,
    z_max: float = DEFAULT_Z_MAX,
    sides: int = DEFAULT_SIDES,
) -> str:
    """File name of the cached crop (``reachable_collider_r1p100_zm0p150_1p450_s24.glb`` in production)."""
    return (
        "reachable_collider_"
        f"r{_safe_float_token(radius)}_"
        f"z{_safe_float_token(z_min)}_{_safe_float_token(z_max)}_"
        f"s{int(sides)}.glb"
    )


def _jsonable_array(values: Any) -> list[float]:
    return [float(v) for v in np.asarray(values, dtype=np.float64).reshape(-1)]


def _vec3_tuple(values: Any) -> tuple[float, float, float]:
    vals = _jsonable_array(values)
    if len(vals) != 3:
        raise ValueError(f"Expected 3 values, got {len(vals)}")
    return (vals[0], vals[1], vals[2])


def _quat_tuple(values: Any) -> tuple[float, float, float, float]:
    vals = _jsonable_array(values)
    if len(vals) != 4:
        raise ValueError(f"Expected 4 quaternion values, got {len(vals)}")
    return (vals[0], vals[1], vals[2], vals[3])


def _metadata_matches(path: str, metadata: dict[str, Any]) -> bool:
    meta_path = path + ".json"
    if not os.path.exists(path) or not os.path.exists(meta_path):
        return False
    try:
        with open(meta_path, "r", encoding="utf-8") as handle:
            existing = json.load(handle)
    except Exception:
        return False
    return existing == metadata


def _write_metadata(path: str, metadata: dict[str, Any]) -> None:
    meta_path = prepare_output(path + ".json")
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


def _export_mesh(mesh, path: str) -> None:
    mesh.export(str(prepare_output(path)))


def _mesh_bounds(mesh) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    bounds = np.asarray(mesh.bounds, dtype=np.float64).reshape(2, 3)
    return tuple(float(v) for v in bounds[0]), tuple(float(v) for v in bounds[1])


def _clean_mesh(mesh):
    if mesh is None:
        return None
    if len(mesh.faces) == 0 or len(mesh.vertices) == 0:
        return None
    try:
        nondegenerate = mesh.nondegenerate_faces()
        mesh.update_faces(nondegenerate)
    except Exception:
        pass
    try:
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass
    if len(mesh.faces) == 0 or len(mesh.vertices) == 0:
        return None
    return mesh


def _clip_polygon_to_halfspace(
    polygon: list[np.ndarray],
    *,
    plane_origin: np.ndarray,
    plane_normal: np.ndarray,
    eps: float = 1e-9,
) -> list[np.ndarray]:
    if not polygon:
        return []

    def signed_distance(point: np.ndarray) -> float:
        return float(np.dot(point - plane_origin, plane_normal))

    clipped: list[np.ndarray] = []
    prev = polygon[-1]
    prev_dist = signed_distance(prev)
    prev_inside = prev_dist >= -eps
    for curr in polygon:
        curr_dist = signed_distance(curr)
        curr_inside = curr_dist >= -eps

        if curr_inside != prev_inside:
            denom = prev_dist - curr_dist
            if abs(denom) > eps:
                t = prev_dist / denom
                clipped.append(prev + t * (curr - prev))
        if curr_inside:
            clipped.append(curr)

        prev = curr
        prev_dist = curr_dist
        prev_inside = curr_inside
    return clipped


def _clip_mesh_to_halfspace(mesh, *, plane_origin: np.ndarray, plane_normal: np.ndarray):
    import trimesh

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    new_vertices: list[np.ndarray] = []
    new_faces: list[list[int]] = []

    for face in faces:
        polygon = [vertices[int(i)] for i in face]
        clipped = _clip_polygon_to_halfspace(
            polygon,
            plane_origin=plane_origin,
            plane_normal=plane_normal,
        )
        if len(clipped) < 3:
            continue
        base_idx = len(new_vertices)
        new_vertices.extend(clipped)
        for i in range(1, len(clipped) - 1):
            new_faces.append([base_idx, base_idx + i, base_idx + i + 1])

    if not new_faces:
        return None
    clipped_mesh = trimesh.Trimesh(
        vertices=np.asarray(new_vertices, dtype=np.float64),
        faces=np.asarray(new_faces, dtype=np.int64),
        process=False,
    )
    try:
        clipped_mesh.merge_vertices(digits_vertex=10)
    except Exception:
        pass
    return _clean_mesh(clipped_mesh)


def transform_mesh_to_world(
    mesh_path: str,
    *,
    scene_scale: float,
    root_pos: np.ndarray | tuple[float, float, float],
    root_quat_wxyz: np.ndarray | tuple[float, float, float, float],
    mesh_up: tuple[float, float, float] = DEFAULT_GLB_UP,
    mesh_front: tuple[float, float, float] = DEFAULT_GLB_FRONT,
):
    """Load a mesh and bake Eden's import/root transform into its vertices."""
    import trimesh

    mesh = load_combined_mesh(mesh_path)
    if len(mesh.faces) == 0:
        raise ValueError(f"Collision mesh has no faces: {mesh_path}")

    mesh_import_rot = align_up_and_front(np.asarray(mesh_up), np.asarray(mesh_front))
    vertices_world = transform_vertices(
        np.asarray(mesh.vertices, dtype=np.float64),
        scale=float(scene_scale),
        root_pos=root_pos,
        root_quat_wxyz=root_quat_wxyz,
        mesh_import_rot=mesh_import_rot,
    )
    return trimesh.Trimesh(
        vertices=vertices_world,
        faces=np.asarray(mesh.faces, dtype=np.int64),
        process=False,
    )


def crop_mesh_to_reachable_cylinder(
    mesh,
    *,
    radius: float = DEFAULT_RADIUS,
    z_min: float = DEFAULT_Z_MIN,
    z_max: float = DEFAULT_Z_MAX,
    sides: int = DEFAULT_SIDES,
):
    """Clip a world-space mesh by a vertical regular-polygon cylinder around the world origin."""
    if radius <= 0.0:
        raise ValueError(f"collision crop radius must be positive, got {radius}")
    if z_max <= z_min:
        raise ValueError(f"collision crop z_max must be greater than z_min, got {z_min}, {z_max}")
    if sides < 4:
        raise ValueError(f"collision crop sides must be >= 4, got {sides}")

    clipped = mesh.copy()
    # The side planes use the polygon apothem so the polygon vertices lie on
    # the requested radius, keeping the crop inside the nominal reach radius.
    apothem = float(radius) * math.cos(math.pi / int(sides))
    for i in range(int(sides)):
        theta = 2.0 * math.pi * i / int(sides)
        outward = np.array([math.cos(theta), math.sin(theta), 0.0], dtype=np.float64)
        inward = -outward
        origin = outward * apothem
        clipped = _clip_mesh_to_halfspace(
            clipped,
            plane_origin=origin,
            plane_normal=inward,
        )
        clipped = _clean_mesh(clipped)
        if clipped is None:
            return None

    clipped = _clip_mesh_to_halfspace(
        clipped,
        plane_origin=np.array([0.0, 0.0, float(z_min)], dtype=np.float64),
        plane_normal=np.array([0.0, 0.0, 1.0], dtype=np.float64),
    )
    clipped = _clean_mesh(clipped)
    if clipped is None:
        return None

    clipped = _clip_mesh_to_halfspace(
        clipped,
        plane_origin=np.array([0.0, 0.0, float(z_max)], dtype=np.float64),
        plane_normal=np.array([0.0, 0.0, -1.0], dtype=np.float64),
    )
    return _clean_mesh(clipped)


def _points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Vectorized even/odd point-in-polygon test in plane-local UV coordinates."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    polygon = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    if len(polygon) < 3:
        return np.zeros(len(points), dtype=bool)
    x, y = points[:, 0], points[:, 1]
    inside = np.zeros(len(points), dtype=bool)
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        crosses = (yi > y) != (yj > y)
        x_cross = (xj - xi) * (y - yi) / (yj - yi + 1e-15) + xi
        inside ^= crosses & (x < x_cross)
        j = i
    return inside


def mask_support_faces_under_footprint(
    mesh,
    footprint: dict[str, Any],
    *,
    vertical_band: float = DEFAULT_SURFACE_MASK_VERTICAL_BAND,
    minimum_normal_alignment: float = 0.75,
):
    """Remove upward background support faces covered by a fitted surface.

    This operates on an already world-space collision mesh. Only nearly
    coplanar upward-facing faces whose centers lie inside the padded physics
    footprint are removed; walls and other scene collision remain untouched.
    """
    vertices_uv = footprint.get("boundary_uv") or footprint.get("vertices_uv")
    if not vertices_uv:
        raise ValueError("Physics footprint has no boundary_uv/vertices_uv polygon")
    origin = np.asarray(footprint["plane_origin"], dtype=np.float64)
    normal = np.asarray(footprint["plane_normal"], dtype=np.float64)
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    basis_u = np.asarray(footprint["basis_u"], dtype=np.float64)
    basis_v = np.asarray(footprint["basis_v"], dtype=np.float64)
    centers = np.asarray(mesh.triangles_center, dtype=np.float64)
    face_normals = np.asarray(mesh.face_normals, dtype=np.float64)
    rel = centers - origin
    distance = rel @ normal
    uv = np.column_stack((rel @ basis_u, rel @ basis_v))
    covered = _points_in_polygon(uv, np.asarray(vertices_uv, dtype=np.float64))
    covered &= np.abs(distance) <= float(vertical_band)
    covered &= (face_normals @ normal) >= float(minimum_normal_alignment)
    masked = mesh.copy()
    masked.update_faces(~covered)
    masked = _clean_mesh(masked)
    if masked is None:
        raise ValueError("Background support masking removed the entire collision mesh")
    return masked, int(covered.sum())


def _baked_result(
    out_path: str,
    original_path: str,
    mesh,
    *,
    original_face_count: int,
    reason: str,
) -> CollisionMeshCropResult:
    bounds_min, bounds_max = _mesh_bounds(mesh)
    return CollisionMeshCropResult(
        mesh_path=out_path,
        original_mesh_path=original_path,
        used_crop=True,
        load_scale=1.0,
        load_pos=(0.0, 0.0, 0.0),
        load_quat_wxyz=GLB_WORLDSPACE_LOAD_QUAT_WXYZ,
        load_up=IDENTITY_UP,
        load_front=IDENTITY_FRONT,
        baked_world_space=True,
        original_face_count=original_face_count,
        cropped_face_count=int(len(mesh.faces)),
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        reason=reason,
    )


def build_surface_masked_collision_mesh(
    crop_result: CollisionMeshCropResult,
    *,
    receiver_descriptor_path: str | Path,
    cache_dir: str | Path,
    vertical_band: float = DEFAULT_SURFACE_MASK_VERTICAL_BAND,
    use_cache: bool = True,
) -> CollisionMeshCropResult:
    """Cache a cropped collider with duplicate table support faces removed.

    Opt-in in production (``--background-operation-surface-mask``, off in every
    launcher). Writes ``<crop stem>_surface_masked.glb`` + ``.glb.json`` in ``cache_dir``.
    """
    import trimesh

    if not crop_result.baked_world_space:
        raise ValueError("Surface masking requires a baked world-space collision crop")
    receiver_descriptor_path = str(receiver_descriptor_path)
    with open(receiver_descriptor_path, "r", encoding="utf-8") as handle:
        descriptor = json.load(handle)
    footprint = ((descriptor.get("selected") or {}).get("footprints") or {}).get("physics")
    if not footprint:
        raise ValueError("Receiver descriptor has no selected physics footprint")
    os.makedirs(cache_dir, exist_ok=True)
    source_stat = os.stat(crop_result.mesh_path)
    descriptor_stat = os.stat(receiver_descriptor_path)
    metadata = {
        "version": 1,
        "source": os.path.abspath(crop_result.mesh_path),
        "source_mtime_ns": int(source_stat.st_mtime_ns),
        "descriptor": os.path.abspath(receiver_descriptor_path),
        "descriptor_mtime_ns": int(descriptor_stat.st_mtime_ns),
        "vertical_band": float(vertical_band),
    }
    stem = os.path.splitext(os.path.basename(crop_result.mesh_path))[0]
    out_path = os.path.abspath(os.path.join(str(cache_dir), f"{stem}_surface_masked.glb"))

    if use_cache and _metadata_matches(out_path, metadata):
        masked = trimesh.load(out_path, force="mesh", process=False)
        reason = "surface_mask_cache"
    else:
        source = trimesh.load(crop_result.mesh_path, force="mesh", process=False)
        masked, removed_count = mask_support_faces_under_footprint(
            source,
            footprint,
            vertical_band=vertical_band,
        )
        _export_mesh(masked, out_path)
        _write_metadata(out_path, metadata)
        reason = f"surface_mask_built:{removed_count}"
    return _baked_result(
        out_path,
        crop_result.original_mesh_path,
        masked,
        original_face_count=crop_result.original_face_count,
        reason=reason,
    )


def build_reachable_collision_mesh(
    collider_path: str | Path,
    *,
    scene_scale: float,
    root_pos: np.ndarray | tuple[float, float, float],
    root_quat_wxyz: np.ndarray | tuple[float, float, float, float],
    cache_dir: str | Path,
    source_mesh_up: np.ndarray | tuple[float, float, float] = DEFAULT_GLB_UP,
    source_mesh_front: np.ndarray | tuple[float, float, float] = DEFAULT_GLB_FRONT,
    radius: float = DEFAULT_RADIUS,
    z_min: float = DEFAULT_Z_MIN,
    z_max: float = DEFAULT_Z_MAX,
    sides: int = DEFAULT_SIDES,
    min_faces: int = DEFAULT_MIN_FACES,
    use_cache: bool = True,
) -> CollisionMeshCropResult:
    """Build or reuse a cropped world-space collision mesh in ``cache_dir``.

    Same algorithm, cache key and file names as the old evaluator. A missing
    collider is an error (the old Eden-asset lookup is gone). When the crop is
    empty or fails, returns ``used_crop=False`` with ``mesh_path`` pointing at the
    original collider, as the evaluator did (``collision.crop`` turns that into an
    error).
    """
    original_path = str(collider_path)
    if not os.path.isfile(original_path):
        raise FileNotFoundError(f"collider not found: {original_path}")
    cache_dir = str(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    source_stat = os.stat(original_path)
    original_load_pos = _vec3_tuple(root_pos)
    original_load_quat = _quat_tuple(root_quat_wxyz)
    original_load_up = _vec3_tuple(source_mesh_up)
    original_load_front = _vec3_tuple(source_mesh_front)
    metadata = {
        "version": 4,
        "source": os.path.abspath(original_path),
        "source_mtime_ns": int(source_stat.st_mtime_ns),
        "source_size": int(source_stat.st_size),
        "baked_world_space": True,
        "scene_scale": float(scene_scale),
        "root_pos": list(original_load_pos),
        "root_quat_wxyz": list(original_load_quat),
        "source_mesh_up": list(original_load_up),
        "source_mesh_front": list(original_load_front),
        "radius": float(radius),
        "z_min": float(z_min),
        "z_max": float(z_max),
        "sides": int(sides),
    }
    out_path = os.path.abspath(os.path.join(cache_dir, crop_glb_name(radius, z_min, z_max, sides)))

    original_face_count = 0
    try:
        if use_cache and _metadata_matches(out_path, metadata):
            import trimesh

            cached = trimesh.load(out_path, force="mesh", process=False)
            return _baked_result(out_path, original_path, cached, original_face_count=-1, reason="cache")

        world_mesh = transform_mesh_to_world(
            original_path,
            scene_scale=scene_scale,
            root_pos=root_pos,
            root_quat_wxyz=root_quat_wxyz,
            mesh_up=original_load_up,
            mesh_front=original_load_front,
        )
        original_face_count = int(len(world_mesh.faces))
        cropped = crop_mesh_to_reachable_cylinder(
            world_mesh,
            radius=radius,
            z_min=z_min,
            z_max=z_max,
            sides=sides,
        )
        if cropped is None or len(cropped.faces) < int(min_faces):
            return CollisionMeshCropResult(
                mesh_path=original_path,
                original_mesh_path=original_path,
                used_crop=False,
                load_scale=float(scene_scale),
                load_pos=original_load_pos,
                load_quat_wxyz=original_load_quat,
                load_up=original_load_up,
                load_front=original_load_front,
                baked_world_space=False,
                original_face_count=original_face_count,
                cropped_face_count=0 if cropped is None else int(len(cropped.faces)),
                bounds_min=None,
                bounds_max=None,
                reason="empty_crop",
            )

        _export_mesh(cropped, out_path)
        _write_metadata(out_path, metadata)
        return _baked_result(out_path, original_path, cropped, original_face_count=original_face_count, reason="built")
    except Exception as exc:
        return CollisionMeshCropResult(
            mesh_path=original_path,
            original_mesh_path=original_path,
            used_crop=False,
            load_scale=float(scene_scale),
            load_pos=original_load_pos,
            load_quat_wxyz=original_load_quat,
            load_up=original_load_up,
            load_front=original_load_front,
            baked_world_space=False,
            original_face_count=original_face_count,
            cropped_face_count=0,
            bounds_min=None,
            bounds_max=None,
            reason=f"failed: {exc}",
        )


# ---------------------------------------------------------------------------
# Evaluator-equivalent collider pose
# ---------------------------------------------------------------------------
def genesis_quat_mul(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """``genesis.utils.geom._np_quat_mul(u, v)`` for one wxyz pair (Genesis 1.4.4), in plain numpy.

    Genesis multiplies with the 8-product formula and normalises the result, so
    ``transform_quat_by_quat(identity, ADJUSTMENT)`` is *not* bit-identical to
    ``ADJUSTMENT`` (it is ``[0.017452406437283522, -6.9e-18, 0.9998476951563913, 0]``).
    The old evaluator built ``orig_collider_quat`` this way; the production crop
    sidecars match this formula, not ``ADJUSTMENT`` itself.
    """
    w1, x1, y1, z1 = (float(c) for c in np.asarray(u, dtype=np.float64).reshape(4))
    w2, x2, y2, z2 = (float(c) for c in np.asarray(v, dtype=np.float64).reshape(4))
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    out = np.array(
        [
            qq - ww + (z1 - y1) * (y2 - z2),
            qq - xx + (x1 + w1) * (x2 + w2),
            qq - yy + (w1 - x1) * (y2 + z2),
            qq - zz + (z1 + y1) * (w2 - x2),
        ],
        dtype=np.float64,
    )
    out /= np.sqrt(np.sum(np.square(out)))
    return out


def evaluator_orig_collider_quat() -> np.ndarray:
    """``transform_quat_by_quat(np.array([1,0,0,0]), ADJUSTMENT)`` = ``_np_quat_mul(ADJUSTMENT, identity)``."""
    return genesis_quat_mul(np.asarray(SCALE_OPT_ADJUSTMENT, dtype=np.float64), np.array([1.0, 0.0, 0.0, 0.0]))


def evaluator_collider_pose(
    pose_path: str | Path, scale_path: str | Path
) -> tuple[tuple[float, float, float], tuple[float, float, float, float], float]:
    """``(collider_pos, collider_quat_wxyz, scene_scale)`` exactly as the old evaluator computes them.

    ``orig_collider_quat`` comes from Genesis' quaternion product (see
    :func:`genesis_quat_mul`); the evaluator's ``apply_relative_transform_to_quat``
    then returns the raw Hamilton product ``q_rel * orig`` without normalising
    (unlike ``r2s.common.transforms.apply_relative_transform_to_quat``). Both are
    repeated here so the crop sidecar and mesh are bit-identical to production.
    """
    t, rx, ry, rz = load_pose(pose_path)
    scene_scale = load_scene_scale(scale_path)
    if rx == 0.0 and ry == 0.0:
        q_rel = axis_quat_np(rz, 2)
    else:
        q_rel = euler_zyx_to_quat(rx, ry, rz)
    quat = quat_mul_np(q_rel, evaluator_orig_collider_quat())
    pos = (float(t[0]), float(t[1]), float(t[2]))
    return pos, tuple(float(v) for v in quat), float(scene_scale)


def crop_collider(
    collider_glb: str | Path,
    pose_path: str | Path,
    scale_path: str | Path,
    out_dir: str | Path,
    *,
    radius: float = DEFAULT_RADIUS,
    z_min: float = DEFAULT_Z_MIN,
    z_max: float = DEFAULT_Z_MAX,
    sides: int = DEFAULT_SIDES,
    min_faces: int = DEFAULT_MIN_FACES,
    use_cache: bool = False,
) -> CollisionMeshCropResult:
    """Place the collider like the evaluator and crop it into ``out_dir``. Raises if the crop fails."""
    pos, quat, scene_scale = evaluator_collider_pose(pose_path, scale_path)
    result = build_reachable_collision_mesh(
        collider_glb,
        scene_scale=scene_scale,
        root_pos=pos,
        root_quat_wxyz=quat,
        cache_dir=out_dir,
        source_mesh_up=(0, 1, 0),
        source_mesh_front=(1, 0, 0),
        radius=radius,
        z_min=z_min,
        z_max=z_max,
        sides=sides,
        min_faces=min_faces,
        use_cache=use_cache,
    )
    if not result.used_crop:
        raise RuntimeError(f"collision crop failed for {collider_glb}: {result.reason}")
    return result


def load_crop_result(crop_glb: str | Path, *, original_mesh_path: str | None = None) -> CollisionMeshCropResult:
    """Result record for a crop already on disk (what the sim assembly loads)."""
    import trimesh

    crop_glb = Path(crop_glb)
    if not crop_glb.is_file():
        raise FileNotFoundError(f"cropped collider not found: {crop_glb}")
    meta_path = crop_glb.with_name(crop_glb.name + ".json")
    if original_mesh_path is None and meta_path.is_file():
        original_mesh_path = json.loads(meta_path.read_text()).get("source")
    mesh = trimesh.load(str(crop_glb), force="mesh", process=False)
    return _baked_result(str(crop_glb.absolute()), str(original_mesh_path or ""), mesh, original_face_count=-1, reason="loaded")


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------
def _out_glb(layout: SceneLayout, args: argparse.Namespace) -> Path:
    return layout.collision_dir / crop_glb_name(args.radius, args.z_min, args.z_max, args.sides)


def _masked_glb(layout: SceneLayout, args: argparse.Namespace) -> Path:
    stem = Path(crop_glb_name(args.radius, args.z_min, args.z_max, args.sides)).stem
    return layout.collision_dir / f"{stem}_surface_masked.glb"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--radius", type=float, default=DEFAULT_RADIUS, help="crop radius (m) around the robot base")
    parser.add_argument("--z-min", type=float, default=DEFAULT_Z_MIN, help="lowest world z kept (m)")
    parser.add_argument("--z-max", type=float, default=DEFAULT_Z_MAX, help="highest world z kept (m)")
    parser.add_argument("--sides", type=int, default=DEFAULT_SIDES, help="sides of the polygonal crop prism")
    parser.add_argument("--min-faces", type=int, default=DEFAULT_MIN_FACES, help="fewer faces than this = failed crop")
    parser.add_argument(
        "--surface-mask",
        action="store_true",
        help="also write the copy with duplicate table faces under the v2 physics footprint removed "
        "(old --background-operation-surface-mask; off in production)",
    )
    parser.add_argument("--surface-mask-vertical-band", type=float, default=DEFAULT_SURFACE_MASK_VERTICAL_BAND)


def run(layout: SceneLayout, args: argparse.Namespace) -> dict[str, Any]:
    result = crop_collider(
        layout.collider_glb,
        layout.refined_pose,
        layout.refined_scale,
        layout.collision_dir,
        radius=args.radius,
        z_min=args.z_min,
        z_max=args.z_max,
        sides=args.sides,
        min_faces=args.min_faces,
    )
    stats: dict[str, Any] = {
        "faces": [result.original_face_count, result.cropped_face_count],
        "bounds_min": result.bounds_min,
        "bounds_max": result.bounds_max,
    }
    print(
        f"[collision-crop] {layout.scene}: faces {result.original_face_count} -> {result.cropped_face_count} "
        f"bounds {result.bounds_min} .. {result.bounds_max} -> {result.mesh_path}"
    )
    if args.surface_mask:
        masked = build_surface_masked_collision_mesh(
            result,
            receiver_descriptor_path=layout.receiver_json,
            cache_dir=layout.collision_dir,
            vertical_band=args.surface_mask_vertical_band,
            use_cache=False,
        )
        stats["surface_mask"] = {"reason": masked.reason, "faces": masked.cropped_face_count}
        print(f"[collision-crop] surface mask {masked.reason} -> {masked.mesh_path}")
    return stats


def _inputs(layout: SceneLayout, args: argparse.Namespace) -> list[Path]:
    paths = [layout.collider_glb, layout.refined_pose, layout.refined_scale]
    if args.surface_mask:
        paths.append(layout.receiver_json)
    return paths


def _outputs(layout: SceneLayout, args: argparse.Namespace) -> list[Path]:
    glb = _out_glb(layout, args)
    out = [glb, glb.with_name(glb.name + ".json")]
    if args.surface_mask:
        masked = _masked_glb(layout, args)
        out += [masked, masked.with_name(masked.name + ".json")]
    return out


STEP = Step(
    name="collision.crop",
    help="crop the background collider to the robot-reachable prism (r 1.10 m, z -0.15..1.45 m), baked to world space",
    add_arguments=add_arguments,
    run=run,
    inputs=_inputs,
    outputs=_outputs,
    needs="cpu",
)
