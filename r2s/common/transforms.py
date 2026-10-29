"""The one place that turns alignment results into world poses.

Alignment (steps 2-3) writes ``aligned_pose*.npy`` (``[tx, ty, tz, rz]`` or
``[tx, ty, tz, rx, ry, rz]``) and ``aligned_pose_scene_scale*.npy`` (one float).
Everything downstream places the splat and the collider with the formulas below.
They are copied from the old ``operation_surface_precompute.py`` /
``operation_surface_render_receiver_fit.py``, which mirror the evaluator:

* collider: ``R_c = q_rel * ADJUSTMENT``, world = ``(v_local * s) @ R_c.T + t``
* splat:    ``R_s = q_rel * (ADJUSTMENT * BASE_ROT)``, ``splat_pos = t / s``,
            world = ``(xyz @ R_s.T + splat_pos) * s``

``q_rel`` is the yaw ``rz`` (plus roll/pitch for 6-value poses). Unlike the old
code, nothing here guesses a scale file: callers pass both paths explicitly, so a
missing scale is an error instead of a silent 0.5 or 1.0.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from r2s.common.quat import (
    axis_quat_np,
    euler_zyx_to_quat,
    matrix_to_quat_wxyz,
    quat_mul_np,
    quat_normalize,
    quat_wxyz_to_matrix,
)

# Base rotation of the splat PLY into the sim frame, and the ~2 deg tilt correction
# shared by the splat and the collider (wxyz).
SCALE_OPT_BASE_ROT = np.array((0.7071067811865476, 0.7071067811865476, 0.0, 0.0), dtype=np.float64)
SCALE_OPT_ADJUSTMENT = np.array((0.01745240643728351, 0.0, 0.9998476951563912, 0.0), dtype=np.float64)
PLACEMENT_LINEAGE = "r2s.align.scale_fit"  # provenance string written into surface JSONs

# GLB files are y-up, x-front; Genesis imports them z-up.
DEFAULT_GLB_UP = (0.0, 1.0, 0.0)
DEFAULT_GLB_FRONT = (1.0, 0.0, 0.0)


# ------------------------------------------------------------------ loading
def load_pose(result_path: str | Path) -> tuple[np.ndarray, float, float, float]:
    """Return ``(t_xyz, rx, ry, rz)`` (radians) from an ``aligned_pose*.npy``."""
    pose = np.load(result_path).astype(np.float64).reshape(-1)
    if pose.shape[0] == 4:
        tx, ty, tz, rz = pose
        rx, ry = 0.0, 0.0
    elif pose.shape[0] == 6:
        tx, ty, tz, rx, ry, rz = pose
    else:
        raise ValueError(
            f"Unexpected pose array length {pose.shape[0]} in {result_path}; "
            "expected 4 ([tx,ty,tz,rz]) or 6 ([tx,ty,tz,rx,ry,rz])."
        )
    return np.array([tx, ty, tz], dtype=np.float64), float(rx), float(ry), float(rz)


def load_scene_scale(scene_scale_path: str | Path) -> float:
    path = Path(scene_scale_path)
    if not path.is_file():
        raise FileNotFoundError(f"scene scale file not found: {path}")
    return float(np.load(path).reshape(-1)[0])


# ------------------------------------------------------------------ rotations
def apply_relative_transform_to_quat(
    orig_quat: np.ndarray,
    rz_rad: float,
    rx_rad: float = 0.0,
    ry_rad: float = 0.0,
) -> np.ndarray:
    if rx_rad == 0.0 and ry_rad == 0.0:
        q_rel = axis_quat_np(rz_rad, 2)
    else:
        q_rel = euler_zyx_to_quat(rx_rad, ry_rad, rz_rad)
    return quat_normalize(quat_mul_np(q_rel, np.asarray(orig_quat, dtype=np.float64)))


def align_up_to_z(up: np.ndarray) -> np.ndarray:
    up = np.asarray(up, dtype=np.float64)
    up /= max(float(np.linalg.norm(up)), 1e-12)
    target = np.array((0.0, 0.0, 1.0), dtype=np.float64)
    if np.allclose(up, target):
        return np.eye(3, dtype=np.float64)
    if np.allclose(up, -target):
        return np.array(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)), dtype=np.float64)
    axis = np.cross(up, target)
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    angle = float(np.arccos(np.clip(np.dot(up, target), -1.0, 1.0)))
    k = np.array(
        ((0.0, -axis[2], axis[1]), (axis[2], 0.0, -axis[0]), (-axis[1], axis[0], 0.0)),
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)


def align_up_and_front(up: np.ndarray, front: np.ndarray) -> np.ndarray:
    """Mirror eden.utils.geom.align_up_and_front without importing Genesis."""
    up = np.asarray(up, dtype=np.float64)
    front = np.asarray(front, dtype=np.float64)
    up /= max(float(np.linalg.norm(up)), 1e-12)
    front /= max(float(np.linalg.norm(front)), 1e-12)

    r1 = align_up_to_z(up)
    front_transformed = r1 @ front
    front_xy = front_transformed.copy()
    front_xy[2] = 0.0
    if float(np.linalg.norm(front_xy)) < 1e-6:
        return r1
    front_xy /= float(np.linalg.norm(front_xy))
    angle = float(np.arctan2(front_xy[1], front_xy[0]))
    r2 = np.array(
        ((math.cos(-angle), -math.sin(-angle), 0.0), (math.sin(-angle), math.cos(-angle), 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    return r2 @ r1


def glb_import_rotation() -> np.ndarray:
    """Rotation Genesis applies when importing a y-up GLB (collider, object meshes)."""
    return align_up_and_front(np.asarray(DEFAULT_GLB_UP), np.asarray(DEFAULT_GLB_FRONT))


# ------------------------------------------------------------------ world poses
def collider_world_transform(
    result_path: str | Path, scene_scale: float
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """World position, rotation matrix and provenance of the background collider."""
    pos, rx, ry, rz = load_pose(result_path)
    # The evaluator builds orig_collider_quat = transform_quat_by_quat(identity, ADJUSTMENT),
    # which is exactly ADJUSTMENT whatever the multiplication convention.
    orig_collider_quat = quat_normalize(SCALE_OPT_ADJUSTMENT)
    collider_quat = apply_relative_transform_to_quat(orig_collider_quat, rz, rx_rad=rx, ry_rad=ry)
    rot = quat_wxyz_to_matrix(collider_quat)
    meta = {
        "placement_lineage": PLACEMENT_LINEAGE,
        "base_rot_wxyz_for_splat_only": SCALE_OPT_BASE_ROT.tolist(),
        "collider_adjustment_wxyz": SCALE_OPT_ADJUSTMENT.tolist(),
        "collider_position_rule": "collider_pos=(tx,ty,tz); do not divide by scene scale",
        "collider_scale_rule": "points_world=(points_local*scene_scale)@R.T+collider_pos",
        "collider_pos": pos.tolist(),
        "collider_quat_wxyz": collider_quat.tolist(),
        "scene_scale": float(scene_scale),
        "pose_rx_ry_rz_rad": [float(rx), float(ry), float(rz)],
        "pose_rx_ry_rz_deg": [float(math.degrees(rx)), float(math.degrees(ry)), float(math.degrees(rz))],
    }
    return pos, rot, meta


def collider_quat(result_path: str | Path) -> np.ndarray:
    _, rx, ry, rz = load_pose(result_path)
    return apply_relative_transform_to_quat(quat_normalize(SCALE_OPT_ADJUSTMENT), rz, rx_rad=rx, ry_rad=ry)


def splat_world_pose(result_path: str | Path, scene_scale_path: str | Path) -> tuple[np.ndarray, np.ndarray, float]:
    """``(splat_pos, splat_quat_wxyz, scene_scale)`` as the evaluator passes them to the splat entity."""
    scene_scale = load_scene_scale(scene_scale_path)
    pos, rx, ry, rz = load_pose(result_path)
    orig_splat_quat = quat_normalize(quat_mul_np(SCALE_OPT_ADJUSTMENT, SCALE_OPT_BASE_ROT))
    splat_quat = apply_relative_transform_to_quat(orig_splat_quat, rz, rx_rad=rx, ry_rad=ry)
    splat_pos = pos / float(scene_scale)
    return splat_pos, splat_quat, float(scene_scale)


def transform_gaussian_centers(
    xyz: np.ndarray,
    *,
    splat_pos: np.ndarray,
    splat_quat_wxyz: np.ndarray,
    scene_scale: float,
) -> np.ndarray:
    rot = quat_wxyz_to_matrix(splat_quat_wxyz)
    return (np.asarray(xyz, dtype=np.float64) @ rot.T + splat_pos[None, :]) * float(scene_scale)


__all__ = [
    "DEFAULT_GLB_FRONT",
    "DEFAULT_GLB_UP",
    "PLACEMENT_LINEAGE",
    "SCALE_OPT_ADJUSTMENT",
    "SCALE_OPT_BASE_ROT",
    "align_up_and_front",
    "align_up_to_z",
    "apply_relative_transform_to_quat",
    "collider_quat",
    "collider_world_transform",
    "glb_import_rotation",
    "load_pose",
    "load_scene_scale",
    "matrix_to_quat_wxyz",
    "quat_mul_np",
    "quat_normalize",
    "quat_wxyz_to_matrix",
    "splat_world_pose",
    "transform_gaussian_centers",
]
