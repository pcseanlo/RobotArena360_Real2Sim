"""DROID camera calibration helpers (``intrinsics.npy`` 3x3 K, ``extrinsics.npy`` 4x4 world-to-camera)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from r2s.common.quat import matrix_to_quat_wxyz

# OpenCV camera axes (x right, y down, z forward) -> Genesis camera axes (y up, looks along -z).
_CV_TO_GENESIS = np.array(((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)), dtype=np.float64)


def load_camera_pose_from_extrinsics(extrinsics_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Camera position and Genesis-convention quaternion (wxyz) from a world-to-camera matrix."""
    t_w2c = np.load(extrinsics_path).astype(np.float64).reshape(4, 4)
    t_c2w = np.linalg.inv(t_w2c)
    t_c2w[:3, :3] = t_c2w[:3, :3] @ _CV_TO_GENESIS
    return t_c2w[:3, 3].copy(), matrix_to_quat_wxyz(t_c2w[:3, :3])


def load_camera_model(
    intrinsics_path: str | Path,
    extrinsics_path: str | Path,
) -> tuple[np.ndarray, int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(K, width, height, T_world_to_camera, cam_pos, cam_lookat, cam_up)``.

    Image size is ``2 * principal point``, as every old script assumed.
    """
    k = np.load(intrinsics_path).astype(np.float64)
    t_world_to_camera = np.load(extrinsics_path).astype(np.float64).reshape(4, 4)
    t_c2w = np.linalg.inv(t_world_to_camera)
    t_c2w[:3, :3] = t_c2w[:3, :3] @ _CV_TO_GENESIS
    width = int(round(2.0 * float(k[0, 2])))
    height = int(round(2.0 * float(k[1, 2])))
    cam_pos = t_c2w[:3, 3].copy()
    forward = -t_c2w[:3, 2]
    cam_up = t_c2w[:3, 1].copy()
    cam_lookat = cam_pos + forward
    return k, width, height, t_world_to_camera, cam_pos, cam_lookat, cam_up
