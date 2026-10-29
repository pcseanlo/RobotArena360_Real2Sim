"""Quaternion helpers. Convention everywhere: ``wxyz``, Hamilton product.

Copied verbatim from the old ``operation_surface_precompute.py`` so numbers match
the old outputs bit for bit.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation as ScipyRotation


def quat_mul_np(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product for wxyz quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def quat_normalize(q: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(q))
    if norm <= 0.0:
        raise ValueError("Cannot normalize zero-length quaternion.")
    return q / norm


def axis_quat_np(theta: float, axis: int) -> np.ndarray:
    """Rotation of ``theta`` radians about axis 0=x, 1=y, 2=z."""
    half = theta * 0.5
    quat = np.array([math.cos(half), 0.0, 0.0, 0.0], dtype=np.float64)
    quat[axis + 1] = math.sin(half)
    return quat


def euler_zyx_to_quat(rx_rad: float, ry_rad: float, rz_rad: float) -> np.ndarray:
    qx = axis_quat_np(rx_rad, 0)
    qy = axis_quat_np(ry_rad, 1)
    qz = axis_quat_np(rz_rad, 2)
    return quat_mul_np(qz, quat_mul_np(qy, qx))


def quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    q = quat_normalize(np.asarray(q, dtype=np.float64))
    return ScipyRotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def matrix_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    q_xyzw = ScipyRotation.from_matrix(rot).as_quat()
    return quat_normalize(np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64))
