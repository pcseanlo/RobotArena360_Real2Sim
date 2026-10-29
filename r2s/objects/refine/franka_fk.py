"""Franka Panda / FR3 forward kinematics (modified DH, Craig convention).

Copied from the old ``ee_error_from_chunks.py`` (``_DH``, ``_FLANGE_D``, ``_HAND_D``,
``_tf``), which ``asset_refine_common`` imported for the fingertip anchor of the
grasp cue. ``tcp.npy`` holds the flange position; the point the gripper closes at
is ``flange + R[:, 2] * HAND_D``.
"""

from __future__ import annotations

import numpy as np

# Franka Panda / FR3 modified DH (Craig): (a, d, alpha) per joint, then flange.
_DH = [
    (0.0, 0.333, 0.0),
    (0.0, 0.0, -np.pi / 2),
    (0.0, 0.316, np.pi / 2),
    (0.0825, 0.0, np.pi / 2),
    (-0.0825, 0.384, -np.pi / 2),
    (0.0, 0.0, np.pi / 2),
    (0.088, 0.0, np.pi / 2),
]
_FLANGE_D = 0.107
_HAND_D = 0.1034  # flange -> fingertip centre (Franka hand TCP)

DH = _DH
FLANGE_D = _FLANGE_D
HAND_D = _HAND_D


def _tf(a, d, alpha, theta):
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [ct, -st, 0.0, a],
            [st * ca, ct * ca, -sa, -d * sa],
            [st * sa, ct * sa, ca, d * ca],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def fk_tcp(q: np.ndarray) -> np.ndarray:
    """Fingertip-centre position (m) in the robot base frame for 7 joint angles (flange + HAND_D)."""
    T = np.eye(4)
    for (a, d, alpha), theta in zip(_DH, q):
        T = T @ _tf(a, d, alpha, theta)
    T = T @ _tf(0.0, _FLANGE_D + _HAND_D, 0.0, 0.0)
    return T[:3, 3]
