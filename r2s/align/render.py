"""Differentiable splat renderer for the pose fit, matching Eden HEAD's gsplat path.

Eden HEAD renders splats with ``gsplat.rasterization`` (``eden/utils/batch_splat_renderer``):
PLY attributes are loaded with ``ply_io.load_ply`` (exp scales, sigmoid opacity,
normalized wxyz quats, SH as ``(N, K, 3)``), placement is baked as
``x_world = R(q) (s x) + pos`` and the camera uses ``fx = fy = H / (2 tan(fovy / 2))``
with the principal point at the image centre and an OpenCV view matrix.

Eden's own ``batch_render`` runs under ``torch.no_grad``, so the pose fit calls
gsplat directly with the same arguments, keeping the gradient path to the pose.

Differences from the old pose fit (2DGS ``diff_surfel`` renderer) are deliberate:
gaussians render as full 3D ellipsoids, exactly as the evaluator sees them at HEAD.
The grey 0.5 background of the old renderer is kept as the default (``bg``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


def quat_mul_t(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of wxyz quaternions, broadcasting over leading dims."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dim=-1,
    )


def quat_rotate_t(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors ``v (N, 3)`` by the unit quaternion ``q (4,)`` (wxyz)."""
    w, xyz = q[0], q[1:]
    t = 2.0 * torch.cross(xyz.expand_as(v), v, dim=-1)
    return v + w * t + torch.cross(xyz.expand_as(v), t, dim=-1)


def axis_quat_t(angle: torch.Tensor, axis: int) -> torch.Tensor:
    """Differentiable rotation about axis 0/1/2 (``angle`` is a 1-element tensor)."""
    half = angle * 0.5
    c, s = torch.cos(half), torch.sin(half)
    zero = torch.zeros_like(half)
    parts = [c, zero, zero, zero]
    parts[axis + 1] = s
    return torch.cat(parts, dim=0)


@dataclass
class Camera:
    """Pinhole camera as Eden HEAD's gsplat path sees it (from DROID calibration)."""

    viewmat: torch.Tensor  # (4, 4) OpenCV world-to-camera
    K: torch.Tensor  # (3, 3) fx = fy from the vertical FOV, principal point at the centre
    width: int
    height: int
    fov_y_deg: float

    @classmethod
    def from_droid(cls, intrinsics: str | Path, extrinsics: str | Path, device: torch.device) -> "Camera":
        k = np.load(intrinsics).astype(np.float64)
        t_w2c = np.load(extrinsics).astype(np.float64).reshape(4, 4)
        fy, cx, cy = float(k[1, 1]), float(k[0, 2]), float(k[1, 2])
        width, height = int(round(2 * cx)), int(round(2 * cy))
        fov_y = 2.0 * math.atan(height / (2.0 * fy))
        focal = height / (2.0 * math.tan(fov_y / 2.0))  # == fy; spelled out to mirror Eden
        K = torch.tensor(
            [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32, device=device,
        )
        viewmat = torch.tensor(t_w2c, dtype=torch.float32, device=device)
        return cls(viewmat, K, width, height, math.degrees(fov_y))

    def position(self) -> np.ndarray:
        return np.linalg.inv(self.viewmat.detach().cpu().double().numpy())[:3, 3]


class SplatModel:
    """One splat PLY on the GPU, base-rotated once, re-posed per render."""

    def __init__(self, ply_path: str | Path, base_quat_wxyz: np.ndarray, device: torch.device):
        from eden.utils.batch_splat_renderer.ply_io import load_ply  # Eden HEAD's loader and activations

        g = load_ply(str(ply_path), activate=True)
        self.device = device
        q = torch.tensor(np.asarray(base_quat_wxyz, dtype=np.float32), device=device)
        xyz = torch.as_tensor(g.xyz, dtype=torch.float32, device=device)
        rot = torch.as_tensor(g.rot, dtype=torch.float32, device=device)
        # Bake the PLY -> sim base rotation (scale 1, pos 0), as Eden's renderer does at load.
        self.xyz = quat_rotate_t(q, xyz)
        self.rot = quat_mul_t(q.expand_as(rot), rot)
        self.scale = torch.as_tensor(g.scale, dtype=torch.float32, device=device)
        self.opacity = torch.as_tensor(g.opacity, dtype=torch.float32, device=device)
        self.sh = torch.as_tensor(g.sh, dtype=torch.float32, device=device)
        self.sh_degree = int(round(math.sqrt(self.sh.shape[1]))) - 1

    @property
    def centroid(self) -> torch.Tensor:
        return self.xyz.mean(dim=0)

    def render(self, camera: Camera, quat: torch.Tensor, trans: torch.Tensor, bg: float | None = 0.5) -> torch.Tensor:
        """``(H, W, 3)`` image in [0, 1] of the splat moved by ``x -> R(quat) x + trans``."""
        from gsplat.rendering import rasterization

        means = quat_rotate_t(quat, self.xyz) + trans
        quats = quat_mul_t(quat.expand_as(self.rot), self.rot)
        backgrounds = None
        if bg is not None:
            backgrounds = torch.full((1, 3), float(bg), dtype=torch.float32, device=self.device)
        renders, _alphas, _meta = rasterization(
            means=means,
            quats=quats,
            scales=self.scale,
            opacities=self.opacity,
            colors=self.sh,
            viewmats=camera.viewmat[None],
            Ks=camera.K[None],
            width=camera.width,
            height=camera.height,
            sh_degree=self.sh_degree,
            render_mode="RGB",
            packed=False,
            backgrounds=backgrounds,
        )
        return renders[0, ..., :3].clamp(0.0, 1.0)
