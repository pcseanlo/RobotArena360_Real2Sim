"""Object placement records shared by the dump, the pose overrides and the object fits.

Copied verbatim from the old ``asset_init_randomization.py`` (an eval-time module):
only :class:`ObjectPlacementSpec`, :class:`PlacedPose` and the wxyz quaternion /
yaw helpers they and the object fits need. The episode randomization itself stays
with the evaluator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from r2s.collision.support import (
    DEFAULT_GLB_FRONT,
    DEFAULT_GLB_UP,
    align_up_and_front,
    load_combined_mesh,
    transform_vertices,
)


# ---------------------------------------------------------------------------
# Small quaternion helpers (wxyz, Hamilton product; same convention as the evaluator's
# ``quat_mul_np`` / ``quat_z_rot``).
# ---------------------------------------------------------------------------
def quat_mul_wxyz(q1, q2) -> np.ndarray:
    w1, x1, y1, z1 = np.asarray(q1, dtype=np.float64).reshape(4)
    w2, x2, y2, z2 = np.asarray(q2, dtype=np.float64).reshape(4)
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def yaw_quat_wxyz(yaw_rad: float) -> np.ndarray:
    half = 0.5 * float(yaw_rad)
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64)


def yaw_matrix(yaw_rad: float) -> np.ndarray:
    c, s = math.cos(float(yaw_rad)), math.sin(float(yaw_rad))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


# ---------------------------------------------------------------------------
# Per-object geometry captured at load time
# ---------------------------------------------------------------------------
@dataclass
class ObjectPlacementSpec:
    """Everything needed to re-pose one object without touching the simulator."""

    safe_key: str
    count: int
    object_name: str
    base_pos: np.ndarray  # (3,) world, after load-time collision correction
    base_quat_wxyz: np.ndarray  # (4,)
    # Vertices/faces in the *base-oriented* frame centred on ``base_pos``: world vertices
    # of the optimized pose minus ``base_pos``. A candidate pose applies a yaw rotation
    # about +z to these and adds the new position.
    centered_vertices: np.ndarray  # (N, 3)
    faces: np.ndarray | None = None  # (M, 3) int
    asset_file: str = ""
    scale: float = 1.0

    @classmethod
    def from_asset(
        cls,
        *,
        safe_key: str,
        count: int,
        object_name: str,
        asset_file: str,
        scale: float,
        base_pos,
        base_quat_wxyz,
        mesh_up=DEFAULT_GLB_UP,
        mesh_front=DEFAULT_GLB_FRONT,
        max_vertices: int = 1500,
    ) -> "ObjectPlacementSpec":
        mesh = load_combined_mesh(asset_file)
        import_rot = align_up_and_front(np.asarray(mesh_up), np.asarray(mesh_front))
        base_pos_arr = np.asarray(base_pos, dtype=np.float64).reshape(3)
        world = transform_vertices(
            np.asarray(mesh.vertices, dtype=np.float64),
            scale=scale,
            root_pos=base_pos_arr,
            root_quat_wxyz=base_quat_wxyz,
            mesh_import_rot=import_rot,
        )
        faces = np.asarray(getattr(mesh, "faces", np.empty((0, 3), dtype=np.int64)), dtype=np.int64)
        spec = cls(
            safe_key=safe_key,
            count=int(count),
            object_name=str(object_name),
            base_pos=base_pos_arr,
            base_quat_wxyz=np.asarray(base_quat_wxyz, dtype=np.float64).reshape(4),
            centered_vertices=world - base_pos_arr[None, :],
            faces=faces if faces.size else None,
            asset_file=asset_file,
            scale=float(scale),
        )
        return spec.subsampled(max_vertices)

    def subsampled(self, max_vertices: int, *, bottom_slab: float = 0.02, seed: int = 0) -> "ObjectPlacementSpec":
        """Cap the vertex count used for per-attempt raycasts (deterministic subsample).

        The full bottom slab (support contacts) is always kept up to half the budget;
        the remainder is drawn uniformly from the rest of the mesh. Faces are dropped
        when subsampling because their indices no longer apply.
        """
        verts = self.centered_vertices
        if max_vertices <= 0 or len(verts) <= max_vertices:
            return self
        rng = np.random.default_rng(seed)
        min_z = float(verts[:, 2].min())
        bottom_idx = np.flatnonzero(verts[:, 2] <= min_z + max(0.0, bottom_slab))
        other_idx = np.flatnonzero(verts[:, 2] > min_z + max(0.0, bottom_slab))
        keep_bottom = min(len(bottom_idx), max(max_vertices // 2, 1))
        if keep_bottom < len(bottom_idx):
            bottom_idx = rng.choice(bottom_idx, size=keep_bottom, replace=False)
        keep_other = max(max_vertices - len(bottom_idx), 0)
        if keep_other < len(other_idx):
            other_idx = rng.choice(other_idx, size=keep_other, replace=False)
        # Always retain the extreme vertices so the AABB is unchanged.
        extremes = np.concatenate([verts.argmin(axis=0), verts.argmax(axis=0)])
        idx = np.unique(np.concatenate([bottom_idx, other_idx, extremes]))
        return ObjectPlacementSpec(
            safe_key=self.safe_key,
            count=self.count,
            object_name=self.object_name,
            base_pos=self.base_pos,
            base_quat_wxyz=self.base_quat_wxyz,
            centered_vertices=verts[idx],
            faces=None,
            asset_file=self.asset_file,
            scale=self.scale,
        )

    @classmethod
    def from_world_vertices(
        cls,
        *,
        safe_key: str,
        count: int,
        object_name: str,
        vertices_world,
        base_pos,
        base_quat_wxyz,
        faces=None,
        asset_file: str = "",
        scale: float = 1.0,
    ) -> "ObjectPlacementSpec":
        base_pos_arr = np.asarray(base_pos, dtype=np.float64).reshape(3)
        verts = np.asarray(vertices_world, dtype=np.float64).reshape(-1, 3)
        return cls(
            safe_key=safe_key,
            count=int(count),
            object_name=str(object_name),
            base_pos=base_pos_arr,
            base_quat_wxyz=np.asarray(base_quat_wxyz, dtype=np.float64).reshape(4),
            centered_vertices=verts - base_pos_arr[None, :],
            faces=None if faces is None else np.asarray(faces, dtype=np.int64),
            asset_file=asset_file,
            scale=float(scale),
        )

    # -- geometry -----------------------------------------------------------
    @property
    def base_min_z(self) -> float:
        return float(self.base_pos[2] + self.centered_vertices[:, 2].min())

    @property
    def base_aabb(self) -> tuple[np.ndarray, np.ndarray]:
        return self.world_aabb(self.base_pos, 0.0)

    def world_vertices(self, pos, yaw_rad: float) -> np.ndarray:
        rot = yaw_matrix(yaw_rad)
        return self.centered_vertices @ rot.T + np.asarray(pos, dtype=np.float64).reshape(1, 3)

    def world_aabb(self, pos, yaw_rad: float) -> tuple[np.ndarray, np.ndarray]:
        verts = self.world_vertices(pos, yaw_rad)
        return verts.min(axis=0), verts.max(axis=0)

    def world_points(self, pos, yaw_rad: float, bottom_slab: float):
        """Mirror ``mesh_world_vertices_and_contact_points`` for a candidate pose."""
        verts = self.world_vertices(pos, yaw_rad)
        aabb_min = verts.min(axis=0)
        aabb_max = verts.max(axis=0)
        slab = max(0.0, float(bottom_slab))
        contact = verts[verts[:, 2] <= float(aabb_min[2]) + slab]
        collision = verts
        if self.faces is not None and self.faces.size:
            tri = verts[self.faces]
            centroids = tri.mean(axis=1)
            bottom = centroids[tri[:, :, 2].min(axis=1) <= float(aabb_min[2]) + slab]
            if bottom.size:
                contact = np.concatenate([contact, bottom], axis=0)
            collision = np.concatenate([verts, centroids], axis=0)
        if contact.size == 0:
            contact = verts
        return verts, contact, collision, aabb_min, aabb_max


@dataclass
class PlacedPose:
    safe_key: str
    object_name: str
    pos: np.ndarray
    quat_wxyz: np.ndarray
    yaw_delta_rad: float
    xy_delta: np.ndarray
    status: str  # sampled | attached | frozen_off_table | fallback_original | skipped_below_surface
    attempts: int = 0
    anchor: str | None = None
    aabb_min: np.ndarray | None = None
    aabb_max: np.ndarray | None = None
    reject_reasons: dict[str, int] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "safe_key": self.safe_key,
            "object_name": self.object_name,
            "pos": [float(v) for v in self.pos],
            "quat_wxyz": [float(v) for v in self.quat_wxyz],
            "yaw_delta_deg": math.degrees(self.yaw_delta_rad),
            "xy_delta_m": [float(v) for v in self.xy_delta],
            "status": self.status,
            "attempts": self.attempts,
            "anchor": self.anchor,
            "reject_reasons": dict(self.reject_reasons),
        }
