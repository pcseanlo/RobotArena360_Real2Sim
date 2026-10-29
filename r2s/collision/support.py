"""Background support queries and the pre-init object lift ("collision guard").

Port of the old ``init_collision_guard.py``. Given the background collider placed in
the world (or its baked, world-space crop), :class:`BackgroundSupportQuery` answers
"what is under / inside this object" and :func:`corrected_init_position` lifts an
object so it starts resting on the support instead of inside it. The sim assembly
calls this for every scene object before ``scene.build``; the object-refinement fit
uses ``support_z_for_aabb`` to re-rest objects at their fitted footprint.

Changes from the old module: the Eden asset lookup (``resolve_eden_asset_path``) is
gone, callers pass real file paths. The quaternion / up-axis / mesh-loading helpers
are the shared, numerically identical ones from ``r2s.common``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from r2s.common.mesh import load_combined_mesh
from r2s.common.quat import quat_wxyz_to_matrix
from r2s.common.transforms import DEFAULT_GLB_FRONT, DEFAULT_GLB_UP, align_up_and_front, align_up_to_z

__all__ = [
    "DEFAULT_GLB_FRONT",
    "DEFAULT_GLB_UP",
    "BackgroundSupportQuery",
    "InitCollisionCorrection",
    "align_up_and_front",
    "align_up_to_z",
    "corrected_init_position",
    "load_combined_mesh",
    "mesh_world_aabb",
    "mesh_world_vertices",
    "mesh_world_vertices_and_contact_points",
    "quat_wxyz_to_matrix",
    "transform_vertices",
]


def _mesh_path(mesh_path: str | Path) -> str:
    path = Path(mesh_path)
    if not path.is_file():
        raise FileNotFoundError(f"mesh not found: {path}")
    return str(path)


def transform_vertices(
    vertices: np.ndarray,
    *,
    scale: float,
    root_pos: np.ndarray | tuple[float, float, float],
    root_quat_wxyz: np.ndarray | tuple[float, float, float, float],
    mesh_import_rot: np.ndarray | None = None,
) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    root_pos = np.asarray(root_pos, dtype=np.float64).reshape(3)
    root_rot = quat_wxyz_to_matrix(np.asarray(root_quat_wxyz, dtype=np.float64).reshape(4))
    if mesh_import_rot is not None:
        vertices = vertices @ np.asarray(mesh_import_rot, dtype=np.float64).reshape(3, 3).T
    return (vertices * float(scale)) @ root_rot.T + root_pos[None, :]


def mesh_world_aabb(
    mesh_path: str | Path,
    *,
    scale: float,
    root_pos: np.ndarray | tuple[float, float, float],
    root_quat_wxyz: np.ndarray | tuple[float, float, float, float],
    mesh_import_rot: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    mesh = load_combined_mesh(_mesh_path(mesh_path))
    vertices_world = transform_vertices(
        np.asarray(mesh.vertices, dtype=np.float64),
        scale=scale,
        root_pos=root_pos,
        root_quat_wxyz=root_quat_wxyz,
        mesh_import_rot=mesh_import_rot,
    )
    return vertices_world.min(axis=0), vertices_world.max(axis=0)


def mesh_world_vertices(
    mesh_path: str | Path,
    *,
    scale: float,
    root_pos: np.ndarray | tuple[float, float, float],
    root_quat_wxyz: np.ndarray | tuple[float, float, float, float],
    mesh_import_rot: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return world-space vertices plus their AABB (min, max)."""
    mesh = load_combined_mesh(_mesh_path(mesh_path))
    vertices_world = transform_vertices(
        np.asarray(mesh.vertices, dtype=np.float64),
        scale=scale,
        root_pos=root_pos,
        root_quat_wxyz=root_quat_wxyz,
        mesh_import_rot=mesh_import_rot,
    )
    return vertices_world, vertices_world.min(axis=0), vertices_world.max(axis=0)


def mesh_world_vertices_and_contact_points(
    mesh_path: str | Path,
    *,
    scale: float,
    root_pos: np.ndarray | tuple[float, float, float],
    root_quat_wxyz: np.ndarray | tuple[float, float, float, float],
    mesh_import_rot: np.ndarray | None = None,
    bottom_slab: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return world vertices, support samples, penetration samples, and AABB."""
    mesh = load_combined_mesh(_mesh_path(mesh_path))
    vertices_world = transform_vertices(
        np.asarray(mesh.vertices, dtype=np.float64),
        scale=scale,
        root_pos=root_pos,
        root_quat_wxyz=root_quat_wxyz,
        mesh_import_rot=mesh_import_rot,
    )
    aabb_min = vertices_world.min(axis=0)
    aabb_max = vertices_world.max(axis=0)

    contact_points = vertices_world[
        vertices_world[:, 2] <= float(aabb_min[2]) + max(0.0, float(bottom_slab))
    ]
    faces = np.asarray(getattr(mesh, "faces", np.empty((0, 3), dtype=np.int64)), dtype=np.int64)
    if faces.size:
        triangles = vertices_world[faces]
        tri_min_z = triangles[:, :, 2].min(axis=1)
        face_centroids = triangles.mean(axis=1)
        bottom_face_centroids = face_centroids[
            tri_min_z <= float(aabb_min[2]) + max(0.0, float(bottom_slab))
        ]
        if bottom_face_centroids.size:
            contact_points = np.concatenate([contact_points, bottom_face_centroids], axis=0)
        collision_points = np.concatenate([vertices_world, face_centroids], axis=0)
    else:
        collision_points = vertices_world

    if contact_points.size == 0:
        contact_points = vertices_world
    return vertices_world, contact_points, collision_points, aabb_min, aabb_max


@dataclass(frozen=True)
class InitCollisionCorrection:
    pos: np.ndarray
    lift: float
    support_z: float | None
    support_source: str
    object_min_z: float | None
    object_max_z: float | None
    # Diagnostics (vertex path only; None for AABB fallback / disabled).
    contact_vertex_count: int = 0
    support_span: float | None = None
    lateral_penetration: float | None = None


class BackgroundSupportQuery:
    """Query local support height from a transformed background collider mesh."""

    def __init__(
        self,
        mesh_path: str | Path,
        *,
        scene_scale: float,
        root_pos: np.ndarray | tuple[float, float, float],
        root_quat_wxyz: np.ndarray | tuple[float, float, float, float],
        footprint_margin: float = 0.01,
        surface_normal_z: float = 0.5,
        max_surface_above_aabb: float = 0.05,
        mesh_up: tuple[float, float, float] = DEFAULT_GLB_UP,
        mesh_front: tuple[float, float, float] = DEFAULT_GLB_FRONT,
    ) -> None:
        self.mesh_path = _mesh_path(mesh_path)
        self.footprint_margin = float(footprint_margin)
        self.surface_normal_z = float(surface_normal_z)
        self.max_surface_above_aabb = float(max_surface_above_aabb)

        mesh = load_combined_mesh(self.mesh_path)
        if len(mesh.faces) == 0:
            raise ValueError(f"Background mesh has no faces: {mesh_path}")

        mesh_import_rot = align_up_and_front(np.asarray(mesh_up), np.asarray(mesh_front))
        self.vertices_world = transform_vertices(
            np.asarray(mesh.vertices, dtype=np.float64),
            scale=float(scene_scale),
            root_pos=root_pos,
            root_quat_wxyz=root_quat_wxyz,
            mesh_import_rot=mesh_import_rot,
        )
        self.faces = np.asarray(mesh.faces, dtype=np.int64)
        triangles = self.vertices_world[self.faces]
        self.tri_xy_min = triangles[:, :, :2].min(axis=1)
        self.tri_xy_max = triangles[:, :, :2].max(axis=1)
        self.tri_z_min = triangles[:, :, 2].min(axis=1)
        self.tri_z_max = triangles[:, :, 2].max(axis=1)

        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        normal_norm = np.linalg.norm(normals, axis=1)
        valid = normal_norm > 1e-12
        normals[valid] /= normal_norm[valid, None]
        self.upward_mask = valid & (normals[:, 2] >= self.surface_normal_z)

    @classmethod
    def from_baked_crop(
        cls,
        crop_glb: str | Path,
        *,
        footprint_margin: float = 0.015,
        surface_normal_z: float = 0.75,
    ) -> "BackgroundSupportQuery":
        """Query on the baked world-space collider crop (``collision.crop`` output).

        Identity root pose and z-up mesh axes: the crop's vertices are already in
        world coordinates (same as the old evaluator's asset-init path and
        ``asset_refine_common.build_collider_query``).
        """
        return cls(
            crop_glb,
            scene_scale=1.0,
            root_pos=(0.0, 0.0, 0.0),
            root_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            footprint_margin=footprint_margin,
            surface_normal_z=surface_normal_z,
            mesh_up=(0.0, 0.0, 1.0),
            mesh_front=(1.0, 0.0, 0.0),
        )

    @classmethod
    def from_arrays(
        cls,
        triangles_world: np.ndarray,
        *,
        footprint_margin: float = 0.01,
        surface_normal_z: float = 0.5,
        max_surface_above_aabb: float = 0.05,
    ) -> "BackgroundSupportQuery":
        query = cls.__new__(cls)
        query.mesh_path = "<arrays>"
        query.footprint_margin = float(footprint_margin)
        query.surface_normal_z = float(surface_normal_z)
        query.max_surface_above_aabb = float(max_surface_above_aabb)
        triangles = np.asarray(triangles_world, dtype=np.float64).reshape(-1, 3, 3)
        query.vertices_world = triangles.reshape(-1, 3)
        query.faces = np.arange(len(query.vertices_world), dtype=np.int64).reshape(-1, 3)
        query.tri_xy_min = triangles[:, :, :2].min(axis=1)
        query.tri_xy_max = triangles[:, :, :2].max(axis=1)
        query.tri_z_min = triangles[:, :, 2].min(axis=1)
        query.tri_z_max = triangles[:, :, 2].max(axis=1)
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        normal_norm = np.linalg.norm(normals, axis=1)
        valid = normal_norm > 1e-12
        normals[valid] /= normal_norm[valid, None]
        query.upward_mask = valid & (normals[:, 2] >= query.surface_normal_z)
        return query

    def support_z_for_aabb(
        self,
        aabb_min: np.ndarray,
        aabb_max: np.ndarray,
        *,
        max_support_z: float | None = None,
    ) -> float | None:
        aabb_min = np.asarray(aabb_min, dtype=np.float64).reshape(3)
        aabb_max = np.asarray(aabb_max, dtype=np.float64).reshape(3)
        margin = max(0.0, self.footprint_margin)
        xy_min = aabb_min[:2] - margin
        xy_max = aabb_max[:2] + margin
        z_limit = aabb_max[2] + max(0.0, self.max_surface_above_aabb)
        if max_support_z is not None and np.isfinite(max_support_z):
            z_limit = min(float(z_limit), float(max_support_z))
        overlap = (
            self.upward_mask
            & (self.tri_xy_max[:, 0] >= xy_min[0])
            & (self.tri_xy_min[:, 0] <= xy_max[0])
            & (self.tri_xy_max[:, 1] >= xy_min[1])
            & (self.tri_xy_min[:, 1] <= xy_max[1])
            & (self.tri_z_max <= z_limit)
        )
        if not np.any(overlap):
            return None
        return float(np.max(self.tri_z_max[overlap]))

    def _ray_intersector(self):
        cached = getattr(self, "_ray_cache", None)
        if cached is not None:
            return cached if cached is not False else None
        try:
            import trimesh

            mesh = trimesh.Trimesh(vertices=self.vertices_world, faces=self.faces, process=False)
            self._trimesh = mesh
            intersector = mesh.ray
            self._ray_cache = intersector
            print(
                f"[init-collision] vertex raycast path enabled "
                f"(intersector={type(intersector).__name__}, faces={len(self.faces)})"
            )
            return intersector
        except Exception as e:
            self._ray_cache = False
            self._trimesh = None
            print(f"[init-collision] WARNING: ray intersector unavailable ({e}); "
                  "falling back to AABB path")
            return None

    def _vertical_intersects_location_numpy(
        self,
        *,
        ray_origins: np.ndarray,
        ray_directions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fallback for z-axis raycasts when trimesh's optional rtree is absent."""
        origins = np.asarray(ray_origins, dtype=np.float64).reshape(-1, 3)
        directions = np.asarray(ray_directions, dtype=np.float64).reshape(-1, 3)
        triangles = self.vertices_world[self.faces]

        locs: list[np.ndarray] = []
        ray_ids: list[int] = []
        tri_ids: list[int] = []
        eps = 1e-9

        for ray_i, (origin, direction) in enumerate(zip(origins, directions)):
            dz = float(direction[2])
            if abs(dz) < eps or abs(direction[0]) > eps or abs(direction[1]) > eps:
                continue

            x, y, z0 = float(origin[0]), float(origin[1]), float(origin[2])
            candidates = np.flatnonzero(
                (self.tri_xy_min[:, 0] <= x + eps)
                & (self.tri_xy_max[:, 0] >= x - eps)
                & (self.tri_xy_min[:, 1] <= y + eps)
                & (self.tri_xy_max[:, 1] >= y - eps)
            )

            best_z: float | None = None
            best_tri: int | None = None
            for tri_i in candidates:
                tri = triangles[tri_i]
                p0, p1, p2 = tri
                normal = np.cross(p1 - p0, p2 - p0)
                if abs(float(normal[2])) < eps:
                    continue

                z = p0[2] - (normal[0] * (x - p0[0]) + normal[1] * (y - p0[1])) / normal[2]
                if dz < 0.0 and z > z0 + eps:
                    continue
                if dz > 0.0 and z < z0 - eps:
                    continue

                v0 = p1[:2] - p0[:2]
                v1 = p2[:2] - p0[:2]
                v2 = np.array([x, y], dtype=np.float64) - p0[:2]
                den = v0[0] * v1[1] - v1[0] * v0[1]
                if abs(float(den)) < eps:
                    continue
                a = (v2[0] * v1[1] - v1[0] * v2[1]) / den
                b = (v0[0] * v2[1] - v2[0] * v0[1]) / den
                if a < -eps or b < -eps or a + b > 1.0 + eps:
                    continue

                if best_z is None:
                    best_z = float(z)
                    best_tri = int(tri_i)
                elif dz < 0.0 and z > best_z:
                    best_z = float(z)
                    best_tri = int(tri_i)
                elif dz > 0.0 and z < best_z:
                    best_z = float(z)
                    best_tri = int(tri_i)

            if best_z is not None and best_tri is not None:
                locs.append(np.array([x, y, best_z], dtype=np.float64))
                ray_ids.append(ray_i)
                tri_ids.append(best_tri)

        return (
            np.asarray(locs, dtype=np.float64).reshape(-1, 3),
            np.asarray(ray_ids, dtype=np.int64),
            np.asarray(tri_ids, dtype=np.int64),
        )

    def support_for_vertices(
        self,
        vertices_world: np.ndarray,
        *,
        contact_points_world: np.ndarray | None = None,
        collision_points_world: np.ndarray | None = None,
        bottom_slab: float = 0.02,
        max_support_z: float | None = None,
        penetration_z_margin: float = 0.01,
    ) -> tuple[float | None, int, float | None, float | None, str]:
        """Per-vertex raycast to compute support_z under the object's contact set.

        For each contact point in the bottom slab of the object: cast a ray
        straight down for support. Separately, cast upward rays from object
        surface samples to catch geometry that is already inside raised
        collision surfaces.

        Returns: (support_z, contact_vertex_count, support_span,
                  lateral_penetration, support_source).
        support_source is "background" when only downward hits were used,
        "background_collision" when at least one ray's contact came from the
        upward (penetration) hit.
        """
        verts = np.asarray(vertices_world, dtype=np.float64).reshape(-1, 3)
        if verts.size == 0:
            return None, 0, None, None, "none"

        intersector = self._ray_intersector()
        if intersector is None:
            return None, 0, None, None, "none"

        obj_min_z = float(verts[:, 2].min())
        obj_max_z = float(verts[:, 2].max())
        slab = max(0.0, float(bottom_slab))
        if contact_points_world is None:
            support_points = verts[verts[:, 2] <= obj_min_z + slab]
            if support_points.shape[0] == 0:
                support_points = verts
        else:
            support_points = np.asarray(contact_points_world, dtype=np.float64).reshape(-1, 3)
            if support_points.size == 0:
                support_points = verts[verts[:, 2] <= obj_min_z + slab]
            if support_points.shape[0] == 0:
                support_points = verts

        if collision_points_world is None:
            collision_points = support_points
        else:
            collision_points = np.asarray(collision_points_world, dtype=np.float64).reshape(-1, 3)
            if collision_points.size == 0:
                collision_points = support_points

        n_support = support_points.shape[0]
        n_collision = collision_points.shape[0]
        directions_down = np.tile(np.array([0.0, 0.0, -1.0]), (n_support, 1))
        directions_up = np.tile(np.array([0.0, 0.0, 1.0]), (n_collision, 1))

        def _first_hits(origins, directions):
            try:
                locs, idx_ray, _ = intersector.intersects_location(
                    ray_origins=origins,
                    ray_directions=directions,
                    multiple_hits=False,
                )
            except Exception as e:
                print(f"[init-collision] WARNING: intersects_location failed: {e}; "
                      "using numpy vertical-ray fallback")
                try:
                    locs, idx_ray, _ = self._vertical_intersects_location_numpy(
                        ray_origins=origins,
                        ray_directions=directions,
                    )
                except Exception as fallback_e:
                    print(f"[init-collision] WARNING: numpy ray fallback failed: {fallback_e}")
                    return {}
            best: dict[int, float] = {}
            for loc, ray_i in zip(locs, idx_ray):
                z = float(loc[2])
                ri = int(ray_i)
                prev = best.get(ri)
                if prev is None:
                    best[ri] = z
                elif directions[ri, 2] < 0.0 and z > prev:
                    best[ri] = z
                elif directions[ri, 2] > 0.0 and z < prev:
                    best[ri] = z
            return best

        down_hits = _first_hits(support_points, directions_down)
        up_hits = _first_hits(collision_points, directions_up)

        # Penetration heuristic: upward ray from the vertex hits a surface
        # within the object's own z-span (plus a small margin). Robust against
        # non-watertight scene meshes where trimesh.contains() gives large
        # false positives. Lateral penetration depth is capped at the object's
        # height so the diagnostic is interpretable.
        z_ceiling = obj_max_z + max(0.0, float(penetration_z_margin))
        obj_height = max(0.0, obj_max_z - obj_min_z)

        # Compute lift per bottom vertex: how much would *this* vertex need
        # to move up to clear its contact (downward support) or escape its
        # penetration (upward hit). Take the max across vertices. This avoids
        # over-lifting objects on legs where a small central penetration would
        # otherwise be combined with the deepest leg vertex to produce a
        # spurious object-height lift.
        contact_zs: list[float] = []
        per_vertex_required: list[float] = []  # contact_z - vz, can be negative
        penetration_used = False
        lateral_pen = 0.0
        for i in range(n_support):
            vz = float(support_points[i, 2])
            sz = down_hits.get(i)
            if sz is None or sz > vz:
                continue
            if (
                max_support_z is not None
                and np.isfinite(max_support_z)
                and sz > float(max_support_z)
            ):
                continue
            contact_zs.append(sz)
            per_vertex_required.append(sz - vz)

        for i in range(n_collision):
            vz = float(collision_points[i, 2])
            pz = up_hits.get(i)
            if pz is None or pz > z_ceiling or pz < vz:
                continue
            penetration_used = True
            pen_depth = min(pz - vz, obj_height) if obj_height > 0 else (pz - vz)
            lateral_pen = max(lateral_pen, pen_depth)
            contact_zs.append(pz)
            per_vertex_required.append(pz - vz)

        if not per_vertex_required:
            pen = lateral_pen if lateral_pen > 0.0 else None
            return None, 0, None, pen, "none"

        # `support_z` is reported such that the caller's existing formula
        #   lift = max(0, support_z + clearance - obj_min_z)
        # produces exactly the per-vertex required lift (plus clearance).
        max_required = max(per_vertex_required)
        support_z = obj_min_z + max_required
        span = max(contact_zs) - min(contact_zs)
        source = "background_collision" if penetration_used else "background"
        pen = lateral_pen if lateral_pen > 0.0 else None
        return support_z, len(contact_zs), float(span), pen, source

    def collision_top_z_for_aabb(self, aabb_min: np.ndarray, aabb_max: np.ndarray) -> float | None:
        aabb_min = np.asarray(aabb_min, dtype=np.float64).reshape(3)
        aabb_max = np.asarray(aabb_max, dtype=np.float64).reshape(3)
        margin = max(0.0, self.footprint_margin)
        xy_min = aabb_min[:2] - margin
        xy_max = aabb_max[:2] + margin
        z_limit = aabb_max[2] + max(0.0, self.max_surface_above_aabb)
        # Note: no `tri_z_max <= z_limit` filter — a tall wall whose top
        # extends above z_limit is still a collision the object must clear.
        # We clip the reported collision-top to z_limit so we don't try to
        # lift the object over the ceiling.
        overlap = (
            (self.tri_xy_max[:, 0] >= xy_min[0])
            & (self.tri_xy_min[:, 0] <= xy_max[0])
            & (self.tri_xy_max[:, 1] >= xy_min[1])
            & (self.tri_xy_min[:, 1] <= xy_max[1])
            & (self.tri_z_min <= z_limit)
            & (self.tri_z_max >= aabb_min[2])
        )
        if not np.any(overlap):
            return None
        clipped = np.minimum(self.tri_z_max[overlap], z_limit)
        return float(np.max(clipped))


def corrected_init_position(
    *,
    pos: np.ndarray | tuple[float, float, float],
    aabb_min: np.ndarray | None,
    aabb_max: np.ndarray | None,
    background_query: BackgroundSupportQuery | None,
    fallback_support_z: float | None,
    clearance: float,
    penetration_clearance: float | None = None,
    background_support_max_z: float | None = None,
    enabled: bool = True,
    object_vertices_world: np.ndarray | None = None,
    object_contact_points_world: np.ndarray | None = None,
    object_collision_points_world: np.ndarray | None = None,
    bottom_slab: float = 0.02,
) -> InitCollisionCorrection:
    """Lift ``pos`` so the object clears the background support / collision.

    Production (old evaluator): ``clearance`` = ``--object-support-clearance`` 0.0005,
    ``penetration_clearance`` = ``--init-collision-clearance`` 0.005 (used only when
    the support came from a penetration hit), ``bottom_slab`` 0.02, background query
    built with ``footprint_margin`` 0.015 and ``surface_normal_z`` 0.75.
    """
    pos_arr = np.asarray(pos, dtype=np.float64).reshape(3).copy()
    if not enabled or aabb_min is None or aabb_max is None:
        return InitCollisionCorrection(pos_arr, 0.0, None, "disabled", None, None)

    aabb_min = np.asarray(aabb_min, dtype=np.float64).reshape(3)
    aabb_max = np.asarray(aabb_max, dtype=np.float64).reshape(3)
    support_z: float | None = None
    support_source = "none"
    contact_vertex_count = 0
    support_span: float | None = None
    lateral_penetration: float | None = None
    object_min_z = float(aabb_min[2])

    used_vertex_path = False
    if background_query is not None and object_vertices_world is not None:
        v_support_z, v_count, v_span, v_pen, v_source = background_query.support_for_vertices(
            object_vertices_world,
            contact_points_world=object_contact_points_world,
            collision_points_world=object_collision_points_world,
            bottom_slab=bottom_slab,
            max_support_z=background_support_max_z,
        )
        if v_count > 0:
            used_vertex_path = True
            support_z = v_support_z
            support_source = v_source
            contact_vertex_count = v_count
            support_span = v_span
            lateral_penetration = v_pen
            verts = np.asarray(object_vertices_world, dtype=np.float64).reshape(-1, 3)
            if verts.size:
                object_min_z = float(verts[:, 2].min())

    if not used_vertex_path and background_query is not None:
        background_support_z = background_query.support_z_for_aabb(
            aabb_min,
            aabb_max,
            max_support_z=background_support_max_z,
        )
        if background_support_z is not None:
            support_z = float(background_support_z)
            support_source = "background"
        background_collision_z = background_query.collision_top_z_for_aabb(aabb_min, aabb_max)
        if background_collision_z is not None and (
            support_z is None or float(background_collision_z) > support_z
        ):
            support_z = float(background_collision_z)
            support_source = "background_collision"

    if fallback_support_z is not None and np.isfinite(fallback_support_z):
        fallback_z = float(fallback_support_z)
        if support_z is None:
            support_z = fallback_z
            support_source = "fallback"
        elif support_source == "background" and math.isclose(fallback_z, support_z):
            support_source = "background+fallback"

    if support_z is None:
        return InitCollisionCorrection(
            pos_arr,
            0.0,
            None,
            support_source,
            object_min_z,
            float(aabb_max[2]),
            contact_vertex_count,
            support_span,
            lateral_penetration,
        )

    effective_clearance = float(clearance)
    if support_source == "background_collision" and penetration_clearance is not None:
        effective_clearance = float(penetration_clearance)
    lift = max(0.0, float(support_z) + effective_clearance - object_min_z)
    if lift > 0.0:
        pos_arr[2] += lift
    return InitCollisionCorrection(
        pos_arr,
        lift,
        float(support_z),
        support_source,
        object_min_z,
        float(aabb_max[2]),
        contact_vertex_count,
        support_span,
        lateral_penetration,
    )
