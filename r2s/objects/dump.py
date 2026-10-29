"""Step 8a ``objects.dump``: the scene objects exactly as the old evaluator loaded them (CPU only).

CPU reproduction of the old evaluator's object loading (old evaluator
``load_scene_objects`` 3111-3439, support plane 2346-2409, ``reproject_to_plane``
2313-2343, quaternion fix-ups 3212-3221, collision guard 3238-3313, object metadata
2926-3009, guard query 6571-6593), writing ``layout.object_dump``
(``objects/stock/object_poses.json``) with ``overrides.write_object_pose_dump``. The
dump never needed Genesis: it was written after ``load_scene_objects`` and before
``en.init``, and its AABBs come from trimesh vertices.

Per key of ``masks/transformations.json`` (in file order):

1. ``count = int(key.split("_")[1])``, GLB ``<assets>/<key>.glb`` (or the asset override);
2. ``reproject_to_plane``: project ``translation`` through K and the raw extrinsics,
   intersect the pixel ray with ``z = z_plane``; ``scale_factor = Zcam_new / Zcam_orig``;
3. ``scale = attr.scale * scale_factor`` (x the scale override's multiplier);
4. ``q = scipy(rotation).wxyz``, then unless the rotation is identity
   ``q = (.7071067811865476, -.7071067811865476, 0, 0) * q``,
   ``q = (0, 0, .7071, .7071) * q``, ``q = q * (0, 0, -.7071, -.7071)`` (not renormalised);
5. collision guard: ``corrected_init_position`` against the cropped collider
   (``clearance`` 0.0005, ``penetration_clearance`` 0.005, ``background_support_max_z
   = z_plane + 0.005``, fallback support ``z_plane``, bottom slab 0.02), then the floor
   lift ``max(0, z_plane + 0.0005 - (min_z + lift))``;
6. ``ObjectPlacementSpec.from_asset(asset_file, scale, base_pos, q)`` (1500-vertex cap).

``z_plane = max(--z-plane 0.02, reference_top + --z-plane-offset 0.01, physics_top)``:
``reference_top`` = highest corner of the plane-fit box, ``physics_top`` = highest
vertex of the v2 receiver's physics footprint.

**Support-query rotation.** The old evaluator built the guard's
``BackgroundSupportQuery`` from the crop with the crop's *Genesis* load pose,
``GLB_WORLDSPACE_LOAD_QUAT_WXYZ`` (-90 deg about x); but the query loads the GLB with
trimesh, which is already world z-up, so the guard saw the collider rotated
-90 deg about x. ``--support-query-rotation legacy`` (default) keeps that, which is
what every on-disk dump used; ``identity`` is the fix.

:func:`load_scene_objects_from_layout` returns everything the sim needs to build the
object entities (specs, scale, pose, mass/friction, collision profile, material)
without re-deriving it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as ScipyRotation

from r2s.collision.crop import GLB_WORLDSPACE_LOAD_QUAT_WXYZ, IDENTITY_FRONT, IDENTITY_UP, evaluator_collider_pose
from r2s.collision.profiles import MeshCollisionQuality, ObjectCollisionSelection, select_object_collision_profile
from r2s.collision.support import (
    DEFAULT_GLB_FRONT,
    DEFAULT_GLB_UP,
    BackgroundSupportQuery,
    InitCollisionCorrection,
    align_up_and_front,
    corrected_init_position,
    mesh_world_vertices_and_contact_points,
)
from r2s.common.quat import quat_mul_np
from r2s.layout import SceneLayout
from r2s.objects import overrides
from r2s.objects.placement_types import ObjectPlacementSpec
from r2s.steps import Step

# Production values (evaluator defaults / evaluate_with_nyxsplat.sh / evaluate_pi_shadow_visual.sh).
DEFAULT_Z_PLANE = 0.02
DEFAULT_Z_PLANE_OFFSET = 0.01
DEFAULT_INIT_COLLISION_CLEARANCE = 0.005
DEFAULT_OBJECT_SUPPORT_CLEARANCE = 0.0005
DEFAULT_FOOTPRINT_MARGIN = 0.015
DEFAULT_SURFACE_NORMAL_Z = 0.75
DEFAULT_BOTTOM_SLAB = 0.02
DEFAULT_OBJECT_MASS_FLOOR_KG = 0.03
DEFAULT_COLLISION_PROFILE = "detailed"
DEFAULT_MATERIAL_MODE = "auto"
DEFAULT_SPLAT_BACKEND = "nyx"  # the preview runs used Nyx (glass -> opaque dielectric)
SUPPORT_QUERY_ROTATIONS = {"legacy": GLB_WORLDSPACE_LOAD_QUAT_WXYZ, "identity": (1.0, 0.0, 0.0, 0.0)}

# Evaluator quaternion fix-ups for the optimized GLBs (3212-3221).
_FIX_PRE_1 = (0.7071067811865476, -0.7071067811865476, 0.0, 0.0)
_FIX_PRE_2 = (0, 0, 0.7071, 0.7071)
_FIX_POST = (0, 0, -0.7071, -0.7071)


# ---------------------------------------------------------------------------
# Geometry helpers (verbatim from the evaluator)
# ---------------------------------------------------------------------------
def rot_matrix_to_wxyz(R: np.ndarray) -> tuple:
    """Evaluator's version: no re-normalisation (unlike ``r2s.common.quat.matrix_to_quat_wxyz``)."""
    quat_xyzw = ScipyRotation.from_matrix(R).as_quat()
    return (float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2]))


def reproject_to_plane(
    X_world: np.ndarray,
    K: np.ndarray,
    T_world2cam: np.ndarray,
    z_plane: float,
) -> tuple[np.ndarray, float]:
    X_world = np.asarray(X_world, dtype=np.float64).reshape(3)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    T_world2cam = np.asarray(T_world2cam, dtype=np.float64).reshape(4, 4)
    R = T_world2cam[:3, :3]
    t = T_world2cam[:3, 3]

    X_cam = R @ X_world + t
    u_h = K @ X_cam
    u = u_h[:2] / u_h[2]

    u_homo = np.array([u[0], u[1], 1.0])
    ray_cam = np.linalg.inv(K) @ u_homo
    ray_world = R.T @ ray_cam
    cam_center_world = -R.T @ t

    if abs(ray_world[2]) < 1e-12:
        raise ValueError("Ray is parallel to the z-plane; intersection undefined.")
    t_scalar = (z_plane - cam_center_world[2]) / ray_world[2]
    X_new = cam_center_world + t_scalar * ray_world

    Z_orig = (R @ X_world + t)[2]
    Z_new = (R @ X_new + t)[2]
    scale_factor = Z_new / Z_orig

    return X_new, scale_factor


def fitted_plane_top_z(descriptor_path) -> float | None:
    """Return the top-face z of the selected plane-fit box, or None."""
    if not descriptor_path or not os.path.exists(descriptor_path):
        return None
    with open(descriptor_path) as f:
        selected = json.load(f).get("selected")
    if not selected:
        return None
    center = np.asarray(selected["center"], dtype=np.float64).reshape(3)
    size = np.asarray(selected["size"], dtype=np.float64).reshape(3)
    quat = selected.get("quat_wxyz")
    if quat is None:
        return float(center[2]) + 0.5 * float(size[2])

    quat = np.asarray(quat, dtype=np.float64).reshape(4)
    quat_norm = float(np.linalg.norm(quat))
    if quat_norm <= 0.0:
        return float(center[2]) + 0.5 * float(size[2])
    quat = quat / quat_norm
    rot = ScipyRotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
    half = 0.5 * size
    corners_local = np.array(
        [
            [sx * half[0], sy * half[1], sz * half[2]]
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ],
        dtype=np.float64,
    )
    corners_world = corners_local @ rot.T + center[None, :]
    return float(corners_world[:, 2].max())


def footprint_top_z(descriptor_path, role: str) -> float | None:
    """Return the actual top-vertex z for one v2 footprint role."""
    if not descriptor_path or not os.path.exists(descriptor_path):
        return None
    with open(descriptor_path, encoding="utf-8") as handle:
        selected = json.load(handle).get("selected") or {}
    footprint = (selected.get("footprints") or {}).get(str(role)) or {}
    vertices = np.asarray(footprint.get("vertices_world") or [], dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) == 0:
        return None
    finite_z = vertices[:, 2][np.isfinite(vertices[:, 2])]
    return float(finite_z.max()) if len(finite_z) else None


def object_support_plane_z(
    operation_surface_path,
    receiver_path,
    *,
    reference_offset: float = DEFAULT_Z_PLANE_OFFSET,
    fallback: float = DEFAULT_Z_PLANE,
) -> tuple[float, float | None, float | None]:
    """``(z_plane, reference_top, physics_top)``: placement plane against the reference box and the v2 collider."""
    reference_top = fitted_plane_top_z(operation_surface_path)
    physics_top = footprint_top_z(receiver_path, "physics")
    candidates = [float(fallback)]
    if reference_top is not None:
        candidates.append(float(reference_top) + float(reference_offset))
    if physics_top is not None:
        candidates.append(float(physics_top))
    return max(candidates), reference_top, physics_top


# ---------------------------------------------------------------------------
# Object metadata: identity, mass, friction, material (evaluator 2819-3091)
# ---------------------------------------------------------------------------
_OBJECT_ROLE_SUFFIXES = {"target", "destination"}
_OBJECT_MATERIAL_PRESETS: dict[str, tuple[str, float | None, float | None]] = {
    # label: (surface kind, metallic, roughness)
    "plastic": ("bsdf", 0.0, 0.45),
    "rough": ("bsdf", 0.0, 0.90),
    "smooth": ("bsdf", 0.0, 0.20),
    "reflective": ("bsdf", 0.0, 0.05),
    "metal": ("bsdf", 1.0, 0.25),
    "iron": ("bsdf", 1.0, 0.30),
    "aluminium": ("bsdf", 1.0, 0.22),
    "aluminum": ("bsdf", 1.0, 0.22),
    "copper": ("bsdf", 1.0, 0.20),
    "gold": ("bsdf", 1.0, 0.18),
    "glass": ("glass", 0.0, 0.05),
    "water": ("glass", 0.0, 0.08),
    "emission": ("emission", None, None),
}


@dataclass(frozen=True)
class ObjectMaterialResolution:
    """Texture-preserving visual surface selected from scene metadata.

    The old evaluator also built the Eden surface here; the sim builds it from
    ``surface_kind``: ``bsdf`` -> ``BSDFSurfaceOptions(metallic, roughness)``,
    ``glass`` -> ``GlassSurfaceOptions(roughness)``, ``emission`` ->
    ``EmissionSurfaceOptions()``; ``status == "asset"`` -> no surface (the GLB's own).
    """

    raw_label: str | None
    normalized_label: str | None
    metadata_source: str | None
    status: str
    surface_kind: str | None = None
    metallic: float | None = None
    roughness: float | None = None
    fallback_reason: str | None = None

    def to_json_dict(self) -> dict[str, object]:
        return asdict(self)


def normalize_object_name(name: object) -> str:
    """Normalize semantic names while dropping dataset-only role suffixes."""
    text = re.sub(r"[^a-z0-9]+", " ", str(name or "").lower()).strip()
    tokens = text.split()
    while tokens and tokens[-1] in _OBJECT_ROLE_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


def resolve_object_mass(phys: dict, *, mass_floor: float = DEFAULT_OBJECT_MASS_FLOOR_KG) -> tuple[float, float]:
    """``(effective_mass, raw_mass)``: metadata mass (default 0.1 kg) clamped to the floor."""
    raw_mass = float(phys.get("mass", 0.1))
    return max(raw_mass, float(mass_floor)), raw_mass


def normalize_object_material_label(label: object) -> str:
    """Normalize Gemini's coarse surface/material label."""
    return re.sub(r"[^a-z0-9]+", " ", str(label or "").lower()).strip()


def _read_object_metadata_file(path) -> list[dict]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[object metadata] WARNING: could not read {path}: {exc}")
        return []
    if not isinstance(data, list):
        print(f"[object metadata] WARNING: expected a list in {path}; ignoring it")
        return []
    return [entry for entry in data if isinstance(entry, dict)]


def _entry_counts(entry: dict) -> list[int]:
    counts = entry.get("count", [])
    if isinstance(counts, (int, float, str)):
        counts = [counts]
    resolved: list[int] = []
    for count in counts:
        try:
            resolved.append(int(count))
        except (TypeError, ValueError):
            continue
    return resolved


def metadata_sources(inputs) -> list[tuple[str, Path]]:
    """The evaluator's metadata files, in priority order (``masks/result.json`` owns the identities)."""
    return [
        ("masks/result.json", inputs.mask_result),
        ("masks/physical_properties.json", inputs.masks_dir / "physical_properties.json"),
        ("physical_properties.json", inputs.meta_dir / "physical_properties.json"),
    ]


def load_scene_object_metadata(source_specs: list[tuple[str, Path]]) -> dict[int, dict]:
    """Resolve canonical object identities and merge richer Gemini properties.

    ``masks/result.json`` owns the segmentation index. Richer files are joined
    by normalized semantic name first, since some root-level enrichment files
    reorder their ``count`` values.
    """
    sources = [(source_name, _read_object_metadata_file(path)) for source_name, path in source_specs]

    canonical_source = next(((source_name, entries) for source_name, entries in sources if entries), None)
    if canonical_source is None:
        return {}

    canonical_name, canonical_entries = canonical_source
    resolved: dict[int, dict] = {}
    for entry in canonical_entries:
        for count in _entry_counts(entry):
            item = dict(entry)
            item["_identity_source"] = canonical_name
            item["_field_sources"] = {key: canonical_name for key in entry if key not in {"count", "object_name"}}
            item["_metadata_sources"] = [canonical_name]
            resolved[count] = item

    for source_name, entries in sources:
        if source_name == canonical_name or not entries:
            continue
        by_name: dict[str, list[dict]] = {}
        by_count: dict[int, list[dict]] = {}
        for entry in entries:
            normalized_name = normalize_object_name(entry.get("object_name"))
            if normalized_name:
                by_name.setdefault(normalized_name, []).append(entry)
            for count in _entry_counts(entry):
                by_count.setdefault(count, []).append(entry)

        for count, item in resolved.items():
            canonical_normalized = normalize_object_name(item.get("object_name"))
            candidates = by_name.get(canonical_normalized, []) if canonical_normalized else []
            match = candidates[0] if len(candidates) == 1 else None
            if match is None:
                count_candidates = by_count.get(count, [])
                if len(count_candidates) == 1:
                    candidate = count_candidates[0]
                    candidate_normalized = normalize_object_name(candidate.get("object_name"))
                    if not canonical_normalized or not candidate_normalized or candidate_normalized == canonical_normalized:
                        match = candidate
            if match is None:
                continue

            item["_metadata_sources"].append(source_name)
            field_sources = item["_field_sources"]
            for key, value in match.items():
                if key in {"count", "object_name"} or value is None:
                    continue
                item[key] = value
                field_sources[key] = source_name

    return resolved


def resolve_object_visual_material(metadata: dict, *, mode: str, splat_backend: str) -> ObjectMaterialResolution:
    """Map coarse Gemini metadata to a texture-preserving surface (evaluator 3012-3074)."""
    label_field = next(
        (f for f in ("material", "surface") if metadata.get(f) is not None and str(metadata[f]).strip()),
        "material",
    )
    raw_label = metadata.get(label_field)
    source = (metadata.get("_field_sources") or {}).get(label_field)
    if mode == "asset":
        return ObjectMaterialResolution(
            raw_label=None if raw_label is None else str(raw_label),
            normalized_label=None if raw_label is None else normalize_object_material_label(raw_label),
            metadata_source=source,
            status="asset",
            fallback_reason="object material mode is asset",
        )
    if raw_label is None or not str(raw_label).strip():
        return ObjectMaterialResolution(
            raw_label=None, normalized_label=None, metadata_source=source, status="asset",
            fallback_reason="material metadata is missing",
        )
    normalized = normalize_object_material_label(raw_label)
    preset = _OBJECT_MATERIAL_PRESETS.get(normalized)
    if preset is None:
        return ObjectMaterialResolution(
            raw_label=str(raw_label), normalized_label=normalized, metadata_source=source, status="asset",
            fallback_reason=f"unknown material label: {raw_label!r}",
        )
    surface_kind, metallic, roughness = preset
    effective_kind = surface_kind
    fallback_reason = None
    if surface_kind == "glass" and splat_backend == "nyx":
        # The Nyx exporter does not expose transmission/IOR.
        effective_kind = "bsdf"
        fallback_reason = "Nyx glass transmission is unavailable; using opaque dielectric"
    return ObjectMaterialResolution(
        raw_label=str(raw_label),
        normalized_label=normalized,
        metadata_source=source,
        status="metadata",
        surface_kind=effective_kind,
        metallic=metallic,
        roughness=roughness,
        fallback_reason=fallback_reason,
    )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
@dataclass
class DumpSettings:
    """Object-loading options; the defaults are the production preview run's."""

    z_plane: float = DEFAULT_Z_PLANE
    z_plane_offset: float = DEFAULT_Z_PLANE_OFFSET
    init_collision: bool = True
    init_collision_clearance: float = DEFAULT_INIT_COLLISION_CLEARANCE
    object_support_clearance: float = DEFAULT_OBJECT_SUPPORT_CLEARANCE
    footprint_margin: float = DEFAULT_FOOTPRINT_MARGIN
    surface_normal_z: float = DEFAULT_SURFACE_NORMAL_Z
    bottom_slab: float = DEFAULT_BOTTOM_SLAB
    support_query_rotation: str = "legacy"
    asset_override: str | None = None
    scale_override: str | None = None
    collision_override: str | None = None
    # entity metadata only (not in the dump)
    collision_profile: str = DEFAULT_COLLISION_PROFILE
    collision_decimate: bool | None = None
    decompose_error_threshold: float = 0.05
    coacd_threshold: float = 0.03
    legacy_face_num: int = 1000
    mass_floor: float = DEFAULT_OBJECT_MASS_FLOOR_KG
    material_mode: str = DEFAULT_MATERIAL_MODE
    splat_backend: str = DEFAULT_SPLAT_BACKEND

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "DumpSettings":
        return cls(**{k: getattr(args, k) for k in cls.__dataclass_fields__ if hasattr(args, k)})


@dataclass
class LoadedObject:
    """One scene object as the evaluator built it (everything ``EntityOptions`` and the dump need)."""

    key: str  # transformations.json key (image_0000_segmented)
    safe_key: str  # entity name (obj_image_0000_segmented)
    count: int
    object_name: str
    asset_file: str
    scale: float
    scale_multiplier: float
    pos: np.ndarray  # default_root_pos (after the guard)
    quat_wxyz: tuple  # default_root_quat
    reprojected_pos: np.ndarray  # before the guard
    spec: ObjectPlacementSpec | None
    mass: float
    raw_mass: float
    friction: float
    collision: ObjectCollisionSelection
    material: ObjectMaterialResolution
    correction: InitCollisionCorrection | None = None
    floor_lift: float = 0.0
    metadata_sources: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        c = self.correction
        return {
            "count": self.count,
            "object_name": self.object_name,
            "scale": self.scale,
            "pos": [float(v) for v in self.pos],
            "lift": None if c is None else float(c.lift),
            "support_source": None if c is None else c.support_source,
            "floor_lift": float(self.floor_lift),
            "mass": self.mass,
            "friction": self.friction,
            "collision_profile": self.collision.selected_profile,
            "material": self.material.surface_kind or self.material.status,
            "warnings": list(self.warnings),
        }


@dataclass
class SceneObjects:
    scene: str
    scene_meta_dir: str
    z_plane: float
    reference_top: float | None
    physics_top: float | None
    objects: dict[str, LoadedObject]
    skipped: list[str] = field(default_factory=list)

    def placement_specs(self) -> dict[str, ObjectPlacementSpec]:
        return {k: o.spec for k, o in self.objects.items() if o.spec is not None}


def build_support_query(
    crop_glb,
    *,
    rotation: str = "legacy",
    footprint_margin: float = DEFAULT_FOOTPRINT_MARGIN,
    surface_normal_z: float = DEFAULT_SURFACE_NORMAL_Z,
) -> BackgroundSupportQuery:
    """The guard's query on the baked crop: identity scale/position, z-up axes, ``rotation`` (see module doc)."""
    if rotation not in SUPPORT_QUERY_ROTATIONS:
        raise ValueError(f"unknown support-query rotation {rotation!r}; choose from {sorted(SUPPORT_QUERY_ROTATIONS)}")
    return BackgroundSupportQuery(
        crop_glb,
        scene_scale=1.0,
        root_pos=(0.0, 0.0, 0.0),
        root_quat_wxyz=SUPPORT_QUERY_ROTATIONS[rotation],
        footprint_margin=footprint_margin,
        surface_normal_z=surface_normal_z,
        mesh_up=IDENTITY_UP,
        mesh_front=IDENTITY_FRONT,
    )


def load_scene_objects(
    *,
    transformations_path,
    intrinsics_path,
    extrinsics_path,
    assets_dir,
    metadata: dict[int, dict],
    z_plane: float,
    support_query: BackgroundSupportQuery | None,
    settings: DumpSettings | None = None,
    scale_multipliers: dict[str, float] | None = None,
    asset_overrides: dict[str, str] | None = None,
    collision_overrides: dict[str, dict] | None = None,
    log=print,
) -> tuple[dict[str, LoadedObject], list[str]]:
    """Old ``load_scene_objects`` without Genesis: ``({safe_key: LoadedObject}, skipped keys)``."""
    st = settings or DumpSettings()
    with open(transformations_path) as f:
        transformations = json.load(f)
    K_scene = np.load(intrinsics_path).astype(np.float64)
    T_w2c_scene = np.load(extrinsics_path).astype(np.float64).reshape(4, 4)
    fallback_support_z = z_plane
    background_support_max_z = z_plane + st.init_collision_clearance

    loaded: dict[str, LoadedObject] = {}
    skipped: list[str] = []
    for key, attributes in transformations.items():
        count = int(key.split("_")[1])
        asset_file = os.path.join(str(assets_dir), key + ".glb")
        warnings: list[str] = []
        if asset_overrides:
            override = overrides.asset_for(asset_overrides, key)
            if override:
                log(f"[object asset override] {key}: {asset_file} -> {override}")
                asset_file = override
        if not os.path.exists(asset_file):
            log(f"[objects] WARNING: GLB not found, skipping — {asset_file}")
            skipped.append(key)
            continue

        raw_pos = np.array(attributes["translation"], dtype=np.float64)
        try:
            pos_ra, scale_factor = reproject_to_plane(raw_pos, K_scene, T_w2c_scene, z_plane)
        except ValueError as e:
            log(f"[objects] WARNING: reprojection failed for {key}: {e}; using raw position")
            warnings.append(f"reprojection failed: {e}")
            pos_ra = raw_pos
            scale_factor = 1.0
        if not np.isfinite(scale_factor) or scale_factor <= 0:
            log(f"[objects] WARNING: non-positive scale_factor={scale_factor} for {key}; skipping")
            skipped.append(key)
            continue
        scale = float(attributes["scale"]) * scale_factor
        scale_mult = 1.0
        if scale_multipliers:
            scale_mult = overrides.multiplier_for(scale_multipliers, key)
            if scale_mult != 1.0:
                log(f"[object scale override] {key}: x{scale_mult:.4f}  scale {scale:.5f} -> {scale * scale_mult:.5f}")
                scale *= scale_mult

        R_mat = np.array(attributes["rotation"], dtype=np.float64)
        quat_wxyz = rot_matrix_to_wxyz(R_mat)
        if np.allclose(R_mat, np.eye(3)):
            log(f"[objects] WARNING: rotation matrix for {key} is identity; using original rotation")
            warnings.append("identity rotation: quaternion fix-ups skipped")
        else:
            quat_wxyz = quat_mul_np(_FIX_PRE_1, quat_wxyz)
            quat_wxyz = quat_mul_np(_FIX_PRE_2, quat_wxyz)
            quat_wxyz = quat_mul_np(quat_wxyz, _FIX_POST)

        phys = metadata.get(count, {})
        obj_name = phys.get("object_name", key)
        mass, raw_mass = resolve_object_mass(phys, mass_floor=st.mass_floor)
        friction = float(phys.get("friction", 0.5))
        material = resolve_object_visual_material(phys, mode=st.material_mode, splat_backend=st.splat_backend)

        pos_eden_arr = np.asarray(pos_ra, dtype=np.float64).reshape(3)
        reprojected = pos_eden_arr.copy()
        correction: InitCollisionCorrection | None = None
        floor_lift = 0.0
        if st.init_collision:
            try:
                init_pos_eden_arr = pos_eden_arr.copy()
                mesh_import_rot = align_up_and_front(np.asarray(DEFAULT_GLB_UP), np.asarray(DEFAULT_GLB_FRONT))
                vertices_world, contact_points_world, collision_points_world, aabb_min, aabb_max = mesh_world_vertices_and_contact_points(
                    asset_file,
                    scale=scale,
                    root_pos=pos_eden_arr,
                    root_quat_wxyz=quat_wxyz,
                    mesh_import_rot=mesh_import_rot,
                    bottom_slab=st.bottom_slab,
                )
                correction = corrected_init_position(
                    pos=pos_eden_arr,
                    aabb_min=aabb_min,
                    aabb_max=aabb_max,
                    background_query=support_query,
                    fallback_support_z=fallback_support_z,
                    clearance=st.object_support_clearance,
                    penetration_clearance=st.init_collision_clearance,
                    background_support_max_z=background_support_max_z,
                    enabled=True,
                    object_vertices_world=vertices_world,
                    object_contact_points_world=contact_points_world,
                    object_collision_points_world=collision_points_world,
                    bottom_slab=st.bottom_slab,
                )
                pos_eden_arr = correction.pos
                if fallback_support_z is not None and np.isfinite(fallback_support_z):
                    object_min_z = correction.object_min_z
                    if object_min_z is None:
                        object_min_z = float(aabb_min[2])
                    applied_lift = float(pos_eden_arr[2] - init_pos_eden_arr[2])
                    current_object_min_z = float(object_min_z) + applied_lift
                    target_floor_z = float(fallback_support_z) + float(st.object_support_clearance)
                    floor_lift = max(0.0, target_floor_z - current_object_min_z)
                    if floor_lift > 0.0:
                        pos_eden_arr = pos_eden_arr.copy()
                        pos_eden_arr[2] += floor_lift
                if correction.lift > 0.0 or floor_lift > 0.0 or correction.support_z is not None:
                    details = [
                        f"lift={correction.lift:.5f}",
                        f"support={correction.support_z if correction.support_z is not None else 'none'}",
                        f"source={correction.support_source}",
                    ]
                    if floor_lift > 0.0:
                        details.append(f"floor_lift={floor_lift:.5f}")
                        details.append(f"floor={float(fallback_support_z):.5f}")
                    if correction.contact_vertex_count:
                        details.append(f"vertices={correction.contact_vertex_count}")
                    if correction.lateral_penetration is not None:
                        details.append(f"penetration={correction.lateral_penetration:.5f}")
                    log(f"[init-collision] {key}: " + "  ".join(details))
            except Exception as e:  # old behaviour: keep the uncorrected z
                log(f"[init-collision] WARNING: failed for {key}: {e}; using uncorrected z")
                warnings.append(f"collision guard failed: {e}")

        pos_eden = tuple(float(v) for v in pos_eden_arr)
        log(f"[objects] {obj_name:40s}  pos={pos_eden} quat={quat_wxyz} scale={scale:.5f}  mass={mass}  friction={friction}")
        safe_key = "obj_" + key.replace("-", "_")
        try:
            spec = ObjectPlacementSpec.from_asset(
                safe_key=safe_key,
                count=count,
                object_name=obj_name,
                asset_file=asset_file,
                scale=scale,
                base_pos=pos_eden_arr,
                base_quat_wxyz=quat_wxyz,
            )
        except Exception as exc:  # old behaviour: the object is missing from the dump
            log(f"[objects] WARNING: could not capture placement geometry for {key}: {exc}")
            warnings.append(f"placement spec failed: {exc}")
            spec = None

        try:
            collision = select_object_collision_profile(
                asset_file,
                requested_profile=st.collision_profile,
                inspect_quality=st.collision_profile == "auto",
                decimate_override=st.collision_decimate,
                detailed_decompose_error_threshold=st.decompose_error_threshold,
                detailed_coacd_threshold=st.coacd_threshold,
                legacy_face_num=st.legacy_face_num,
            )
        except Exception as exc:
            log(f"[collision profile] WARNING: inspection failed for {key}: {exc}; using legacy")
            collision = ObjectCollisionSelection(
                requested_profile=st.collision_profile,
                selected_profile="legacy",
                decimate=True if st.collision_decimate is None else bool(st.collision_decimate),
                decimate_face_num=int(st.legacy_face_num),
                decimate_aggressiveness=3,
                decompose_object_error_threshold=0.15,
                coacd_threshold=0.05,
                quality=MeshCollisionQuality(0, 0, 0, False, 0.0, 0.0, 99, ("inspection-failed",)),
            )
        if collision_overrides:
            fields = overrides.collision_override_for(collision_overrides, key)
            if fields:
                collision = overrides.apply_collision_override(collision, fields)
                log(f"[object collision override] {key}: {fields}")

        loaded[safe_key] = LoadedObject(
            key=key,
            safe_key=safe_key,
            count=count,
            object_name=str(obj_name),
            asset_file=asset_file,
            scale=float(scale),
            scale_multiplier=float(scale_mult),
            pos=np.asarray(pos_eden_arr, dtype=np.float64).copy(),
            quat_wxyz=tuple(float(v) for v in quat_wxyz),
            reprojected_pos=reprojected,
            spec=spec,
            mass=float(mass),
            raw_mass=float(raw_mass),
            friction=friction,
            collision=collision,
            material=material,
            correction=correction,
            floor_lift=float(floor_lift),
            metadata_sources=list(phys.get("_metadata_sources", [])),
            warnings=warnings,
        )
    return loaded, skipped


def _load_override(path, scene: str, what: str, reader):
    path = overrides.require_file(overrides.scene_path(path, scene), what)
    return None if path is None else reader(path)


def check_crop_is_current(layout: SceneLayout) -> None:
    """Fail when the crop's sidecar was built for another collider pose than ``layout.refined_pose``."""
    sidecar = layout.cropped_collider_glb.with_name(layout.cropped_collider_glb.name + ".json")
    meta = json.loads(sidecar.read_text())
    pos, quat, scale = evaluator_collider_pose(layout.refined_pose, layout.refined_scale)
    expected = {"root_pos": list(pos), "root_quat_wxyz": list(quat), "scene_scale": scale, "baked_world_space": True}
    stale = {k: (meta.get(k), v) for k, v in expected.items() if meta.get(k) != v}
    if stale:
        raise ValueError(f"{layout.cropped_collider_glb} is stale for {layout.refined_pose} (rerun collision.crop): {stale}")


def load_scene_objects_from_layout(
    layout: SceneLayout, settings: DumpSettings | None = None, *, log=print
) -> SceneObjects:
    """Everything the sim needs to build the scene objects, from the layout's files.

    Reads the step-0 inputs, ``layout.plane_fit_json`` / ``layout.receiver_json``
    (support plane) and ``layout.cropped_collider_glb`` (collision guard).
    """
    st = settings or DumpSettings()
    inputs = layout.inputs
    z_plane, reference_top, physics_top = object_support_plane_z(
        layout.plane_fit_json, layout.receiver_json, reference_offset=st.z_plane_offset, fallback=st.z_plane
    )
    log(
        f"[objects] z_plane = {z_plane:.5f} (reference_top={reference_top if reference_top is not None else 'none'} "
        f"offset={st.z_plane_offset:.5f} v2_physics_top={physics_top if physics_top is not None else 'none'})"
    )
    query = None
    if st.init_collision:
        check_crop_is_current(layout)
        query = build_support_query(
            layout.cropped_collider_glb,
            rotation=st.support_query_rotation,
            footprint_margin=st.footprint_margin,
            surface_normal_z=st.surface_normal_z,
        )
    metadata = load_scene_object_metadata(metadata_sources(inputs))
    objects, skipped = load_scene_objects(
        transformations_path=inputs.transformations,
        intrinsics_path=inputs.intrinsics,
        extrinsics_path=inputs.extrinsics,
        assets_dir=inputs.assets_dir,
        metadata=metadata,
        z_plane=z_plane,
        support_query=query,
        settings=st,
        scale_multipliers=_load_override(st.scale_override, layout.scene, "scale override", overrides.load_object_scale_multipliers),
        asset_overrides=_load_override(st.asset_override, layout.scene, "asset override", overrides.load_object_asset_overrides),
        collision_overrides=_load_override(st.collision_override, layout.scene, "collision override", overrides.load_object_collision_overrides),
        log=log,
    )
    return SceneObjects(
        scene=layout.scene,
        scene_meta_dir=str(inputs.meta_dir),
        z_plane=z_plane,
        reference_top=reference_top,
        physics_top=physics_top,
        objects=objects,
        skipped=skipped,
    )


def dump_scene_objects(scene: SceneObjects, out_path) -> str:
    return overrides.write_object_pose_dump(
        out_path,
        scene.placement_specs(),
        scene_name=scene.scene,
        scene_meta_dir=scene.scene_meta_dir,
        z_plane=scene.z_plane,
        physics_top=scene.physics_top,
        reference_top=scene.reference_top,
    )


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------
def add_arguments(parser: argparse.ArgumentParser) -> None:
    d = DumpSettings()
    parser.add_argument("--z-plane", type=float, default=d.z_plane, help="fallback placement plane (m)")
    parser.add_argument("--z-plane-offset", type=float, default=d.z_plane_offset, help="offset above the plane-fit box top (m)")
    parser.add_argument("--no-init-collision", dest="init_collision", action="store_false", help="skip the load-time collision guard")
    parser.add_argument("--init-collision-clearance", type=float, default=d.init_collision_clearance, help="clearance when escaping a penetration (m)")
    parser.add_argument("--object-support-clearance", type=float, default=d.object_support_clearance, help="resting gap above a support (m)")
    parser.add_argument("--footprint-margin", type=float, default=d.footprint_margin)
    parser.add_argument("--surface-normal-z", type=float, default=d.surface_normal_z)
    parser.add_argument("--bottom-slab", type=float, default=d.bottom_slab)
    parser.add_argument(
        "--support-query-rotation",
        choices=sorted(SUPPORT_QUERY_ROTATIONS),
        default=d.support_query_rotation,
        help="legacy: the old evaluator's -90 deg x rotation of the crop in the guard (reproduces every old dump); identity: the fix",
    )
    parser.add_argument("--asset-override", default=None, help="candidate JSON with asset_override GLBs ('{scene}' is replaced)")
    parser.add_argument("--scale-override", default=None, help="candidate JSON with scale_multiplier per object ('{scene}' is replaced)")
    parser.add_argument("--collision-override", default=None, help="candidate JSON with collision blocks (metadata only; '{scene}' is replaced)")


def run(layout: SceneLayout, args: argparse.Namespace) -> dict[str, Any]:
    scene = load_scene_objects_from_layout(layout, DumpSettings.from_args(args))
    if scene.skipped:
        print(f"[objects] WARNING: skipped {scene.skipped}")
    out = dump_scene_objects(scene, layout.object_dump)
    print(f"[object pose] dumped {len(scene.placement_specs())} loaded pose(s) -> {out}")
    return {
        "z_plane": scene.z_plane,
        "reference_top": scene.reference_top,
        "physics_top": scene.physics_top,
        "skipped": scene.skipped,
        "objects": {k: o.summary() for k, o in scene.objects.items()},
    }


def _inputs(layout: SceneLayout, args: argparse.Namespace) -> list[Path]:
    inputs = layout.inputs
    out = [inputs.transformations, inputs.intrinsics, inputs.extrinsics, inputs.assets_dir, layout.plane_fit_json, layout.receiver_json]
    if args.init_collision:
        crop = layout.cropped_collider_glb
        out += [crop, crop.with_name(crop.name + ".json"), layout.refined_pose, layout.refined_scale]
    out += [p for _, p in metadata_sources(inputs)]
    for p in (args.asset_override, args.scale_override, args.collision_override):
        if p:
            out.append(Path(overrides.scene_path(p, layout.scene)))
    return out


def _optional_inputs(layout: SceneLayout, args: argparse.Namespace) -> list[Path]:
    # metadata files: the first non-empty one owns the identities; any may be absent (old behaviour)
    return [p for _, p in metadata_sources(layout.inputs)]


STEP = Step(
    name="objects.dump",
    help="load the scene objects as the old evaluator did (plane reprojection, collision guard) -> objects/stock/object_poses.json",
    add_arguments=add_arguments,
    run=run,
    inputs=_inputs,
    outputs=lambda layout, args: [layout.object_dump],
    needs="cpu",
    optional_inputs=_optional_inputs,
)
