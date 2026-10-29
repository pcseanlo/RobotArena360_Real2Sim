"""Per-object overrides read from a candidate / refined-poses JSON: pose, scale, asset, collision.

Port of the old ``object_pose_override.py``, ``object_scale_override.py``,
``object_asset_override.py`` and ``object_collision_override.py`` (library parts;
the evaluator argparse glue is gone). Every reader accepts ``{"objects": [...]}``
(``refined_poses.json``, ``candidate.json``) or a bare list, keyed by ``safe_key``
(``obj_image_0000_segmented``; the ``obj_`` prefix is optional for the scale /
asset / collision readers, which return ``transformations.json`` keys).

How the old evaluator applied them (the sim assembly must do the same):

* **asset** (``load_object_asset_overrides``) - at load time, the GLB used for
  rendering *and* collision; everything downstream (collision profile, dump,
  guard) sees the replacement.
* **scale** (``load_object_scale_multipliers``) - at load time, the mesh scale
  (``transformations.json`` scale x plane-reprojection factor) is multiplied
  *before* the collision guard, the placement spec and ``EntityOptions``.
* **collision** (``load_object_collision_overrides`` + ``apply_collision_override``)
  - replaces fields of the automatic ``ObjectCollisionSelection`` before the
  entity is built.
* **pose** (``load_object_pose_override`` -> ``resolve_override_for_run`` ->
  ``apply_object_pose_override``) - after every ``env.reset()``. The refine
  previews (candidate-settle) used ``mode="absolute"`` and
  ``max_displacement=0.05``; the evaluator CLI default is ``delta`` / 0.01.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from r2s.common.io import prepare_output
from r2s.objects.placement_types import ObjectPlacementSpec, PlacedPose

SAFE_KEY_PREFIX = "obj_"


def scene_path(path: str | os.PathLike[str] | None, scene: str | None) -> str | None:
    """Replace ``{scene}`` in a per-scene override path (the old launchers' convention)."""
    if path is None:
        return None
    path = str(path)
    if "{scene}" in path and scene:
        path = path.replace("{scene}", str(scene))
    return path


def normalize_key(key: str) -> str:
    """Map either key form to the evaluator's ``transformations.json`` key."""
    key = str(key)
    return key[len(SAFE_KEY_PREFIX):] if key.startswith(SAFE_KEY_PREFIX) else key


def _entries(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload["objects"] if isinstance(payload, dict) else payload


# ===========================================================================
# Pose dump / override (old object_pose_override.py)
# ===========================================================================
def write_object_pose_dump(
    path: str | os.PathLike[str],
    placement_specs: dict[str, ObjectPlacementSpec],
    *,
    scene_name: str | None,
    scene_meta_dir: str | None,
    z_plane: float | None,
    physics_top: float | None = None,
    reference_top: float | None = None,
) -> str:
    """Write the poses the evaluator loaded (``object_poses.json``, the input of steps 8b and 9).

    ``placement_specs`` are captured at load time exactly as the evaluator did
    (``ObjectPlacementSpec.from_asset(asset_file, scale, base_pos=<pos after the
    collision guard>, base_quat_wxyz=<load quat>)`` with the default 1500-vertex cap).
    """
    objects = []
    for safe_key, spec in placement_specs.items():
        aabb_min, aabb_max = spec.base_aabb
        objects.append(
            {
                "safe_key": safe_key,
                "count": int(spec.count),
                "object_name": spec.object_name,
                "asset_file": spec.asset_file,
                "scale": float(spec.scale),
                "pos": [float(v) for v in spec.base_pos],
                "quat_wxyz": [float(v) for v in spec.base_quat_wxyz],
                "aabb_min": [float(v) for v in aabb_min],
                "aabb_max": [float(v) for v in aabb_max],
            }
        )
    payload = {
        "version": 1,
        "scene": scene_name,
        "scene_meta_dir": scene_meta_dir,
        "z_plane": None if z_plane is None else float(z_plane),
        # Height of the physics receiver surface the objects actually settle on
        # at runtime; z_plane (used for the load-time placement) can sit several
        # cm above it, so the fit must rest objects on physics_top instead.
        "physics_top": None if physics_top is None or not math.isfinite(float(physics_top)) else float(physics_top),
        "reference_top": None if reference_top is None or not math.isfinite(float(reference_top)) else float(reference_top),
        "objects": objects,
    }
    out = prepare_output(path)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return str(out)


@dataclass
class ObjectPoseOverride:
    """Poses to apply plus each object's original (load-time) pose for reverts."""

    placed: list[PlacedPose]
    originals: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    source: str = ""


def load_object_pose_override(path: str | os.PathLike[str]) -> ObjectPoseOverride:
    path = str(path)
    entries = _entries(path)
    placed: list[PlacedPose] = []
    originals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for entry in entries:
        if entry.get("apply", True) is False:
            continue
        pos = np.asarray(entry["pos"], dtype=np.float64).reshape(3)
        quat = np.asarray(entry["quat_wxyz"], dtype=np.float64).reshape(4)
        quat = quat / max(float(np.linalg.norm(quat)), 1e-12)
        xy_delta = np.asarray(entry.get("xy_delta_m", [0.0, 0.0]), dtype=np.float64).reshape(2)
        key = str(entry["safe_key"])
        placed.append(
            PlacedPose(
                safe_key=key,
                object_name=str(entry.get("object_name", key)),
                pos=pos,
                quat_wxyz=quat,
                yaw_delta_rad=math.radians(float(entry.get("yaw_delta_deg", 0.0))),
                xy_delta=xy_delta,
                status=str(entry.get("status", "override")),
            )
        )
        if "base_pos" in entry and "base_quat_wxyz" in entry:
            # The dump's load-time pose (before levelling/dropping), i.e. what the
            # evaluator placed on its own; used to revert an object that the
            # simulator rejects (collision hull contact -> ejection).
            originals[key] = (
                np.asarray(entry.get("load_pos") or entry["base_pos"], dtype=np.float64).reshape(3),
                np.asarray(entry.get("load_quat_wxyz") or entry["base_quat_wxyz"], dtype=np.float64).reshape(4),
            )
    return ObjectPoseOverride(placed=placed, originals=originals, source=path)


def _quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def resolve_override_for_run(
    override: ObjectPoseOverride,
    placement_specs: dict[str, ObjectPlacementSpec] | None,
    *,
    mode: str = "absolute",
) -> tuple[ObjectPoseOverride, list[str]]:
    """Adapt the file's poses to this run's own load-time poses.

    ``absolute`` keeps the file's poses and only swaps in this run's load-time
    pose as the revert target.  ``delta`` re-applies the file's change (the
    fitted pose relative to the pose the fit started from) on top of the pose
    this run loaded: same XY shift and the same world-frame rotation, keeping
    this run's own height, so a launcher whose ground or collision guard
    differs keeps its own resting height (the settle steps handle the drop).
    """
    if mode not in {"absolute", "delta"}:
        raise ValueError(f"unknown override mode {mode!r}")
    lines: list[str] = []
    if not placement_specs:
        if mode == "delta":
            lines.append("WARNING: delta mode requested but this run captured no load-time poses; using absolute poses")
        return override, lines
    placed: list[PlacedPose] = []
    originals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for pose in override.placed:
        spec = placement_specs.get(pose.safe_key)
        if spec is None:
            lines.append(f"WARNING: {pose.safe_key} not loaded in this run; skipping its override")
            continue
        own_pos = np.asarray(spec.base_pos, dtype=np.float64).reshape(3)
        own_quat = np.asarray(spec.base_quat_wxyz, dtype=np.float64).reshape(4)
        originals[pose.safe_key] = (own_pos, own_quat)
        new_pos, new_quat = pose.pos, pose.quat_wxyz
        if mode == "delta":
            file_orig = override.originals.get(pose.safe_key)
            if file_orig is None:
                lines.append(f"WARNING: {pose.safe_key} has no load-time pose in the file; applying its absolute pose")
            else:
                file_pos, file_quat = file_orig
                d_pos = pose.pos - file_pos
                d_quat = _quat_mul(pose.quat_wxyz, _quat_conj(file_quat))  # world-frame rotation
                # XY only: the fit's z change (resting on the fit run's physics
                # surface) does not transfer to a run whose ground differs, and
                # the settle steps drop the object onto this run's surface anyway.
                new_pos = own_pos + np.array([d_pos[0], d_pos[1], 0.0])
                new_quat = _quat_mul(d_quat, own_quat)
                new_quat = new_quat / max(float(np.linalg.norm(new_quat)), 1e-12)
                shift = np.linalg.norm(own_pos - file_pos)
                if shift > 0.02:
                    lines.append(
                        f"note: {pose.safe_key} load-time pose differs from the fit's by {100 * shift:.1f} cm; "
                        "delta applied on top of this run's pose"
                    )
        placed.append(
            PlacedPose(
                safe_key=pose.safe_key,
                object_name=pose.object_name,
                pos=np.asarray(new_pos, dtype=np.float64),
                quat_wxyz=np.asarray(new_quat, dtype=np.float64),
                yaw_delta_rad=pose.yaw_delta_rad,
                xy_delta=pose.xy_delta,
                status=pose.status,
            )
        )
    return ObjectPoseOverride(placed=placed, originals=originals, source=override.source), lines


def _entity_pos(entity) -> np.ndarray | None:
    getter = getattr(entity, "get_pos", None)
    if getter is None:
        return None
    try:
        value = getter(relative=True)
    except TypeError:
        value = getter()
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64).reshape(-1, 3)[0]


def _set_entity_pose(entity, pos, quat) -> None:
    # relative=True: the pose is expressed in Genesis's user (morph) frame, i.e.
    # the mesh origin the offline fit and the load-time default_root_pos use.
    # relative=False would write the solver frame, which Genesis offsets by the
    # inertial alignment: objects whose mesh origin is far from their centre of
    # mass (brush, scissors) then land several cm away from the fitted pose.
    entity.set_pos(tuple(float(v) for v in pos), relative=True)
    entity.set_quat(tuple(float(v) for v in quat), relative=True)


def apply_object_pose_override(
    env,
    override: ObjectPoseOverride | list[PlacedPose],
    *,
    settle_steps: int = 0,
    max_displacement: float = 0.01,
) -> list[str]:
    """Push the poses into the simulator; returns log lines.

    ``env`` needs ``env.entities`` (a mapping safe_key -> Genesis entity) and
    ``env.step(None)`` (one control step holding the reset action). The old
    evaluator called this after every reset with ``settle_steps=5``.

    With ``settle_steps`` > 0 the simulator is stepped (holding the reset
    action) and every overridden object is read back: one that drifted more
    than ``max_displacement`` from its commanded position was rejected by the
    physics (typically a collision-hull contact with a neighbour) and is
    reverted to its load-time pose so the scene is never worse than stock.
    """
    if isinstance(override, list):
        override = ObjectPoseOverride(placed=override)
    lines: list[str] = []
    missing: list[str] = []
    entities = {}
    for pose in override.placed:
        entity = env.entities.get(pose.safe_key) if hasattr(env.entities, "get") else None
        if entity is None:
            missing.append(pose.safe_key)
            continue
        entities[pose.safe_key] = entity
        _set_entity_pose(entity, pose.pos, pose.quat_wxyz)
        lines.append(
            f"{pose.safe_key:24s} {pose.object_name:24s} status={pose.status:14s} "
            f"dxy=({pose.xy_delta[0]:+.3f},{pose.xy_delta[1]:+.3f})m "
            f"dyaw={math.degrees(pose.yaw_delta_rad):+6.1f}deg "
            f"pos=({pose.pos[0]:.3f},{pose.pos[1]:.3f},{pose.pos[2]:.3f})"
        )
    if missing:
        lines.append(f"WARNING: entities not found in the scene: {missing}")
    if settle_steps <= 0 or not entities:
        return lines

    def settle() -> None:
        for _ in range(settle_steps):
            env.step(None)

    settle()
    reverted = []
    for pose in override.placed:
        entity = entities.get(pose.safe_key)
        if entity is None:
            continue
        now = _entity_pos(entity)
        if now is None:
            continue
        # Judge on the horizontal jump (plus a large vertical one): a few cm of
        # vertical settling onto the support is normal, since the collision
        # hull and the visual mesh do not share a bottom.
        delta = now - pose.pos
        disp = float(np.linalg.norm(delta[:2]))
        if abs(float(delta[2])) > max(0.05, 5 * max_displacement):
            disp = max(disp, abs(float(delta[2])))
        if disp <= max_displacement:
            lines.append(
                f"settled {pose.safe_key}: xy {100 * disp:.2f} cm, z {100 * float(delta[2]):+.2f} cm "
                f"in {settle_steps} steps"
            )
            continue
        original = override.originals.get(pose.safe_key)
        if original is None:
            lines.append(
                f"WARNING: {pose.safe_key} moved {100 * disp:.1f} cm during settle "
                "(physics rejected the pose) and no load-time pose is known to revert to"
            )
            continue
        _set_entity_pose(entity, original[0], original[1])
        reverted.append((pose, disp))
        lines.append(
            f"WARNING: {pose.safe_key} ({pose.object_name}) jumped {100 * disp:.1f} cm during settle "
            f"(z {100 * float(delta[2]):+.1f} cm); reverted to its load-time pose"
        )
    if reverted:
        settle()
        for pose, _ in reverted:
            now = _entity_pos(entities[pose.safe_key])
            if now is not None:
                delta = now - override.originals[pose.safe_key][0]
                lines.append(
                    f"after revert {pose.safe_key}: xy {100 * float(np.linalg.norm(delta[:2])):.2f} cm, "
                    f"z {100 * float(delta[2]):+.2f} cm from load-time pose"
                )
    return lines


# ===========================================================================
# Scale multipliers (old object_scale_override.py)
# ===========================================================================
SCALE_MULTIPLIER_MIN = 0.2
SCALE_MULTIPLIER_MAX = 5.0


def load_object_scale_multipliers(path: str | os.PathLike[str]) -> dict[str, float]:
    """Return ``{transformations_key: multiplier}`` for entries that change the scale.

    ``scale_apply: false`` entries and multipliers of exactly 1.0 are skipped; a
    multiplier outside [0.2, 5.0] is an error.
    """
    path = str(path)
    entries = _entries(path)
    multipliers: dict[str, float] = {}
    for entry in entries:
        if entry.get("scale_apply", True) is False:
            continue
        raw = entry.get("scale_multiplier")
        if raw is None:
            continue
        mult = float(raw)
        if not (SCALE_MULTIPLIER_MIN <= mult <= SCALE_MULTIPLIER_MAX):
            raise ValueError(
                f"{path}: {entry.get('safe_key')}: scale_multiplier {mult} outside "
                f"[{SCALE_MULTIPLIER_MIN}, {SCALE_MULTIPLIER_MAX}]"
            )
        if abs(mult - 1.0) < 1e-9:
            continue
        key = normalize_key(entry["safe_key"])
        multipliers[key] = mult
    return multipliers


def multiplier_for(multipliers: dict[str, float] | None, key: str) -> float:
    if not multipliers:
        return 1.0
    return float(multipliers.get(normalize_key(key), 1.0))


# ===========================================================================
# Asset replacement (old object_asset_override.py)
# ===========================================================================
def load_object_asset_overrides(path: str | os.PathLike[str]) -> dict[str, str]:
    """``{transformations_key: glb_path}`` for entries with an ``asset_override``; paths must exist.

    Relative ``asset_override`` paths are resolved against the JSON's folder.
    """
    path = str(path)
    entries = _entries(path)
    base_dir = os.path.dirname(os.path.abspath(path))
    out: dict[str, str] = {}
    for entry in entries:
        glb = entry.get("asset_override")
        if not glb or entry.get("asset_apply", True) is False:
            continue
        glb = glb if os.path.isabs(glb) else os.path.join(base_dir, glb)
        if not os.path.isfile(glb):
            raise FileNotFoundError(f"{path}: {entry.get('safe_key')}: asset_override not found: {glb}")
        out[normalize_key(entry["safe_key"])] = glb
    return out


def asset_for(overrides: dict[str, str] | None, key: str) -> str | None:
    if not overrides:
        return None
    return overrides.get(normalize_key(key))


# ===========================================================================
# Collision-hull settings (old object_collision_override.py)
# ===========================================================================
COLLISION_FIELDS = (
    "coacd_threshold",
    "decompose_object_error_threshold",
    "decimate",
    "decimate_face_num",
    "decimate_aggressiveness",
)
CONVEX_HULL_ERROR_THRESHOLD = 1.0


def load_object_collision_overrides(path: str | os.PathLike[str]) -> dict[str, dict[str, Any]]:
    """``{transformations_key: {field: value}}`` for entries with a non-empty ``collision`` block.

    ``collision: {"convex_hull": true}`` means one solid convex hull (no
    decomposition): ``decompose_object_error_threshold`` = 1.0. Why: an open,
    non-watertight cup decomposes into ~50 convex pieces whose lowest pieces form
    a support patch on one side of the centre of mass, so it topples (droid2 white
    cup); a single hull has the flat bottom the real cup has.
    """
    path = str(path)
    entries = _entries(path)
    out: dict[str, dict[str, Any]] = {}
    for entry in entries:
        block = entry.get("collision")
        if not isinstance(block, dict) or not block:
            continue
        if block.get("collision_apply", True) is False:
            continue
        fields: dict[str, Any] = {}
        if block.get("convex_hull"):
            fields["decompose_object_error_threshold"] = CONVEX_HULL_ERROR_THRESHOLD
        for name in COLLISION_FIELDS:
            if name in block and block[name] is not None:
                fields[name] = block[name]
        unknown = set(block) - set(COLLISION_FIELDS) - {"convex_hull", "collision_apply", "reason"}
        if unknown:
            raise ValueError(f"{path}: {entry.get('safe_key')}: unknown collision fields {sorted(unknown)}")
        if fields:
            out[normalize_key(entry["safe_key"])] = fields
    return out


def collision_override_for(overrides: dict[str, dict[str, Any]] | None, key: str) -> dict[str, Any] | None:
    if not overrides:
        return None
    return overrides.get(normalize_key(key))


def apply_collision_override(selection, fields: dict[str, Any]):
    """New ``ObjectCollisionSelection`` with ``fields`` replaced and the profile tagged."""
    kwargs = {}
    for name, value in fields.items():
        if name == "decimate":
            kwargs[name] = bool(value)
        elif name in ("decimate_face_num", "decimate_aggressiveness"):
            kwargs[name] = int(value)
        else:
            kwargs[name] = float(value)
    return replace(selection, selected_profile=f"{selection.selected_profile}+override", **kwargs)


def require_file(path: str | os.PathLike[str] | None, what: str) -> str | None:
    """Fail loudly on a missing override file (the old ``validate_arguments``)."""
    if path is None:
        return None
    if not Path(path).is_file():
        raise FileNotFoundError(f"{what}: file not found: {path}")
    return str(path)
