"""Shared pieces of step 9 (asset refinement: size + orientation + position per object).

Port of the old ``asset_refine_common.py``. Successor to the one-shot
``objects.posefit``: the same silhouette machinery, plus per-object *scale*, the
real trajectory (where the gripper closed) and the simulator's *settled* pose as
evidence. Candidates are applied in the sim through the pose / scale overrides
(``r2s.objects.overrides``) and judged through the settle probe
(``r2s.objects.settle``).

Vocabulary
----------
dump        ``object_poses.json`` (``objects.dump``): the poses the evaluator
            loaded, at whatever scale it loaded them.
candidate   one proposed placement of every object: ``candidate.json`` (see
            ``Candidate``), readable by both override readers.
settle      ``settle.json`` from the settle probe: commanded vs settled pose
            after N control steps.

Changes from the old module: ``ScenePaths`` (repo-relative defaults, hardcoded
data roots) is replaced by :class:`RefinePaths`, built from ``SceneLayout``; the
reachable collider is the explicit ``collision.crop`` output instead of a glob.
"""

from __future__ import annotations

import ast
import copy
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from r2s.common.io import prepare_output
from r2s.objects.placement_types import ObjectPlacementSpec, quat_mul_wxyz
from r2s.objects.pose_fit import (
    ObjectModel,
    PinholeCamera,
    load_object_pose_dump,
    load_real_masks,
)

GRIPPER_CLOSED_THRESHOLD = 0.5  # same rule as compare_real_sim_policy.real_grasp_onset
REPLAY_WRIST = 1.0


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RefinePaths:
    """Every file step 9 reads or writes for one scene (replaces the old ``ScenePaths``)."""

    scene: str
    # real data (step-0 inputs)
    intrinsics: Path
    extrinsics: Path
    bbox_masks_dir: Path
    images_dir: Path
    depth_dir: Path
    tcp: Path
    grippers: Path
    joints_old: Path
    joints: Path
    phys_props: Path
    task_spec: Path
    # earlier steps
    cropped_collider: Path
    object_dump: Path
    refined_poses: Path
    # step 9 outputs: <refine_root>/<tag>/
    refine_root: Path
    # optional replay runs (``<replay_root>/<scene>/lead5_wrist1.0[_<tag>]``); None = no replay terms
    replay_root: Path | None = None

    @classmethod
    def from_layout(cls, layout, *, replay_root: str | os.PathLike[str] | None = None) -> "RefinePaths":
        inputs = layout.inputs
        return cls(
            scene=layout.scene,
            intrinsics=inputs.intrinsics,
            extrinsics=inputs.extrinsics,
            bbox_masks_dir=inputs.bbox_masks_dir,
            images_dir=inputs.images_dir,
            depth_dir=inputs.depth_dir,
            # DROID/<s>/ trajectory files (the old ScenePaths read these, not DROID_original/)
            tcp=inputs.tcp,
            grippers=inputs.grippers,
            joints_old=inputs.joints_old,
            joints=inputs.joints,
            # The old code read only masks/physical_properties.json (no fallback to the
            # copy next to masks/), so InputLayout.physical_properties is not used here.
            phys_props=inputs.masks_dir / "physical_properties.json",
            task_spec=layout.paths.tasks_dir / f"{layout.scene}.py",
            cropped_collider=layout.cropped_collider_glb,
            object_dump=layout.object_dump,
            refined_poses=layout.refined_poses,
            refine_root=layout.refine_dir("stock").parent,
            replay_root=None if replay_root is None else Path(replay_root),
        )

    def depth_path(self, frame: int = 0) -> Path:
        return self.depth_dir / f"{frame}.npy"

    def candidate_dir(self, tag: str) -> Path:
        return self.refine_root / tag

    def replay_dir(self, variant: str | None, *, lead: int = 5, wrist: float = REPLAY_WRIST) -> Path | None:
        if self.replay_root is None:
            return None
        name = f"lead{lead}_wrist{wrist}" + (f"_{variant}" if variant else "")
        return self.replay_root / self.scene / name


# ---------------------------------------------------------------------------
# Priors: physical_properties dims and the task spec
# ---------------------------------------------------------------------------
def load_phys_dims(path) -> dict[int, dict[str, Any]]:
    """``{count: {"name", "dims_m": sorted (a<=b<=c) or None, "mass"}}`` from masks/physical_properties.json.

    A missing file means no dims prior (as in the old code).
    """
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as handle:
        entries = json.load(handle)
    if isinstance(entries, dict):
        entries = entries.get("objects", [])
    out: dict[int, dict[str, Any]] = {}
    for entry in entries:
        count = entry.get("count")
        if isinstance(count, list):
            count = count[0] if count else None
        if count is None:
            continue
        dims = [entry.get(k) for k in ("width", "height", "length")]
        dims_m = sorted(float(d) for d in dims) if all(isinstance(d, (int, float)) for d in dims) else None
        out[int(count)] = {"name": str(entry.get("object_name", count)), "dims_m": dims_m, "mass": entry.get("mass")}
    return out


_DIMS3_RE = re.compile(r"(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)\s*cm", re.IGNORECASE)
_DIMS2_RE = re.compile(r"(?<![\dx.])(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)\s*cm", re.IGNORECASE)
_DIMS1_RE = re.compile(r"(?<![\dx.])(\d+(?:\.\d+)?)\s*cm", re.IGNORECASE)
NOTE_DIMS_EXCLUDE_WORDS = ("proxy", "collision", "mesh", "reconstruct", "in sim", "sim ", "asset", "hull")


@dataclass
class TaskInfo:
    scene: str
    instruction: str = ""
    objects: dict[str, int] = field(default_factory=dict)  # role -> count
    notes: str = ""
    dims_cm: dict[int, list[float]] = field(default_factory=dict)  # count -> [a,b,c] or [d]
    source: str = ""

    @property
    def task_counts(self) -> list[int]:
        seen: list[int] = []
        for c in self.objects.values():
            if c not in seen:
                seen.append(c)
        return seen


def _const_int_map(tree: ast.Module) -> dict[str, Any]:
    consts: dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                consts[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, SyntaxError):
                continue
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Tuple):
            try:
                values = ast.literal_eval(node.value)
            except (ValueError, SyntaxError):
                continue
            for target, value in zip(node.targets[0].elts, values):
                if isinstance(target, ast.Name):
                    consts[target.id] = value
    return consts


def _eval_with_consts(node: ast.AST, consts: dict[str, Any]) -> Any:
    if isinstance(node, ast.Name):
        return consts[node.id]
    if isinstance(node, ast.Dict):
        return {_eval_with_consts(k, consts): _eval_with_consts(v, consts) for k, v in zip(node.keys, node.values)}
    if isinstance(node, (ast.Tuple, ast.List)):
        return [_eval_with_consts(e, consts) for e in node.elts]
    return ast.literal_eval(node)


def load_task_info(path, *, scene: str = "", object_names: dict[int, str] | None = None) -> TaskInfo:
    """Instruction, object roles, notes and any '(AxBxC cm)' size hints from a v2 task file.

    Sizes are looked up in comments as well (``# 0 pink bowl (14.4cm)``); each
    hint is attributed to the object whose name appears closest before it.
    """
    info = TaskInfo(scene=scene, source=str(path))
    if not os.path.isfile(path):
        return info
    with open(path, encoding="utf-8") as handle:
        source = handle.read()
    tree = ast.parse(source)
    consts = _const_int_map(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            try:
                if kw.arg == "instruction" and not info.instruction:
                    info.instruction = str(_eval_with_consts(kw.value, consts))
                elif kw.arg == "objects" and not info.objects:
                    value = _eval_with_consts(kw.value, consts)
                    if isinstance(value, dict):
                        info.objects = {str(k): int(v) for k, v in value.items() if isinstance(v, int)}
                elif kw.arg == "notes":
                    text = _eval_with_consts(kw.value, consts)
                    if isinstance(text, str) and text not in info.notes:
                        info.notes = (info.notes + " " + text).strip()
            except (ValueError, KeyError, SyntaxError, TypeError):
                continue
    # Aliases per count: the mask/phys name plus the task's role names ("bear" -> 2).
    # Matching is case-sensitive on lower-case tokens so the upper-case index
    # constants (BEAR, BOWL = 2, 0) never count as mentions.
    aliases: dict[int, set[str]] = {}
    for c, n in (object_names or {}).items():
        aliases.setdefault(int(c), set()).add(str(n).lower())
    for role, c in info.objects.items():
        aliases.setdefault(int(c), set()).add(role.replace("_", " ").lower())

    def owner_for(pos: int) -> int | None:
        window = source[max(0, pos - 80):pos]
        lowered = window.lower()
        best: tuple[int, int, int] | None = None  # (token distance, full-name distance, count)
        for count, names in aliases.items():
            tok_dist, full_dist = 10**9, 10**9
            for name in names:
                for token in {name, *name.split()}:
                    if len(token) < 3:
                        continue
                    idx = lowered.rfind(token)
                    # skip ALL-CAPS mentions (index constants such as BEAR, BOWL)
                    while idx >= 0 and window[idx : idx + len(token)] == token.upper():
                        idx = lowered.rfind(token, 0, idx)
                    if idx < 0:
                        continue
                    dist = len(window) - idx
                    tok_dist = min(tok_dist, dist)
                    if token == name:
                        full_dist = min(full_dist, dist)
            if tok_dist < 10**9:
                key = (tok_dist, full_dist, count)
                if best is None or key < best:
                    best = key
        return None if best is None else best[2]

    taken: list[tuple[int, int]] = []

    def claim(m: re.Match, values: list[float]) -> None:
        if any(a <= m.start() < b for a, b in taken):
            return
        taken.append((m.start(), m.end()))
        # a size that describes the sim asset, not the real object ("collision proxy is only
        # 5x4x3.2cm", "mesh is 7 cm") must not become a prior
        context = source[max(0, m.start() - 70):m.start()].lower()
        if any(w in context for w in NOTE_DIMS_EXCLUDE_WORDS):
            return
        owner = owner_for(m.start())
        if owner is not None and owner not in info.dims_cm:
            info.dims_cm[owner] = sorted(values)

    for m in _DIMS3_RE.finditer(source):
        claim(m, [float(v) for v in m.groups()])
    for m in _DIMS2_RE.finditer(source):
        claim(m, [float(v) for v in m.groups()])
    for m in _DIMS1_RE.finditer(source):
        claim(m, [float(m.group(1))])
    return info


# ---------------------------------------------------------------------------
# Candidate = one proposed placement for every object
# ---------------------------------------------------------------------------
CANDIDATE_VERSION = 2


@dataclass
class Candidate:
    scene: str
    candidate_id: str
    method: str = ""  # stock | posefit | B | A
    parent: str = ""
    source_dump: str = ""
    settings: dict[str, Any] = field(default_factory=dict)
    objects: list[dict[str, Any]] = field(default_factory=list)
    scores: dict[str, Any] = field(default_factory=dict)

    def entry(self, safe_key: str) -> dict[str, Any] | None:
        for obj in self.objects:
            if obj.get("safe_key") == safe_key:
                return obj
        return None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "version": CANDIDATE_VERSION,
            "scene": self.scene,
            "candidate_id": self.candidate_id,
            "method": self.method,
            "parent": self.parent,
            "source_dump": self.source_dump,
            "settings": self.settings,
            "objects": self.objects,
            "scores": self.scores,
        }

    def save(self, path) -> str:
        out = prepare_output(path)
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(self.to_json_dict(), handle, indent=2)
        return str(out)

    @classmethod
    def load(cls, path) -> "Candidate":
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(
            scene=str(payload.get("scene", "")),
            candidate_id=str(payload.get("candidate_id", os.path.basename(os.path.dirname(os.path.abspath(path))))),
            method=str(payload.get("method", "")),
            parent=str(payload.get("parent", "")),
            source_dump=str(payload.get("source_dump", "")),
            settings=dict(payload.get("settings", {})),
            objects=list(payload.get("objects", [])),
            scores=dict(payload.get("scores", {})),
        )


def _f(v) -> list[float]:
    return [float(x) for x in np.asarray(v, dtype=np.float64).reshape(-1)]


def candidate_object_entry(
    model: ObjectModel,
    *,
    xy_delta=(0.0, 0.0),
    yaw_rad: float = 0.0,
    scale_multiplier: float = 1.0,
    status: str = "candidate",
    apply: bool = True,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One ``objects[]`` entry: a superset of ``refined_poses.json`` fields so the
    pose override reads pos/quat/xy_delta/base+load poses and the scale override
    reads ``scale_multiplier``."""
    spec = model.spec
    xy = np.asarray(xy_delta, dtype=np.float64).reshape(2)
    entry = {
        "safe_key": spec.safe_key,
        "count": int(spec.count),
        "object_name": spec.object_name,
        "status": status,
        "apply": bool(apply),
        "xy_delta_m": _f(xy),
        "shift_cm": float(100.0 * np.linalg.norm(xy)),
        "yaw_delta_deg": float(math.degrees(yaw_rad)),
        "pos": _f(model.pos_for(xy)),
        "quat_wxyz": _f(model.quat_for(yaw_rad)),
        "base_pos": _f(spec.base_pos),
        "base_quat_wxyz": _f(spec.base_quat_wxyz),
        "load_pos": None if model.load_pos is None else _f(model.load_pos),
        "load_quat_wxyz": None if model.load_quat is None else _f(model.load_quat),
        "asset_file": spec.asset_file,
        "scale": float(spec.scale),
        "scale_multiplier": float(scale_multiplier),
        "scale_apply": abs(float(scale_multiplier) - 1.0) > 1e-9,
        "level_deg": float(model.level_deg),
        "z_delta_m": float(model.z_delta),
    }
    if extra:
        entry.update(extra)
    return entry


def candidate_stock(dump: dict, camera: PinholeCamera, *, scene: str, source_dump: str = "") -> tuple[Candidate, dict[str, ObjectModel]]:
    """The evaluator's own placement: no override, multipliers 1."""
    models = {}
    objects = []
    for entry in dump["objects"]:
        model = ObjectModel.from_dump_entry(entry, camera, z_plane=None, level_max_deg=0.0)
        models[entry["safe_key"]] = model
        objects.append(candidate_object_entry(model, status="stock", apply=False))
    cand = Candidate(scene=scene, candidate_id="stock", method="stock", source_dump=source_dump, objects=objects)
    return cand, models


def candidate_from_refined_poses(path, *, scene: str, candidate_id: str = "posefit") -> Candidate:
    """Wrap an existing ``refined_poses.json`` (object_pose_refine.py) as a candidate."""
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    objects = []
    for obj in payload.get("objects", []):
        entry = copy.deepcopy(obj)
        entry.setdefault("scale_multiplier", 1.0)
        entry.setdefault("scale_apply", False)
        objects.append(entry)
    return Candidate(
        scene=scene,
        candidate_id=candidate_id,
        method="posefit",
        source_dump=str(payload.get("source_dump", "")),
        settings=dict(payload.get("settings", {})),
        objects=objects,
    )


# ---------------------------------------------------------------------------
# Models with scale, from a dump and (optionally) a settle dump
# ---------------------------------------------------------------------------
def scaled_model(model: ObjectModel, multiplier: float) -> ObjectModel:
    """Copy of ``model`` with the mesh scaled about its root by ``multiplier``.

    Matches what ``--object-scale-override`` does at load time (the evaluator
    multiplies the EntityOptions scale, i.e. every vertex offset from the mesh
    root).  The object is then re-rested so its lowest vertex keeps the height
    it had before scaling (the load-time guard's choice), exactly like
    ``ObjectModel.from_dump_entry`` preserves ``minz_before``.
    """
    m = float(multiplier)
    spec = model.spec
    if abs(m - 1.0) < 1e-12:
        return model
    old_minz = spec.base_min_z
    new_spec = ObjectPlacementSpec(
        safe_key=spec.safe_key,
        count=spec.count,
        object_name=spec.object_name,
        base_pos=spec.base_pos.copy(),
        base_quat_wxyz=spec.base_quat_wxyz.copy(),
        centered_vertices=spec.centered_vertices * m,
        faces=None if spec.faces is None else spec.faces,
        asset_file=spec.asset_file,
        scale=spec.scale * m,
    )
    new_spec.base_pos[2] += old_minz - new_spec.base_min_z
    return ObjectModel(
        spec=new_spec,
        camera=model.camera,
        level_deg=model.level_deg,
        z_delta=model.z_delta + float(new_spec.base_pos[2] - spec.base_pos[2]),
        minz_before=model.minz_before,
        load_pos=model.load_pos,
        load_quat=model.load_quat,
    )


def model_at_settled_pose(model: ObjectModel, settled_pos, settled_quat_wxyz) -> ObjectModel:
    """Re-express ``model`` at the pose the simulator settled it to.

    The settled quaternion becomes the new base orientation (so a later
    XY/yaw fit starts from the physics rest pose instead of the commanded one)
    and the vertex cloud is re-derived from the *unrotated* base mesh.  We only
    know vertices in the commanded base frame, so the settled rotation is
    applied relative to the commanded one.
    """
    from r2s.objects.pose_fit import _quat_to_matrix
    from r2s.objects.settle import _quat_normalize

    q_cmd = _quat_normalize(model.spec.base_quat_wxyz)
    q_set = _quat_normalize(settled_quat_wxyz)
    # rotation taking the commanded orientation to the settled one (world frame)
    conj = np.array([q_cmd[0], -q_cmd[1], -q_cmd[2], -q_cmd[3]])
    w1, x1, y1, z1 = q_set
    w2, x2, y2, z2 = conj
    d = np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )
    R = _quat_to_matrix(d)
    spec = model.spec
    new_spec = ObjectPlacementSpec(
        safe_key=spec.safe_key,
        count=spec.count,
        object_name=spec.object_name,
        base_pos=np.asarray(settled_pos, dtype=np.float64).reshape(3).copy(),
        base_quat_wxyz=q_set.copy(),
        centered_vertices=spec.centered_vertices @ R.T,
        faces=spec.faces,
        asset_file=spec.asset_file,
        scale=spec.scale,
    )
    return ObjectModel(
        spec=new_spec,
        camera=model.camera,
        level_deg=model.level_deg,
        z_delta=model.z_delta,
        minz_before=model.minz_before,
        load_pos=model.load_pos,
        load_quat=model.load_quat,
    )


def build_models(
    dump: dict,
    camera: PinholeCamera,
    *,
    candidate: Candidate | None = None,
    settle: dict[str, dict[str, Any]] | None = None,
    z_plane: float | None = None,
    level_max_deg: float = 30.0,
    clearance: float = 0.0005,
    use_settled: bool = True,
    asset_overrides: dict[str, str] | None = None,
    orientation_overrides: dict[str, Any] | None = None,
) -> dict[str, ObjectModel]:
    """``{safe_key: ObjectModel}`` for a candidate.

    Without a candidate this is the dump as loaded (rested like the old fit).
    With one, each object is scaled by its ``scale_multiplier`` and placed at
    the candidate's pos/quat; with a settle dump and ``use_settled`` the
    *settled* pose wins, because that is what the policy sees.
    """
    models: dict[str, ObjectModel] = {}
    for entry in dump["objects"]:
        key = entry["safe_key"]
        cand_entry = candidate.entry(key) if candidate is not None else None
        override_asset = (asset_overrides or {}).get(key)
        if cand_entry is not None and cand_entry.get("asset_override") and cand_entry.get("asset_apply", True):
            override_asset = cand_entry["asset_override"]
        if override_asset:
            # the candidate replaces the reconstruction (flattened copy or a parametric primitive)
            entry = {**entry, "asset_file": override_asset}
        orient = (orientation_overrides or {}).get(key)
        if cand_entry is not None and cand_entry.get("orientation_quat_wxyz"):
            orient = cand_entry["orientation_quat_wxyz"]
        load_quat_original = np.asarray(entry["quat_wxyz"], dtype=np.float64).reshape(4)
        if orient is not None:
            # a primitive's own upright pose (identity in world); the run's load-time quaternion is
            # kept as load_quat so the delta-mode pose override lands exactly on this orientation
            entry = {**entry, "quat_wxyz": [float(v) for v in orient]}
        model = ObjectModel.from_dump_entry(entry, camera, z_plane=z_plane, clearance=clearance, level_max_deg=0.0 if orient is not None else level_max_deg)
        if orient is not None:
            model.load_quat = load_quat_original
        if cand_entry is not None:
            mult = float(cand_entry.get("scale_multiplier", 1.0)) if cand_entry.get("scale_apply", True) else 1.0
            model = scaled_model(model, mult)
            if cand_entry.get("apply", True) and "pos" in cand_entry and "quat_wxyz" in cand_entry:
                model.spec.base_pos = np.asarray(cand_entry["pos"], dtype=np.float64).reshape(3).copy()
                # the candidate's quaternion already contains any yaw change
                model = model_at_settled_pose(model, cand_entry["pos"], cand_entry["quat_wxyz"])
        if settle and use_settled and key in settle:
            s = settle[key]
            model = model_at_settled_pose(model, s["settled_pos"], s["settled_quat_wxyz"])
        models[key] = model
    return models


def rest_plane(dump: dict) -> float | None:
    """Height objects should rest on: the physics receiver top when the dump carries it.

    ``z_plane`` (0.02 on droid2) is the calibration/reprojection plane; the
    simulated table top (``physics_top``, ~0.0 on droid2) is where physics puts
    objects, and resting on ``z_plane`` instead leaves them floating 2 cm, which
    the settle probe then reports as a drop or a tip.
    """
    for key in ("physics_top", "reference_top", "z_plane"):
        value = dump.get(key)
        if value is not None and math.isfinite(float(value)):
            return float(value)
    return None


def orientation_overrides_of(candidate: "Candidate | None") -> dict[str, Any]:
    if candidate is None:
        return {}
    return {o["safe_key"]: o["orientation_quat_wxyz"] for o in candidate.objects if o.get("orientation_quat_wxyz")}


def asset_overrides_of(candidate: "Candidate | None") -> dict[str, str]:
    if candidate is None:
        return {}
    return {o["safe_key"]: o["asset_override"] for o in candidate.objects if o.get("asset_override") and o.get("asset_apply", True)}


def object_names_from_dump(dump: dict) -> dict[int, str]:
    return {int(o["count"]): str(o.get("object_name", o["safe_key"])) for o in dump["objects"]}


def safe_key_by_count(dump: dict) -> dict[int, str]:
    return {int(o["count"]): str(o["safe_key"]) for o in dump["objects"]}


# ---------------------------------------------------------------------------
# Real trajectory helpers
# ---------------------------------------------------------------------------
def fk_flange(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Franka flange pose (pos, R) in the base frame from 7 joint angles.

    Same modified-DH table as ``ee_error_from_chunks``; ``tcp.npy`` holds this
    flange position (checked on droid2: the projected point sits on the wrist,
    ~10 cm above the fingers), so the point the gripper actually closes at is
    ``flange + R[:, 2] * HAND_D``.
    """
    from r2s.objects.refine.franka_fk import _DH, _FLANGE_D, _tf

    T = np.eye(4)
    for (a, d, alpha), theta in zip(_DH, np.asarray(q, dtype=np.float64).reshape(7)):
        T = T @ _tf(a, d, alpha, theta)
    T = T @ _tf(0.0, _FLANGE_D, 0.0, 0.0)
    return T[:3, 3].copy(), T[:3, :3].copy()


def grasp_points_from_joints(joints: np.ndarray, hand_offset: float | None = None) -> np.ndarray:
    """Fingertip-centre positions (N,3) for recorded joint rows."""
    from r2s.objects.refine.franka_fk import _HAND_D

    d = _HAND_D if hand_offset is None else float(hand_offset)
    Q = np.asarray(joints, dtype=np.float64).reshape(-1, 7)
    out = np.empty((len(Q), 3))
    for i, q in enumerate(Q):
        pos, R = fk_flange(q)
        out[i] = pos + R[:, 2] * d
    return out


def load_real_trajectory(paths: RefinePaths) -> dict[str, np.ndarray]:
    """tcp (flange), grippers, joints, and ``grasp_point`` = fingertip centre from FK.

    Falls back to ``tcp`` when the joint file is missing or not frame-aligned.
    """
    from r2s.objects.refine.franka_fk import _HAND_D

    out: dict[str, Any] = {}
    if os.path.isfile(paths.grippers):
        out["grippers"] = np.load(paths.grippers).reshape(-1)
    n = len(out["grippers"]) if "grippers" in out else None
    tcp_rot = None
    if os.path.isfile(paths.tcp):
        raw = np.load(paths.tcp)
        if raw.ndim == 3 and raw.shape[1:] == (4, 4):
            # some scenes store the flange as a 4x4 pose per frame
            tcp_rot = raw[:, :3, :3].astype(np.float64)
            out["tcp"] = raw[:, :3, 3].astype(np.float64)
        else:
            out["tcp"] = raw.reshape(len(raw), -1)[:, :3].astype(np.float64)
    # joints: joints_old.npy when present, else joints.npy if it is frame-aligned with the gripper
    for cand in (paths.joints_old, paths.joints):
        if os.path.isfile(cand):
            j = np.load(cand)
            if j.ndim == 2 and j.shape[1] == 7 and (n is None or len(j) == n):
                out["joints"] = j.astype(np.float64)
                out["joints_source"] = os.path.basename(cand)
                break
    joints = out.get("joints")
    tcp = out.get("tcp")
    if joints is not None and (tcp is None or len(joints) == len(tcp)):
        out["grasp_point"] = grasp_points_from_joints(joints)
        if tcp is not None:
            flange = np.array([fk_flange(q)[0] for q in joints])
            out["fk_flange_rms_m"] = float(np.sqrt(np.mean(np.sum((flange - tcp) ** 2, axis=1))))
    elif tcp is not None and tcp_rot is not None:
        out["grasp_point"] = tcp + tcp_rot[:, :, 2] * _HAND_D  # fingertip along the flange z axis
        out["fk_flange_rms_m"] = None
    elif tcp is not None:
        out["grasp_point"] = tcp.copy()  # flange only; ~10 cm above the fingers
        out["fk_flange_rms_m"] = None
    return out


def grasp_events(grippers: np.ndarray, *, threshold: float = GRIPPER_CLOSED_THRESHOLD, min_gap: int = 15) -> list[dict[str, int]]:
    """Rising/falling edges of the gripper signal: one dict per close..open episode."""
    g = np.asarray(grippers, dtype=np.float64).reshape(-1)
    closed = g > threshold
    events: list[dict[str, int]] = []
    i = 0
    n = len(g)
    while i < n:
        if closed[i] and (i == 0 or not closed[i - 1]):
            j = i
            while j < n and closed[j]:
                j += 1
            if events and i - events[-1]["release"] < min_gap:
                events[-1]["release"] = j
            else:
                events.append({"onset": i, "release": j})
            i = j
        else:
            i += 1
    return events


def mask_ground_extent(mask: np.ndarray, camera: PinholeCamera, z_plane: float) -> dict[str, float] | None:
    """Footprint size implied by the mask's bottom edge back-projected onto the table.

    For each image column inside the mask, the lowest pixel is (approximately)
    where the object meets the table; ray-plane intersection turns that into
    world XY, whose principal extents are a cheap, depth-free size cue.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) < 10:
        return None
    cols = np.unique(xs)
    bottom = np.array([[c, ys[xs == c].max()] for c in cols], dtype=np.float64)
    xy = camera.ray_plane_xy(bottom, z_plane)
    c = xy - xy.mean(axis=0)
    cov = c.T @ c / max(len(c), 1)
    evals, _ = np.linalg.eigh(cov)
    return {
        "major_m": float(4.0 * math.sqrt(max(evals[1], 0.0))),  # ~full length for a uniform segment
        "minor_m": float(4.0 * math.sqrt(max(evals[0], 0.0))),
        "centroid_xy": _f(xy.mean(axis=0)),
    }


__all__ = [
    "Candidate",
    "RefinePaths",
    "TaskInfo",
    "build_models",
    "candidate_from_refined_poses",
    "candidate_object_entry",
    "candidate_stock",
    "grasp_events",
    "load_object_pose_dump",
    "load_phys_dims",
    "load_real_masks",
    "load_real_trajectory",
    "load_task_info",
    "mask_ground_extent",
    "model_at_settled_pose",
    "object_names_from_dump",
    "safe_key_by_count",
    "scaled_model",
]


# ---------------------------------------------------------------------------
# Asset repair: flatten a reconstruction's hidden, domed bottom
# ---------------------------------------------------------------------------
def flatten_bottom_asset(
    asset_file: str,
    out_file: str,
    *,
    scale: float,
    root_quat_wxyz,
    flatten_m: float = 0.012,
    mode: str = "fill",
) -> dict[str, Any]:
    """Write a copy of the GLB whose lowest ``flatten_m`` (at the given orientation) is made flat.

    ``mode="fill"`` (default) pushes the dome down to the lowest point, so the
    object keeps its height and silhouette; ``mode="clip"`` cuts the slab off
    (shorter by ``flatten_m``).

    Image-to-3D never sees the underside, so cups/bottles get a rounded bottom
    and rock over in physics.  Every vertex below ``lowest + flatten_m`` (world
    z at the pose the object will be placed in) is moved up to that plane, in
    the mesh's own frame, so the object rests on a flat face whose area is the
    mesh's cross-section at that height.  Textures/materials are kept when the
    GLB holds one geometry (the DROID assets do).
    """
    import trimesh

    from r2s.collision.support import DEFAULT_GLB_FRONT, DEFAULT_GLB_UP, align_up_and_front, quat_wxyz_to_matrix

    scene = trimesh.load(str(asset_file), force="scene")
    geoms = list(scene.geometry.items())
    if len(geoms) != 1:
        raise ValueError(f"{asset_file}: expected one geometry, found {len(geoms)}; flatten-bottom keeps textures only for single-geometry GLBs")
    name, geom = geoms[0]
    node_tf = np.eye(4)
    for node in scene.graph.nodes_geometry:
        tf, gname = scene.graph.get(node)
        if gname == name:
            node_tf = np.asarray(tf, dtype=np.float64)
            break
    import_rot = align_up_and_front(np.asarray(DEFAULT_GLB_UP), np.asarray(DEFAULT_GLB_FRONT))
    root_rot = quat_wxyz_to_matrix(np.asarray(root_quat_wxyz, dtype=np.float64))
    # combined-frame vertices (what load_combined_mesh returns) -> world z
    v_local = np.asarray(geom.vertices, dtype=np.float64)
    v_comb = v_local @ node_tf[:3, :3].T + node_tf[:3, 3]
    v_world = (v_comb @ import_rot.T * float(scale)) @ root_rot.T
    z = v_world[:, 2]
    z0 = float(z.min())
    cut = z0 + float(flatten_m)
    below = z < cut
    # world +z expressed in the combined mesh frame (unit vector; scale is uniform)
    up_world = np.array([0.0, 0.0, 1.0])
    up_comb = import_rot.T @ (root_rot.T @ up_world)
    up_comb /= max(np.linalg.norm(up_comb), 1e-12)
    if mode == "fill":
        # extend the dome DOWN to a plane at the lowest point: height and the visible
        # silhouette stay identical, only the hidden underside changes
        shift = (z0 - z[below]) / float(scale)
    elif mode == "clip":
        # raise the dome UP to the cut plane: shortens the object by flatten_m (visible)
        shift = (cut - z[below]) / float(scale)
    else:
        raise ValueError(f"unknown mode {mode!r}; expected 'fill' or 'clip'")
    v_comb_new = v_comb.copy()
    v_comb_new[below] += shift[:, None] * up_comb[None, :]
    inv = np.linalg.inv(node_tf)
    v_local_new = v_comb_new @ inv[:3, :3].T + inv[:3, 3]
    geom.vertices = v_local_new
    scene.export(str(prepare_output(out_file)))
    # contact patch at the new bottom
    import cv2

    flat_xy = v_world[below][:, :2].astype(np.float32).reshape(-1, 1, 2)
    area = float(cv2.contourArea(cv2.convexHull(flat_xy))) if len(flat_xy) >= 3 else 0.0
    return {
        "asset_file": os.path.abspath(asset_file),
        "out_file": os.path.abspath(out_file),
        "mode": mode,
        "flatten_m": float(flatten_m),
        "n_vertices_moved": int(below.sum()),
        "n_vertices": int(len(v_local)),
        "flat_patch_area_cm2": 1e4 * area,
        "raised_m": float(flatten_m) if mode == "clip" else 0.0,
    }


# ---------------------------------------------------------------------------
# Asset replacement: parametric primitives for simple objects
# ---------------------------------------------------------------------------
def make_cup_asset(
    out_file: str,
    *,
    height_m: float,
    top_diameter_m: float,
    bottom_diameter_m: float | None = None,
    wall_m: float = 0.002,
    color_rgb=(220, 220, 220),
    entry_scale: float = 1.0,
    segments: int = 64,
) -> dict[str, Any]:
    """Write a hollow, tapered, flat-bottomed cup GLB the evaluator can load in place of a reconstruction.

    Image-to-3D turns a plain cup into a squat mug with a hallucinated handle
    and a domed underside; for objects this simple a parametric mesh sized
    from the real evidence (mask footprint, physical_properties dims) is more
    faithful.  Vertices are divided by ``entry_scale`` so that the evaluator's
    ``scale`` (from transformations.json x reprojection) restores metric size,
    exactly like the reconstruction it replaces.  The mesh frame follows the
    GLB convention the loader expects (+Y up), matching the original assets.
    """
    import trimesh

    from r2s.collision.support import DEFAULT_GLB_FRONT, DEFAULT_GLB_UP, align_up_and_front

    bottom_diameter_m = top_diameter_m * 0.82 if bottom_diameter_m is None else bottom_diameter_m
    r_top, r_bot, h = top_diameter_m / 2.0, bottom_diameter_m / 2.0, height_m
    ang = np.linspace(0.0, 2 * np.pi, segments, endpoint=False)
    c, s = np.cos(ang), np.sin(ang)
    rings = []  # (radius, z)
    rings.append((r_bot, 0.0))
    rings.append((r_top, h))
    rings.append((r_top - wall_m, h))
    rings.append((max(r_bot - wall_m, 1e-3), wall_m * 1.5))
    verts = []
    for r, z in rings:
        verts.append(np.stack([r * c, r * s, np.full_like(c, z)], axis=1))
    verts = np.concatenate(verts, axis=0)
    n = segments
    faces = []

    def quad(a, b):  # connect ring a to ring b
        for i in range(n):
            j = (i + 1) % n
            faces.append([a * n + i, a * n + j, b * n + j])
            faces.append([a * n + i, b * n + j, b * n + i])

    quad(0, 1)  # outer wall
    quad(1, 2)  # rim
    quad(2, 3)  # inner wall
    # bottom disc (outer) and inner floor
    bottom_center = len(verts)
    verts = np.concatenate([verts, [[0.0, 0.0, 0.0]], [[0.0, 0.0, wall_m * 1.5]]], axis=0)
    inner_center = bottom_center + 1
    for i in range(n):
        j = (i + 1) % n
        faces.append([bottom_center, j, i])  # outer bottom, facing down
        faces.append([inner_center, 3 * n + i, 3 * n + j])  # inner floor, facing up
    faces = np.asarray(faces, dtype=np.int64)
    # metric z-up -> the loader's GLB frame: undo the import rotation the evaluator applies
    import_rot = align_up_and_front(np.asarray(DEFAULT_GLB_UP), np.asarray(DEFAULT_GLB_FRONT))
    verts_glb = (verts @ import_rot) / float(entry_scale)  # world = import_rot @ v_glb * scale
    mesh = trimesh.Trimesh(vertices=verts_glb, faces=faces, process=False)
    mesh.fix_normals()
    rgba = [int(color_rgb[0]), int(color_rgb[1]), int(color_rgb[2]), 255]
    mesh.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(baseColorFactor=rgba, metallicFactor=0.0, roughnessFactor=0.6))
    mesh.export(str(prepare_output(out_file)))
    return {"out_file": os.path.abspath(out_file), "height_m": h, "top_diameter_m": top_diameter_m, "bottom_diameter_m": bottom_diameter_m, "wall_m": wall_m, "color_rgb": list(map(int, color_rgb)), "entry_scale": float(entry_scale), "n_vertices": int(len(verts_glb))}


# ---------------------------------------------------------------------------
# Semantic uprighting: cups/mugs/bottles stand on their symmetry axis
# ---------------------------------------------------------------------------
UPRIGHT_CLASSES = ("cup", "mug", "bottle", "can", "thermos", "glass", "jar", "tumbler")
UPRIGHT_MIN_DEG = 5.0
UPRIGHT_MAX_DEG = 85.0  # ~90 deg = lying on its side: may be intentional (a bottle on the table); leave it
UPRIGHT_MAX_CV = 0.30  # radial spread above this = not a clean cylinder (mug with handle, blob): do not auto-upright


def symmetry_axis(vertices: np.ndarray) -> tuple[np.ndarray, float]:
    """Unit axis (pointing up) about which the mesh is most rotationally symmetric.

    Among the three PCA axes, the one with the smallest coefficient of variation
    of the radial distance: a cup or bottle has a near-constant radius about its
    axis.  The cv is returned as a confidence (cylinders ~0.15-0.25; a mug with
    a handle ~0.3; a non-symmetric object higher).
    """
    v = np.asarray(vertices, dtype=np.float64)
    c = v - v.mean(axis=0)
    _, U = np.linalg.eigh(c.T @ c)
    best = None
    for i in range(3):
        a = U[:, i]
        # roundness of the projection onto the plane perpendicular to the axis:
        # 1.0 for a circle (true symmetry axis), well below 1 for the side view of a cylinder
        radial = c - np.outer(c @ a, a)
        w, _ = np.linalg.eigh(radial.T @ radial)
        w = np.sort(np.abs(w))[::-1]
        roundness = float(math.sqrt(max(w[1], 0.0) / max(w[0], 1e-12)))
        r = np.linalg.norm(radial, axis=1)
        cv = float(r.std() / max(r.mean(), 1e-9))
        # decide by radial spread (a cup has one axis with constant radius); roundness only
        # rules out clearly non-round projections (the side view of a tall cylinder)
        candidates_ok = roundness >= 0.8
        key = (0 if candidates_ok else 1, cv)
        if best is None or key < best[0]:
            best = (key, cv, a)
    _, cv, a = best
    return (a if a[2] >= 0 else -a), cv


def upright_by_symmetry_axis(model: ObjectModel, *, min_deg: float = UPRIGHT_MIN_DEG, max_deg: float = UPRIGHT_MAX_DEG, max_cv: float = UPRIGHT_MAX_CV, z_plane: float | None = None, clearance: float = 0.0005) -> tuple[ObjectModel, dict[str, Any]]:
    """Rotate a cup-like model so its symmetry axis is vertical (semantics before silhouette).

    A reconstruction whose axis is 10-60 deg off vertical still matches its
    mask well enough, but it is placed leaning and falls over as soon as the
    physics releases it (droid69 orange cup: 38 deg at placement -> 65 deg
    settled).  The rotation is about the horizontal axis through the mesh
    root; the object is then re-rested.  Meshes at ~90 deg are left alone.
    """
    from r2s.objects.pose_fit import _axis_angle_quat_wxyz, _quat_to_matrix

    verts = model.spec.centered_vertices
    axis, cv = symmetry_axis(verts)
    tilt = math.degrees(math.acos(float(np.clip(axis[2], -1.0, 1.0))))
    info = {"axis_tilt_deg_before": tilt, "radial_cv": cv, "applied": False}
    if not (min_deg <= tilt <= max_deg):
        info["reason"] = "tilt outside range"
        return model, info
    if cv > max_cv:
        info["reason"] = f"radial spread {cv:.2f} > {max_cv}: axis not trusted (force with decisions.json 'upright': true)"
        return model, info
    rot_axis = np.cross(axis, np.array([0.0, 0.0, 1.0]))
    q = _axis_angle_quat_wxyz(rot_axis, math.radians(tilt))
    R = _quat_to_matrix(q)
    spec = model.spec
    new_spec = ObjectPlacementSpec(
        safe_key=spec.safe_key,
        count=spec.count,
        object_name=spec.object_name,
        base_pos=spec.base_pos.copy(),
        base_quat_wxyz=quat_mul_wxyz(q, spec.base_quat_wxyz),
        centered_vertices=verts @ R.T,
        faces=spec.faces,
        asset_file=spec.asset_file,
        scale=spec.scale,
    )
    target = spec.base_min_z if z_plane is None else float(z_plane) + clearance
    new_spec.base_pos[2] += target - new_spec.base_min_z
    axis2, _ = symmetry_axis(new_spec.centered_vertices)
    info.update({"applied": True, "axis_tilt_deg_after": math.degrees(math.acos(float(np.clip(axis2[2], -1.0, 1.0)))), "z_delta_m": float(new_spec.base_pos[2] - spec.base_pos[2])})
    out = ObjectModel(spec=new_spec, camera=model.camera, level_deg=model.level_deg + tilt, z_delta=model.z_delta + info["z_delta_m"], minz_before=model.minz_before, load_pos=model.load_pos, load_quat=model.load_quat)
    return out, info


def is_upright_class(object_name: str, classes=UPRIGHT_CLASSES) -> bool:
    n = str(object_name).lower()
    return any(w in n for w in classes)


# ---------------------------------------------------------------------------
# Rest height from the background collider at the fitted footprint
# ---------------------------------------------------------------------------
def build_collider_query(paths: RefinePaths, *, footprint_margin: float = 0.015, surface_normal_z: float = 0.75):
    """`BackgroundSupportQuery` on the baked crop (``collision.crop`` output) with an identity root pose
    (same as the evaluator's asset-init path: the trimesh export is already world z-up, so no import
    rotation). A missing crop is an error."""
    from r2s.collision.support import BackgroundSupportQuery

    path = Path(paths.cropped_collider)
    if not path.is_file():
        raise FileNotFoundError(f"reachable collider crop not found: {path} (run collision.crop)")
    return BackgroundSupportQuery.from_baked_crop(path, footprint_margin=footprint_margin, surface_normal_z=surface_normal_z)


def rest_height_on_collider(query, model: ObjectModel, xy_delta, yaw_rad: float, *, plane_z: float | None, clearance: float = 0.0005, max_above_plane: float = 0.15) -> tuple[float | None, float | None]:
    """(support_z, dz) for the object's fitted footprint: the collider surface under it, but never
    below the physics plane and never a surface more than ``max_above_plane`` above it (that would be
    another object or a wall).  ``dz`` is the vertical move that puts the lowest vertex ``clearance``
    above the support.  Returns (None, None) when the collider has nothing under the footprint."""
    lo, hi = model.spec.world_aabb(model.pos_for(xy_delta), yaw_rad)
    limit = None if plane_z is None else float(plane_z) + max_above_plane
    support = query.support_z_for_aabb(lo, hi, max_support_z=limit)
    if support is None:
        return None, None
    if plane_z is not None:
        support = max(float(support), float(plane_z))
    dz = float(support + clearance - lo[2])
    return float(support), dz


# ---------------------------------------------------------------------------
# Local textured render (painter's algorithm) for appearance checks without a GPU
# ---------------------------------------------------------------------------
def mesh_vertex_colors(asset_file: str) -> np.ndarray | None:
    """(N,3) uint8 vertex colours of a single-geometry textured GLB, in load_combined_mesh order."""
    import trimesh

    scene = trimesh.load(str(asset_file), force="scene")
    geoms = list(scene.geometry.values())
    if len(geoms) != 1:
        return None
    try:
        return np.asarray(geoms[0].visual.to_color().vertex_colors)[:, :3].astype(np.uint8)
    except Exception:
        return None


def render_textured(model: ObjectModel, xy_delta, yaw_rad: float, colors: np.ndarray, background: np.ndarray | None = None) -> np.ndarray:
    """RGB render of one object at a pose: faces sorted far-to-near and filled with their mean colour.

    Occlusion-correct for opaque meshes (unlike a point splat, which lets the back wall of an
    open shell show through), no GPU or GL context needed, ~50 ms for 20k faces.
    """
    import cv2

    cam = model.camera
    verts = model.spec.world_vertices(model.pos_for(xy_delta), yaw_rad)
    faces = model.spec.faces
    uv, depth = cam.project(verts)
    img = np.zeros((cam.height, cam.width, 3), dtype=np.uint8) if background is None else np.ascontiguousarray(background).copy()
    if faces is None or len(faces) == 0:
        return img
    tri_depth = depth[faces].mean(axis=1)
    keep = (depth[faces] > 1e-6).all(axis=1)
    order = np.argsort(-tri_depth[keep])
    tri_uv = np.round(uv[faces[keep]]).astype(np.int32)[order]
    tri_col = colors[faces[keep]].astype(np.float64).mean(axis=1)[order]
    for poly, col in zip(tri_uv, tri_col):
        cv2.fillConvexPoly(img, poly, tuple(float(c) for c in col))
    return img


def appearance_score(render: np.ndarray, real: np.ndarray, mask: np.ndarray, silhouette: np.ndarray) -> float:
    """Mean absolute colour difference inside mask ∩ silhouette (lower is better)."""
    both = mask & silhouette
    if both.sum() < 50:
        return float("inf")
    return float(np.abs(render[both].astype(np.float64) - real[both].astype(np.float64)).mean())
