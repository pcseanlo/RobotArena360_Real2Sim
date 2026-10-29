"""Step 9 fit ("Method B"): joint scale + XY + yaw fit per object with a multi-cue objective.

Port of the old ``asset_refine_fit.py`` (``fit-joint``; the ``review``,
``diagnose-hull``, ``flatten-bottom`` and ``replace-primitive`` commands are kept as
library functions). Production (``asset_refine_batch.sh``)::

    asset_refine_fit.py fit-joint --scene S --object-poses asset_refine/S/stock/object_poses.json \
        --out asset_refine/S/B_v1/candidate.json --seed object_pose_fit/S/refined_poses.json \
        [--weights B_v1/weights.json] [--decisions B_v1/decisions.json]

is :func:`fit_joint` with :class:`FitJointSettings` defaults.

Per object the search is: for each mesh scale in ``--scales`` (0.70..1.40),
scale the mesh about its root, re-rest it, align XY to the mask centroid
(Gauss-Newton, ``object_pose_refine.align_centroid``), polish XY by IoU, and
evaluate the objective ``J`` from ``asset_refine_metrics.objective_terms``
(IoU, area ratio, dims prior, gated depth residual, grasp anchor, stability).
The yaw sweep runs once at the seed scale (the mask's shape, not its size,
decides yaw) and is refined at the winning scale.  A final coordinate descent
over (scale, xy, yaw) polishes the winner.  Footprint conflicts are resolved
with the old rule (the object that moved further loses).

``decisions.json`` (hand-edited between rounds) can lock objects, pin a scale,
skip yaw, or override weights per object::

    {"obj_image_0002_segmented": {"scale": 1.2, "skip_yaw": true},
     "obj_image_0004_segmented": {"lock": true},
     "obj_image_0000_segmented": {"weights": {"dims": 0.0}}}

CPU only; no simulator.  The settle dump (``--settle``) from a previous round
of the same candidate feeds the stability term and the settle-then-refit step:
an object the physics tipped or slid is re-fitted from its *settled*
orientation instead of the commanded one.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from r2s.common.io import prepare_output
from r2s.objects.placement_types import ObjectPlacementSpec, quat_mul_wxyz
from r2s.objects.pose_fit import (
    FitResult,
    ObjectModel,
    PinholeCamera,
    align_centroid,
    mask_centroid,
    mask_iou,
    polish_xy_iou,
    resolve_footprint_conflicts,
)
from r2s.objects.refine.common import (
    UPRIGHT_CLASSES,
    appearance_score,
    mesh_vertex_colors,
    render_textured,
    UPRIGHT_MAX_DEG,
    UPRIGHT_MIN_DEG,
    Candidate,
    RefinePaths,
    asset_overrides_of,
    build_collider_query,
    build_models,
    is_upright_class,
    rest_height_on_collider,
    upright_by_symmetry_axis,
    orientation_overrides_of,
    candidate_from_refined_poses,
    candidate_object_entry,
    load_object_pose_dump,
    model_at_settled_pose,
    rest_plane,
    scaled_model,
)
from r2s.objects.refine.metrics import (
    DEFAULT_WEIGHTS,
    GRASP_HIT_M,
    SceneEvidence,
    footprint_distance,
    objective_terms,
    score_candidate,
    score_object,
)
from r2s.objects.settle import load_settle_dump

DEFAULT_SCALES = [round(s, 3) for s in np.arange(0.70, 1.40 + 1e-9, 0.05)]
ORIENTATION_IOU_TOLERANCE = 0.05
PUSH_DISAGREE_M = 0.05
SCALE_KEEP_AREA_RANGE = (0.65, 1.50)
SCALE_EXTEND_BEYOND = 1.5
MIN_VALID_IOU = 0.30  # candidates below this silhouette overlap are never accepted  # dims prior suggesting more than this (or less than 1/this) widens the sweep
SETTLE_Z_CORRECT_MIN_M = 0.01  # a previous-round drop/lift larger than this corrects the rest height
SETTLE_Z_CORRECT_MAX_XY_M = 0.03  # ... provided the object is being placed at (about) the same spot  # silhouette/mask area at scale 1.0 inside this: do not rescale
YAW_AMBIGUITY_IOU = 0.03
YAW_APPEARANCE_IOU = 0.10
YAW_APPEARANCE_WEIGHT = 0.01  # IoU units per unit of mean colour difference (0-255)  # with a textured mesh, every yaw within this IoU band is judged by appearance  # sweep peaks closer than this are not evidence; prefer the original orientation
OVERLAP_3D_FRACTION = 0.10  # vertices of one object inside another's convex hull
OVERLAP_PUSH_STEP_M = 0.01
OVERLAP_PUSH_MAX_M = 0.12  # mask-best pose vs fingertip anchor: beyond this the object was probably pushed


@dataclass
class JointFitResult:
    safe_key: str
    count: int
    object_name: str
    status: str = "unchanged"
    scale_multiplier: float = 1.0
    xy_delta: np.ndarray = field(default_factory=lambda: np.zeros(2))
    yaw_rad: float = 0.0
    objective: float = float("inf")
    objective_before: float = float("inf")
    terms: dict[str, float] = field(default_factory=dict)
    cues: dict[str, Any] = field(default_factory=dict)
    cues_before: dict[str, Any] = field(default_factory=dict)
    scale_curve: list[dict[str, float]] = field(default_factory=list)
    yaw_curve: list[tuple[float, float]] = field(default_factory=list)
    settled_refit: bool = False
    notes: list[str] = field(default_factory=list)
    orientation: str = "as-is"
    model: Any = None  # the (possibly re-oriented) base model the pose is relative to
    pushed: bool = False  # grasp anchor ignored because the object was likely moved before the grasp

    def as_fit_result(self) -> FitResult:
        """Shape the old conflict resolver understands."""
        res = FitResult(safe_key=self.safe_key, count=self.count, object_name=self.object_name, status="refined" if self.status == "refined" else self.status)
        res.xy_delta = self.xy_delta
        res.yaw_rad = self.yaw_rad
        res.notes = self.notes
        return res


class Objective:
    """Evaluates J for one object at (scale, xy, yaw) with the scene evidence."""

    def __init__(self, base: ObjectModel, evidence: SceneEvidence, *, weights: dict[str, float], settle_entry: dict | None, target_points: list[np.ndarray], key: str):
        self.base = base
        self.evidence = evidence
        self.weights = weights
        self.settle_entry = settle_entry
        self.target_points = target_points
        self.key = key
        self.count = int(base.spec.count)
        real = evidence.real_masks.get(self.count)
        self.real_mask = None if real is None else real[1]
        self._models: dict[float, ObjectModel] = {}
        self.evaluations = 0
        self._colors = None
        self._colors_loaded = False

    def colors(self):
        if not self._colors_loaded:
            self._colors_loaded = True
            try:
                self._colors = mesh_vertex_colors(self.base.spec.asset_file)
            except Exception:
                self._colors = None
        return self._colors

    def appearance(self, model: ObjectModel, xy, yaw: float) -> float | None:
        """Mean colour difference between a local textured render and the photo inside the mask."""
        colors = self.colors()
        if colors is None or self.real_mask is None or self.evidence.image is None:
            return None
        if len(colors) != len(model.spec.centered_vertices):
            return None
        render = render_textured(model, xy, yaw, colors, background=self.evidence.image)
        return appearance_score(render, self.evidence.image, self.real_mask, model.silhouette(xy, yaw))

    def model(self, scale: float) -> ObjectModel:
        s = round(float(scale), 4)
        if s not in self._models:
            self._models[s] = scaled_model(self.base, s)
        return self._models[s]

    def anchor(self, model: ObjectModel, xy, yaw: float) -> dict[str, Any] | None:
        """Closest fingertip anchor (over every grasp on this object): d_foot + height above the top."""
        if not self.target_points:
            return None
        from r2s.objects.refine.metrics import _anchor_key

        best = None
        for p in self.target_points:
            fd = footprint_distance(model, xy, yaw, p)
            if best is None or _anchor_key(fd) < _anchor_key(best):
                best = fd
        return best

    def d_foot(self, model: ObjectModel, xy, yaw: float) -> float | None:
        a = self.anchor(model, xy, yaw)
        return None if a is None else float(a["d_foot_m"])

    def evaluate(self, scale: float, xy, yaw: float) -> tuple[float, dict[str, float], dict[str, Any]]:
        self.evaluations += 1
        model = self.model(scale)
        cues = score_object(model, xy, yaw, self.evidence, scale_multiplier=scale, settle_entry=self.settle_entry)
        anchor = self.anchor(model, xy, yaw)
        if anchor is not None:
            cues["grasp_anchor"] = anchor
        terms = objective_terms(cues, anchor, self.weights)
        j = float(sum(terms.values()))
        # The mask is the primary evidence: a pose that no longer overlaps it is invalid whatever
        # the priors say (a wrong dims prior once drove a mug to x3.5 with IoU 0.000).
        iou = cues.get("iou")
        if self.real_mask is not None and self.real_mask.any() and (iou is None or iou < MIN_VALID_IOU):
            j = float("inf")  # no silhouette in view, or almost no overlap
        return j, terms, cues


def _fit_xy_at(obj: Objective, scale: float, yaw: float, xy0) -> np.ndarray:
    model = obj.model(scale)
    if obj.real_mask is None or not obj.real_mask.any():
        return np.asarray(xy0, dtype=np.float64)
    real_c = mask_centroid(obj.real_mask)
    xy, _ = align_centroid(model, real_c, xy0, yaw)
    xy, _ = polish_xy_iou(model, obj.real_mask, xy, yaw)
    return np.asarray(xy, dtype=np.float64)


def _yaw_sweep(obj: Objective, scale: float, xy0, *, step_deg: float, mode: str) -> tuple[list[tuple[float, float, np.ndarray]], float]:
    model = obj.model(scale)
    mask = obj.real_mask
    real_c = mask_centroid(mask)
    candidates = [0.0, 90.0, 180.0, 270.0] if mode == "c4" else list(np.arange(0.0, 360.0, float(step_deg)))
    curve = []
    for deg in candidates:
        yaw = math.radians(deg)
        xy_c, _ = align_centroid(model, real_c, xy0, yaw, max_iters=4)
        curve.append((deg, mask_iou(model.silhouette(xy_c, yaw), mask), xy_c))
    ious = np.array([c[1] for c in curve])
    return curve, float(ious.max() - ious.min())


def principal_orientations(model: ObjectModel, *, z_plane: float | None = None, clearance: float = 0.0005) -> list[tuple[str, ObjectModel]]:
    """The six 'a principal axis points up' rest orientations of the mesh, each re-rested.

    A reconstruction whose up-axis is wrong (a plush standing on its nose while
    the real one sits) cannot be fixed by yaw; these candidates let the objective
    (mask shape, dims prior, fingertip height) pick the right one.  ``as-is`` is
    always the first entry.
    """
    from r2s.objects.pose_fit import _axis_angle_quat_wxyz, _quat_to_matrix

    verts = model.spec.centered_vertices
    c = verts - verts.mean(axis=0)
    _, U = np.linalg.eigh(c.T @ c)
    out: list[tuple[str, ObjectModel]] = [("as-is", model)]
    z = np.array([0.0, 0.0, 1.0])
    for i in range(3):
        for sign, tag in ((1.0, "+"), (-1.0, "-")):
            axis = sign * U[:, i]
            axis = axis / max(np.linalg.norm(axis), 1e-12)
            cos = float(np.clip(axis @ z, -1.0, 1.0))
            if cos > 0.999:
                continue  # already the as-is up axis
            if cos < -0.999:
                rot_axis, angle = np.array([1.0, 0.0, 0.0]), math.pi
            else:
                rot_axis, angle = np.cross(axis, z), math.acos(cos)
            q = _axis_angle_quat_wxyz(rot_axis, angle)
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
            # rest: keep the original lowest-vertex height (the guard's choice) unless a plane is given
            target = model.spec.base_min_z if z_plane is None else float(z_plane) + clearance
            new_spec.base_pos[2] += target - new_spec.base_min_z
            out.append((f"pca{i}{tag}", ObjectModel(spec=new_spec, camera=model.camera, level_deg=model.level_deg, z_delta=model.z_delta + float(new_spec.base_pos[2] - spec.base_pos[2]), minz_before=model.minz_before, load_pos=model.load_pos, load_quat=model.load_quat)))
    return out


def _search(
    obj: Objective,
    res: JointFitResult,
    *,
    scales: list[float],
    seed_scale: float,
    seed_xy,
    seed_yaw: float,
    yaw_mode: str,
    yaw_step_deg: float,
    yaw_min_gain: float,
    fixed_scale: float | None,
    polish_rounds: int,
) -> tuple[float, float, np.ndarray, float, dict[str, float], dict[str, Any]]:
    """Yaw at the seed scale -> scale sweep with XY refit -> fine yaw -> coordinate descent."""
    def decide_yaw(scale: float, yaw_now: float, xy_now, label: str):
        """Coarse yaw sweep at ``scale``; ambiguous peaks are settled by appearance + IoU (textured
        mesh) or by staying closest to the reconstruction's own yaw."""
        curve, yaw_range = _yaw_sweep(obj, scale, xy_now, step_deg=yaw_step_deg, mode=yaw_mode)
        res.yaw_curve = [(c[0], c[1]) for c in curve]
        iou_now = mask_iou(obj.model(scale).silhouette(xy_now, yaw_now), obj.real_mask)
        best_iou = max(c[1] for c in curve)
        has_appearance = obj.colors() is not None and obj.evidence.image is not None
        tol = YAW_APPEARANCE_IOU if has_appearance else YAW_AMBIGUITY_IOU
        near = [c for c in curve if c[1] >= best_iou - tol]
        deg, iou_k, xy_k = min(near, key=lambda c: abs(((c[0] + 180.0) % 360.0) - 180.0))
        if len(near) > 1:
            scored = []
            for c in near:
                a = obj.appearance(obj.model(scale), c[2], math.radians(c[0]))
                if a is not None:
                    # combined: silhouette overlap minus a colour-difference penalty (0.01 per unit)
                    scored.append((c[1] - YAW_APPEARANCE_WEIGHT * a, a, c))
            if scored:
                _, a_best, (deg, iou_k, xy_k) = max(scored, key=lambda t: t[0])
                res.notes.append(f"yaw ambiguous at {label}: {len(near)} peaks within {tol} IoU; IoU+appearance picks {int(deg)} deg (IoU {iou_k:.2f}, colour diff {a_best:.1f}; " + ", ".join(f"{int(c[0])}:{c[1]:.2f}/{a:.0f}" for _, a, c in sorted(scored, key=lambda t: -t[0])[:4]) + ")")
            else:
                res.notes.append(f"yaw ambiguous at {label}: {len(near)} peaks within {tol} IoU ({', '.join(f'{int(c[0])}:{c[1]:.2f}' for c in near)}); keeping the one nearest 0 deg")
        if iou_k - iou_now > yaw_min_gain or (has_appearance and len(near) > 1):
            new_yaw = math.radians(((deg + 180.0) % 360.0) - 180.0)
            res.notes.append(f"yaw {math.degrees(new_yaw):+.0f} deg at {label}: IoU {iou_now:.3f} -> {iou_k:.3f} (sweep range {yaw_range:.3f})")
            return new_yaw, xy_k
        res.notes.append(f"yaw kept at {math.degrees(yaw_now):+.0f} deg at {label}: best sweep gain {iou_k - iou_now:+.3f} < {yaw_min_gain} (range {yaw_range:.3f})")
        return yaw_now, xy_now

    # 1. yaw at the seed scale (shape decides yaw, size does not)
    best_yaw = float(seed_yaw)
    xy_seed = _fit_xy_at(obj, seed_scale, best_yaw, seed_xy)
    if yaw_mode != "off":
        best_yaw, xy_seed = decide_yaw(seed_scale, best_yaw, xy_seed, f"seed scale x{seed_scale:.2f}")

    # 2. scale sweep with XY re-fitted per scale.  Semantics first: the reconstruction's own
    # size is kept unless the silhouette at that size is clearly the wrong area (outside
    # SCALE_KEEP_AREA_RANGE), because a silhouette trades size for depth along the view ray
    # (droid108: cups inflated 15-20 % and pushed back for +0.05 IoU).
    xy_1 = _fit_xy_at(obj, 1.0, best_yaw, xy_seed)
    sil_1 = obj.model(1.0).silhouette(xy_1, best_yaw)
    area_1 = float(sil_1.sum() / max(int(obj.real_mask.sum()), 1))
    # (a tiny mesh can still match the mask area by sitting close to the camera, so the area rule
    # is skipped when the dims prior says the size is far off: the extended sweep decides instead)
    extended = len(scales) > len(DEFAULT_SCALES)
    if fixed_scale is None and not extended and SCALE_KEEP_AREA_RANGE[0] <= area_1 <= SCALE_KEEP_AREA_RANGE[1]:
        scales = [1.0]
        res.notes.append(f"scale kept at x1.00: silhouette area at the reconstruction's size is {area_1:.2f} of the mask (within {SCALE_KEEP_AREA_RANGE})")
    best = None
    for s in scales:
        xy_s = _fit_xy_at(obj, s, best_yaw, xy_seed)
        j, terms, cues = obj.evaluate(s, xy_s, best_yaw)
        res.scale_curve.append({"scale": float(s), "objective": j, "iou": cues.get("iou"), "area_ratio": cues.get("area_ratio"), "d_foot_m": obj.d_foot(obj.model(s), xy_s, best_yaw)})
        if best is None or j < best[0]:
            best = (j, s, xy_s, best_yaw, terms, cues)

    # 3. yaw again at the winning scale (a wrong size can favour the wrong yaw), then fine
    j, s, xy, yaw, terms, cues = best
    if yaw_mode != "off" and abs(s - seed_scale) > 1e-6:
        yaw2, xy2 = decide_yaw(s, yaw, xy, f"fitted scale x{s:.2f}")
        if yaw2 != yaw:
            xy2 = _fit_xy_at(obj, s, yaw2, xy2)
            j2, terms2, cues2 = obj.evaluate(s, xy2, yaw2)
            j, xy, yaw, terms, cues = j2, xy2, yaw2, terms2, cues2
    if yaw_mode == "sweep":
        for d in np.arange(-yaw_step_deg, yaw_step_deg + 1e-9, max(yaw_step_deg / 5.0, 1.0)):
            yaw_c = yaw + math.radians(d)
            xy_c = _fit_xy_at(obj, s, yaw_c, xy)
            j_c, terms_c, cues_c = obj.evaluate(s, xy_c, yaw_c)
            if j_c < j - 1e-6:
                j, xy, yaw, terms, cues = j_c, xy_c, yaw_c, terms_c, cues_c

    # 4. coordinate descent polish (scale +-0.02, xy +-5 mm, yaw +-3 deg)
    for _ in range(polish_rounds):
        improved = False
        moves = []
        if fixed_scale is None and len(scales) > 1:
            moves += [(ds, np.zeros(2), 0.0) for ds in (-0.02, 0.02)]
        moves += [(0.0, np.array(d), 0.0) for d in ((0.005, 0), (-0.005, 0), (0, 0.005), (0, -0.005))]
        if yaw_mode != "off":
            moves += [(0.0, np.zeros(2), math.radians(a)) for a in (-3.0, 3.0)]
        for ds, dxy, dyaw in moves:
            s_c = round(s + ds, 4)
            if not (0.2 <= s_c <= 5.0):
                continue
            j_c, terms_c, cues_c = obj.evaluate(s_c, xy + dxy, yaw + dyaw)
            if j_c < j - 1e-6:
                j, s, xy, yaw, terms, cues = j_c, s_c, xy + dxy, yaw + dyaw, terms_c, cues_c
                improved = True
        if not improved:
            break
    return j, s, np.asarray(xy, dtype=np.float64), float(yaw), terms, cues


def fit_object_joint(
    base: ObjectModel,
    evidence: SceneEvidence,
    *,
    key: str,
    weights: dict[str, float] | None = None,
    scales: list[float] | None = None,
    seed_scale: float = 1.0,
    seed_xy=(0.0, 0.0),
    seed_yaw: float = 0.0,
    yaw_mode: str = "sweep",
    yaw_step_deg: float = 15.0,
    yaw_min_gain: float = 0.02,
    fixed_scale: float | None = None,
    settle_entry: dict | None = None,
    target_points: list[np.ndarray] | None = None,
    polish_rounds: int = 2,
    orientation_mode: str = "yaw",
    orientation_top_k: int = 2,
    z_plane: float | None = None,
) -> JointFitResult:
    """Joint (orientation, scale, XY, yaw) search for one object; see the module docstring."""
    weights = weights or DEFAULT_WEIGHTS
    scales = [float(fixed_scale)] if fixed_scale is not None else list(scales or DEFAULT_SCALES)
    obj0 = Objective(base, evidence, weights=weights, settle_entry=settle_entry, target_points=target_points or [], key=key)
    res = JointFitResult(safe_key=key, count=obj0.count, object_name=base.spec.object_name)
    res.model = base
    j0, terms0, cues0 = obj0.evaluate(1.0, np.zeros(2), 0.0)
    res.objective_before, res.cues_before = j0, cues0
    # A reconstruction whose size is far from the physical dims (droid35 mug: 3.8 cm mesh, 12 cm real)
    # needs a sweep around the dims-implied multiplier; the default 0.70-1.40 cannot reach it.
    sm = (cues0.get("dims") or {}).get("suggested_multiplier")
    if fixed_scale is None and sm and (sm > SCALE_EXTEND_BEYOND or sm < 1.0 / SCALE_EXTEND_BEYOND):
        extra = [round(float(v), 3) for v in np.arange(0.6 * sm, 1.4 * sm + 1e-9, 0.05 * max(sm, 1.0))]
        scales = sorted(set(scales) | set(extra))
        res.notes.append(f"dims prior ({(cues0.get('dims') or {}).get('source')}) suggests x{sm:.2f}: scale sweep extended to {min(scales):.2f}-{max(scales):.2f}")
    if obj0.real_mask is None or not obj0.real_mask.any():
        res.status = "skipped_no_mask"
        res.objective, res.terms, res.cues = j0, terms0, cues0
        return res
    if not base.silhouette(np.zeros(2), 0.0).any():
        res.status = "skipped_no_silhouette"
        res.objective, res.terms, res.cues = j0, terms0, cues0
        res.notes.append("sim silhouette is empty (object outside the calibrated view?)")
        return res

    # Orientation candidates: as-is only, or the principal-axis rest poses ranked by a
    # cheap probe (XY refit + 4 yaws at the seed scale) with the top-k searched fully.
    orientations = [("as-is", base)]
    if orientation_mode == "principal":
        probes = []
        for label, model in principal_orientations(base, z_plane=z_plane):
            obj_o = Objective(model, evidence, weights=weights, settle_entry=settle_entry, target_points=target_points or [], key=key)
            best_probe = None
            for deg in (0.0, 90.0, 180.0, 270.0):
                yaw = math.radians(deg) + (float(seed_yaw) if label == "as-is" else 0.0)
                xy = _fit_xy_at(obj_o, seed_scale, yaw, seed_xy if label == "as-is" else (0.0, 0.0))
                j, _, cues = obj_o.evaluate(seed_scale, xy, yaw)
                if best_probe is None or j < best_probe[0]:
                    best_probe = (j, yaw, xy, cues.get("iou"))
            probes.append((best_probe[0], label, model, best_probe[1], best_probe[2], best_probe[3]))
        probes.sort(key=lambda p: p[0])
        res.notes.append("orientation probe: " + ", ".join(f"{p[1]} J {p[0]:.2f} IoU {p[5] or 0:.2f}" for p in probes))
        chosen = probes[: max(1, orientation_top_k)]
        if not any(p[1] == "as-is" for p in chosen):
            chosen.append(next(p for p in probes if p[1] == "as-is"))
        orientations = [(p[1], p[2]) for p in chosen]
        probe_seed = {p[1]: (p[3], p[4]) for p in probes}
    else:
        probe_seed = {"as-is": (float(seed_yaw), np.asarray(seed_xy, dtype=np.float64))}

    def run_searches(active_weights: dict[str, float]) -> tuple[tuple, list[str]]:
        """Search every orientation with the given weights; apply the mask-evidence gate."""
        best_local = None
        as_is_local = None
        notes: list[str] = []
        for label, model in orientations:
            obj = Objective(model, evidence, weights=active_weights, settle_entry=settle_entry, target_points=target_points or [], key=key)
            sub = JointFitResult(safe_key=key, count=obj.count, object_name=model.spec.object_name)
            yaw_seed, xy_seed = probe_seed.get(label, (0.0, np.zeros(2)))
            j, s, xy, yaw, terms, cues = _search(
                obj,
                sub,
                scales=scales,
                seed_scale=seed_scale,
                seed_xy=xy_seed,
                seed_yaw=yaw_seed,
                yaw_mode=yaw_mode,
                yaw_step_deg=yaw_step_deg,
                yaw_min_gain=yaw_min_gain,
                fixed_scale=fixed_scale,
                polish_rounds=polish_rounds,
            )
            if len(orientations) > 1:
                notes.append(f"orientation {label}: J {j:.2f} scale x{s:.2f} IoU {cues.get('iou') or 0:.3f}")
            entry = (j, s, xy, yaw, terms, cues, label, model, sub)
            if best_local is None or j < best_local[0]:
                best_local = entry
            if label == "as-is":
                as_is_local = entry
        # Mask-evidence gate: a different rest orientation is only adopted when the
        # silhouette still explains the mask about as well as the as-is one; the
        # priors (dims, fingertip height) may break ties but not overrule the image.
        if best_local[6] != "as-is" and as_is_local is not None:
            iou_best = best_local[5].get("iou") or 0.0
            iou_as_is = as_is_local[5].get("iou") or 0.0
            if iou_best < iou_as_is - ORIENTATION_IOU_TOLERANCE:
                notes.append(f"orientation {best_local[6]} rejected: IoU {iou_best:.3f} < as-is {iou_as_is:.3f} - {ORIENTATION_IOU_TOLERANCE}")
                best_local = as_is_local
        return best_local, notes

    # Push detection: the mask-driven (anchor-free) fit is judged first.  When its
    # footprint is more than PUSH_DISAGREE_M from the fingertip anchor, the real
    # hand most likely moved the object before closing (droid2 plush: ~8 cm); the
    # frame-0 mask is then the only evidence for the initial pose, so the anchor
    # is dropped for this object.  Otherwise the anchored search decides.
    is_target = bool(target_points) and weights.get("grasp", 0) > 0
    if is_target:
        no_grasp = {**weights, "grasp": 0.0}
        alt_best, alt_notes = run_searches(no_grasp)
        a2 = alt_best[5].get("grasp_anchor") or {}
        if a2.get("d_foot_m", 0.0) > PUSH_DISAGREE_M:
            res.notes.extend(alt_notes)
            res.notes.append(
                f"grasp anchor ignored: mask-best pose is {100 * a2.get('d_foot_m', 0):.1f} cm from the closing point "
                f"(IoU {alt_best[5].get('iou') or 0:.3f}); object likely pushed before the grasp"
            )
            best_overall = alt_best
            j0 = sum(objective_terms(cues0, None, no_grasp).values())
            terms0 = objective_terms(cues0, None, no_grasp)
            res.objective_before = j0
            res.pushed = True
        else:
            best_overall, notes = run_searches(weights)
            res.notes.extend(notes)
    else:
        best_overall, notes = run_searches(weights)
        res.notes.extend(notes)

    j, s, xy, yaw, terms, cues, label, model, sub = best_overall

    res.model = model
    res.orientation = label
    res.scale_curve, res.yaw_curve = sub.scale_curve, sub.yaw_curve
    res.notes.extend(sub.notes)
    res.scale_multiplier = float(s)
    res.xy_delta = np.asarray(xy, dtype=np.float64)
    res.yaw_rad = float(((yaw + math.pi) % (2 * math.pi)) - math.pi)
    res.objective, res.terms, res.cues = j, terms, cues
    res.status = "refined" if (np.isfinite(j) and j < j0 - 1e-6) else "unchanged"
    if res.status == "unchanged":
        res.model, res.orientation = base, "as-is"
        res.scale_multiplier, res.xy_delta, res.yaw_rad = 1.0, np.zeros(2), 0.0
        res.objective, res.terms, res.cues = j0, terms0, cues0
        res.notes.append("no candidate beat the original objective; keeping the original")
    res.notes.append(f"{obj0.evaluations} objective evaluations")
    return res


def _hull_contains_3d(container_pts: np.ndarray, pts: np.ndarray) -> float:
    from scipy.spatial import ConvexHull, Delaunay

    if len(container_pts) < 4 or len(pts) == 0:
        return 0.0
    try:
        hull = ConvexHull(container_pts)
        tri = Delaunay(container_pts[hull.vertices])
    except Exception:
        return 0.0
    return float((tri.find_simplex(pts) >= 0).mean())


def resolve_3d_overlaps(models: dict[str, ObjectModel], results: dict[str, JointFitResult], evidence: SceneEvidence, log=print) -> None:
    """Before placement: no refined object may sit inside another (an open box over a cup).

    For each overlapping pair the object with the larger extent is pushed away
    from the camera along its ground view ray in 1 cm steps (the silhouette
    barely changes, the depth does) until fewer than OVERLAP_3D_FRACTION of
    either object's vertices lie inside the other's hull, or the push budget is
    spent, in which case that object keeps its original pose.
    """
    def world_pts(k):
        r = results[k]
        return scaled_model(models[k], r.scale_multiplier).spec.world_vertices(scaled_model(models[k], r.scale_multiplier).pos_for(r.xy_delta), r.yaw_rad)

    keys = [k for k, r in results.items() if r.status == "refined"]
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            if results[a].status != "refined" or results[b].status != "refined":
                continue
            pa, pb = world_pts(a), world_pts(b)
            step = max(1, len(pa) // 2000), max(1, len(pb) // 2000)
            frac = max(_hull_contains_3d(pb, pa[:: step[0]]), _hull_contains_3d(pa, pb[:: step[1]]))
            if frac < OVERLAP_3D_FRACTION:
                continue
            ext = {k: float(np.ptp(models[k].spec.centered_vertices, axis=0).max()) for k in (a, b)}
            mover = a if ext[a] >= ext[b] else b
            other = b if mover == a else a
            r = results[mover]
            cam_c = -evidence.camera.T_w2c[:3, :3].T @ evidence.camera.T_w2c[:3, 3]
            m = scaled_model(models[mover], r.scale_multiplier)
            ray = m.pos_for(r.xy_delta)[:2] - cam_c[:2]
            ray /= max(np.linalg.norm(ray), 1e-9)
            real = evidence.real_masks.get(int(m.spec.count))
            iou0 = mask_iou(m.silhouette(r.xy_delta, r.yaw_rad), real[1]) if real else None
            pushed = None
            for d in np.arange(OVERLAP_PUSH_STEP_M, OVERLAP_PUSH_MAX_M + 1e-9, OVERLAP_PUSH_STEP_M):
                xy = r.xy_delta + ray * d
                pts = m.spec.world_vertices(m.pos_for(xy), r.yaw_rad)
                po = world_pts(other)
                f2 = max(_hull_contains_3d(po, pts[:: step[0]]), _hull_contains_3d(pts, po[:: step[1]]))
                if f2 < OVERLAP_3D_FRACTION:
                    iou = mask_iou(m.silhouette(xy, r.yaw_rad), real[1]) if real else None
                    pushed = (d, xy, f2, iou)
                    break
            if pushed is None:
                r.status = "conflict_not_applied"
                r.scale_multiplier, r.xy_delta, r.yaw_rad = 1.0, np.zeros(2), 0.0
                r.notes.append(f"3-D overlap with {results[other].object_name} [{results[other].count}] ({frac:.2f}) could not be cleared within {100 * OVERLAP_PUSH_MAX_M:.0f} cm; keeping the original pose")
                log(f"[fit-joint] overlap: {r.object_name} [{r.count}] inside {results[other].object_name}: {frac:.2f}, unresolved -> original pose")
            else:
                d, xy, f2, iou = pushed
                r.xy_delta = np.asarray(xy, dtype=np.float64)
                r.notes.append(f"3-D overlap with {results[other].object_name} [{results[other].count}] ({frac:.2f}): pushed {100 * d:.0f} cm away from the camera (overlap {f2:.2f}, IoU {iou0 if iou0 is None else round(iou0, 3)} -> {iou if iou is None else round(iou, 3)})")
                if r.cues is not None and iou is not None:
                    r.cues["iou"] = iou
                log(f"[fit-joint] overlap: {r.object_name} [{r.count}] inside {results[other].object_name}: {frac:.2f} -> pushed {100 * d:.0f} cm along the view ray (IoU {iou0 if iou0 is None else round(iou0, 3)} -> {round(iou, 3) if iou is not None else None})")


# ---------------------------------------------------------------------------
# Scene-level driver
# ---------------------------------------------------------------------------
def load_json(path) -> dict:
    if not path:
        return {}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def seed_pose_for(seed: Candidate | None, key: str) -> tuple[float, np.ndarray, float]:
    """(scale, xy, yaw) a seed candidate applied to this object, relative to the dump pose."""
    if seed is None:
        return 1.0, np.zeros(2), 0.0
    e = seed.entry(key)
    if e is None or not e.get("apply", True):
        return 1.0, np.zeros(2), 0.0
    scale = float(e.get("scale_multiplier", 1.0)) if e.get("scale_apply", True) else 1.0
    return scale, np.asarray(e.get("xy_delta_m", [0.0, 0.0]), dtype=np.float64), math.radians(float(e.get("yaw_delta_deg", 0.0)))


def fit_scene(
    paths: RefinePaths,
    dump: dict,
    *,
    candidate_id: str,
    seed: Candidate | None = None,
    settle: dict[str, dict] | None = None,
    weights: dict[str, float] | None = None,
    decisions: dict[str, dict] | None = None,
    scales: list[float] | None = None,
    yaw_mode: str = "sweep",
    yaw_step_deg: float = 15.0,
    allow_overlap: bool = False,
    max_overlap: float = 0.25,
    level_max_deg: float = 30.0,
    orientation_mode: str = "yaw",
    orientation_top_k: int = 2,
    upright_classes=UPRIGHT_CLASSES,
    rest_on_collider: bool = True,
    log=print,
) -> tuple[Candidate, dict[str, ObjectModel], dict[str, JointFitResult], SceneEvidence]:
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    decisions = decisions or {}
    z_plane = rest_plane(dump)  # physics table top when the dump carries it, else z_plane
    cal_plane = dump.get("z_plane")
    evidence = SceneEvidence.load(paths, z_plane=float(cal_plane if cal_plane is not None else 0.02))
    if z_plane is not None and cal_plane is not None and abs(z_plane - cal_plane) > 0.005:
        log(f"[fit-joint] resting objects on physics_top {z_plane:.4f} (calibration z_plane {cal_plane:.4f})")
    log(f"[fit-joint] {paths.scene}: {len(dump['objects'])} objects, {len(evidence.events)} grasp event(s), targets {[(t['onset'], t.get('object_name')) for t in evidence.targets]}")
    base_models = build_models(dump, evidence.camera, z_plane=z_plane, level_max_deg=level_max_deg, asset_overrides=asset_overrides_of(seed), orientation_overrides=orientation_overrides_of(seed))
    if seed is not None and asset_overrides_of(seed):
        log(f"[fit-joint] asset overrides from seed: {sorted(asset_overrides_of(seed))}")
    target_points_by_count: dict[int, list[np.ndarray]] = {}
    for t in evidence.targets:
        if t.get("count") is not None:
            pts = t.get("anchor_points") or [t["tcp_onset"]]
            target_points_by_count.setdefault(int(t["count"]), []).extend(np.asarray(p, dtype=np.float64) for p in pts)

    results: dict[str, JointFitResult] = {}
    fitted_models: dict[str, ObjectModel] = {}
    upright_info: dict[str, dict] = {}
    for key, base in base_models.items():
        dec = decisions.get(key, {})
        settle_entry = (settle or {}).get(key)
        started = time.perf_counter()
        if upright_classes and is_upright_class(base.spec.object_name, upright_classes) and not dec.get("lock") and not dec.get("skip_upright") and not dec.get("upright"):
            base, info = upright_by_symmetry_axis(base, z_plane=z_plane)
            upright_info[key] = info
            if info["applied"]:
                log(f"[fit-joint] {base.spec.object_name} [{base.spec.count}] uprighted: symmetry axis {info['axis_tilt_deg_before']:.1f} deg -> {info['axis_tilt_deg_after']:.1f} deg off vertical (radial cv {info['radial_cv']:.2f})")
            elif info["axis_tilt_deg_before"] > UPRIGHT_MIN_DEG:
                log(f"[fit-joint] {base.spec.object_name} [{base.spec.count}] symmetry axis {info['axis_tilt_deg_before']:.1f} deg off vertical, NOT uprighted: {info.get('reason')}")
        elif dec.get("upright"):
            base, info = upright_by_symmetry_axis(base, z_plane=z_plane, max_deg=180.0, max_cv=10.0)
            upright_info[key] = info
            log(f"[fit-joint] {base.spec.object_name} [{base.spec.count}] uprighted by decision: {info['axis_tilt_deg_before']:.1f} -> {info.get('axis_tilt_deg_after', info['axis_tilt_deg_before']):.1f} deg")
        if dec.get("orientation"):
            # a principal-axis rest orientation chosen by a human / appearance check (labels as in
            # principal_orientations: as-is, pca0+, pca0-, pca1+, pca1-, pca2+, pca2-)
            choices = dict(principal_orientations(base, z_plane=z_plane))
            if dec["orientation"] not in choices:
                raise SystemExit(f"decisions.json: unknown orientation {dec['orientation']!r} for {key}; choose from {sorted(choices)}")
            base = choices[dec["orientation"]]
            upright_info[key] = {"applied": True, "by_decision": dec["orientation"]}
            log(f"[fit-joint] {base.spec.object_name} [{base.spec.count}] orientation {dec['orientation']} by decision")
        if settle_entry and settle_entry.get("unstable") and not dec.get("lock"):
            # An object that tipped in the previous round is NOT refit from its fallen pose (the
            # real object is upright; adopting the fall made droid2's tiger a lying blob).  It keeps
            # the commanded orientation and is flagged: the remedy is the asset's bottom
            # (flatten-bottom) or a collision override, not a new orientation.
            log(f"[fit-joint] {base.spec.object_name} [{base.spec.count}] tipped {settle_entry.get('tilt_deg', 0):.0f} deg last round -> keeping its orientation; run diagnose-hull / flatten-bottom")
        if dec.get("lock"):
            res = JointFitResult(safe_key=key, count=int(base.spec.count), object_name=base.spec.object_name, status="locked")
            j0, terms0, cues0 = Objective(base, evidence, weights=weights, settle_entry=settle_entry, target_points=target_points_by_count.get(int(base.spec.count), []), key=key).evaluate(1.0, np.zeros(2), 0.0)
            res.objective = res.objective_before = j0
            res.terms, res.cues, res.cues_before = terms0, cues0, cues0
            res.notes.append("locked by decisions.json")
        else:
            s0, xy0, yaw0 = seed_pose_for(seed, key)
            if dec.get("yaw_deg") is not None:
                yaw0 = math.radians(float(dec["yaw_deg"]))  # pinned yaw (relative to the base orientation); no sweep
            res = fit_object_joint(
                base,
                evidence,
                key=key,
                weights={**weights, **dec.get("weights", {})},
                scales=scales,
                seed_scale=s0,
                seed_xy=xy0,
                seed_yaw=yaw0,
                yaw_mode="off" if (dec.get("skip_yaw") or dec.get("yaw_deg") is not None) else yaw_mode,
                yaw_step_deg=yaw_step_deg,
                fixed_scale=dec.get("scale"),
                settle_entry=settle_entry,
                target_points=target_points_by_count.get(int(base.spec.count), []),
                orientation_mode="yaw" if dec.get("skip_orientation") else orientation_mode,
                orientation_top_k=orientation_top_k,
                z_plane=z_plane,
            )
            if settle_entry and settle_entry.get("unstable"):
                res.notes.append(f"unstable last round (tilt {settle_entry.get('tilt_deg', 0):.1f} deg, drift {100 * settle_entry.get('xy_drift_m', 0):.1f} cm): orientation kept; fix the bottom (flatten-bottom) rather than the pose")
        results[key] = res
        fitted_models[key] = res.model if res.model is not None else base
        c = res.cues
        anchor = c.get("grasp_anchor") or {}
        log(
            f"[fit-joint] {res.object_name:22s} [{res.count}] {res.status:12s} J {res.objective_before:6.2f} -> {res.objective:6.2f}  "
            f"orient {res.orientation} scale x{res.scale_multiplier:.2f} shift ({100 * res.xy_delta[0]:+.1f},{100 * res.xy_delta[1]:+.1f}) cm yaw {math.degrees(res.yaw_rad):+.0f} deg  "
            f"IoU {res.cues_before.get('iou') or 0:.3f} -> {c.get('iou') or 0:.3f} area {c.get('area_ratio') or 0:.2f}"
            + (f"  d_foot {100 * anchor.get('d_foot_m', 0):.1f} cm dz_top {100 * anchor.get('dz_top_m', 0):+.1f} cm" if anchor else "")
            + f"  ({time.perf_counter() - started:.1f} s)"
        )
        for note in res.notes:
            log(f"       {note}")

    if not allow_overlap:
        resolve_3d_overlaps(fitted_models, results, evidence, log=log)
        scaled = {k: scaled_model(fitted_models[k], results[k].scale_multiplier) for k in results}
        old_results = {k: results[k].as_fit_result() for k in results}
        for note in resolve_footprint_conflicts(scaled, old_results, max_overlap=max_overlap):
            log(f"[fit-joint] conflict: {note}")
        for k, fr in old_results.items():
            if fr.status == "conflict_not_applied":
                results[k].status = "conflict_not_applied"
                results[k].scale_multiplier, results[k].xy_delta, results[k].yaw_rad = 1.0, np.zeros(2), 0.0
                results[k].notes.extend(fr.notes)

    # Rest every fitted object on the background collider under its NEW footprint (the load-time
    # guard only knew the load-time XY; a cup moved 10 cm may now sit over a dip or a bump).
    collider = build_collider_query(paths) if rest_on_collider else None
    if rest_on_collider and collider is None:
        log("[fit-joint] no reachable collider crop found; resting on the plane only")
    objects = []
    final_models: dict[str, ObjectModel] = {}
    for key, res in results.items():
        model = scaled_model(fitted_models[key], res.scale_multiplier)
        prev = (settle or {}).get(key)
        if prev is not None and prev.get("z_drop_m") is not None and abs(float(prev["z_drop_m"])) > SETTLE_Z_CORRECT_MIN_M and float(np.linalg.norm(np.asarray(prev.get("commanded_pos", [0, 0, 0])[:2]) - model.pos_for(res.xy_delta)[:2])) < SETTLE_Z_CORRECT_MAX_XY_M:
            # The previous round measured where the physics surface really is under this spot
            # (the object sank or was lifted by that much); the collider crop we can read offline
            # does not always show it.  Correct the height only, never the orientation.
            dz = float(prev["z_drop_m"])
            model.spec.base_pos = model.spec.base_pos.copy()
            model.spec.base_pos[2] += dz
            model.z_delta += dz
            res.notes.append(f"height corrected by the previous settle: physics surface {100 * dz:+.1f} cm from the commanded rest")
            log(f"[fit-joint] {res.object_name} [{res.count}] previous settle sank {100 * dz:+.1f} cm at this spot -> rest height corrected")
        elif collider is not None:
            support, dz = rest_height_on_collider(collider, model, res.xy_delta, res.yaw_rad, plane_z=z_plane)
            if dz is not None and abs(dz) > 0.002:
                model.spec.base_pos = model.spec.base_pos.copy()
                model.spec.base_pos[2] += dz
                model.z_delta += dz
                res.notes.append(f"rested on the collider at the fitted footprint: support z {support:.3f}, moved {100 * dz:+.1f} cm")
                log(f"[fit-joint] {res.object_name} [{res.count}] collider support under the fitted footprint {support:.3f} m -> z {100 * dz:+.1f} cm")
        apply = res.status in {"refined"} or res.settled_refit
        entry = candidate_object_entry(
            model,
            xy_delta=res.xy_delta,
            yaw_rad=res.yaw_rad,
            scale_multiplier=res.scale_multiplier,
            status=res.status,
            apply=apply,
            extra={
                "orientation": res.orientation,
                "upright": upright_info.get(key),
                "pushed_before_grasp": res.pushed,
                "objective_before": res.objective_before,
                "objective": res.objective,
                "terms": res.terms,
                "cues": _jsonable(res.cues),
                "cues_before": _jsonable(res.cues_before),
                "scale_curve": res.scale_curve,
                "yaw_curve_deg_iou": [[float(a), float(b)] for a, b in res.yaw_curve],
                "settled_refit": res.settled_refit,
                "notes": res.notes,
            },
        )
        seed_entry = seed.entry(key) if seed is not None else None
        if seed_entry is not None:
            for carried in ("asset_override", "asset_override_info", "asset_apply", "collision", "orientation_quat_wxyz"):
                if carried in seed_entry:
                    entry[carried] = seed_entry[carried]
        objects.append(entry)
        final_models[key] = model_at_settled_pose(model, entry["pos"], entry["quat_wxyz"])
    cand = Candidate(
        scene=paths.scene,
        candidate_id=candidate_id,
        method="B",
        parent=seed.candidate_id if seed is not None else "stock",
        source_dump="",
        settings={"weights": weights, "decisions": decisions, "scales": scales or DEFAULT_SCALES, "yaw_mode": yaw_mode, "yaw_step_deg": yaw_step_deg, "orientation_mode": orientation_mode, "orientation_top_k": orientation_top_k, "allow_overlap": allow_overlap, "max_overlap": max_overlap, "level_max_deg": level_max_deg, "settle_used": bool(settle)},
        objects=objects,
    )
    return cand, final_models, results, evidence


def _jsonable(value):
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


# ---------------------------------------------------------------------------
# Review outputs (for the human / Claude pass between rounds)
# ---------------------------------------------------------------------------
def write_review(out_dir, paths: RefinePaths, evidence: SceneEvidence, dump: dict, candidates: list[tuple[Candidate, dict[str, ObjectModel], dict[str, Any]]]) -> list[str]:
    """residual_table.md + silhouette overlays per candidate + a plan view with the real TCP path."""
    import cv2

    from r2s.objects.pose_fit import draw_contour, load_first_real_image

    out_dir = str(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    written = []
    image = load_first_real_image(paths.images_dir)
    if image is None:
        image = np.zeros((evidence.camera.height, evidence.camera.width, 3), dtype=np.uint8)
    colors = [(255, 60, 60), (60, 255, 60), (60, 120, 255), (255, 200, 40), (200, 60, 255)]

    # 1. residual table
    lines = [f"# {paths.scene} asset refinement review", "", f"instruction: {evidence.task.instruction or '(none)'}", ""]
    if evidence.targets:
        events = []
        for t in evidence.targets:
            dist = "-" if t.get("dist_px") is None else format(t["dist_px"], ".0f") + " px"
            events.append(f"frame {t['onset']} -> {t.get('object_name')} [{t.get('count')}] ({dist} from mask)")
        lines.append("grasp events (real): " + "; ".join(events))
        lines.append("")
    header = "| object | candidate | scale | shift cm | yaw | IoU | area | dims src | dims resid | depth q | front res cm | h_est/h_sim cm | d_foot cm | tilt | drift cm | J |"
    lines += [header, "|" + "---|" * 16]
    for cand, models, score in candidates:
        for key, obj in score["objects"].items():
            e = cand.entry(key) or {}
            d = obj.get("depth") or {}
            st = obj.get("stability") or {}
            dfoot = next((o.get("d_foot_m") for o in score["grasp_offline"] if o.get("safe_key") == key), None)
            dims = obj.get("dims") or {}
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"{obj['object_name']} [{obj['count']}]",
                        cand.candidate_id,
                        f"x{e.get('scale_multiplier', 1.0):.2f}",
                        f"{e.get('shift_cm', 0.0):.1f}",
                        f"{e.get('yaw_delta_deg', 0.0):+.0f}",
                        "-" if obj.get("iou") is None else f"{obj['iou']:.3f}",
                        "-" if obj.get("area_ratio") is None else f"{obj['area_ratio']:.2f}",
                        str(dims.get("source")),
                        f"{dims.get('residual', 0.0):.2f}",
                        str(d.get("quality")),
                        "-" if d.get("front_residual_m") is None else f"{100 * d['front_residual_m']:+.1f}",
                        ("-" if d.get("height_est_m") is None else f"{100 * d['height_est_m']:.1f}") + "/" + ("-" if d.get("height_sim_m") is None else f"{100 * d['height_sim_m']:.1f}"),
                        "-" if dfoot is None else f"{100 * dfoot:.1f}",
                        "-" if st.get("tilt_deg") is None else f"{st['tilt_deg']:.1f}",
                        "-" if st.get("xy_drift_m") is None else f"{100 * st['xy_drift_m']:.1f}",
                        f"{obj.get('objective', 0.0):.2f}",
                    ]
                )
                + " |"
            )
        g = score["grasp"]
        lines.append(f"| **{cand.candidate_id} totals** | | | | | mean {score['mean_iou'] or 0:.3f} | | | | | | | G {g['G'] if g['G'] is not None else '-'} | unstable {score['n_unstable']} | | {score['objective_total']:.2f} |")
    path = os.path.join(out_dir, "residual_table.md")
    with open(prepare_output(path), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    written.append(path)

    # 2. silhouette overlays: real mask white, each candidate a colour
    canvas = image.copy()
    for i, (cand, models, _) in enumerate(candidates):
        for key, model in models.items():
            draw_contour(canvas, model.silhouette(np.zeros(2), 0.0), colors[i % len(colors)], 1)
    for count, (name, mask) in evidence.real_masks.items():
        draw_contour(canvas, mask, (255, 255, 255), 2)
    legend = "white=real  " + "  ".join(f"{cand.candidate_id}={['red','green','blue','yellow','purple'][i % 5]}" for i, (cand, _, _) in enumerate(candidates))
    cv2.putText(canvas, legend, (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1, cv2.LINE_AA)
    if evidence.tcp is not None:
        for t in evidence.targets:
            uv, _ = evidence.camera.project(np.asarray(t["tcp_onset"])[None])
            cv2.circle(canvas, (int(uv[0, 0]), int(uv[0, 1])), 5, (0, 0, 255), 2)
    path = os.path.join(out_dir, "overlay_candidates.png")
    cv2.imwrite(str(prepare_output(path)), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    written.append(path)

    # 3. plan view: footprints + TCP path around each grasp
    xs, ys = [], []
    for _, models, _ in candidates:
        for m in models.values():
            lo, hi = m.spec.base_aabb
            xs += [lo[0], hi[0]]
            ys += [lo[1], hi[1]]
    if evidence.tcp is not None:
        xs += [evidence.tcp[:, 0].min(), evidence.tcp[:, 0].max()]
        ys += [evidence.tcp[:, 1].min(), evidence.tcp[:, 1].max()]
    if xs:
        x0, x1, y0, y1 = min(xs) - 0.05, max(xs) + 0.05, min(ys) - 0.05, max(ys) + 0.05
        size = 800
        sx = size / max(x1 - x0, 1e-3)
        sy = size / max(y1 - y0, 1e-3)
        plan = np.full((size, size, 3), 255, dtype=np.uint8)

        def to_px(x, y):  # x up (away from the robot), +y to the left as in the image frame
            return int((y1 - y) * sy), int((x1 - x) * sx)

        for i, (cand, models, _) in enumerate(candidates):
            for m in models.values():
                foot = m.spec.world_vertices(m.spec.base_pos, 0.0)[:, :2]
                hull = cv2.convexHull(np.array([to_px(x, y) for x, y in foot], dtype=np.int32))
                cv2.polylines(plan, [hull], True, colors[i % len(colors)][::-1], 1 + (i == len(candidates) - 1))
                if i == len(candidates) - 1:
                    cx, cy = to_px(*m.spec.base_pos[:2])
                    cv2.putText(plan, f"{m.spec.count}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        if evidence.tcp is not None:
            for t in evidence.targets:
                k0, k1 = max(t["onset"] - 15, 0), min(t["release"] + 5, len(evidence.tcp) - 1)
                pts = np.array([to_px(x, y) for x, y in evidence.tcp[k0:k1, :2]], dtype=np.int32)
                cv2.polylines(plan, [pts], False, (0, 0, 0), 1)
                cv2.circle(plan, to_px(*evidence.tcp[t["onset"], :2]), 6, (0, 0, 255), 2)
                cv2.putText(plan, f"close f{t['onset']}", to_px(*evidence.tcp[t["onset"], :2]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200), 1, cv2.LINE_AA)
        cv2.putText(plan, "plan view: x up (away from robot), y left; " + legend, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        path = os.path.join(out_dir, "plan_view.png")
        cv2.imwrite(str(prepare_output(path)), plan)
        written.append(path)
    return written


# ---------------------------------------------------------------------------
# Drivers (the old CLI commands as functions)
# ---------------------------------------------------------------------------
def load_candidate_any(path, scene: str) -> Candidate:
    """A ``candidate.json`` (version >= 2) or a ``refined_poses.json`` wrapped as a candidate.

    A wrapped ``refined_poses.json`` is named after its folder (old behaviour):
    ``posefit`` for ``objects/posefit/refined_poses.json`` (the old repo's
    ``object_pose_fit/<scene>/`` gave ``<scene>``).
    """
    payload = load_json(path)
    if "candidate_id" in payload or payload.get("version", 0) >= 2:
        return Candidate.load(path)
    cid = os.path.basename(os.path.dirname(os.path.abspath(path))) or "posefit"
    return candidate_from_refined_poses(path, scene=scene, candidate_id=cid)


@dataclass
class FitJointSettings:
    """``asset_refine_fit.py fit-joint`` options; the defaults are the production values."""

    scales: list[float] | None = None  # None = DEFAULT_SCALES (0.70..1.40 step 0.05)
    yaw: str = "sweep"
    yaw_step_deg: float = 15.0
    orientations: str = "yaw"
    orientation_top_k: int = 2
    upright_classes: tuple[str, ...] = UPRIGHT_CLASSES
    no_upright: bool = False
    no_collider_rest: bool = False
    allow_overlap: bool = False
    max_overlap: float = 0.25
    no_level: bool = False
    level_max_deg: float = 30.0

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "FitJointSettings":
        kwargs = {k: getattr(args, k) for k in cls.__dataclass_fields__ if hasattr(args, k)}
        if kwargs.get("scales"):
            kwargs["scales"] = [float(s) for s in kwargs["scales"]]
        if "upright_classes" in kwargs:
            kwargs["upright_classes"] = tuple(kwargs["upright_classes"])
        return cls(**kwargs)


def fit_joint(
    paths: RefinePaths,
    *,
    dump_path,
    out_path,
    candidate_id: str | None = None,
    seed_path=None,
    settle_path=None,
    weights_path=None,
    decisions_path=None,
    settings: FitJointSettings | None = None,
    review_dir=None,
    log=print,
) -> Candidate:
    """Old ``fit-joint``: fit, score offline, write ``candidate.json`` and (with ``review_dir``) the review."""
    st = settings or FitJointSettings()
    dump = load_object_pose_dump(dump_path)
    seed = load_candidate_any(seed_path, paths.scene) if seed_path else None
    settle = load_settle_dump(settle_path) if settle_path else None
    weights = load_json(weights_path)
    decisions = load_json(decisions_path)
    scales = [float(s) for s in st.scales] if st.scales else None
    cand, models, results, evidence = fit_scene(
        paths,
        dump,
        candidate_id=candidate_id or os.path.basename(os.path.dirname(os.path.abspath(out_path))),
        seed=seed,
        settle=settle,
        weights=weights,
        decisions=decisions,
        scales=scales,
        yaw_mode=st.yaw,
        yaw_step_deg=st.yaw_step_deg,
        allow_overlap=st.allow_overlap,
        max_overlap=st.max_overlap,
        level_max_deg=0.0 if st.no_level else st.level_max_deg,
        orientation_mode=st.orientations,
        orientation_top_k=st.orientation_top_k,
        upright_classes=() if st.no_upright else tuple(st.upright_classes),
        rest_on_collider=not st.no_collider_rest,
        log=log,
    )
    cand.source_dump = os.path.abspath(str(dump_path))
    score = score_candidate(cand, models, evidence, settle=None, replay_dir=None)
    cand.scores = {"offline": _jsonable({k: v for k, v in score.items() if k not in {"objects"}}), "mean_iou": score["mean_iou"], "G_offline": score["grasp"]["G"], "objective_total": score["objective_total"]}
    written = cand.save(out_path)
    log(f"[fit-joint] wrote {written}  mean IoU {score['mean_iou']:.3f}  G(offline) {score['grasp']['G']}  J {score['objective_total']:.2f}")
    if review_dir is not None:
        entries = []
        if seed is not None:
            seed_models = build_models(dump, evidence.camera, candidate=seed, z_plane=rest_plane(dump))
            entries.append((seed, seed_models, score_candidate(seed, seed_models, evidence)))
        entries.append((cand, models, score))
        for path in write_review(review_dir, paths, evidence, dump, entries):
            log(f"[fit-joint] review {path}")
    return cand


def review_candidates(paths: RefinePaths, *, dump_path, candidate_paths: list, out_dir, lead: int = 5, log=print) -> list[str]:
    """Old ``review``: residual table + overlays + plan view for candidates (settle.json next to each is used)."""
    dump = load_object_pose_dump(dump_path)
    z_plane = rest_plane(dump)
    cal_plane = dump.get("z_plane")
    evidence = SceneEvidence.load(paths, z_plane=float(cal_plane if cal_plane is not None else 0.02))
    entries = []
    for path in candidate_paths:
        cand = load_candidate_any(path, paths.scene)
        settle_path = os.path.join(os.path.dirname(os.path.abspath(path)), "settle.json")
        settle = load_settle_dump(settle_path) if os.path.isfile(settle_path) else None
        models = build_models(dump, evidence.camera, candidate=cand, settle=settle, z_plane=z_plane)
        replay_dir = paths.replay_dir(None if cand.candidate_id == "stock" else cand.candidate_id, lead=lead)
        has_replay = replay_dir is not None and os.path.isdir(replay_dir)
        entries.append((cand, models, score_candidate(cand, models, evidence, settle=settle, replay_dir=str(replay_dir) if has_replay else None)))
    written = write_review(out_dir, paths, evidence, dump, entries)
    for path in written:
        log(f"[review] {path}")
    return written


def _selected(selectors, count, key, name) -> bool:
    return any(s == str(count) or s == key or s in str(name).lower() for s in selectors)


def diagnose_hull(
    paths: RefinePaths,
    *,
    dump_path,
    candidate_path=None,
    objects: list[str] | None = None,
    coacd_threshold: float = 0.03,
    out_path=None,
    apply_to=None,
    log=print,
) -> dict[str, Any]:
    """Old ``diagnose-hull``: COACD support-patch diagnosis per object; optionally adds 'collision' blocks."""
    from r2s.objects.refine.metrics import hull_support_diagnosis

    dump = load_object_pose_dump(dump_path)
    cand = load_candidate_any(candidate_path, paths.scene) if candidate_path else None
    camera = PinholeCamera.from_files(paths.intrinsics, paths.extrinsics)
    models = build_models(dump, camera, candidate=cand, z_plane=rest_plane(dump))
    report = {}
    for key, model in models.items():
        if objects and not _selected(objects, model.spec.count, key, model.spec.object_name):
            continue
        started = time.perf_counter()
        d = hull_support_diagnosis(model, coacd_threshold=coacd_threshold)
        d["safe_key"], d["object_name"], d["count"] = key, model.spec.object_name, int(model.spec.count)
        report[key] = d
        if d.get("available"):
            log(
                f"[diagnose-hull] {model.spec.object_name:22s} [{model.spec.count}] pieces {d['n_pieces']:3d} watertight={d['watertight']} "
                f"support {d['support_area_cm2']:5.1f} cm2 (visual bottom {d['visual_bottom_area_cm2']:5.1f}) COM margin {100 * d['com_margin_m']:+.2f} cm "
                f"contact@2mm {d['contact_fraction']:.2f} of section, half-section at {'-' if d['half_section_height_m'] is None else format(1000 * d['half_section_height_m'], '.0f') + ' mm'} "
                f"-> {'FLATTEN BOTTOM' if d['recommend_flatten_bottom'] else ''}{' CONVEX HULL' if d['recommend_convex_hull'] else ''}{' ok' if not (d['recommend_flatten_bottom'] or d['recommend_convex_hull']) else ''}  ({time.perf_counter() - started:.0f} s)"
            )
        else:
            log(f"[diagnose-hull] {model.spec.object_name}: {d.get('reason')}")
    out = out_path or os.path.join(os.path.dirname(os.path.abspath(str(candidate_path or dump_path))), "hull_diagnosis.json")
    with open(prepare_output(out), "w", encoding="utf-8") as handle:
        json.dump(_jsonable(report), handle, indent=2)
    log(f"[diagnose-hull] wrote {out}")
    if apply_to and cand is not None:
        n = 0
        for key, d in report.items():
            if d.get("recommend_convex_hull"):
                entry = cand.entry(key)
                if entry is not None:
                    entry["collision"] = {"convex_hull": True, "reason": f"COACD support patch margin {100 * d['com_margin_m']:+.1f} cm, {d['n_pieces']} pieces, not watertight"}
                    n += 1
        cand.save(apply_to)
        log(f"[diagnose-hull] wrote {apply_to} with {n} collision block(s)")
    return report


def flatten_bottom(
    paths: RefinePaths,
    *,
    dump_path,
    candidate_path,
    asset_dir,
    objects: list[str] | None = None,
    flatten_mm: float = 12.0,
    mode: str = "fill",
    apply_to=None,
    log=print,
) -> Candidate:
    """Old ``flatten-bottom``: flatten the hidden bottom of reconstructions, point the candidate at the copies."""
    from r2s.objects.refine.common import flatten_bottom_asset

    dump = load_object_pose_dump(dump_path)
    cand = load_candidate_any(candidate_path, paths.scene)
    n = 0
    for entry in dump["objects"]:
        key = entry["safe_key"]
        ce = cand.entry(key)
        if ce is None:
            continue
        if objects and not _selected(objects, entry["count"], key, entry.get("object_name", "")):
            continue
        quat = ce.get("quat_wxyz") or entry["quat_wxyz"]
        scale = float(entry["scale"]) * (float(ce.get("scale_multiplier", 1.0)) if ce.get("scale_apply", True) else 1.0)
        stem = os.path.splitext(os.path.basename(entry["asset_file"]))[0]
        out = os.path.abspath(os.path.join(str(asset_dir), f"{stem}_{mode}{int(round(flatten_mm))}mm.glb"))
        info = flatten_bottom_asset(entry["asset_file"], out, scale=scale, root_quat_wxyz=quat, flatten_m=flatten_mm / 1000.0, mode=mode)
        ce["asset_override"] = out
        ce["asset_override_info"] = info
        n += 1
        log(f"[flatten-bottom] {entry.get('object_name', key):22s} [{entry['count']}] moved {info['n_vertices_moved']}/{info['n_vertices']} vertices, flat patch {info['flat_patch_area_cm2']:.1f} cm2 -> {out}")
    target = apply_to or candidate_path
    cand.save(target)
    log(f"[flatten-bottom] wrote {target} with {n} asset override(s)")
    return cand


def replace_primitive(
    paths: RefinePaths,
    *,
    dump_path,
    candidate_path,
    asset_dir,
    objects: list[str],
    shape: str = "cup",
    height_mm: float | None = None,
    diameter_mm: float | None = None,
    bottom_diameter_mm: float | None = None,
    color=None,
    color_from_mask: bool = False,
    color_percentile: float = 75.0,
    apply_to=None,
    log=print,
) -> Candidate:
    """Old ``replace-primitive``: swap a reconstruction for a parametric cup sized from the real evidence."""
    from r2s.objects.pose_fit import load_first_real_image
    from r2s.objects.refine.common import load_phys_dims, load_real_masks, make_cup_asset

    dump = load_object_pose_dump(dump_path)
    cand = load_candidate_any(candidate_path, paths.scene)
    phys = load_phys_dims(paths.phys_props)
    masks = load_real_masks(paths.bbox_masks_dir)
    n = 0
    for entry in dump["objects"]:
        key = entry["safe_key"]
        ce = cand.entry(key)
        if ce is None or not _selected(objects, entry["count"], key, entry.get("object_name", "")):
            continue
        dims = (phys.get(int(entry["count"])) or {}).get("dims_m")
        height = height_mm / 1000.0 if height_mm else (max(dims) if dims else 0.10)
        diameter = diameter_mm / 1000.0 if diameter_mm else (sorted(dims)[1] if dims else 0.08)
        rgb = tuple(color) if color else (220, 220, 220)
        if color_from_mask and int(entry["count"]) in masks:
            import cv2

            img = load_first_real_image(paths.images_dir)
            if img is None:
                raise FileNotFoundError(f"no first frame under {paths.images_dir}")
            m = masks[int(entry["count"])][1]
            er = cv2.erode(m.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
            px = img[er] if er.any() else img[m]
            rgb = tuple(int(v) for v in np.percentile(px, color_percentile, axis=0))  # lit side, not the shadowed one
        stem = os.path.splitext(os.path.basename(entry["asset_file"]))[0]
        out = os.path.abspath(os.path.join(str(asset_dir), f"{stem}_{shape}_{int(round(1000 * height))}x{int(round(1000 * diameter))}mm.glb"))
        if shape == "cup":
            info = make_cup_asset(out, height_m=height, top_diameter_m=diameter, bottom_diameter_m=(bottom_diameter_mm / 1000.0 if bottom_diameter_mm else None), color_rgb=rgb, entry_scale=float(entry["scale"]))
        else:
            raise ValueError(f"unsupported shape {shape}")
        ce["asset_override"] = out
        ce["asset_override_info"] = {"kind": "primitive", "shape": shape, **info}
        # a symmetric primitive has no yaw and is upright at the identity world rotation
        ce["orientation_quat_wxyz"] = [1.0, 0.0, 0.0, 0.0]
        ce["quat_wxyz"] = [1.0, 0.0, 0.0, 0.0]
        ce["yaw_delta_deg"] = 0.0
        ce["scale_multiplier"] = 1.0
        n += 1
        log(f"[replace-primitive] {entry.get('object_name', key):22s} [{entry['count']}] -> {shape} {100 * height:.1f} cm tall, {100 * diameter:.1f} cm wide, colour {rgb} -> {out}")
    target = apply_to or candidate_path
    cand.save(target)
    log(f"[replace-primitive] wrote {target} with {n} replacement(s)")
    return cand
