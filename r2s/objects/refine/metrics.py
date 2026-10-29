"""Cues and scores for step 9 candidates (port of the old ``asset_refine_metrics.py``).

Every cue is a plain function of numpy data so the optimizer (``asset_refine_fit``)
and the harness (``asset_refine_harness score``) agree by construction.

Cues per object
---------------
silhouette   IoU / area ratio of the mesh silhouette vs the real first-frame mask
             (``object_pose_refine``).
depth        real metric depth inside the mask vs the rendered mesh depth, after a
             per-object *ring* calibration against the table plane.  The DROID
             depth is metric but its extrinsics put the table ~5 cm off, so only
             the local affine-corrected residual is trusted, and only when the
             ring fit is tight (``quality``).  Used for height/scale, never XY.
dims prior   physical_properties.json width/height/length (soft, sigma 0.3 in
             log space) or the task note's "AxBxC cm" (tighter, sigma 0.15).
grasp        the real gripper closed at ``tcp[onset]``: the grasped object's
             footprint must contain that point (``d_foot``); with a replay run,
             the sim object must be touched at that frame and move with the hand
             (``contact_onset``, ``comotion``), and nothing else may be hit
             before it (``false_contacts``).
stability    commanded vs settled pose from ``--object-settle-dump`` (or the
             warmup of any sim_stats.npz): xy drift, z drop, tilt, residual motion.
"""

from __future__ import annotations

import glob
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from r2s.objects.pose_fit import ObjectModel, PinholeCamera, mask_iou
from r2s.objects.refine.common import (
    Candidate,
    RefinePaths,
    TaskInfo,
    grasp_events,
)
from r2s.objects.settle import UNSTABLE_TILT_DEG, UNSTABLE_XY_DRIFT_M, stability_from_sim_stats

# thresholds
DEPTH_RING_DILATE_PX = 41
DEPTH_RING_EXCLUDE_PX = 9
DEPTH_RING_MIN_PIXELS = 500
DEPTH_RING_MAX_RMS_M = 0.015
DEPTH_RESIDUAL_SCALE_M = 0.02
GRASP_HIT_M = 0.03
GRASP_Z_GATE_M = (-0.10, 0.05)  # fingertip centre relative to the object's top: fingers wrap below the top
GRASP_DZ_TARGET_M = -0.02  # a top grasp closes ~2 cm below the object's top
GRASP_DZ_SIGMA_M = 0.04
GRASP_ANCHOR_WINDOW_FRAMES = 10  # frames before closing whose fingertip points all count as anchors
GRASP_CONTACT_WINDOW_FRAMES = 10
GRASP_COMOTION_FRAMES = 40
GRASP_LIFT_M = 0.02
DIMS_SIGMA_PHYS = 0.3
DIMS_SIGMA_NOTE = 0.15
DIMS_RESIDUAL_CAP = 4.0  # a wildly wrong prior (toy wheel 0.7 cm) must not dominate the fit
GRASP_TERM_CAP = 4.0


# ---------------------------------------------------------------------------
# Depth cue
# ---------------------------------------------------------------------------
@dataclass
class DepthCue:
    camera: PinholeCamera
    depth: np.ndarray  # (H,W) metric depth along the camera z axis
    z_plane: float
    ray_dirs_world: np.ndarray = field(init=False)  # (H,W,3) unit-free (z_cam=1) directions in world
    cam_center: np.ndarray = field(init=False)
    plane_depth: np.ndarray = field(init=False)  # (H,W) camera depth at which each ray hits z_plane

    def __post_init__(self) -> None:
        K = self.camera.K
        R = self.camera.T_w2c[:3, :3]
        t = self.camera.T_w2c[:3, 3]
        self.cam_center = -R.T @ t
        h, w = self.depth.shape
        us, vs = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
        dirs_cam = np.stack([(us - K[0, 2]) / K[0, 0], (vs - K[1, 2]) / K[1, 1], np.ones_like(us)], axis=-1)
        self.ray_dirs_world = dirs_cam @ R  # R.T @ d per pixel
        dz = self.ray_dirs_world[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            self.plane_depth = np.where(np.abs(dz) > 1e-9, (self.z_plane - self.cam_center[2]) / dz, np.nan)

    @classmethod
    def from_scene(cls, paths: RefinePaths, camera: PinholeCamera, z_plane: float, *, frame: int = 0) -> "DepthCue | None":
        path = paths.depth_path(frame)
        if not os.path.isfile(path):
            return None
        depth = np.load(path).astype(np.float64)
        if depth.shape != (camera.height, camera.width):
            return None
        return cls(camera=camera, depth=depth, z_plane=float(z_plane))

    def world_points(self, pixel_mask: np.ndarray, depth_values: np.ndarray) -> np.ndarray:
        """World XYZ of the masked pixels at the given (camera-z) depths."""
        d = self.ray_dirs_world[pixel_mask]
        return self.cam_center[None, :] + d * depth_values[:, None]


def _dilate(mask: np.ndarray, px: int) -> np.ndarray:
    import cv2

    if px <= 0:
        return mask.copy()
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask.astype(np.uint8), k).astype(bool)


def ring_calibration(
    mask: np.ndarray,
    other_masks: np.ndarray | None,
    cue: DepthCue,
    *,
    dilate_px: int = DEPTH_RING_DILATE_PX,
    exclude_px: int = DEPTH_RING_EXCLUDE_PX,
) -> dict[str, Any]:
    """Affine map measured depth -> true depth fitted on the table ring around the mask.

    Ring = dilate(mask, dilate_px) minus dilate(mask ∪ others, exclude_px); on it
    the true depth is the table plane's.  ``quality`` is True when the fit is
    tight enough (rms < 1.5 cm on ≥ 500 pixels) for the residual to mean anything.
    """
    ring = _dilate(mask, dilate_px)
    excl = _dilate(mask if other_masks is None else (mask | other_masks), exclude_px)
    ring &= ~excl
    meas = cue.depth[ring]
    true = cue.plane_depth[ring]
    ok = np.isfinite(meas) & np.isfinite(true) & (meas > 0.05) & (true > 0.05)
    meas, true = meas[ok], true[ok]
    n = int(ok.sum())
    if n < 20:
        return {"scale": 1.0, "offset": 0.0, "rms_m": float("nan"), "n": n, "quality": False}
    A = np.stack([meas, np.ones_like(meas)], axis=1)
    (s, b), *_ = np.linalg.lstsq(A, true, rcond=None)
    resid = true - (s * meas + b)
    # one robust re-fit without the worst 10 %
    keep = np.abs(resid) <= np.quantile(np.abs(resid), 0.9)
    if keep.sum() >= 20:
        (s, b), *_ = np.linalg.lstsq(A[keep], true[keep], rcond=None)
        resid = true - (s * meas + b)
    rms = float(np.sqrt(np.mean(resid**2)))
    return {
        "scale": float(s),
        "offset": float(b),
        "rms_m": rms,
        "n": n,
        "quality": bool(rms < DEPTH_RING_MAX_RMS_M and n >= DEPTH_RING_MIN_PIXELS),
    }


def render_depth_splat(model: ObjectModel, xy_delta, yaw_rad: float, *, fill_px: int = 1) -> np.ndarray:
    """Per-pixel nearest vertex depth (camera z) of the mesh at the candidate pose; inf where empty.

    A vertex z-buffer plus a small min-filter is enough for a median residual
    over a mask; no triangle rasterizer with depth interpolation is needed.
    """
    from scipy.ndimage import minimum_filter

    cam = model.camera
    verts = model.spec.world_vertices(model.pos_for(xy_delta), yaw_rad)
    uv, depth = cam.project(verts)
    faces = model.spec.faces
    if faces is not None and len(faces):
        # Coarse meshes (a box has 8 vertices) need samples on the faces: draw
        # barycentric points per triangle proportional to its projected area.
        tri_uv = uv[faces]
        area = 0.5 * np.abs((tri_uv[:, 1, 0] - tri_uv[:, 0, 0]) * (tri_uv[:, 2, 1] - tri_uv[:, 0, 1]) - (tri_uv[:, 2, 0] - tri_uv[:, 0, 0]) * (tri_uv[:, 1, 1] - tri_uv[:, 0, 1]))
        n_samples = np.minimum(np.ceil(area / 1.5).astype(np.int64), 4000)
        total = int(n_samples.sum())
        if total > 0:
            budget = 300_000
            if total > budget:
                n_samples = np.floor(n_samples * (budget / total)).astype(np.int64)
            rng = np.random.default_rng(0)
            face_idx = np.repeat(np.arange(len(faces)), n_samples)
            r1 = np.sqrt(rng.random(len(face_idx)))
            r2 = rng.random(len(face_idx))
            b0, b1, b2 = 1.0 - r1, r1 * (1.0 - r2), r1 * r2
            tri_xyz = verts[faces[face_idx]]
            pts = b0[:, None] * tri_xyz[:, 0] + b1[:, None] * tri_xyz[:, 1] + b2[:, None] * tri_xyz[:, 2]
            uv_s, depth_s = cam.project(pts)
            uv = np.concatenate([uv, uv_s], axis=0)
            depth = np.concatenate([depth, depth_s], axis=0)
    u = np.round(uv[:, 0]).astype(np.int64)
    v = np.round(uv[:, 1]).astype(np.int64)
    ok = (depth > 1e-6) & (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)
    zbuf = np.full((cam.height, cam.width), np.inf, dtype=np.float64)
    if ok.any():
        np.minimum.at(zbuf, (v[ok], u[ok]), depth[ok])
    if fill_px > 0:
        zbuf = minimum_filter(zbuf, size=2 * fill_px + 1, mode="nearest")
    return zbuf


def depth_residual(
    model: ObjectModel,
    xy_delta,
    yaw_rad: float,
    mask: np.ndarray,
    cue: DepthCue,
    calib: dict[str, Any],
    *,
    silhouette: np.ndarray | None = None,
) -> dict[str, Any]:
    """Front-surface residual (real − sim, m) and a depth-based height estimate for one object."""
    sil = model.silhouette(xy_delta, yaw_rad) if silhouette is None else silhouette
    both = mask & sil
    out: dict[str, Any] = {
        "quality": bool(calib.get("quality", False)),
        "ring_rms_m": calib.get("rms_m"),
        "overlap_px": int(both.sum()),
        "front_residual_m": None,
        "height_est_m": None,
        "height_sim_m": float(np.ptp(model.spec.world_vertices(model.pos_for(xy_delta), yaw_rad)[:, 2])),
    }
    if not calib.get("quality", False):
        return out
    corrected = calib["scale"] * cue.depth + calib["offset"]
    valid = mask & np.isfinite(cue.depth) & (cue.depth > 0.05)
    if valid.sum() >= 20:
        pts = cue.world_points(valid, corrected[valid])
        out["height_est_m"] = float(np.quantile(pts[:, 2], 0.9) - cue.z_plane)
    if both.sum() >= 20:
        zsim = render_depth_splat(model, xy_delta, yaw_rad)
        ok = both & np.isfinite(zsim) & np.isfinite(corrected)
        if ok.sum() >= 20:
            out["front_residual_m"] = float(np.median(corrected[ok] - zsim[ok]))
            out["front_residual_px"] = int(ok.sum())
    return out


# ---------------------------------------------------------------------------
# Dims prior
# ---------------------------------------------------------------------------
def model_extents(model: ObjectModel, yaw_rad: float = 0.0) -> list[float]:
    """Sorted world AABB extents (m) at the base pose (yaw only changes XY, so extents are yaw-invariant up to rotation)."""
    lo, hi = model.spec.world_aabb(model.spec.base_pos, yaw_rad)
    return sorted(float(v) for v in (hi - lo))


def dims_prior_residual(
    extents_m: list[float],
    *,
    phys_dims_m: list[float] | None = None,
    note_dims_cm: list[float] | None = None,
) -> dict[str, Any]:
    """Sum of squared log-ratios between the mesh extents and the prior, in sigma units.

    The task note (a human size judgment) outranks the Gemini-estimated
    physical_properties dims.  A single-number note ("14.4cm") is compared
    with the largest extent only.
    """
    ext = sorted(float(v) for v in extents_m)
    if note_dims_cm:
        prior = sorted(float(v) / 100.0 for v in note_dims_cm)
        sigma, source = DIMS_SIGMA_NOTE, "note"
    elif phys_dims_m:
        prior = sorted(float(v) for v in phys_dims_m)
        sigma, source = DIMS_SIGMA_PHYS, "phys"
    else:
        return {"residual": 0.0, "source": None, "log_ratio": None, "suggested_multiplier": None}
    if len(prior) == 1:
        ratios = [math.log(max(ext[-1], 1e-4) / max(prior[0], 1e-4))]
    else:
        ratios = [math.log(max(e, 1e-4) / max(p, 1e-4)) for e, p in zip(ext[-len(prior):], prior)]
    resid = float(sum((r / sigma) ** 2 for r in ratios) / len(ratios))
    mean_log = float(np.mean(ratios))
    return {
        "residual": resid,
        "source": source,
        "prior_m": prior,
        "log_ratio": mean_log,
        "suggested_multiplier": float(math.exp(-mean_log)),
    }


# ---------------------------------------------------------------------------
# Grasp consistency
# ---------------------------------------------------------------------------
def identify_grasp_targets(
    camera: PinholeCamera,
    grasp_points: np.ndarray,
    events: list[dict[str, int]],
    real_masks: dict[int, tuple[str, np.ndarray]],
    *,
    task_counts: list[int] | None = None,
    max_px: float = 60.0,
    anchor_window: int = GRASP_ANCHOR_WINDOW_FRAMES,
) -> list[dict[str, Any]]:
    """Which real object each grasp closed on, from the fingertip centre projected into the image.

    Candidate-independent: uses only the real trajectory (FK of the recorded
    joints) and the real masks (restricted to the task's objects when a task
    spec names them), so a bad fit cannot change which object we hold the fit
    to.  ``anchor_points`` are the fingertip positions over the last
    ``anchor_window`` frames before closing: the real hand often nudges the
    object a few cm on the way in, so the footprint only has to contain one of
    them.
    """
    from scipy.ndimage import distance_transform_edt

    counts = [c for c in (task_counts or list(real_masks)) if c in real_masks] or list(real_masks)
    dist_maps = {c: distance_transform_edt(~real_masks[c][1]) for c in counts}
    out = []
    for ev in events:
        k = ev["onset"]
        uv, depth = camera.project(grasp_points[k : k + 1])
        u, v = uv[0]
        k0 = max(k - int(anchor_window), 0)
        entry = {
            "onset": k,
            "release": ev["release"],
            "tcp_onset": [float(x) for x in grasp_points[k]],
            "anchor_points": [[float(x) for x in p] for p in grasp_points[k0 : k + 1]],
            "tcp_px": [float(u), float(v)],
            "count": None,
            "dist_px": None,
        }
        if depth[0] > 0 and 0 <= u < camera.width and 0 <= v < camera.height:
            ui, vi = int(round(u)), int(round(v))
            best = min(counts, key=lambda c: dist_maps[c][vi, ui])
            d = float(dist_maps[best][vi, ui])
            if d <= max_px:
                entry["count"] = int(best)
                entry["dist_px"] = d
                entry["object_name"] = real_masks[best][0]
        out.append(entry)
    return out


def _hull_distance_xy(points_xy: np.ndarray, p: np.ndarray) -> float:
    """Distance from ``p`` to the convex hull of ``points_xy`` (0 inside)."""
    import cv2

    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 1, 2)
    if len(pts) < 3:
        return float(np.min(np.linalg.norm(points_xy - p[None, :], axis=1))) if len(pts) else float("inf")
    hull = cv2.convexHull(pts)
    return float(max(-cv2.pointPolygonTest(hull, (float(p[0]), float(p[1])), True), 0.0))


def footprint_distance(model: ObjectModel, xy_delta, yaw_rad: float, point_xyz) -> dict[str, float | bool]:
    """XY distance from ``point`` to the object's footprint (convex hull; 0 inside) plus the z gate."""
    verts = model.spec.world_vertices(model.pos_for(xy_delta), yaw_rad)
    p = np.asarray(point_xyz, dtype=np.float64).reshape(3)
    d = _hull_distance_xy(verts[:, :2], p[:2])
    top = float(verts[:, 2].max())
    dz_top = float(p[2] - top)
    return {"d_foot_m": d, "dz_top_m": dz_top, "z_ok": GRASP_Z_GATE_M[0] <= dz_top <= GRASP_Z_GATE_M[1], "hit": d < GRASP_HIT_M}


def _anchor_key(fd: dict[str, Any]) -> float:
    """Closest approach in 3-D: footprint distance plus height above the grasp band."""
    dz = fd.get("dz_top_m")
    return (float(fd["d_foot_m"]) / GRASP_HIT_M) ** 2 + (0.0 if dz is None else ((float(dz) - GRASP_DZ_TARGET_M) / GRASP_DZ_SIGMA_M) ** 2)


def anchor_distance(model: ObjectModel, xy_delta, yaw_rad: float, target: dict[str, Any]) -> dict[str, float | bool]:
    """``footprint_distance`` at the anchor-window point of closest 3-D approach before closing."""
    points = target.get("anchor_points") or [target["tcp_onset"]]
    best = None
    for p in points:
        fd = footprint_distance(model, xy_delta, yaw_rad, p)
        if best is None or _anchor_key(fd) < _anchor_key(best):
            best = fd
    assert best is not None
    return best


def grasp_consistency_offline(
    models: dict[str, ObjectModel],
    poses: dict[str, tuple[np.ndarray, float]],
    targets: list[dict[str, Any]],
    key_by_count: dict[int, str],
) -> list[dict[str, Any]]:
    """Per grasp event: does the target's footprint contain the real TCP at closing?"""
    out = []
    for tgt in targets:
        entry = dict(tgt)
        c = tgt.get("count")
        key = key_by_count.get(c) if c is not None else None
        if key is None or key not in models:
            entry.update({"safe_key": None, "d_foot_m": None, "hit": False})
        else:
            xy, yaw = poses.get(key, (np.zeros(2), 0.0))
            entry["safe_key"] = key
            entry.update(anchor_distance(models[key], xy, yaw, tgt))
        # nearest object of any kind, for the "what would be grabbed instead" note
        best = None
        for k, m in models.items():
            xy, yaw = poses.get(k, (np.zeros(2), 0.0))
            fd = anchor_distance(m, xy, yaw, tgt)
            if best is None or fd["d_foot_m"] < best[1]:
                best = (k, fd["d_foot_m"])
        entry["nearest_safe_key"] = None if best is None else best[0]
        entry["nearest_d_foot_m"] = None if best is None else best[1]
        out.append(entry)
    return out


def _replay_sim_stats(replay_dir: str) -> str | None:
    hits = sorted(glob.glob(os.path.join(replay_dir, "*", "*_sim_stats.npz")))
    return hits[-1] if hits else None


def grasp_consistency_replay(replay_dir: str, targets: list[dict[str, Any]], tcp: np.ndarray, key_by_count: dict[int, str]) -> list[dict[str, Any]]:
    """Replay-side grasp terms from ``replay_frames.npz`` (frame→tick) and ``*_sim_stats.npz``."""
    frames_path = os.path.join(replay_dir, "replay_frames.npz")
    stats_path = _replay_sim_stats(replay_dir)
    if not (os.path.isfile(frames_path) and stats_path):
        return []
    rf = np.load(frames_path)
    ticks = np.asarray(rf["control_tick"]).astype(int)
    st = np.load(stats_path, allow_pickle=True)
    names = [str(n) for n in np.asarray(st["object_names"]).reshape(-1)]
    pos = np.asarray(st["object_pos"], dtype=np.float64)
    contact = np.asarray(st["robot_object_contact_count"]).astype(int)
    T = pos.shape[0]

    def tick(frame: int) -> int:
        frame = min(max(frame, 0), len(ticks) - 1)
        return int(min(ticks[frame], T - 1))

    out = []
    for tgt in targets:
        entry = {"onset": tgt["onset"], "count": tgt.get("count")}
        key = key_by_count.get(tgt.get("count"))
        if key is None or key not in names:
            entry.update({"contact_onset": None, "comotion": None, "lifted": None, "false_contacts": None})
            out.append(entry)
            continue
        i = names.index(key)
        k = int(tgt["onset"])
        t0, t1 = tick(k), tick(k + GRASP_CONTACT_WINDOW_FRAMES)
        entry["contact_onset"] = bool(contact[t0 : t1 + 1, i].max() > 0)
        k2 = min(k + GRASP_COMOTION_FRAMES, len(ticks) - 1, len(tcp) - 1)
        t2 = tick(k2)
        obj_disp = float(np.linalg.norm(pos[t2, i] - pos[t0, i]))
        tcp_disp = float(np.linalg.norm(tcp[k2] - tcp[k]))
        entry["object_disp_m"] = obj_disp
        entry["tcp_disp_m"] = tcp_disp
        entry["comotion"] = float(obj_disp / tcp_disp) if tcp_disp > 1e-3 else None
        entry["lifted"] = bool(pos[t2, i, 2] - pos[t0, i, 2] > GRASP_LIFT_M)
        others = [j for j in range(len(names)) if j != i]
        pre = contact[: t0 + 1]
        entry["false_contacts"] = int(sum(1 for j in others if pre[:, j].max() > 0)) if len(pre) else 0
        entry["false_contact_names"] = [names[j] for j in others if len(pre) and pre[:, j].max() > 0]
        out.append(entry)
    return out


def grasp_score(offline: list[dict[str, Any]], replay: list[dict[str, Any]] | None = None, *, unstable_keys: set[str] | None = None) -> dict[str, Any]:
    """``G`` per event and averaged: 1[hit] + 0.5 contact + 0.5 min(comotion/0.5, 1) − 0.5 false_contacts."""
    replay_by_onset = {r["onset"]: r for r in (replay or [])}
    per_event = []
    for ev in offline:
        g = 1.0 if ev.get("hit") and ev.get("z_ok", True) else 0.0
        r = replay_by_onset.get(ev["onset"])
        if r is not None:
            if r.get("contact_onset"):
                g += 0.5
            if r.get("comotion") is not None:
                g += 0.5 * min(r["comotion"] / 0.5, 1.0)
            if r.get("false_contacts"):
                g -= 0.5 * r["false_contacts"]
        if unstable_keys and ev.get("safe_key") in unstable_keys:
            g = min(g, 0.0)
        per_event.append(g)
    return {"G": float(np.mean(per_event)) if per_event else None, "per_event": per_event, "n_events": len(per_event)}


# ---------------------------------------------------------------------------
# Replay policy metrics + stability
# ---------------------------------------------------------------------------
def replay_policy_metrics(replay_dir: str, policies=("pi", "pi_local")) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for pol in policies:
        path = os.path.join(replay_dir, "compare", pol, "metrics.json")
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as handle:
            m = json.load(handle)
        phases = m.get("phases", {})
        out[pol] = {
            ph: {
                "ratio_pair_over_temporal": phases[ph].get("ratio_pair_over_temporal"),
                "d_pair": phases[ph].get("d_pair"),
                "temporal": phases[ph].get("temporal"),
            }
            for ph in ("all", "pre_grasp", "grasp")
            if ph in phases
        }
        out[pol]["close_onset"] = m.get("gripper_close_onset_frame")
    return out


def stability_from_settle(settle: dict[str, dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    if not settle:
        return {}
    return {
        k: {
            "xy_drift_m": v.get("xy_drift_m"),
            "z_drop_m": v.get("z_drop_m"),
            "tilt_deg": v.get("tilt_deg"),
            "settled": v.get("settled"),
            "unstable": v.get("unstable"),
            "reverted": v.get("reverted"),
        }
        for k, v in settle.items()
    }


def stability_from_replay(replay_dir: str, n_steps: int = 100) -> dict[str, dict[str, Any]]:
    path = _replay_sim_stats(replay_dir)
    if not path:
        return {}
    return stability_from_sim_stats(np.load(path, allow_pickle=True), n_steps=n_steps)


STABILITY_TERM_CAP = 16.0  # a fallen object (100 deg) is "very unstable"; beyond that degree does not matter


def stability_penalty(entry: dict[str, Any] | None) -> float:
    if not entry:
        return 0.0
    tilt = float(entry.get("tilt_deg") or 0.0)
    drift = float(entry.get("xy_drift_m") or 0.0)
    return min((tilt / UNSTABLE_TILT_DEG) ** 2 + (drift / UNSTABLE_XY_DRIFT_M) ** 2, STABILITY_TERM_CAP)


# ---------------------------------------------------------------------------
# Whole-candidate scoring
# ---------------------------------------------------------------------------
@dataclass
class SceneEvidence:
    """Everything candidate-independent, loaded once per scene."""

    paths: RefinePaths
    camera: PinholeCamera
    z_plane: float
    real_masks: dict[int, tuple[str, np.ndarray]]
    phys: dict[int, dict[str, Any]]
    task: TaskInfo
    tcp: np.ndarray | None
    grippers: np.ndarray | None
    depth: DepthCue | None
    events: list[dict[str, int]] = field(default_factory=list)
    targets: list[dict[str, Any]] = field(default_factory=list)
    ring_calib: dict[int, dict[str, Any]] = field(default_factory=dict)
    fk_flange_rms_m: float | None = None
    image: np.ndarray | None = None  # real frame 0 (RGB) for appearance checks

    @classmethod
    def load(cls, paths: RefinePaths, *, z_plane: float, height: int = 360, width: int = 640) -> "SceneEvidence":
        from r2s.objects.refine.common import load_phys_dims, load_real_masks, load_real_trajectory, load_task_info

        camera = PinholeCamera.from_files(paths.intrinsics, paths.extrinsics, height=height, width=width)
        real_masks = load_real_masks(paths.bbox_masks_dir)
        phys = load_phys_dims(paths.phys_props)
        task = load_task_info(paths.task_spec, scene=paths.scene, object_names={c: v["name"] for c, v in phys.items()})
        traj = load_real_trajectory(paths)
        tcp = traj.get("grasp_point")  # fingertip centre (FK of joints_old); flange only as a fallback
        grippers = traj.get("grippers")
        depth = DepthCue.from_scene(paths, camera, z_plane)
        ev = cls(paths=paths, camera=camera, z_plane=z_plane, real_masks=real_masks, phys=phys, task=task, tcp=tcp, grippers=grippers, depth=depth)
        ev.fk_flange_rms_m = traj.get("fk_flange_rms_m")
        from r2s.objects.pose_fit import load_first_real_image

        ev.image = load_first_real_image(paths.images_dir)
        if tcp is not None and grippers is not None and len(tcp) == len(grippers):
            ev.events = grasp_events(grippers)
            ev.targets = identify_grasp_targets(camera, tcp, ev.events, real_masks, task_counts=task.task_counts or None)
        if depth is not None:
            all_masks = {c: m for c, (_, m) in real_masks.items()}
            for c, m in all_masks.items():
                others = np.zeros_like(m)
                for c2, m2 in all_masks.items():
                    if c2 != c:
                        others |= m2
                ev.ring_calib[c] = ring_calibration(m, others, depth)
        return ev

    def target_counts(self) -> list[int]:
        return [t["count"] for t in self.targets if t.get("count") is not None]


def score_object(
    model: ObjectModel,
    xy_delta,
    yaw_rad: float,
    evidence: SceneEvidence,
    *,
    scale_multiplier: float = 1.0,
    settle_entry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """All per-object cues at one pose (used both by the optimizer and the harness)."""
    count = int(model.spec.count)
    real = evidence.real_masks.get(count)
    sil = model.silhouette(xy_delta, yaw_rad)
    out: dict[str, Any] = {"safe_key": model.spec.safe_key, "count": count, "object_name": model.spec.object_name, "scale_multiplier": float(scale_multiplier)}
    if real is not None and real[1].any() and sil.any():
        out["iou"] = mask_iou(sil, real[1])
        out["area_ratio"] = float(sil.sum() / max(int(real[1].sum()), 1))
    else:
        out["iou"] = None
        out["area_ratio"] = None
    ext = model_extents(model, yaw_rad)
    out["extents_m"] = ext
    phys = evidence.phys.get(count, {})
    out["dims"] = dims_prior_residual(ext, phys_dims_m=phys.get("dims_m"), note_dims_cm=evidence.task.dims_cm.get(count))
    if evidence.depth is not None and real is not None and count in evidence.ring_calib:
        out["depth"] = depth_residual(model, xy_delta, yaw_rad, real[1], evidence.depth, evidence.ring_calib[count], silhouette=sil)
    else:
        out["depth"] = None
    out["stability"] = None if settle_entry is None else {k: settle_entry.get(k) for k in ("xy_drift_m", "z_drop_m", "tilt_deg", "settled", "unstable", "reverted")}
    out["is_grasp_target"] = count in evidence.target_counts()
    return out


GRASP_DEAD_ZONE_M = GRASP_HIT_M  # no penalty while the closing point is within 3 cm of the footprint
GRASP_SOFT_SCALE_M = 0.05  # 1 unit of penalty per further 5 cm
GRASP_DZ_GATE_M = 0.05  # the height cue only means something when the hand is over the object


def grasp_penalty(anchor: dict[str, Any] | None) -> float:
    """Dead-zoned consistency term, not a pull.

    The real hand often nudges the object on the way in (droid2: the plush moved
    ~8 cm before the fingers closed), so the frame-0 mask and the closing point
    can both be right.  Inside 3 cm nothing is charged; beyond that the penalty
    grows as ((d − 3 cm) / 5 cm)² and is capped, so a gross error (17 cm) still
    costs but the fit is never dragged off the mask.  The fingertip-height cue
    ((dz_top − target) / 4 cm)² is added only when the hand is within 5 cm of the
    footprint.
    """
    if not anchor:
        return 0.0
    d = float(anchor.get("d_foot_m") or 0.0)
    dz = anchor.get("dz_top_m")
    pen = (max(d - GRASP_DEAD_ZONE_M, 0.0) / GRASP_SOFT_SCALE_M) ** 2
    if dz is not None and d < GRASP_DZ_GATE_M:
        pen += ((float(dz) - GRASP_DZ_TARGET_M) / GRASP_DZ_SIGMA_M) ** 2
    return min(pen, GRASP_TERM_CAP)


def objective_terms(obj: dict[str, Any], grasp_anchor: dict[str, Any] | float | None, weights: dict[str, float]) -> dict[str, float]:
    """The per-object objective the optimizer minimizes; summed over terms by the caller.

    ``grasp_anchor`` is the ``anchor_distance`` dict for grasp targets (None for
    the rest); a bare float is accepted as ``d_foot`` for backwards compatibility.
    """
    w = weights
    terms: dict[str, float] = {}
    if isinstance(grasp_anchor, (int, float)):
        grasp_anchor = {"d_foot_m": float(grasp_anchor), "dz_top_m": None}
    iou = obj.get("iou")
    terms["iou"] = w.get("iou", 1.0) * (1.0 - (iou if iou is not None else 0.0))
    ar = obj.get("area_ratio")
    terms["area"] = w.get("area", 0.3) * (math.log(ar) ** 2 if ar else 0.0)
    terms["dims"] = w.get("dims", 0.2) * min(float(obj["dims"]["residual"]), DIMS_RESIDUAL_CAP) if obj.get("dims") else 0.0
    dep = obj.get("depth") or {}
    r = dep.get("front_residual_m")
    terms["depth"] = w.get("depth", 0.1) * (min((r / DEPTH_RESIDUAL_SCALE_M) ** 2, 4.0) if (r is not None and dep.get("quality")) else 0.0)
    terms["grasp"] = w.get("grasp", 0.5) * grasp_penalty(grasp_anchor)
    stab = obj.get("stability")
    terms["stability"] = w.get("stability", 0.5) * stability_penalty(stab)
    return terms


# The mask (IoU + area) is the primary frame-0 evidence; dims/depth/grasp are priors and
# consistency checks; stability is charged from the settle dump of the previous round.
DEFAULT_WEIGHTS = {"iou": 1.0, "area": 1.0, "dims": 0.2, "depth": 0.1, "grasp": 0.25, "stability": 0.5}


def score_candidate(
    candidate: Candidate,
    models: dict[str, ObjectModel],
    evidence: SceneEvidence,
    *,
    settle: dict[str, dict[str, Any]] | None = None,
    replay_dir: str | None = None,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Offline cues for every object + grasp/stability/replay summaries for the candidate.

    ``models`` must already be at the candidate's (settled) poses, so every
    object is scored with ``xy_delta=0, yaw=0``.
    """
    weights = weights or DEFAULT_WEIGHTS
    key_by_count = {int(m.spec.count): k for k, m in models.items()}
    per_object = {}
    poses = {k: (np.zeros(2), 0.0) for k in models}
    offline = grasp_consistency_offline(models, poses, evidence.targets, key_by_count)
    anchor_by_key: dict[str, dict[str, Any]] = {}
    for o in offline:
        if o.get("safe_key") and (o["safe_key"] not in anchor_by_key or o["d_foot_m"] < anchor_by_key[o["safe_key"]]["d_foot_m"]):
            anchor_by_key[o["safe_key"]] = o
    total = 0.0
    for key, model in models.items():
        entry = candidate.entry(key) or {}
        obj = score_object(model, np.zeros(2), 0.0, evidence, scale_multiplier=float(entry.get("scale_multiplier", 1.0)), settle_entry=(settle or {}).get(key))
        obj["terms"] = objective_terms(obj, anchor_by_key.get(key), weights)
        obj["objective"] = float(sum(obj["terms"].values()))
        total += obj["objective"]
        per_object[key] = obj
    replay = grasp_consistency_replay(replay_dir, evidence.targets, evidence.tcp, key_by_count) if (replay_dir and evidence.tcp is not None) else []
    unstable = {k for k, v in (settle or {}).items() if v.get("unstable")}
    g = grasp_score(offline, replay, unstable_keys=unstable)
    ious = [o["iou"] for o in per_object.values() if o.get("iou") is not None]
    return {
        "candidate_id": candidate.candidate_id,
        "scene": candidate.scene,
        "objects": per_object,
        "grasp_offline": offline,
        "grasp_replay": replay,
        "grasp": g,
        "mean_iou": float(np.mean(ious)) if ious else None,
        "min_iou": float(np.min(ious)) if ious else None,
        "objective_total": float(total),
        "n_unstable": len(unstable),
        "stability": stability_from_settle(settle) if settle else (stability_from_replay(replay_dir) if replay_dir else {}),
        "policy": replay_policy_metrics(replay_dir) if replay_dir else {},
        "weights": weights,
    }


def summarize_scores(scores: list[dict[str, Any]]) -> str:
    """Markdown table across candidates (one scene)."""
    rows = ["| candidate | G | hit | mean IoU | min IoU | objective | unstable | pi_local grasp pair/temporal | pi grasp pair/temporal |", "|---|---|---|---|---|---|---|---|---|"]

    def fmt(value, spec: str) -> str:
        return "-" if value is None else format(value, spec)

    for s in scores:
        g = s["grasp"]
        hits = sum(1 for o in s["grasp_offline"] if o.get("hit"))
        pol = s.get("policy", {})

        def ratio(p):
            try:
                return format(pol[p]["grasp"]["ratio_pair_over_temporal"], ".2f")
            except (KeyError, TypeError):
                return "-"

        cells = [
            s["candidate_id"],
            fmt(g["G"], ".2f"),
            f"{hits}/{g['n_events']}",
            fmt(s["mean_iou"], ".3f"),
            fmt(s["min_iou"], ".3f"),
            fmt(s["objective_total"], ".2f"),
            str(s["n_unstable"]),
            ratio("pi_local"),
            ratio("pi"),
        ]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Collision-hull support diagnosis (offline COACD, ~1 min per object)
# ---------------------------------------------------------------------------
def hull_support_diagnosis(
    model: ObjectModel,
    *,
    coacd_threshold: float = 0.03,
    slab_m: float = 0.004,
    min_margin_m: float = 0.01,
) -> dict[str, Any]:
    """Does the decomposed collision hull give the object a support patch under its centre of mass?

    Runs COACD on the visual mesh with the evaluator's threshold, takes the
    pieces whose lowest vertices sit within ``slab_m`` of the overall bottom at
    the model's current orientation, and measures how far the centre of mass
    (vertex centroid) projects inside the convex hull of that support patch.
    A negative or small margin means the object rests on a few nubs on one
    side and will topple however carefully it is placed; a single convex hull
    (``collision: {"convex_hull": true}``) then gives it the flat bottom the
    real object has.
    """
    import cv2

    try:
        import coacd
    except ImportError:
        return {"available": False, "reason": "coacd not importable"}
    from r2s.collision.support import DEFAULT_GLB_FRONT, DEFAULT_GLB_UP, align_up_and_front, load_combined_mesh, transform_vertices

    spec = model.spec
    mesh = load_combined_mesh(spec.asset_file)
    import_rot = align_up_and_front(np.asarray(DEFAULT_GLB_UP), np.asarray(DEFAULT_GLB_FRONT))
    verts_all = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    coacd.set_log_level("error") if hasattr(coacd, "set_log_level") else None
    parts = coacd.run_coacd(coacd.Mesh(verts_all, faces), threshold=float(coacd_threshold))

    def to_world(v):
        return transform_vertices(np.asarray(v, dtype=np.float64), scale=spec.scale, root_pos=spec.base_pos, root_quat_wxyz=spec.base_quat_wxyz, mesh_import_rot=import_rot)

    world_all = to_world(verts_all)
    min_z = float(world_all[:, 2].min())
    com = world_all.mean(axis=0)
    support = []
    piece_rows = []
    for i, (v, _f) in enumerate(parts):
        w = to_world(v)
        mz = float(w[:, 2].min())
        piece_rows.append({"piece": i, "min_z_m": mz, "above_bottom_m": mz - min_z})
        if mz - min_z <= slab_m:
            low = w[w[:, 2] <= mz + slab_m]
            support.append(low[:, :2])
    if support:
        pts = np.concatenate(support, axis=0).astype(np.float32).reshape(-1, 1, 2)
        hull = cv2.convexHull(pts)
        margin = float(cv2.pointPolygonTest(hull, (float(com[0]), float(com[1])), True))
        area = float(cv2.contourArea(hull))
    else:
        margin, area = float("-inf"), 0.0
    visual_low = world_all[world_all[:, 2] <= min_z + 2 * slab_m][:, :2].astype(np.float32).reshape(-1, 1, 2)
    visual_area = float(cv2.contourArea(cv2.convexHull(visual_low))) if len(visual_low) >= 3 else 0.0
    watertight = bool(getattr(mesh, "is_watertight", False))
    unstable = margin < min_margin_m
    dome = bottom_dome_profile(world_all)
    return {
        "available": True,
        "watertight": watertight,
        **dome,
        "n_pieces": len(parts),
        "n_support_pieces": len(support),
        "support_area_cm2": 1e4 * area,
        "visual_bottom_area_cm2": 1e4 * visual_area,
        "com_margin_m": margin,
        "com_height_m": float(com[2] - min_z),
        "unstable_on_hull": bool(unstable),
        "recommend_convex_hull": bool(unstable and not watertight),
        "recommend_flatten_bottom": bool(dome["domed_bottom"]),
        "lowest_pieces": sorted(piece_rows, key=lambda r: r["min_z_m"])[:5],
    }


DOME_CONTACT_SLAB_M = 0.002
DOME_FILL_FRACTION = 0.5
DOME_MAX_HEIGHT_M = 0.005
DOME_MIN_CONTACT_FRACTION = 0.3


def bottom_dome_profile(world_vertices: np.ndarray) -> dict[str, Any]:
    """Is the bottom rounded?  Contact patch within 2 mm vs the widest horizontal section,
    and the height at which the section reaches half of it.

    A cup reconstructed from one photo has an unseen, domed underside: its
    2 mm contact patch is a few cm² (10-25 % of the section) and the section
    only fills in 1-3 cm up, so under gravity it touches at a point and rocks
    over from any orientation.  ``domed_bottom`` flags contact fraction below
    0.3 or fill height above 5 mm.
    """
    import cv2

    v = np.asarray(world_vertices, dtype=np.float64)
    z0 = float(v[:, 2].min())
    height = float(v[:, 2].max() - z0)

    def area_below(h: float) -> float:
        sl = v[v[:, 2] <= z0 + h][:, :2].astype(np.float32).reshape(-1, 1, 2)
        return float(cv2.contourArea(cv2.convexHull(sl))) if len(sl) >= 3 else 0.0

    full = max(area_below(height), 1e-9)
    contact = area_below(DOME_CONTACT_SLAB_M)
    fill_h = None
    for h in np.arange(0.001, height + 1e-9, 0.001):
        if area_below(h) >= DOME_FILL_FRACTION * full:
            fill_h = float(h)
            break
    frac = contact / full
    domed = frac < DOME_MIN_CONTACT_FRACTION and (fill_h is None or fill_h > DOME_MAX_HEIGHT_M)
    return {
        "contact_area_2mm_cm2": 1e4 * contact,
        "max_section_area_cm2": 1e4 * full,
        "contact_fraction": float(frac),
        "half_section_height_m": fill_h,
        "domed_bottom": bool(domed),
    }
