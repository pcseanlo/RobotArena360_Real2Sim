"""Settle probe: commanded vs settled object poses and the stability metrics built on them.

Port of the old ``object_settle_probe.py``. Everything here is plain numpy except
:func:`probe_settle`, which steps an already-built Genesis scene; the sim side
calls it with the scene's object entities and a one-control-step callable (see
its docstring for the exact procedure the old evaluator ran).

Production settings used by step 9 (``asset_refine_batch.sh``):

* stock-settle:     ``mode="free"``,    ``steps=150`` (hold 0), ``sample_every=5``
* candidate-settle: ``mode="upright"``, ``steps=150``, ``hold_steps=20``, ``sample_every=5``

An object that slides, sinks or tips over after the physics starts is not at the
pose the fit chose; the real objects in frame 0 are at rest, so the *settled* sim
pose is what the fit has to match. The metrics feed
``r2s.objects.refine.metrics.stability_penalty`` and the settle-then-refit step.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np

from r2s.common.io import prepare_output

DEFAULT_SETTLE_STEPS = 50
DEFAULT_SAMPLE_EVERY = 5
DEFAULT_HOLD_STEPS = 20
SETTLE_MODES = ("free", "gentle", "upright")
RESIDUAL_WINDOW_STEPS = 10
SETTLED_RESIDUAL_M = 0.001
UNSTABLE_XY_DRIFT_M = 0.01
UNSTABLE_TILT_DEG = 5.0

# step 9 production settle settings (asset_refine_batch.sh)
STOCK_SETTLE_MODE = "free"
CANDIDATE_SETTLE_MODE = "upright"
REFINE_SETTLE_STEPS = 150


def write_settle_video(path: str, frames: list, fps: int = 15) -> str | None:
    """mp4 (libx264, 15 fps) of the frames collected by :func:`probe_settle`."""
    if not frames:
        return None
    import imageio.v2 as imageio

    path = str(prepare_output(path))
    first = np.asarray(frames[0])
    h, w = first.shape[0] - first.shape[0] % 2, first.shape[1] - first.shape[1] % 2  # libx264 needs even sizes
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(np.ascontiguousarray(np.asarray(frame)[:h, :w]))
    return path


# ---------------------------------------------------------------------------
# Quaternion helpers (wxyz, numpy only)
# ---------------------------------------------------------------------------
def _quat_normalize(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return q / max(float(np.linalg.norm(q)), 1e-12)


def quat_to_matrix(q) -> np.ndarray:
    w, x, y, z = _quat_normalize(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _quat_mul(a, b) -> np.ndarray:
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


def relative_rotation_matrix(q_from, q_to) -> np.ndarray:
    """World-frame rotation taking orientation ``q_from`` to ``q_to`` (R_to · R_fromᵀ).

    Independent of which body frame the quaternion describes (mesh frame or
    Genesis's inertially aligned solver frame), so the same numbers come out
    of the settle probe and of a sim_stats.npz.
    """
    qf = _quat_normalize(q_from)
    conj = np.array([qf[0], -qf[1], -qf[2], -qf[3]])
    return quat_to_matrix(_quat_mul(_quat_normalize(q_to), conj))


def tilt_deg(q_from, q_to) -> float:
    """How far (deg) the object rotated about a horizontal axis between the two orientations."""
    R = relative_rotation_matrix(q_from, q_to)
    return math.degrees(math.acos(float(np.clip(R[2, 2], -1.0, 1.0))))


def yaw_drift_deg(q_from, q_to) -> float:
    """Signed rotation (deg) about world +z between the two orientations."""
    R = relative_rotation_matrix(q_from, q_to)
    return math.degrees(math.atan2(R[1, 0], R[0, 0]))


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------
def _to_numpy(value, n: int) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64).reshape(-1, n)[0]


def read_entity_pose(entity) -> tuple[np.ndarray, np.ndarray] | None:
    get_pos = getattr(entity, "get_pos", None)
    get_quat = getattr(entity, "get_quat", None)
    if get_pos is None or get_quat is None:
        return None
    try:
        pos = get_pos(relative=True)
        quat = get_quat(relative=True)
    except TypeError:
        pos = get_pos()
        quat = get_quat()
    return _to_numpy(pos, 3), _to_numpy(quat, 4)


@dataclass
class ObjectSettle:
    safe_key: str
    commanded_pos: np.ndarray
    commanded_quat_wxyz: np.ndarray
    settled_pos: np.ndarray
    settled_quat_wxyz: np.ndarray
    samples: list[tuple[int, np.ndarray, np.ndarray]] = field(default_factory=list)
    object_name: str = ""
    reverted: bool = False
    mode: str = "free"
    hold_steps: int = 0
    released_pos: np.ndarray | None = None  # pose at the end of the hold (upright/gentle modes)
    released_quat_wxyz: np.ndarray | None = None

    @property
    def tilt_after_release(self) -> float | None:
        """Tilt (deg) accumulated after the hold ended: the object's own doing, not the drop's."""
        if self.released_quat_wxyz is None:
            return None
        return tilt_deg(self.released_quat_wxyz, self.settled_quat_wxyz)

    @property
    def topples_after_release(self) -> bool | None:
        t = self.tilt_after_release
        return None if t is None else t > UNSTABLE_TILT_DEG

    @property
    def xy_drift(self) -> float:
        return float(np.linalg.norm((self.settled_pos - self.commanded_pos)[:2]))

    @property
    def z_drop(self) -> float:
        return float(self.settled_pos[2] - self.commanded_pos[2])

    @property
    def tilt(self) -> float:
        return tilt_deg(self.commanded_quat_wxyz, self.settled_quat_wxyz)

    @property
    def yaw_drift(self) -> float:
        return yaw_drift_deg(self.commanded_quat_wxyz, self.settled_quat_wxyz)

    @property
    def residual_motion(self) -> float | None:
        """Position change over the last ``RESIDUAL_WINDOW_STEPS`` steps (None if unsampled)."""
        if len(self.samples) < 2:
            return None
        last_step = self.samples[-1][0]
        earlier = [s for s in self.samples if s[0] <= last_step - RESIDUAL_WINDOW_STEPS]
        ref = earlier[-1] if earlier else self.samples[0]
        return float(np.linalg.norm(self.samples[-1][1] - ref[1]))

    @property
    def settled(self) -> bool:
        r = self.residual_motion
        return r is not None and r <= SETTLED_RESIDUAL_M

    @property
    def unstable(self) -> bool:
        return self.xy_drift > UNSTABLE_XY_DRIFT_M or self.tilt > UNSTABLE_TILT_DEG or not self.settled

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "safe_key": self.safe_key,
            "object_name": self.object_name,
            "commanded_pos": [float(v) for v in self.commanded_pos],
            "commanded_quat_wxyz": [float(v) for v in self.commanded_quat_wxyz],
            "settled_pos": [float(v) for v in self.settled_pos],
            "settled_quat_wxyz": [float(v) for v in self.settled_quat_wxyz],
            "xy_drift_m": self.xy_drift,
            "z_drop_m": self.z_drop,
            "tilt_deg": self.tilt,
            "yaw_drift_deg": self.yaw_drift,
            "residual_motion_m": self.residual_motion,
            "settled": bool(self.settled),
            "unstable": bool(self.unstable),
            "reverted": bool(self.reverted),
            "mode": self.mode,
            "hold_steps": int(self.hold_steps),
            "released_pos": None if self.released_pos is None else [float(v) for v in self.released_pos],
            "released_quat_wxyz": None if self.released_quat_wxyz is None else [float(v) for v in self.released_quat_wxyz],
            "tilt_after_release_deg": self.tilt_after_release,
            "topples_after_release": self.topples_after_release,
            "samples": [
                {"step": int(step), "pos": [float(v) for v in pos], "quat_wxyz": [float(v) for v in quat]}
                for step, pos, quat in self.samples
            ],
        }


@dataclass
class SettleReport:
    steps: int
    sample_every: int
    objects: list[ObjectSettle]
    missing: list[str] = field(default_factory=list)

    def by_key(self) -> dict[str, ObjectSettle]:
        return {o.safe_key: o for o in self.objects}

    def log_lines(self) -> list[str]:
        lines = []
        for o in self.objects:
            flag = " UNSTABLE" if o.unstable else ""
            r = o.residual_motion
            after = "" if o.tilt_after_release is None else f"  after-release tilt {o.tilt_after_release:5.1f} deg{' TOPPLES' if o.topples_after_release else ''}"
            lines.append(
                f"{o.safe_key:24s} {o.object_name:20s} xy {100 * o.xy_drift:5.2f} cm  z {100 * o.z_drop:+5.2f} cm  "
                f"tilt {o.tilt:5.1f} deg  yaw {o.yaw_drift:+6.1f} deg  "
                f"residual {'n/a' if r is None else f'{1000 * r:.2f} mm'} in {self.steps} steps ({o.mode}){after}{flag}"
            )
        if self.missing:
            lines.append(f"WARNING: entities not found in the scene: {self.missing}")
        return lines


def _zero_velocity(entity) -> None:
    fn = getattr(entity, "zero_all_dofs_velocity", None)
    if fn is not None:
        fn()


def _hold_pose(entity, pos_xy: np.ndarray, quat) -> None:
    """Re-apply the commanded XY and orientation, keeping whatever height physics reached."""
    pose = read_entity_pose(entity)
    z = pose[0][2] if pose is not None else 0.0
    entity.set_pos((float(pos_xy[0]), float(pos_xy[1]), float(z)), relative=True)
    entity.set_quat(tuple(float(v) for v in quat), relative=True)
    _zero_velocity(entity)


def probe_settle(
    entities_by_key: Mapping[str, Any],
    step_fn: Callable[[], None],
    safe_keys: list[str],
    *,
    steps: int = DEFAULT_SETTLE_STEPS,
    sample_every: int = DEFAULT_SAMPLE_EVERY,
    object_names: dict[str, str] | None = None,
    reverted_keys: set[str] | None = None,
    mode: str = "free",
    hold_steps: int = DEFAULT_HOLD_STEPS,
    frame_callback=None,
    frames_out: list | None = None,
) -> SettleReport:
    """Hold the reset action for ``steps`` control steps and record every object's pose.

    ``entities_by_key`` maps ``safe_key`` to an already-built Genesis rigid entity
    (``get_pos/get_quat(relative=True)``, ``set_pos/set_quat(..., relative=True)``,
    ``zero_all_dofs_velocity()``); ``step_fn()`` advances the simulation by one
    control step while holding the reset action (the old evaluator's
    ``env.step(None)``: 50 Hz, robot at its reset pose).

    Procedure (unchanged from the old evaluator):

    1. commanded pose = what each entity reports *before* the first step (the
       load-time pose, or the pose override's result);
    2. for step = 1..``steps``: ``step_fn()``; during the first ``hold_steps``
       steps (``gentle``/``upright`` only) zero every object's velocity, and in
       ``upright`` mode first re-apply the commanded XY and orientation keeping
       the height physics reached; at ``step == hold_steps`` record the
       *released* pose;
    3. sample every object's pose every ``sample_every`` steps and at the last
       step; the last sample is the settled pose.

    ``frame_callback``/``frames_out`` (optional) collect one camera frame before
    the first step and after every step, for :func:`write_settle_video`.

    ``mode`` (see ``--object-settle-mode``): ``gentle`` zeroes every object's
    velocity after each of the first ``hold_steps`` steps so the drop is
    quasi-static; ``upright`` additionally re-applies the commanded orientation
    and XY during the hold, so only the height settles, then releases.  The pose
    at release is recorded so ``tilt_after_release`` separates "fell because it
    was dropped from 2 cm" from "cannot stand on its collision hull".
    """
    if mode not in SETTLE_MODES:
        raise ValueError(f"unknown settle mode {mode!r}; expected one of {SETTLE_MODES}")
    hold = max(0, int(hold_steps)) if mode != "free" else 0
    entities = {}
    missing = []
    for key in safe_keys:
        entity = entities_by_key.get(key) if hasattr(entities_by_key, "get") else None
        if entity is None:
            missing.append(key)
            continue
        entities[key] = entity
    commanded = {}
    for key, entity in entities.items():
        pose = read_entity_pose(entity)
        if pose is not None:
            commanded[key] = pose
    samples: dict[str, list] = {key: [] for key in commanded}
    released: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    sample_every = max(1, int(sample_every))
    if frame_callback is not None and frames_out is not None:
        frames_out.append(frame_callback())  # the commanded scene, before any physics
    for step in range(1, int(steps) + 1):
        step_fn()
        if frame_callback is not None and frames_out is not None:
            frames_out.append(frame_callback())
        if step <= hold:
            for key, entity in entities.items():
                if key not in commanded:
                    continue
                if mode == "upright":
                    _hold_pose(entity, commanded[key][0][:2], commanded[key][1])
                else:
                    _zero_velocity(entity)
            if step == hold:
                for key, entity in entities.items():
                    if key in commanded:
                        pose = read_entity_pose(entity)
                        if pose is not None:
                            released[key] = pose
        if step % sample_every == 0 or step == int(steps):
            for key, entity in entities.items():
                if key not in commanded:
                    continue
                pose = read_entity_pose(entity)
                if pose is not None:
                    samples[key].append((step, pose[0], pose[1]))
    objects = []
    for key, (c_pos, c_quat) in commanded.items():
        if samples[key]:
            s_pos, s_quat = samples[key][-1][1], samples[key][-1][2]
        else:
            s_pos, s_quat = c_pos, c_quat
        rel = released.get(key)
        objects.append(
            ObjectSettle(
                safe_key=key,
                object_name=(object_names or {}).get(key, ""),
                commanded_pos=np.asarray(c_pos, dtype=np.float64),
                commanded_quat_wxyz=np.asarray(c_quat, dtype=np.float64),
                settled_pos=np.asarray(s_pos, dtype=np.float64),
                settled_quat_wxyz=np.asarray(s_quat, dtype=np.float64),
                samples=samples[key],
                reverted=key in (reverted_keys or set()),
                mode=mode,
                hold_steps=hold,
                released_pos=None if rel is None else np.asarray(rel[0], dtype=np.float64),
                released_quat_wxyz=None if rel is None else np.asarray(rel[1], dtype=np.float64),
            )
        )
    return SettleReport(steps=int(steps), sample_every=sample_every, objects=objects, missing=missing)


def probe_settle_env(env, safe_keys: list[str], **kwargs) -> SettleReport:
    """:func:`probe_settle` on an old-evaluator-style env (``env.entities``, ``env.step(None)``)."""
    return probe_settle(env.entities, lambda: env.step(None), safe_keys, **kwargs)


def write_settle_dump(path, report: SettleReport, *, scene_name: str | None = None, source: str = "") -> str:
    payload = {
        "version": 1,
        "scene": scene_name,
        "source": source,
        "steps": int(report.steps),
        "sample_every": int(report.sample_every),
        "objects": [o.to_json_dict() for o in report.objects],
        "missing": list(report.missing),
    }
    out = prepare_output(path)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return str(out)


def load_settle_dump(path) -> dict[str, dict[str, Any]]:
    """``{safe_key: entry}`` from a settle dump."""
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    return {str(o["safe_key"]): o for o in payload.get("objects", [])}


# ---------------------------------------------------------------------------
# Same metrics from an existing rollout (sim_stats.npz)
# ---------------------------------------------------------------------------
def stability_from_sim_stats(npz, n_steps: int = 100, *, reference_step: int = 0) -> dict[str, dict[str, Any]]:
    """Per-object drift/tilt over the first ``n_steps`` of a recorded rollout.

    Works on any ``*_sim_stats.npz`` (``object_names``, ``object_pos``,
    ``object_quat_wxyz`` per control step), so stock/posefit runs get a
    stability column without a new GPU run.  Only the warmup is meaningful:
    once the robot touches an object, motion is no longer settling.
    """
    names = [str(n) for n in np.asarray(npz["object_names"]).reshape(-1)]
    pos = np.asarray(npz["object_pos"], dtype=np.float64)
    quat = np.asarray(npz["object_quat_wxyz"], dtype=np.float64)
    last = min(int(n_steps), pos.shape[0] - 1)
    ref = min(int(reference_step), last)
    win = max(ref, last - RESIDUAL_WINDOW_STEPS)
    out: dict[str, dict[str, Any]] = {}
    for i, name in enumerate(names):
        d = pos[last, i] - pos[ref, i]
        residual = float(np.linalg.norm(pos[last, i] - pos[win, i]))
        entry = {
            "object_name": name,
            "xy_drift_m": float(np.linalg.norm(d[:2])),
            "z_drop_m": float(d[2]),
            "tilt_deg": tilt_deg(quat[ref, i], quat[last, i]),
            "yaw_drift_deg": yaw_drift_deg(quat[ref, i], quat[last, i]),
            "residual_motion_m": residual,
            "settled": residual <= SETTLED_RESIDUAL_M,
            "steps": last - ref,
        }
        entry["unstable"] = (
            entry["xy_drift_m"] > UNSTABLE_XY_DRIFT_M or entry["tilt_deg"] > UNSTABLE_TILT_DEG or not entry["settled"]
        )
        out[name] = entry
    return out
