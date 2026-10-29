"""Ported from the old test_object_settle_probe.py (r2s.objects.settle; fake env, no simulator)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from r2s.objects import settle as osp  # noqa: E402


def _quat_axis_angle(axis, deg):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    h = math.radians(deg) / 2
    return np.array([math.cos(h), *(math.sin(h) * axis)])



def test_tilt_and_yaw_drift():
    ident = np.array([1.0, 0.0, 0.0, 0.0])
    assert osp.tilt_deg(ident, ident) == pytest.approx(0.0)
    tilted = _quat_axis_angle([1, 0, 0], 18.0)
    assert osp.tilt_deg(ident, tilted) == pytest.approx(18.0, abs=1e-6)
    # yaw alone does not tilt the up axis
    yawed = _quat_axis_angle([0, 0, 1], 40.0)
    assert osp.tilt_deg(ident, yawed) == pytest.approx(0.0, abs=1e-9)
    assert osp.yaw_drift_deg(ident, yawed) == pytest.approx(40.0, abs=1e-6)
    assert osp.yaw_drift_deg(yawed, ident) == pytest.approx(-40.0, abs=1e-6)
    # wraps to (-180, 180]
    assert abs(osp.yaw_drift_deg(ident, _quat_axis_angle([0, 0, 1], 350.0)) - (-10.0)) < 1e-6


class _FakeEntity:
    """Rigid body that drifts linearly for ``moving_steps`` steps, then stops."""

    def __init__(self, pos, quat, *, drift_per_step=(0.0, 0.0, 0.0), tilt_per_step=0.0, moving_steps=10):
        self.pos = np.asarray(pos, dtype=np.float64)
        self.quat = np.asarray(quat, dtype=np.float64)
        self.drift = np.asarray(drift_per_step, dtype=np.float64)
        self.tilt = float(tilt_per_step)
        self.moving_steps = int(moving_steps)
        self.step_count = 0
        self.relative_calls = []

    def get_pos(self, relative=False):
        self.relative_calls.append(relative)
        return self.pos.copy()

    def get_quat(self, relative=False):
        return self.quat.copy()

    def advance(self):
        self.step_count += 1
        if self.step_count <= self.moving_steps:
            self.pos = self.pos + self.drift
            if self.tilt:
                q = _quat_axis_angle([1, 0, 0], self.tilt)
                w1, x1, y1, z1 = q
                w2, x2, y2, z2 = self.quat
                self.quat = np.array(
                    [
                        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                    ]
                )


class _FakeEnv:
    def __init__(self, entities):
        self.entities = entities
        self.steps = 0

    def step(self, action):
        assert action is None
        self.steps += 1
        for e in self.entities.values():
            e.advance()


def test_probe_metrics_and_flags(tmp_path):
    stable = _FakeEntity([0.5, 0.0, 0.05], [1, 0, 0, 0], drift_per_step=(0, 0, -0.001), moving_steps=5)
    slider = _FakeEntity([0.6, 0.1, 0.05], [1, 0, 0, 0], drift_per_step=(0.004, 0.0, 0.0), moving_steps=10)
    tipper = _FakeEntity([0.4, -0.1, 0.05], [1, 0, 0, 0], tilt_per_step=2.0, moving_steps=10)
    mover = _FakeEntity([0.3, 0.2, 0.05], [1, 0, 0, 0], drift_per_step=(0.0, 0.0005, 0.0), moving_steps=10_000)
    env = _FakeEnv({"obj_stable": stable, "obj_slider": slider, "obj_tipper": tipper, "obj_mover": mover})
    report = osp.probe_settle_env(
        env,
        ["obj_stable", "obj_slider", "obj_tipper", "obj_mover", "obj_missing"],
        steps=50,
        sample_every=5,
        object_names={"obj_slider": "cup"},
        reverted_keys={"obj_tipper"},
    )
    assert env.steps == 50
    assert report.missing == ["obj_missing"]
    assert all(stable.relative_calls), "poses must be read in the mesh (relative) frame"
    by = report.by_key()
    s = by["obj_stable"]
    assert s.z_drop == pytest.approx(-0.005)
    assert s.xy_drift == pytest.approx(0.0)
    assert s.settled and not s.unstable
    sl = by["obj_slider"]
    assert sl.object_name == "cup"
    assert sl.xy_drift == pytest.approx(0.04)
    assert sl.settled and sl.unstable  # came to rest, but 4 cm from where it was commanded
    t = by["obj_tipper"]
    assert t.tilt == pytest.approx(20.0, abs=1e-6)
    assert t.unstable and t.reverted
    m = by["obj_mover"]
    assert not m.settled and m.unstable
    assert m.residual_motion == pytest.approx(0.005, abs=1e-9)  # 10 steps x 0.5 mm
    assert len(s.samples) == 10 and s.samples[-1][0] == 50
    lines = report.log_lines()
    assert any("UNSTABLE" in ln for ln in lines) and any("obj_missing" in ln for ln in lines)

    path = osp.write_settle_dump(str(tmp_path / "settle.json"), report, scene_name="droidX", source="cand.json")
    payload = json.loads(open(path).read())
    assert payload["scene"] == "droidX" and payload["steps"] == 50
    loaded = osp.load_settle_dump(path)
    assert set(loaded) == {"obj_stable", "obj_slider", "obj_tipper", "obj_mover"}
    assert loaded["obj_tipper"]["reverted"] is True
    assert loaded["obj_slider"]["settled_pos"][0] == pytest.approx(0.64)
    assert loaded["obj_mover"]["settled"] is False


def test_probe_zero_steps_reports_commanded():
    e = _FakeEntity([0.1, 0.2, 0.3], [1, 0, 0, 0], drift_per_step=(1, 1, 1))
    report = osp.probe_settle_env(_FakeEnv({"obj_a": e}), ["obj_a"], steps=0)
    o = report.objects[0]
    assert np.allclose(o.settled_pos, o.commanded_pos) and o.residual_motion is None and not o.settled


def test_stability_from_sim_stats():
    T, n = 120, 2
    names = np.array(["obj_a", "obj_b"])
    pos = np.zeros((T, n, 3))
    quat = np.tile(np.array([1.0, 0, 0, 0]), (T, n, 1))
    # obj_a sinks 3 mm then stays; obj_b slides 2 cm over the first 40 steps and tips 10 deg
    pos[:, 0, 2] = np.where(np.arange(T) < 5, -0.0006 * np.arange(T), -0.003)
    pos[:, 1, 0] = np.minimum(np.arange(T), 40) * 0.0005
    quat[40:, 1] = _quat_axis_angle([0, 1, 0], 10.0)
    npz = {"object_names": names, "object_pos": pos, "object_quat_wxyz": quat}
    out = osp.stability_from_sim_stats(npz, n_steps=100)
    a, b = out["obj_a"], out["obj_b"]
    assert a["z_drop_m"] == pytest.approx(-0.003) and a["settled"] and not a["unstable"]
    assert b["xy_drift_m"] == pytest.approx(0.02) and b["tilt_deg"] == pytest.approx(10.0, abs=1e-6) and b["unstable"]
    assert b["settled"]  # motion stopped at step 40
    assert out["obj_a"]["steps"] == 100


class _TippyEntity(_FakeEntity):
    """Tips 3 deg/step whenever it is free (no hold), and records set_pos/set_quat calls."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.holds = 0
        self.zeroed = 0
        self.held_until = 0

    def set_pos(self, pos, relative=False):
        assert relative
        self.pos = np.asarray(pos, dtype=np.float64)
        self.holds += 1

    def set_quat(self, quat, relative=True):
        self.quat = np.asarray(quat, dtype=np.float64)

    def zero_all_dofs_velocity(self):
        self.zeroed += 1
        self.held_until = self.step_count


def test_settle_modes_upright_hold_and_release():
    stable = _TippyEntity([0.5, 0.0, 0.05], [1, 0, 0, 0], drift_per_step=(0, 0, -0.001), moving_steps=5)
    tippy = _TippyEntity([0.6, 0.1, 0.05], [1, 0, 0, 0], drift_per_step=(0.002, 0, 0), tilt_per_step=3.0, moving_steps=10_000)
    env = _FakeEnv({"obj_stable": stable, "obj_tippy": tippy})
    report = osp.probe_settle_env(env, ["obj_stable", "obj_tippy"], steps=40, sample_every=5, mode="upright", hold_steps=20)
    t = report.by_key()["obj_tippy"]
    assert t.mode == "upright" and t.hold_steps == 20 and t.released_quat_wxyz is not None
    # during the hold the commanded orientation was re-applied every step
    assert tippy.holds == 20 and tippy.zeroed >= 20
    assert osp.tilt_deg(t.commanded_quat_wxyz, t.released_quat_wxyz) < 1e-6
    # after release it tipped on its own: 20 free steps x 3 deg
    assert t.tilt_after_release == pytest.approx(60.0, abs=1e-6) and t.topples_after_release
    s = report.by_key()["obj_stable"]
    assert s.tilt_after_release == pytest.approx(0.0, abs=1e-9) and not s.topples_after_release
    assert "TOPPLES" in "\n".join(report.log_lines())
    d = t.to_json_dict()
    assert d["mode"] == "upright" and d["topples_after_release"] is True and d["released_pos"] is not None

    # gentle: velocities zeroed during the hold, no pose re-application
    tippy2 = _TippyEntity([0.6, 0.1, 0.05], [1, 0, 0, 0], tilt_per_step=3.0, moving_steps=10_000)
    report = osp.probe_settle_env(_FakeEnv({"obj_tippy": tippy2}), ["obj_tippy"], steps=30, mode="gentle", hold_steps=10)
    assert tippy2.holds == 0 and tippy2.zeroed == 10
    assert report.objects[0].released_quat_wxyz is not None
    # free: nothing held, no release pose
    tippy3 = _TippyEntity([0.6, 0.1, 0.05], [1, 0, 0, 0], tilt_per_step=3.0, moving_steps=10_000)
    report = osp.probe_settle_env(_FakeEnv({"obj_tippy": tippy3}), ["obj_tippy"], steps=10, mode="free")
    assert tippy3.zeroed == 0 and report.objects[0].tilt_after_release is None
    with pytest.raises(ValueError):
        osp.probe_settle_env(_FakeEnv({}), [], steps=1, mode="bogus")


def test_probe_settle_takes_entities_and_step_callable():
    """The sim-side entry point: a mapping of entities plus a one-control-step callable."""
    env = _FakeEnv({"obj_a": _FakeEntity([0.5, 0.0, 0.05], [1, 0, 0, 0])})
    calls = []

    def step():
        calls.append(1)
        env.step(None)

    report = osp.probe_settle(env.entities, step, ["obj_a", "obj_missing"], steps=12, sample_every=5)
    assert len(calls) == 12
    assert [s[0] for s in report.objects[0].samples] == [5, 10, 12]
    assert report.missing == ["obj_missing"]
