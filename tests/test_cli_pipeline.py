"""``r2s pipeline``: range selection, --skip-done, fail-fast checks, stop at the first failure,
and the scripts/run_scene.sh compatibility wrapper. No step really runs."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from r2s import cli
from r2s.config import Paths
from r2s.layout import SceneLayout
from r2s.manifest import Manifest
from r2s.steps import OPTIONAL_STEPS, STEP_MODULES, load_step

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def run_env(tmp_path, monkeypatch):
    """A fresh work run and conda root; every step process / GPU probe is faked."""
    monkeypatch.setenv("R2S_WORK", str(tmp_path / "work"))
    monkeypatch.setenv("R2S_CONDA_ROOT", str(tmp_path / "conda"))
    monkeypatch.setenv("R2S_PYTHON", "/fake/python")
    calls: list[str] = []
    state = {"fail": set(), "gpu": True}

    def fake_run(python, step, scene, extra):
        assert python == "/fake/python" and scene == "droidX"
        calls.append(step)
        return 3 if step in state["fail"] else 0

    def fake_gpu(python):
        state["gpu_probes"] = state.get("gpu_probes", 0) + 1
        return state["gpu"], "Fake GPU" if state["gpu"] else "none"

    monkeypatch.setattr(cli, "run_step_process", fake_run)
    monkeypatch.setattr(cli, "gpu_available", fake_gpu)
    return calls, state, tmp_path


def _conda_env(tmp_path: Path, env: str) -> None:
    py = tmp_path / "conda" / "envs" / env / "bin" / "python"
    py.parent.mkdir(parents=True, exist_ok=True)
    py.touch()


def test_step_range_order_and_optional():
    full = cli.step_range(None, None)
    assert [n for n, _ in full] == list(STEP_MODULES)
    assert [n for n, runs in full if not runs] == sorted(OPTIONAL_STEPS)
    assert cli.step_range("surface.plane", "surface.receiver") == [
        ("surface.plane", True), ("surface.trim", True), ("surface.receiver", True)]
    # an optional step runs when it bounds the range, or with include_optional
    assert dict(cli.step_range("inputs.superres", "inputs.inpaint"))["inputs.superres"] is True
    assert dict(cli.step_range("inputs.segment", "inputs.inpaint"))["inputs.superres"] is False
    assert dict(cli.step_range("inputs.segment", "inputs.inpaint", include_optional=True))["inputs.superres"] is True
    with pytest.raises(SystemExit, match="unknown step"):
        cli.step_range("surface.nope", None)
    with pytest.raises(SystemExit, match="comes after"):
        cli.step_range("surface.receiver", "surface.plane")


def test_align_check_runs_right_after_align_scale():
    order = list(STEP_MODULES)
    assert order[order.index("align.scale") + 1] == "align.check"
    assert load_step("align.check").needs == "cpu"


def test_needs_vocabulary():
    from r2s.steps import Step

    def make(needs):
        return Step("x", "", lambda p: None, lambda l, a: None, lambda l, a: [], lambda l, a: [], needs=needs)

    for needs, gpu, env in (("cpu", False, None), ("gpu", True, None), ("conda:lama", False, "lama"),
                            ("gpu+conda:sam3d-objects", True, "sam3d-objects")):
        assert (make(needs).gpu, make(needs).conda_env) == (gpu, env)
    for bad in ("GPU", "conda", "conda:", "gpu+cpu", "cuda", "gpu,conda:match"):
        with pytest.raises(ValueError):
            make(bad)
    expected = {"inputs.segment": "gpu+conda:gemini", "inputs.physics": "conda:gemini", "inputs.inpaint": "conda:lama",
                "inputs.superres": "gpu+conda:invsr", "inputs.mesh_sheet": "gpu+conda:match",
                "world.collider": "cpu", "align.check": "cpu", "objects.refine": "gpu"}
    assert {n: load_step(n).needs for n in expected} == expected


def test_step_done(tmp_path):
    fields = {f: tmp_path / f for f in Paths.__dataclass_fields__}
    layout = SceneLayout.create("droidX", Paths(**fields))
    out = tmp_path / "out.json"
    out.write_text("{}")
    m = Manifest(layout.manifest)
    m.record("surface.plane", {"status": "ok", "outputs": [{"path": str(out), "exists": True}]})
    m.record("surface.trim", {"status": "ok", "outputs": [{"path": str(tmp_path / "gone.ply"), "exists": True}]})
    m.record("surface.receiver", {"status": "failed", "outputs": [{"path": str(out), "exists": True}]})
    m.record("collision.crop", {"status": "ok", "outputs": []})
    assert cli.step_done(layout, "surface.plane")
    assert not cli.step_done(layout, "surface.trim")  # an output was deleted since
    assert not cli.step_done(layout, "surface.receiver")
    assert not cli.step_done(layout, "collision.crop")
    assert not cli.step_done(layout, "objects.dump")  # never ran


def test_runs_range_in_order(run_env, capsys):
    calls, state, _ = run_env
    assert cli.main(["pipeline", "--scene", "droidX", "--from", "surface.plane", "--to", "objects.posefit"]) == 0
    assert calls == ["surface.plane", "surface.trim", "surface.receiver", "collision.crop", "objects.dump",
                     "objects.posefit"]
    assert "gpu_probes" not in state  # CPU-only range: no GPU probe
    assert "done, 6 step(s)" in capsys.readouterr().out


def test_stops_at_first_failure(run_env, capsys):
    calls, state, _ = run_env
    state["fail"] = {"surface.trim"}
    code = cli.main(["pipeline", "--scene", "droidX", "--from", "surface.plane", "--to", "collision.crop"])
    assert code == 3
    assert calls == ["surface.plane", "surface.trim"]
    err = capsys.readouterr().err
    assert "FAILED at surface.trim" in err and "--from surface.trim --to collision.crop" in err


def test_gpu_step_fails_fast_without_gpu(run_env, capsys):
    calls, state, tmp = run_env
    state["gpu"] = False
    code = cli.main(["pipeline", "--scene", "droidX", "--from", "world.import", "--to", "surface.plane"])
    assert code == 2 and calls == []  # nothing ran, not even the CPU steps before align.pose
    err = capsys.readouterr().err
    assert "align.pose needs a GPU node" in err and "align.scale" in err
    state["gpu"] = True
    assert cli.main(["pipeline", "--scene", "droidX", "--from", "world.import", "--to", "surface.plane"]) == 0
    assert calls == ["world.import", "world.collider", "align.pose", "align.scale", "align.check", "surface.plane"]
    assert state["gpu_probes"] == 2  # once per pipeline call


def test_missing_conda_env_fails_fast(run_env, capsys):
    calls, _, _ = run_env
    assert cli.main(["pipeline", "--scene", "droidX", "--from", "inputs.inpaint", "--to", "inputs.inpaint"]) == 2
    assert calls == [] and "needs conda env 'lama'" in capsys.readouterr().err


def test_skip_done_dry_run_and_optional(run_env, capsys):
    calls, _, tmp = run_env
    layout = SceneLayout.create("droidX")
    out = tmp / "plane.json"
    out.write_text("{}")
    Manifest(layout.manifest).record("surface.plane", {"status": "ok", "outputs": [{"path": str(out)}]})
    assert cli.main(["pipeline", "--scene", "droidX", "--from", "surface.plane", "--to", "surface.trim",
                     "--skip-done", "--dry-run"]) == 0
    assert calls == []
    text = capsys.readouterr().out
    assert "surface.plane        already done (--skip-done)" in text and "surface.trim         run" in text
    assert cli.main(["pipeline", "--scene", "droidX", "--from", "surface.plane", "--to", "surface.trim",
                     "--skip-done"]) == 0
    assert calls == ["surface.trim"]
    calls.clear()
    _conda_env(tmp, "gemini")
    _conda_env(tmp, "lama")
    assert cli.main(["pipeline", "--scene", "droidX", "--from", "inputs.physics", "--to", "inputs.inpaint"]) == 0
    assert calls == ["inputs.physics", "inputs.inpaint"]  # inputs.superres is optional


def test_run_step_process_uses_the_repo(monkeypatch):
    seen = {}

    def fake(cmd, cwd):
        seen.update(cmd=cmd, cwd=cwd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(cli.subprocess, "run", fake)
    assert cli.run_step_process("/py", "surface.plane", "droid2", ["--config", "/c.yaml"]) == 0
    assert seen["cmd"] == ["/py", "-m", "r2s", "run", "surface.plane", "--scene", "droid2", "--config", "/c.yaml"]
    assert Path(seen["cwd"]) == REPO


@pytest.mark.parametrize("args,expected", [
    (["droid2"], ["pipeline", "--scene", "droid2"]),
    (["droid2", "align.pose"], ["pipeline", "--scene", "droid2", "--from", "align.pose"]),
    (["droid2", "align.pose", "objects.dump"], ["pipeline", "--scene", "droid2", "--from", "align.pose", "--to", "objects.dump"]),
    (["droid2", "world.import", "objects.dump", "--skip-done"],
     ["pipeline", "--scene", "droid2", "--from", "world.import", "--to", "objects.dump", "--skip-done"]),
    (["droid2", "--skip-done", "align.pose"], ["pipeline", "--scene", "droid2", "--from", "align.pose", "--skip-done"]),
])
def test_run_scene_wrapper_maps_to_pipeline(tmp_path, args, expected):
    fake = tmp_path / "python"
    log = tmp_path / "argv.json"
    fake.write_text(f"#!/bin/sh\nprintf '%s\\n' \"$@\" > {log}\n")
    fake.chmod(0o755)
    env = dict(os.environ, R2S_PYTHON=str(fake))
    subprocess.run(["bash", str(REPO / "scripts" / "run_scene.sh"), *args], check=True, env=env)
    assert log.read_text().split("\n")[:-1] == ["-m", "r2s", *expected]
