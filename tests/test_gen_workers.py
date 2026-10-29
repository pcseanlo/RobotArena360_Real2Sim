"""Every step-0 worker starts in its own conda env (``--help``) and finds ``sg_glue``.

Skipped when an env is absent. The imports that load torch / Genesis / Open3D in the
``match`` / ``moge`` envs (~10 s each) run with ``R2S_TEST_CONDA=1``.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from r2s.config import Paths

REPO = Path(__file__).resolve().parents[1]
GEN = REPO / "r2s" / "gen"
SLOW = os.environ.get("R2S_TEST_CONDA") == "1"

# worker -> conda env (the env the step passes to run_in_env)
WORKERS = {
    "r2s/gen/workers/segment_gembox.py": "gemini",
    "r2s/gen/workers/physics_gemini.py": "gemini",
    "r2s/gen/workers/superres_invsr.py": "invsr",
    "r2s/gen/workers/lama_inpaint.py": "lama",
    "r2s/gen/workers/sam3d_meshes.py": "sam3d-objects",
    "r2s/gen/workers/match_place.py": "match",
    "r2s/gen/workers/align_table_depth.py": "moge",
    "r2s/gen/workers/sheet_wrap.py": "match",
}
# sg_glue modules each env imports (as the workers do: r2s/gen appended to sys.path)
SG_GLUE = {
    "gemini": (["gembox", "gemini_utils", "pointcloud", "segment_replay", "physics_utils"], False),
    "match": (["match_find_rot", "matching_utils", "dataloader_utils", "minima_sp_lg", "simulator_genesis", "hunyuanworld"], True),
    "moge": (["align_inpaint"], True),
}


def _python(env: str) -> Path:
    try:
        root = Paths.load().conda_root
    except Exception:  # pragma: no cover - config not loadable
        pytest.skip("paths.yaml not loadable")
    py = root / "envs" / env / "bin" / "python"
    if not py.exists():
        pytest.skip(f"conda env {env} not available")
    return py


def _run(py: Path, *args: str) -> subprocess.CompletedProcess:
    env = {"PATH": f"{py.parent}:/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1",
           "HOME": os.environ.get("HOME", "/tmp")}
    return subprocess.run([str(py), *args], capture_output=True, text=True, env=env, cwd="/", timeout=600)


def test_every_worker_is_listed():
    found = {str(p.relative_to(REPO)) for p in (REPO / "r2s/gen/workers").glob("*.py")}
    assert found == set(WORKERS)


@pytest.mark.parametrize("worker", sorted(WORKERS))
def test_worker_help_in_its_env(worker):
    proc = _run(_python(WORKERS[worker]), str(REPO / worker), "--help")
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "usage:" in proc.stdout


@pytest.mark.parametrize("env", sorted(SG_GLUE))
def test_sg_glue_imports_in_its_env(env):
    modules, slow = SG_GLUE[env]
    if slow and not SLOW:
        pytest.skip("set R2S_TEST_CONDA=1 (imports torch / Genesis / Open3D)")
    sg = Paths.load().scene_generation  # LightGlue, as match_place.py adds it
    code = (f"import sys; sys.path.append({str(GEN)!r}); sys.path.append({str(sg)!r}); "
            f"from sg_glue import {', '.join(modules)}; print('ok')")
    proc = _run(_python(env), "-c", code)
    assert proc.returncode == 0 and proc.stdout.strip().endswith("ok"), proc.stderr[-2000:]
