"""``r2s doctor``: report levels from a canned env probe, the HF cache lookup, and a real (small) probe."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from r2s import doctor
from r2s.config import Paths
from r2s.steps import STEP_MODULES, load_step

GOOD_PROBE = {
    "python": sys.executable,
    "version": "3.12.0",
    "modules": {m: {"ok": True, "version": "1.0", "file": f"/x/{m}/__init__.py", "seconds": 0.1}
                for m in doctor.REPO_MODULES},
    "torch_cuda": "13.0",
    "cuda_available": True,
    "gpus": ["Fake GPU"],
    "gsplat_ext": {"ok": True, "detail": "/cache/gsplat_cuda/gsplat_cuda.so"},
}


def _paths(tmp: Path) -> Paths:
    """Every root exists, every conda env and model file is present."""
    fields = {f: tmp / f for f in Paths.__dataclass_fields__}
    for p in fields.values():
        p.mkdir(parents=True)
    paths = Paths(**fields)
    envs = {load_step(n).conda_env for n in STEP_MODULES} - {None}
    for env in envs:
        (paths.conda_root / "envs" / env / "bin").mkdir(parents=True)
        (paths.conda_root / "envs" / env / "bin" / "python").touch()
    sg = paths.scene_generation
    for rel in ("lama/big-lama/models/best.ckpt", "lama/big-lama/config.yaml", "MINIMA/weights/minima_lightglue.pth",
                "third_party/LightGlue/lightglue/__init__.py", "InvSR/inference_invsr.py"):
        (sg / rel).parent.mkdir(parents=True, exist_ok=True)
        (sg / rel).touch()
    for rel in ("checkpoints/hf/pipeline.yaml", "notebook/inference.py"):
        (paths.sam3d_objects / rel).parent.mkdir(parents=True, exist_ok=True)
        (paths.sam3d_objects / rel).touch()
    return paths


def _hf_cache(root: Path, repo_ids) -> Path:
    for repo_id in repo_ids:
        snap = root / ("models--" + repo_id.replace("/", "--")) / "snapshots" / "abc123"
        snap.mkdir(parents=True)
        (snap / "config.json").write_text("{}")
    return root


def _by_item(findings):
    out = {}
    for f in findings:
        out.setdefault(f.item, []).append(f.level)
    return out


def _git_commits(repo: Path, n: int) -> list[str]:
    """``n`` empty commits in ``repo`` (a fresh git repo); their hashes, oldest first."""
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]
    subprocess.run(git + ["init", "-q", "-b", doctor.EDEN_BRANCH], check=True)
    for i in range(n):
        subprocess.run(git + ["commit", "-q", "--allow-empty", "-m", f"c{i}"], check=True)
    out = subprocess.run(git + ["rev-list", "--reverse", "HEAD"], check=True, capture_output=True, text=True)
    return out.stdout.split()


def test_eden_version(tmp_path):
    (repo := tmp_path / "eden").mkdir()
    first, second = _git_commits(repo, 2)
    assert [f.level for f in doctor.eden_findings(repo, first)] == ["OK"]  # HEAD contains it
    subprocess.run(["git", "-C", str(repo), "checkout", "-q", first], check=True)
    (found,) = doctor.eden_findings(repo, second)
    assert found.level == "WARN" and second[:7] in found.message
    assert doctor.eden_findings(tmp_path / "missing") == []  # path_findings reports a missing checkout
    (plain := tmp_path / "plain").mkdir()
    assert [f.level for f in doctor.eden_findings(plain)] == ["WARN"]  # not a git checkout


def test_all_good(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(_hf_cache(tmp_path / "hf", doctor.HF_MODELS)))
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    paths = _paths(tmp_path / "roots")
    monkeypatch.setattr(doctor, "EDEN_VERIFIED", _git_commits(paths.eden, 1)[0])
    findings = doctor.run_doctor(paths, probe=GOOD_PROBE)
    assert {f.level for f in findings} == {"OK"}, [f for f in findings if f.level != "OK"]
    items = _by_item(findings)
    assert {"import torch", "torch CUDA", "gsplat ext", "GPU", "path work", "conda match", "big-lama", "MINIMA",
            "sam3d", "hf cache", "GEMINI_API_KEY"} <= set(items)
    assert len(items["hf cache"]) == len(doctor.HF_MODELS)


def test_problems_are_reported(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(_hf_cache(tmp_path / "hf", ["facebook/sam3"])))
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("TRANSFORMERS_CACHE", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    paths = _paths(tmp_path / "roots")
    (paths.conda_root / "envs" / "match" / "bin" / "python").unlink()
    (paths.conda_root / "envs" / "invsr" / "bin" / "python").unlink()  # only the optional superres step uses it
    (paths.scene_generation / "lama" / "big-lama" / "models" / "best.ckpt").unlink()
    paths.tasks_dir.rmdir()
    paths.work.rmdir()  # an output root: created by the first run
    probe = dict(GOOD_PROBE, cuda_available=False, gpus=[],
                 gsplat_ext={"ok": False, "detail": "not built (/cache/gsplat_cuda/gsplat_cuda.so missing)"},
                 modules=dict(GOOD_PROBE["modules"], genesis={"ok": False, "error": "ImportError: nope"}))
    items = _by_item(doctor.run_doctor(paths, probe=probe))
    assert items["import genesis"] == ["FAIL"] and items["import torch"] == ["OK"]
    assert items["GPU"] == ["WARN"] and items["gsplat ext"] == ["FAIL"]
    assert items["conda match"] == ["FAIL"] and items["conda invsr"] == ["WARN"]
    assert items["big-lama"] == ["FAIL"] and items["path tasks_dir"] == ["FAIL"] and items["path work"] == ["OK"]
    assert items["hf cache"] == ["OK", "FAIL", "FAIL"] and items["GEMINI_API_KEY"] == ["WARN"]


def test_hf_cache_dirs_order(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "a"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "b"))
    monkeypatch.setenv("TRANSFORMERS_CACHE", str(tmp_path / "c"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "d"))
    assert doctor.hf_cache_dirs() == [tmp_path / "a", tmp_path / "b" / "hub", tmp_path / "c",
                                      tmp_path / "d" / "huggingface" / "hub"]
    _hf_cache(tmp_path / "c", ["org/model"])
    (tmp_path / "a" / "models--org--model" / "snapshots" / "empty").mkdir(parents=True)  # no files: not a hit
    assert doctor.hf_model_dir("org/model", doctor.hf_cache_dirs()) == tmp_path / "c" / "models--org--model"
    assert doctor.hf_model_dir("org/other", doctor.hf_cache_dirs()) is None


def test_real_probe_imports_in_a_child_process():
    probe = doctor.probe_repo_env(Path(sys.executable), modules=("cma", "r2s_no_such_module"))
    assert probe["modules"]["cma"]["ok"] and not probe["modules"]["r2s_no_such_module"]["ok"]
    assert "torch_cuda" not in probe  # torch (and the gsplat build) only when asked for
