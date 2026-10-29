"""``r2s doctor``: can this machine run the pipeline? One OK / WARN / FAIL line per check.

Nothing is built and no model is loaded:

* the repo env (``paths.python``) imports eden, genesis, gsplat, torch (built with
  CUDA), cma, trimesh, shapely, plyfile, cv2 and open3d (world.collider) -- probed in
  a child process, so this process stays light;
* gsplat's CUDA extension is built: the JIT build ``<TORCH_EXTENSIONS_DIR>/gsplat_cuda/
  gsplat_cuda.so`` exists and loads (importing ``gsplat.cuda._backend`` would compile
  it when missing);
* a GPU is visible (WARN: the GPU steps need a GPU node);
* the Eden checkout contains the commit r2s was verified on (``EDEN_BRANCH`` @
  ``EDEN_VERIFIED``; WARN otherwise -- read-only git queries);
* every root in ``configs/paths.yaml`` exists (``work`` / ``scene_data_out`` may be
  created by the first run);
* every conda env a step runs in exists under ``conda_root`` (WARN for envs only
  optional steps use);
* model files: big-lama, MINIMA + LightGlue, SAM 3D Objects, InvSR (optional) and the
  Hugging Face cache entries of SAM3, Depth-Anything-V2 metric indoor and MoGe
  (``$HF_HUB_CACHE``, ``$HF_HOME/hub``, ``$TRANSFORMERS_CACHE``, ``~/.cache/huggingface/hub``);
* ``GEMINI_API_KEY`` (WARN: only a live ``inputs.segment`` / ``inputs.physics`` needs it).

Exit 1 on any FAIL.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

from r2s.check import Finding, print_findings
from r2s.config import Paths

REPO_MODULES = ("eden", "genesis", "gsplat", "torch", "cma", "trimesh", "shapely", "plyfile", "cv2", "open3d")
HF_MODELS = {
    "facebook/sam3": "inputs.segment (SAM3)",
    "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf": "inputs.table_depth",
    "Ruicheng/moge-vitl": "inputs.meshes (SAM 3D's MoGe)",
}
OUTPUT_ROOTS = ("work", "scene_data_out")  # made by the first run
# The Eden commit the verification in docs/changes.md ran on (README.md "Dependencies").
EDEN_REPO = "https://github.com/Kashu7100/Eden"
EDEN_BRANCH = "robotarena360"
EDEN_VERIFIED = "b7c85ccc5677a7d8c186e9a05c29a7aac961c4ec"

# Runs inside the repo env; prints one JSON line. Imports only, nothing is compiled.
_PROBE = r'''
import importlib, importlib.util, json, os, sys, time
out = {"python": sys.executable, "version": sys.version.split()[0], "modules": {}}
for name in sys.argv[1].split(","):
    t = time.time()
    try:
        m = importlib.import_module(name)
        out["modules"][name] = {"ok": True, "version": str(getattr(m, "__version__", "") or ""),
                                "file": getattr(m, "__file__", None), "seconds": round(time.time() - t, 1)}
    except BaseException as exc:
        out["modules"][name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
torch = sys.modules.get("torch")
if torch is not None:
    out["torch_cuda"] = torch.version.cuda
    out["cuda_available"] = bool(torch.cuda.is_available())
    out["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if out["cuda_available"] else []
    so = None
    try:
        if "gsplat" in out["modules"] and out["modules"]["gsplat"]["ok"] and importlib.util.find_spec("gsplat.csrc"):
            out["gsplat_ext"] = {"ok": True, "detail": "prebuilt gsplat.csrc"}
        else:
            from torch.utils import cpp_extension
            root = os.environ.get("TORCH_EXTENSIONS_DIR")
            if not root:  # torch's default build root (cpp_extension._get_build_directory, without mkdir)
                acc = "cu" + torch.version.cuda.replace(".", "") if torch.version.cuda else "cpu"
                tag = f"py{sys.version_info.major}{sys.version_info.minor}{getattr(sys, 'abiflags', '')}_{acc}"
                root = os.path.join(cpp_extension.get_default_build_root(), tag)
            so = os.path.join(root, "gsplat_cuda", "gsplat_cuda.so")
            if not os.path.isfile(so):
                out["gsplat_ext"] = {"ok": False, "detail": f"not built ({so} missing)"}
            else:
                cpp_extension._import_module_from_library("gsplat_cuda", os.path.dirname(so), True)
                out["gsplat_ext"] = {"ok": True, "detail": so}
    except BaseException as exc:
        out["gsplat_ext"] = {"ok": False, "detail": f"{so}: {type(exc).__name__}: {exc}"}
print("R2S_DOCTOR " + json.dumps(out))
'''


def probe_repo_env(python: Path, modules=REPO_MODULES, timeout: float = 900) -> dict:
    """Import ``modules`` in ``python`` (a child process) and report versions, CUDA and the gsplat build."""
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    proc = subprocess.run([str(python), "-c", _PROBE, ",".join(modules)], capture_output=True, text=True,
                          timeout=timeout, env=env, cwd=str(Path(__file__).resolve().parents[1]))
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("R2S_DOCTOR "):
            return json.loads(line[len("R2S_DOCTOR "):])
    tail = (proc.stderr.strip().splitlines() or [f"exit {proc.returncode}"])[-1]
    raise RuntimeError(f"probe failed in {python}: {tail}")


def hf_cache_dirs() -> list[Path]:
    """Hugging Face hub cache folders to search, in the hub library's order (then the legacy variable)."""
    env = os.environ
    cands = []
    if env.get("HF_HUB_CACHE"):
        cands.append(Path(env["HF_HUB_CACHE"]))
    if env.get("HF_HOME"):
        cands.append(Path(env["HF_HOME"]) / "hub")
    if env.get("TRANSFORMERS_CACHE"):
        cands.append(Path(env["TRANSFORMERS_CACHE"]))
    xdg = Path(env["XDG_CACHE_HOME"]) if env.get("XDG_CACHE_HOME") else Path.home() / ".cache"
    cands.append(xdg / "huggingface" / "hub")
    seen, out = set(), []
    for c in cands:
        key = c.expanduser().resolve() if c.expanduser().exists() else c.expanduser()
        if key not in seen:
            seen.add(key)
            out.append(c.expanduser())
    return out


def hf_model_dir(repo_id: str, caches: list[Path]) -> Path | None:
    """``<cache>/models--org--name`` with at least one non-empty snapshot, or None."""
    name = "models--" + repo_id.replace("/", "--")
    for cache in caches:
        snaps = cache / name / "snapshots"
        if snaps.is_dir() and any(any(s.iterdir()) for s in snaps.iterdir() if s.is_dir()):
            return cache / name
    return None


def _file_check(item: str, paths: list[Path], used_by: str, *, optional: bool = False) -> Finding:
    missing = [p for p in paths if not p.exists()]
    if not missing:
        return Finding("OK", item, f"{paths[0]}  ({used_by})")
    return Finding("WARN" if optional else "FAIL", item, f"missing {missing[0]}  ({used_by})")


def env_findings(probe: dict, paths: Paths, gpu_steps: list[str]) -> list[Finding]:
    out = []
    note = "" if Path(sys.executable).resolve() == Path(probe["python"]).resolve() else f"; this command runs {sys.executable}"
    out.append(Finding("OK", "repo env", f"{probe['python']} (Python {probe['version']}){note}"))
    for name, m in probe["modules"].items():
        if m["ok"]:
            where = m["version"] or m["file"] or "-"  # the file when there is no version (eden: which checkout)
            out.append(Finding("OK", f"import {name}", f"{where}  ({m['seconds']} s)"))
        else:
            out.append(Finding("FAIL", f"import {name}", m["error"]))
    if "torch_cuda" in probe:
        if probe["torch_cuda"]:
            out.append(Finding("OK", "torch CUDA", f"built for CUDA {probe['torch_cuda']}"))
        else:
            out.append(Finding("FAIL", "torch CUDA", "torch is a CPU-only build; gsplat and the GPU steps need CUDA"))
        ext = probe.get("gsplat_ext") or {"ok": False, "detail": "not probed"}
        hint = "" if ext["ok"] else "; build once on a GPU node: python -c 'from gsplat.cuda._backend import _C'"
        out.append(Finding("OK" if ext["ok"] else "FAIL", "gsplat ext", ext["detail"] + hint))
        if probe["cuda_available"]:
            out.append(Finding("OK", "GPU", ", ".join(probe["gpus"])))
        else:
            out.append(Finding("WARN", "GPU", f"none visible; these steps need a GPU node: {', '.join(gpu_steps)}"))
    return out


def path_findings(paths: Paths) -> list[Finding]:
    out = []
    for f in dataclasses.fields(paths):
        p = getattr(paths, f.name)
        if p.exists():
            out.append(Finding("OK", f"path {f.name}", str(p)))
            continue
        if f.name in OUTPUT_ROOTS:
            parent = next((q for q in p.parents if q.exists()), None)
            if parent is not None and os.access(parent, os.W_OK):
                out.append(Finding("OK", f"path {f.name}", f"{p} (created by the first run)"))
                continue
        out.append(Finding("FAIL", f"path {f.name}", f"{p} does not exist (configs/paths.yaml / env override)"))
    return out


def eden_findings(eden: Path, verified: str | None = None) -> list[Finding]:
    """Which commit the Eden checkout is on, against the one r2s was verified on (``EDEN_VERIFIED``)."""
    if not eden.is_dir():
        return []  # path_findings reports it
    verified = verified or EDEN_VERIFIED

    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", "-C", str(eden), *args], capture_output=True, text=True, timeout=60)

    want = f"{EDEN_BRANCH} @ {verified[:7]}"
    head = git("rev-parse", "--short", "HEAD")
    if head.returncode != 0:
        return [Finding("WARN", "eden version", f"cannot read {eden} with git; r2s was verified on {want} ({EDEN_REPO})")]
    where = f"{git('rev-parse', '--abbrev-ref', 'HEAD').stdout.strip()} @ {head.stdout.strip()}"
    if git("merge-base", "--is-ancestor", verified, "HEAD").returncode == 0:
        return [Finding("OK", "eden version", f"{where} (contains the verified {want})")]
    return [Finding("WARN", "eden version", f"{where} does not contain {want}, the commit r2s was verified on")]


def conda_findings(paths: Paths, steps: dict) -> list[Finding]:
    """``steps``: name -> Step. One line per conda env, listing the steps that use it."""
    from r2s.steps import OPTIONAL_STEPS

    users: dict[str, list[str]] = {}
    for name, step in steps.items():
        if step.conda_env:
            users.setdefault(step.conda_env, []).append(name)
    out = []
    for env, names in users.items():
        py = paths.conda_root / "envs" / env / "bin" / "python"
        optional = all(n in OPTIONAL_STEPS for n in names)
        if py.exists():
            out.append(Finding("OK", f"conda {env}", f"{py.parent.parent}  ({', '.join(names)})"))
        else:
            out.append(Finding("WARN" if optional else "FAIL", f"conda {env}", f"missing {py}  ({', '.join(names)})"))
    return out


def weight_findings(paths: Paths) -> list[Finding]:
    sg = paths.scene_generation
    out = [
        _file_check("big-lama", [sg / "lama" / "big-lama" / "models" / "best.ckpt", sg / "lama" / "big-lama" / "config.yaml"],
                    "inputs.inpaint"),
        _file_check("MINIMA", [sg / "MINIMA" / "weights" / "minima_lightglue.pth"], "inputs.place"),
        _file_check("LightGlue", [sg / "third_party" / "LightGlue" / "lightglue" / "__init__.py"], "inputs.place"),
        _file_check("sam3d", [paths.sam3d_objects / "checkpoints" / "hf" / "pipeline.yaml",
                              paths.sam3d_objects / "notebook" / "inference.py"], "inputs.meshes"),
        _file_check("InvSR", [sg / "InvSR" / "inference_invsr.py"], "inputs.superres, optional", optional=True),
    ]
    caches = hf_cache_dirs()
    for repo_id, used_by in HF_MODELS.items():
        found = hf_model_dir(repo_id, caches)
        if found is not None:
            out.append(Finding("OK", "hf cache", f"{repo_id}: {found}  ({used_by})"))
        else:
            out.append(Finding("FAIL", "hf cache", f"{repo_id} not in {', '.join(map(str, caches))}  ({used_by})"))
    return out


def run_doctor(paths: Paths, probe: dict | None = None) -> list[Finding]:
    from r2s.steps import STEP_MODULES, load_step

    steps = {name: load_step(name) for name in STEP_MODULES}
    gpu_steps = [name for name, step in steps.items() if step.gpu]
    findings: list[Finding] = []
    if probe is None:
        if not paths.python.exists():
            findings.append(Finding("FAIL", "repo env", f"{paths.python} does not exist (scripts/setup_env.sh)"))
        else:
            try:
                probe = probe_repo_env(paths.python)
            except Exception as exc:
                findings.append(Finding("FAIL", "repo env", str(exc)))
    if probe is not None:
        findings += env_findings(probe, paths, gpu_steps)
    findings += eden_findings(paths.eden)
    findings += path_findings(paths)
    findings += conda_findings(paths, steps)
    findings += weight_findings(paths)
    if os.environ.get("GEMINI_API_KEY"):
        findings.append(Finding("OK", "GEMINI_API_KEY", "set"))
    else:
        findings.append(Finding("WARN", "GEMINI_API_KEY", "not set: live inputs.segment / inputs.physics need it "
                                                          "(--replay-gemini / --replay work without)"))
    return findings


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m r2s doctor", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="paths.yaml (default: configs/paths.yaml or $R2S_CONFIG)")
    args = parser.parse_args(argv)
    paths = Paths.load(args.config)
    fails = print_findings(run_doctor(paths), f"[r2s doctor] work {paths.work}")
    return 1 if fails else 0
