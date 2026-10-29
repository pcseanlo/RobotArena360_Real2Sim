"""Run a step-0 worker inside one of the model conda environments.

Workers live in ``r2s/gen/workers/`` and are plain scripts: they import only the
standard library, what their own env provides and ``r2s/gen/sg_glue`` (as the
top-level package ``sg_glue``, with ``r2s/gen`` appended to ``sys.path``) -- never
the ``r2s`` package -- and take every path on the command line. The orchestrating step (running in the repo
env) builds that command line from the scene layout and calls :func:`run_in_env`.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

from r2s.config import Paths

WORKERS_DIR = Path(__file__).resolve().parent / "workers"


def env_prefix(paths: Paths, env: str) -> Path:
    prefix = paths.conda_root / "envs" / env
    if not (prefix / "bin" / "python").exists():
        raise FileNotFoundError(f"conda env {env!r} not found at {prefix}")
    return prefix


def run_in_env(
    paths: Paths,
    env: str,
    argv: Sequence[str | os.PathLike[str]],
    *,
    cwd: str | os.PathLike[str] | None = None,
    extra_env: Mapping[str, str] | None = None,
    pythonpath: Sequence[str | os.PathLike[str]] = (),
    log_file: str | os.PathLike[str] | None = None,
) -> None:
    """Run ``python <argv...>`` with the env's interpreter; raise on a non-zero exit.

    The env's ``bin`` goes first on ``PATH`` (ffmpeg, CUDA wrappers) and
    ``CONDA_PREFIX`` is set, which is what ``conda activate`` would do for these envs.
    """
    prefix = env_prefix(paths, env)
    env_vars = dict(os.environ)
    env_vars.pop("PYTHONHOME", None)
    env_vars["PATH"] = f"{prefix / 'bin'}{os.pathsep}{env_vars.get('PATH', '')}"
    env_vars["CONDA_PREFIX"] = str(prefix)
    env_vars["CONDA_DEFAULT_ENV"] = env
    env_vars["PYTHONDONTWRITEBYTECODE"] = "1"  # never drop __pycache__ into third-party checkouts
    env_vars["PYTHONNOUSERSITE"] = "1"
    if pythonpath:
        env_vars["PYTHONPATH"] = os.pathsep.join(str(p) for p in pythonpath)
    else:
        env_vars.pop("PYTHONPATH", None)
    env_vars.update(extra_env or {})

    cmd = [str(prefix / "bin" / "python"), *[str(a) for a in argv]]
    print(f"[r2s:{env}] {shlex.join(cmd)}", file=sys.stderr, flush=True)
    if log_file is None:
        subprocess.run(cmd, cwd=cwd, env=env_vars, check=True)
        return
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log:
        log.write(f"\n$ {shlex.join(cmd)}\n".encode())
        log.flush()
        proc = subprocess.run(cmd, cwd=cwd, env=env_vars, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        tail = log_path.read_text(errors="replace").splitlines()[-30:]
        raise RuntimeError(f"{env} worker failed (exit {proc.returncode}); last log lines:\n" + "\n".join(tail))


def worker(name: str) -> Path:
    path = WORKERS_DIR / name
    if not path.exists():
        raise FileNotFoundError(path)
    return path
