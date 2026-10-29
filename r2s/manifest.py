"""Per-scene run manifest: what each step ran with, read and wrote.

``<work>/scenes/<s>/manifest.json`` keeps the latest record of every step, so a
result can always be traced back to its exact arguments. The old repo lost those
(pose-fit arguments, scale truncation per scene); this file is the fix.
"""

from __future__ import annotations

import datetime as _dt
import getpass
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from r2s.config import REPO_ROOT

_HASH_HEAD = 1 << 20  # hash the first and last MiB of big files: fast and good enough to detect changes


def file_fingerprint(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "exists": False}
    if path.is_dir():
        return {"path": str(path), "exists": True, "dir": True}
    st = path.stat()
    h = hashlib.sha256()
    with path.open("rb") as f:
        if st.st_size <= 2 * _HASH_HEAD:
            h.update(f.read())
        else:
            h.update(f.read(_HASH_HEAD))
            f.seek(-_HASH_HEAD, os.SEEK_END)
            h.update(f.read(_HASH_HEAD))
    return {
        "path": str(path),
        "exists": True,
        "size": st.st_size,
        "mtime": _dt.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
        "sha256_head_tail": h.hexdigest()[:16],
    }


def _git_rev() -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        dirty = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        rev = out.stdout.strip() or None
        return f"{rev}-dirty" if rev and dirty.stdout.strip() else rev
    except (OSError, subprocess.SubprocessError):
        return None


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


class Manifest:
    def __init__(self, path: Path):
        self.path = Path(path)

    def load(self) -> dict[str, Any]:
        if self.path.exists():
            return json.loads(self.path.read_text())
        return {"version": 1, "steps": {}}

    def record(self, step: str, entry: dict[str, Any]) -> None:
        data = self.load()
        data.setdefault("steps", {})[step] = _jsonable(entry)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n")
        tmp.replace(self.path)

    def get(self, step: str) -> dict[str, Any] | None:
        return self.load().get("steps", {}).get(step)


@contextmanager
def step_record(
    manifest_path: Path,
    step: str,
    *,
    args: dict[str, Any],
    inputs: list[Path],
    outputs: list[Path],
    inputs_root: Path | None = None,
) -> Iterator[dict[str, Any]]:
    """Record one step run. The yielded dict can carry extra ``notes``/``stats``.

    ``inputs_root`` is the step-0 root the scene's inputs were read from
    (``SceneLayout.inputs_root_source``).
    """
    extra: dict[str, Any] = {}
    started = time.time()
    entry: dict[str, Any] = {
        "started": _dt.datetime.now().isoformat(timespec="seconds"),
        "user": getpass.getuser(),
        "host": socket.gethostname(),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "git_rev": _git_rev(),
        "python": sys.executable,
        "argv": sys.argv,
        "args": args,
        "inputs_root": inputs_root,
        "inputs": [file_fingerprint(p) for p in inputs],
    }
    try:
        yield extra
    except BaseException as exc:
        entry.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        entry["seconds"] = round(time.time() - started, 2)
        Manifest(manifest_path).record(step, {**entry, **extra})
        raise
    entry.update(status="ok", seconds=round(time.time() - started, 2))
    entry["outputs"] = [file_fingerprint(p) for p in outputs]
    Manifest(manifest_path).record(step, {**entry, **extra})
