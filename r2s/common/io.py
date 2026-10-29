"""Small file helpers. Every step writes its outputs through :func:`prepare_output`."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def prepare_output(path: str | os.PathLike[str]) -> Path:
    """Make ``path`` safe to write: create its folder and drop a symlink sitting there.

    Work folders are often seeded with symlinks to read-only inputs (another run,
    the old repo, Eden assets). Writing through such a link would overwrite the
    original, so an existing symlink is unlinked first and a fresh file is written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        path.unlink()
    return path


def write_json(path: str | os.PathLike[str], data: Any, *, indent: int = 2) -> Path:
    path = prepare_output(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=indent) + "\n")
    tmp.replace(path)
    return path


def read_json(path: str | os.PathLike[str]) -> Any:
    return json.loads(Path(path).read_text())


def save_npy(path: str | os.PathLike[str], array: np.ndarray) -> Path:
    path = prepare_output(path)
    np.save(path, np.asarray(array))
    return path
