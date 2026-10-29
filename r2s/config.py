"""Load ``configs/paths.yaml`` and ``configs/scenes.yaml``.

Paths support ``${VAR}``, ``${VAR:-default}`` (environment first) and ``{key}``
(another key of the same file). Nothing here touches the filesystem beyond reading
the two YAML files.
"""

from __future__ import annotations

import copy
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATHS_FILE = REPO_ROOT / "configs" / "paths.yaml"
DEFAULT_SCENES_FILE = REPO_ROOT / "configs" / "scenes.yaml"

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-((?:[^{}]|\{[^{}]*\})*))?\}")
_KEY_RE = re.compile(r"\{([a-z_][a-z0-9_]*)\}")
LOCAL_ENV_HINT = ("set them in configs/local.env (cp configs/local.env.example configs/local.env) "
                  "and `source scripts/env.sh`")


class ConfigError(KeyError):
    def __str__(self) -> str:
        return str(self.args[0])


def _expand_env(value: str, unset: list[str]) -> str:
    """Expand ``${VAR}`` / ``${VAR:-default}``; an unset ``${VAR}`` is appended to ``unset``."""

    def repl(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        env = os.environ.get(name)
        if env:
            return env
        if default is None:
            unset.append(name)
            return f"<{name} unset>"
        return default

    previous = None
    while previous != value:
        previous, value = value, _ENV_RE.sub(repl, value)
    return value


def _expand_keys(raw: dict[str, str]) -> dict[str, str]:
    out = dict(raw)
    for _ in range(len(out) + 1):
        changed = False
        for key, value in out.items():
            new = _KEY_RE.sub(lambda m: out[m.group(1)] if m.group(1) in out else m.group(0), value)
            if new != value:
                out[key], changed = new, True
        if not changed:
            return out
    raise ValueError("circular {key} reference in paths config")


@dataclass(frozen=True)
class Paths:
    scene_data: Path
    scene_data_out: Path
    work: Path
    splat_source_dir: Path
    eden: Path
    python: Path
    scene_generation: Path
    sam3d_objects: Path
    conda_root: Path
    layout_refiner: Path
    tasks_dir: Path

    @classmethod
    def load(cls, path: str | os.PathLike[str] | None = None) -> "Paths":
        path = _paths_file(path)
        unset: list[str] = []
        expanded = _expand_keys({k: _expand_env(str(v), unset) for k, v in _read_paths(path).items()})
        if unset:
            raise ConfigError(f"{path} needs {', '.join(sorted(set(unset)))}: {LOCAL_ENV_HINT}")
        missing = [f for f in cls.__dataclass_fields__ if f not in expanded]
        if missing:
            raise ConfigError(f"{path} is missing keys: {', '.join(missing)}")
        return cls(**{f: Path(expanded[f]).expanduser() for f in cls.__dataclass_fields__})


def _paths_file(path: str | os.PathLike[str] | None) -> Path:
    return Path(path or os.environ.get("R2S_CONFIG") or DEFAULT_PATHS_FILE)


def _read_paths(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text()) or {}


def unset_variables(path: str | os.PathLike[str] | None = None) -> list[str]:
    """The environment variables the paths file needs that are not set."""
    unset: list[str] = []
    for value in _read_paths(_paths_file(path)).values():
        _expand_env(str(value), unset)
    return sorted(set(unset))


@dataclass(frozen=True)
class SceneConfig:
    """Merged settings for one scene (``defaults`` overlaid by the scene entry)."""

    name: str
    values: dict[str, Any] = field(default_factory=dict)

    @property
    def scene_type(self) -> str:
        return str(self.values.get("scene_type", "DROID"))

    @property
    def robot(self) -> str:
        return str(self.values.get("robot", "franka"))

    def step_options(self, step: str) -> dict[str, Any]:
        """Per-scene CLI option overrides for ``step`` (keys use underscores)."""
        opts = self.values.get(step) or {}
        return {str(k).replace("-", "_"): v for k, v in opts.items()}


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (over or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


class SceneRegistry:
    def __init__(self, path: str | os.PathLike[str] | None = None):
        self.path = Path(path or os.environ.get("R2S_SCENES") or DEFAULT_SCENES_FILE)
        data = yaml.safe_load(self.path.read_text()) or {}
        self._defaults: dict[str, Any] = data.get("defaults") or {}
        self._scenes: dict[str, dict[str, Any]] = {k: (v or {}) for k, v in (data.get("scenes") or {}).items()}

    def names(self, scene_type: str | None = None) -> list[str]:
        return [n for n in self._scenes if scene_type is None or self.get(n).scene_type == scene_type]

    def get(self, name: str) -> SceneConfig:
        if name not in self._scenes:
            # Unknown scenes still work with the defaults, so a new scene needs no edit here.
            return SceneConfig(name, copy.deepcopy(self._defaults))
        return SceneConfig(name, _merge(self._defaults, self._scenes[name]))
