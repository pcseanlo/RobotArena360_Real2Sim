"""Step registry.

Every pipeline step is a module exposing ``STEP = Step(...)``. Modules are imported
lazily, so ``python -m r2s run surface.plane`` never imports torch or Genesis.
"""

from __future__ import annotations

import argparse
import importlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from r2s.layout import SceneLayout

PathsFn = Callable[[SceneLayout, argparse.Namespace], list[Path]]

# Where a step runs (``Step.needs``): "cpu" (repo env), "gpu" (repo env on a GPU node),
# "conda:<env>" (CPU, a model in its own conda env) or "gpu+conda:<env>" (GPU, conda env).
_NEEDS_RE = re.compile(r"^(?:cpu|gpu|(?:gpu\+)?conda:[A-Za-z0-9_.-]+)$")


@dataclass(frozen=True)
class Step:
    name: str
    help: str
    add_arguments: Callable[[argparse.ArgumentParser], None]
    run: Callable[[SceneLayout, argparse.Namespace], dict[str, Any] | None]
    inputs: PathsFn
    outputs: PathsFn
    # Where it runs: "cpu", "gpu", "conda:<env>" or "gpu+conda:<env>" (see _NEEDS_RE).
    needs: str = "cpu"
    # Inputs that may be missing without failing the step (fallbacks, optional seeds).
    optional_inputs: PathsFn = field(default=lambda layout, args: [])

    def __post_init__(self) -> None:
        if not _NEEDS_RE.match(self.needs):
            raise ValueError(f"step {self.name}: needs={self.needs!r}; use cpu, gpu, conda:<env> or gpu+conda:<env>")

    @property
    def gpu(self) -> bool:
        """True when the step needs a visible CUDA device."""
        return self.needs == "gpu" or self.needs.startswith("gpu+")

    @property
    def conda_env(self) -> str | None:
        """The model conda env the step's worker runs in, if any."""
        return self.needs.split("conda:", 1)[1] if "conda:" in self.needs else None


# Run order. The number is the step in docs/pipeline.md.
STEP_MODULES: dict[str, tuple[str, str]] = {
    "inputs.episode": ("0a", "r2s.gen.episode"),
    "inputs.segment": ("0c-1", "r2s.gen.segment"),
    "inputs.physics": ("0c-2", "r2s.gen.physics"),
    "inputs.superres": ("0c-3", "r2s.gen.superres"),
    "inputs.inpaint": ("0c-4", "r2s.gen.inpaint"),
    "inputs.meshes": ("0c-5", "r2s.gen.meshes"),
    "inputs.place": ("0c-6", "r2s.gen.place"),
    "inputs.table_depth": ("0c-7", "r2s.gen.table_depth"),
    "inputs.mesh_sheet": ("0c-8", "r2s.gen.mesh_sheet"),
    "world.import": ("1a", "r2s.world.import_splat"),
    "world.collider": ("1b", "r2s.world.collider"),
    "align.pose": ("2", "r2s.align.pose_fit"),
    "align.scale": ("3", "r2s.align.scale_fit"),
    "align.check": ("3b", "r2s.align.depth_check"),
    "surface.plane": ("4", "r2s.surface.plane_fit"),
    "surface.trim": ("5", "r2s.surface.trim"),
    "surface.receiver": ("6", "r2s.surface.receiver_fit"),
    "collision.crop": ("7", "r2s.collision.crop"),
    "objects.dump": ("8a", "r2s.objects.dump"),
    "objects.posefit": ("8b", "r2s.objects.pose_fit"),
    "objects.refine": ("9", "r2s.objects.refine.pipeline"),
    "objects.freeze": ("10", "r2s.objects.freeze"),
}


# Steps that are not part of the reference chain; ``r2s pipeline`` skips them unless
# they are named as the first or last step of the range (or --include-optional).
OPTIONAL_STEPS: frozenset[str] = frozenset({"inputs.superres"})


def load_step(name: str) -> Step:
    if name not in STEP_MODULES:
        raise SystemExit(f"unknown step {name!r}; run `python -m r2s list`")
    module = importlib.import_module(STEP_MODULES[name][1])
    step = getattr(module, "STEP")
    assert step.name == name, f"{module.__name__}.STEP is named {step.name!r}, expected {name!r}"
    return step
