"""Step 1b ``world.collider``: collision mesh of the background from the Gaussian splat.

Port of the user's ply2glb tool (``run.sh`` -> ``ply2glb.py``, 2026-05-17): the
mesher is ``r2s/world/poisson_mesher.py`` (the original code, unchanged) and runs in
the repo env, so neither the tool nor its conda env is needed. Both presets are
screened Poisson on the Gaussian centres (opacity > threshold, largest 0.5% of
splats dropped), density trim, Taubin smoothing, small-component removal, quadric
decimation and trimesh hole filling:

  surface (default)  thinnest-Gaussian-axis normals, depth 10, k 30, density q 0.05,
                     200k faces, min cluster 500, Taubin 8
  planar             PCA normals, opacity > 0.15, depth 9, k 60, density q 0.02,
                     150k faces, min cluster 500, Taubin 20, ``--meshfix``

``--meshfix`` is a no-op: ``pymeshfix`` is not installed (it was not in the old
``ply2glb`` env either), so the mesher prints "pymeshfix not installed - skip".
Every production planar collider was made that way; the flag is passed anyway to
keep the preset identical to run.sh.

The mode comes from ``configs/scenes.yaml`` (``world.collider: {mode: planar}``).
Input ``world/gaussian.ply`` (step 1a), output ``world/collider.glb``.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from r2s.common.io import prepare_output
from r2s.layout import SceneLayout
from r2s.steps import Step

MESHER = "r2s.world.poisson_mesher"

# run.sh presets (2026-05-17), flags verbatim.
PRESETS: dict[str, list[str]] = {
    "surface": [
        "--method", "poisson",
        "--poisson-normal-source", "gaussian-axis",
        "--poisson-depth", "10",
        "--poisson-normal-k", "30",
        "--poisson-density-quantile", "0.05",
        "--target-faces", "200000",
        "--min-cluster", "500",
        "--taubin-iters", "8",
    ],
    "planar": [
        "--method", "poisson",
        "--opacity-thresh", "0.15",
        "--poisson-depth", "9",
        "--poisson-normal-k", "60",
        "--poisson-density-quantile", "0.02",
        "--target-faces", "150000",
        "--min-cluster", "500",
        "--taubin-iters", "20",
        "--meshfix",
    ],
}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mode", choices=sorted(PRESETS), default="surface",
                        help="collider preset (per scene in scenes.yaml)")


def collider_argv(gaussian_ply: Path, collider_glb: Path, mode: str) -> list[str]:
    """Python arguments of one mesher run (a child process: Open3D's threads and memory stay out of r2s)."""
    return ["-m", MESHER, str(gaussian_ply), str(collider_glb), *PRESETS[mode]]


def build_collider(layout: SceneLayout, args: argparse.Namespace) -> dict:
    out = prepare_output(layout.collider_glb)
    cmd = [sys.executable, *collider_argv(layout.gaussian_ply, out, args.mode)]
    print(f"[r2s:world.collider] {shlex.join(cmd)}", file=sys.stderr, flush=True)
    subprocess.run(cmd, check=True)
    import trimesh

    mesh = trimesh.load(str(out), force="mesh")
    return {
        "mode": args.mode,
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
    }


STEP = Step(
    name="world.collider",
    help="Poisson collision mesh (GLB) of the splat with the scene's collider preset",
    add_arguments=add_arguments,
    run=build_collider,
    inputs=lambda layout, args: [layout.gaussian_ply],
    outputs=lambda layout, args: [layout.collider_glb],
    needs="cpu",
)
