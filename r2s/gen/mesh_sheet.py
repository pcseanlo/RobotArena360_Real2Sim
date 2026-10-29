"""Step 0c-8 ``inputs.mesh_sheet``: textured sheet mesh of the background from the inpainted depth.

Port of scene_generation ``src/pipeline/sheet_wraping_updated.py`` with the
production flags (``--depth_source aligned``, ``--target_tris 200000``; the
config had no ``sheet_wrap:`` block so ``beta 10``, ``alpha_threshold 0.15``,
``depth_eps 1e-4``). Env ``match``, worker ``r2s/gen/workers/sheet_wrap.py``.
Needs a GPU: SG's ``get_filtered_mask`` and ``sheet_warping_helper`` default to
``device="cuda"``.

Every pixel of ``inpainted_depth.npy`` becomes a vertex (pinhole unprojection
with ``intrinsics.npy``), depth discontinuities are cut with a Sobel filter on
disparity (DROID scenes only), small boundary spikes are bridged, the mesh is
cleaned and decimated to 200k triangles, coloured from the background mask
image and rotated 180 deg about X and Y. Output: ``DROID/<s>/mesh_sheet.glb``.

Deviation: the colour image is the background mask written by
``inputs.table_depth`` (``masks/image_<n>_mask.png``, ``n`` = number of objects),
passed explicitly; the old script took the last sorted ``masks/*_mask.png``,
which is the same file unless stale background masks were left behind.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from r2s.common.io import prepare_output
from r2s.gen.conda import run_in_env, worker
from r2s.gen.meshes import background_index, object_mask
from r2s.layout import SceneLayout
from r2s.steps import Step


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--target-tris", type=int, default=200_000, help="quadric decimation target (<=0: off)")
    parser.add_argument("--beta", type=float, default=10.0, help="disparity Sobel decay")
    parser.add_argument("--alpha-threshold", type=float, default=0.15, help="disparity edge cut threshold")
    parser.add_argument("--depth-eps", type=float, default=1e-4, help="minimum valid depth")
    parser.add_argument("--dataset", default=None,
                        help="old --dataset value (default: the scene_type); the edge filter runs only for DROID")


def _rgb(layout: SceneLayout) -> Path | None:
    io = layout.inputs_out
    return object_mask(io, background_index(io)) if io.mask_result.exists() else None


def inputs(layout: SceneLayout, args: argparse.Namespace) -> list[Path]:
    io = layout.inputs_out
    files = [io.mask_result, io.inpainted_depth, io.intrinsics]
    rgb = _rgb(layout)
    return files + ([rgb] if rgb is not None else [])


def outputs(layout: SceneLayout, args: argparse.Namespace) -> list[Path]:
    return [layout.inputs_out.mesh_sheet]


def build_mesh_sheet(layout: SceneLayout, args: argparse.Namespace) -> dict:
    io = layout.inputs_out
    rgb = _rgb(layout)
    out = prepare_output(io.mesh_sheet)
    run_in_env(
        layout.paths,
        "match",
        [
            worker("sheet_wrap.py"),
            "--rgb", rgb,
            "--depth", io.inpainted_depth,
            "--intrinsics", io.intrinsics,
            "--out", out,
            "--dataset", args.dataset or layout.config.scene_type,
            "--target_tris", str(args.target_tris),
            "--beta", str(args.beta),
            "--alpha_threshold", str(args.alpha_threshold),
            "--depth_eps", str(args.depth_eps),
        ],
    )
    import trimesh

    mesh = trimesh.load(str(out), force="mesh")
    return {"rgb": str(rgb), "vertices": int(len(mesh.vertices)), "faces": int(len(mesh.faces))}


STEP = Step(
    name="inputs.mesh_sheet",
    help="Sheet-wrap the inpainted background depth into DROID/<s>/mesh_sheet.glb",
    add_arguments=add_arguments,
    run=build_mesh_sheet,
    inputs=inputs,
    outputs=outputs,
    needs="gpu+conda:match",
)
