"""Step 0c-7 ``inputs.table_depth``: metric depth of the inpainted background + the background "object".

Port of scene_generation ``src/pipeline/align_inpaint_updated.py`` (env ``moge``,
worker ``r2s/gen/workers/align_table_depth.py``, Depth Anything V2 Metric Indoor
Large from the Hugging Face cache). Depth is predicted for the LaMa background
``scene_background/<s>/background.png`` and the real ``images0/1.jpg``; the
prediction of the background is affinely fitted to that of the real frame
(RANSAC, ``random_state=0``) and then scaled to ``depth/1.npy`` (trimmed LS)
on pixels away from the union of ``input_mask/<s>/*_mask.png``.

Writes, under ``DROID/<s>/``: ``inpainted_depth.npy``, ``whole_mask.png``,
``masks/image_<n>_mask.png`` (background RGB) and ``pcds/<n>_frame1.ply``
(its unprojection with the new depth), ``n`` = number of objects in
``masks/result.json``.

Deviation: the old script used ``n = max(existing image_*_mask.png) + 1``, so each
re-run added another background mask/cloud (and later steps picked up the wrong
one). Here ``n`` is fixed by ``result.json`` and re-runs overwrite.
Verified bit-identical (inpainted_depth.npy, whole_mask.png, background mask
and cloud) against production for droid820 and droid2. droid2/droid34 also carry
a ``table`` entry in ``result.json`` from an earlier run of the older
``align_inpaint.py``; ``object_counts`` skips it, and this step never edits
``result.json``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from r2s.common.io import prepare_output, read_json
from r2s.gen.conda import run_in_env, worker
from r2s.gen.meshes import background_index, object_mask, prepare_worker_dir
from r2s.layout import InputLayout, SceneLayout
from r2s.steps import Step


def input_mask_dir(io: InputLayout) -> Path:
    """``<scene_data>/input_mask/<s>``: the per-object inpainting masks given to LaMa."""
    return io.root / "input_mask" / io.scene


def whole_mask(io: InputLayout) -> Path:
    return io.meta_dir / "whole_mask.png"


def background_pcd(io: InputLayout, index: int) -> Path:
    return io.meta_dir / "pcds" / f"{index:04d}_frame1.ply"


def _index_or_none(io: InputLayout) -> int | None:
    return background_index(io) if io.mask_result.exists() else None


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="cuda")


def inputs(layout: SceneLayout, args: argparse.Namespace) -> list[Path]:
    io = layout.inputs_out
    return [io.mask_result, io.background_png, io.images_dir / "1.jpg", io.depth_dir / "1.npy",
            input_mask_dir(io), io.intrinsics, io.extrinsics]


def fit_summary(io: InputLayout) -> Path:
    """New: the fitted scale/shift (the old script only printed them)."""
    return io.meta_dir / "inpainted_depth_fit.json"


def outputs(layout: SceneLayout, args: argparse.Namespace) -> list[Path]:
    io = layout.inputs_out
    files = [io.inpainted_depth, whole_mask(io), fit_summary(io)]
    n = _index_or_none(io)
    if n is not None:
        files += [object_mask(io, n), background_pcd(io, n)]
    return files


def align_table_depth(layout: SceneLayout, args: argparse.Namespace) -> dict:
    io = layout.inputs_out
    n = background_index(io)
    if not any(input_mask_dir(io).glob("*_mask.png")):
        raise FileNotFoundError(f"no *_mask.png in {input_mask_dir(io)}")
    root = layout.paths.scene_data_out
    for d in (io.meta_dir, io.masks_dir, io.meta_dir / "pcds"):
        prepare_worker_dir(d, root)
    for p in (io.inpainted_depth, whole_mask(io), object_mask(io, n), background_pcd(io, n)):
        prepare_output(p)
    summary = prepare_output(fit_summary(io))

    run_in_env(
        layout.paths,
        "moge",
        [
            worker("align_table_depth.py"),
            "--scene-dir", io.meta_dir,
            "--background", io.background_png,
            "--image", io.images_dir / "1.jpg",
            "--depth", io.depth_dir / "1.npy",
            "--inpaint-mask-dir", input_mask_dir(io),
            "--index", str(n),
            "--masks-dir", io.masks_dir,
            "--pcds-dir", io.meta_dir / "pcds",
            "--out-depth", io.inpainted_depth,
            "--out-json", summary,
            "--device", args.device,
        ],
    )
    fit = read_json(summary)
    return {"background_index": n, "scale": fit["scale"], "shift": fit["shift"]}


STEP = Step(
    name="inputs.table_depth",
    help="Metric depth of the inpainted background (Depth Anything V2) + background mask/point cloud",
    add_arguments=add_arguments,
    run=align_table_depth,
    inputs=inputs,
    outputs=outputs,
    needs="gpu+conda:moge",
)
