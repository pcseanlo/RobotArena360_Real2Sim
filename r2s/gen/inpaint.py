"""Step 0c-4 ``inputs.inpaint``: LaMa background inpainting -> ``scene_background/<s>/background.png``.

Port of SG's LaMa fork (``<scene_generation>/lama/bin/predict.py`` with
``big-lama`` + ``best.ckpt``, prediction config ``configs/prediction/default.yaml``:
``pad_out_to_modulo 8``, ``out_key inpainted``, no refinement, CPU). The worker
``r2s/gen/workers/lama_inpaint.py`` runs in the ``lama`` env (Python 3.6, torch 1.10).

Inputs: the masks ``<scene_data_out>/input_mask/<s>/**/*mask*.png`` written by
``inputs.segment`` (one per object + the robot/human union) and the anchor
``image_0000.png`` next to them. Masks are processed in sorted order and each
result is the input of the next, so the background has every object and the
robot/human removed. Intermediates stay in ``<work>/scenes/<s>/gen/inpaint/``
(SG overwrote ``input_mask/<s>/image_k.png`` for k >= 1 with them).

Determinism: LaMa on CPU is deterministic, but MKL/oneDNN pick ISA-specific
kernels, so the bits depend on the CPU family. Run on the old run's family the
background is byte-identical (droid820: Intel Xeon AVX-512, any ``--threads``;
droid2: AMD EPYC without AVX-512). Across families it differs by at most 1 grey
level on ~150 of 230k pixels after six chained masks.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

from r2s.common.io import prepare_output
from r2s.gen.conda import run_in_env, worker
from r2s.gen.episode import input_mask_dir, output_inputs
from r2s.layout import SceneLayout
from r2s.steps import Step


def mask_files(mask_dir: Path) -> list[Path]:
    # LaMa's InpaintingDataset glob, so the order is the old one.
    return [Path(p) for p in sorted(glob.glob(os.path.join(str(mask_dir), "**", "*mask*.png"), recursive=True))]


def anchor_path(layout: SceneLayout, args: argparse.Namespace) -> Path:
    if args.anchor:
        return Path(args.anchor)
    masks = mask_files(input_mask_dir(layout.inputs_out))
    first = masks[0] if masks else input_mask_dir(layout.inputs_out) / "image_0000_mask.png"
    return first.with_name(first.name.rsplit("_mask", 1)[0] + ".png")


def work_dir(layout: SceneLayout) -> Path:
    return layout.root / "gen" / "inpaint"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--anchor", default=None,
                        help="image the first mask is applied to (default: input_mask/<s>/image_0000.png)")
    parser.add_argument("--device", default="cpu", help="torch device (SG: cpu)")
    parser.add_argument("--threads", type=int, default=None, help="torch intra-op threads (default: torch's, as SG)")
    parser.add_argument("--checkpoint", default="best.ckpt", help="checkpoint in big-lama/models")


def run(layout: SceneLayout, args: argparse.Namespace) -> dict:
    out = output_inputs(layout)
    masks = mask_files(input_mask_dir(out))
    if not masks:
        raise FileNotFoundError(f"no *mask*.png in {input_mask_dir(out)}; run inputs.segment first")
    lama = layout.paths.scene_generation / "lama"
    work = work_dir(layout)
    summary = prepare_output(work / "summary.json")
    background = prepare_output(out.background_png)
    cmd = [worker("lama_inpaint.py"), "--lama-dir", lama, "--model-dir", lama / "big-lama",
           "--checkpoint", args.checkpoint, "--anchor-image", anchor_path(layout, args),
           "--out", background, "--work-dir", work / "steps", "--device", args.device, "--summary", summary]
    for m in masks:
        cmd += ["--mask", m]
    if args.threads:
        cmd += ["--threads", str(args.threads)]
    # TORCH_HOME only matters if LaMa wanted to download something; keep that out of SG.
    run_in_env(layout.paths, "lama", cmd, extra_env={"TORCH_HOME": str(work / "torch_home")},
               log_file=work / "worker.log")
    s = json.loads(summary.read_text())
    return {"masks": s["order"], "torch_threads": s["torch_threads"], "device": s["device"]}


def _inputs(layout: SceneLayout, args: argparse.Namespace) -> list[Path]:
    lama = layout.paths.scene_generation / "lama"
    return [input_mask_dir(layout.inputs_out), anchor_path(layout, args),
            lama / "big-lama" / "config.yaml", lama / "big-lama" / "models" / args.checkpoint]


STEP = Step(
    name="inputs.inpaint",
    help="Sequential LaMa inpainting of the object + robot/human masks -> scene_background/<s>/background.png",
    add_arguments=add_arguments,
    run=run,
    inputs=_inputs,
    outputs=lambda layout, args: [layout.inputs_out.background_png],
    needs="conda:lama",
)
