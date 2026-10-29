"""Step 0c-3 ``inputs.superres`` (optional, not in the production chain): InvSR super-resolution.

SG's driver (``bash_scripts/pipeline_droid_test.sh``) has two InvSR blocks, both
gated by ``use_bypass == false && use_molmo == false``. ``configs/DROID_config.yaml``
had ``use_molmo: null``, so the test ``"null" == "false"`` failed and **neither
block ran for any production scene** (droid2, droid820, ...). The chain
episode -> segment -> physics -> inpaint therefore skips this step; run it only to
reproduce the older scenes whose ``images0/super.png`` exists (droid26/32/34/35/37/45).

``--stage anchor`` (run before ``inputs.segment``): ``images0/1.jpg`` -> InvSR
(``--num_steps 2``, SG's checkout ``<scene_generation>/InvSR``, env ``invsr``, GPU)
-> ffmpeg ``scale`` to the size of ``images0/0.jpg`` -> ``images0/super.png``, which
segmentation then uses as its anchor (SG's ``_pick_image``).

``--stage masks`` (run after ``inputs.inpaint``): every image in ``masks/`` is
super-resolved in place and scaled back to the frame size, and ``masks/cropped/`` ->
``<scene_data_out>/asset_images/<s>/enhanced/`` (only the Hunyuan texturing path,
unused by sam3d, reads that folder). Note this rewrites ``masks/image_k_mask.png``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from r2s.gen.conda import run_in_env, worker
from r2s.gen.episode import output_inputs
from r2s.layout import SceneLayout
from r2s.steps import Step


def enhanced_dir(layout: SceneLayout) -> Path:
    out = layout.inputs_out
    return out.root / "asset_images" / out.scene / "enhanced"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--stage", choices=("anchor", "masks"), default="anchor")
    parser.add_argument("--num-steps", type=int, default=2, help="InvSR inference steps (SG: 2)")


def run(layout: SceneLayout, args: argparse.Namespace) -> dict:
    out = output_inputs(layout)
    invsr = layout.paths.scene_generation / "InvSR"
    work = layout.root / "gen" / "superres"
    cmd = [worker("superres_invsr.py"), "--invsr-dir", invsr, "--stage", args.stage, "--meta-dir", out.meta_dir,
           "--work-dir", work, "--num-steps", str(args.num_steps)]
    if args.stage == "masks":
        cmd += ["--enhanced-dir", enhanced_dir(layout)]
    # InvSR loads SD-turbo with cache_dir=<InvSR>/weights; offline, nothing is fetched or written there.
    run_in_env(layout.paths, "invsr", cmd, extra_env={"HF_HUB_OFFLINE": "1"}, log_file=work / f"worker_{args.stage}.log")
    return {"stage": args.stage}


def _inputs(layout: SceneLayout, args: argparse.Namespace) -> list[Path]:
    out = layout.inputs_out
    paths = [out.images_dir / "0.jpg", out.images_dir / "1.jpg", layout.paths.scene_generation / "InvSR"]
    if args.stage == "masks":
        paths += [out.masks_dir / "cropped"]
    return paths


def _outputs(layout: SceneLayout, args: argparse.Namespace) -> list[Path]:
    out = layout.inputs_out
    if args.stage == "anchor":
        return [out.images_dir / "super.png"]
    return [out.masks_dir, enhanced_dir(layout)]


STEP = Step(
    name="inputs.superres",
    help="(optional, skipped in production) InvSR super-resolution of the anchor (or the masks)",
    add_arguments=add_arguments,
    run=run,
    inputs=_inputs,
    outputs=_outputs,
    needs="gpu+conda:invsr",
)
