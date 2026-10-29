#!/usr/bin/env python3
"""Step 0c-3 worker: InvSR super-resolution (env ``invsr``, GPU).

The two InvSR blocks of SG ``bash_scripts/pipeline_droid_test.sh`` (both gated by
``use_bypass == false && use_molmo == false``; skipped in every production run
because ``use_molmo`` was ``null``):

``--stage anchor`` (before segmentation)
    ``images0/1.jpg`` -> InvSR (``--num_steps 2``) -> ffmpeg ``scale=W:H`` to the
    size of ``images0/0.jpg`` -> ``images0/super.png``. Segmentation then uses
    ``super.png`` as its anchor.
``--stage masks`` (after LaMa)
    every image in ``masks/`` is super-resolved in place and scaled back to W x H
    with ffmpeg, then ``masks/cropped/`` -> InvSR -> ``asset_images/<s>/enhanced/``.

InvSR runs from its checkout (``cwd``: its configs and weights are relative). The
anchor stage stages its input in ``--work-dir`` instead of ``images0/super_res``.
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile


def run_invsr(invsr_dir, in_dir, out_dir, num_steps):
    cmd = [sys.executable, os.path.join(invsr_dir, "inference_invsr.py"),
           "-i", in_dir, "-o", out_dir, "--num_steps", str(num_steps)]
    print("[invsr]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=invsr_dir, check=True)


def ffmpeg_scale(src, dst, w, h):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("ffmpeg not found on PATH")
    subprocess.run([ffmpeg, "-loglevel", "error", "-i", src, "-vf", f"scale={w}:{h}", "-y", dst], check=True)


def image_size(path):
    from PIL import Image

    with Image.open(path) as im:
        return im.size


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--invsr-dir", required=True)
    p.add_argument("--stage", choices=("anchor", "masks"), default="anchor")
    p.add_argument("--meta-dir", required=True, help="<scene_data>/DROID/<s>")
    p.add_argument("--work-dir", required=True)
    p.add_argument("--enhanced-dir", default=None, help="masks stage: asset_images/<s>/enhanced")
    p.add_argument("--num-steps", type=int, default=2)
    args = p.parse_args(argv)

    images0 = os.path.join(args.meta_dir, "images0")
    w, h = image_size(os.path.join(images0, "0.jpg"))
    os.makedirs(args.work_dir, exist_ok=True)

    if args.stage == "anchor":
        sr = os.path.join(args.work_dir, "super_res")
        if os.path.isdir(sr):
            shutil.rmtree(sr)
        os.makedirs(sr)
        shutil.copyfile(os.path.join(images0, "1.jpg"), os.path.join(sr, "1.jpg"))
        run_invsr(args.invsr_dir, sr, sr, args.num_steps)
        out = os.path.join(images0, "super.png")
        if os.path.islink(out):
            os.unlink(out)
        ffmpeg_scale(os.path.join(sr, "1.png"), out, w, h)
        print("[invsr] wrote", out)
        return

    mask_dir = os.path.join(args.meta_dir, "masks")
    run_invsr(args.invsr_dir, mask_dir, mask_dir, args.num_steps)
    for img in sorted(glob.glob(os.path.join(mask_dir, "*.png"))):
        fd, tmp = tempfile.mkstemp(suffix=".png", dir=args.work_dir)
        os.close(fd)
        ffmpeg_scale(img, tmp, w, h)
        shutil.move(tmp, img)
    if not args.enhanced_dir:
        raise SystemExit("--stage masks needs --enhanced-dir")
    run_invsr(args.invsr_dir, os.path.join(mask_dir, "cropped"), args.enhanced_dir, args.num_steps)


if __name__ == "__main__":
    main()
