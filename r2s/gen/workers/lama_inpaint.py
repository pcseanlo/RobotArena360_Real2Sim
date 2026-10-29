#!/usr/bin/env python3
"""Step 0c-4 worker: sequential LaMa background inpainting (env ``lama``, Python 3.6).

Moved from SG ``lama/bin/predict.py`` (the scene_generation fork). Same model
(``<model-dir>/config.yaml`` + ``models/best.ckpt``), same prediction config
(``configs/prediction/default.yaml``: dataset kind ``default``, ``img_suffix .png``,
``pad_out_to_modulo 8``, ``out_key inpainted``, no refinement) and the same
sequential semantics: masks are taken in sorted order, the result of mask ``i`` is
written as ``image_{i+1:04d}.png`` and the dataset is re-globbed, so every step
inpaints the previous result; the last result is the background.

Differences from SG (plumbing only):
- no hydra: the two configs are read directly and every path is a CLI argument,
  so nothing depends on hydra's run-dir cwd (SG wrote
  ``../../../../data/scene_background/<s>/background.png`` relative to it);
- the masks and one copy of the anchor per mask are staged in ``--work-dir`` and
  every intermediate image is written there (SG overwrote ``input_mask/<s>/``);
- the empty ``outdir`` and the ``./out_mask/original*.png`` debug write are gone;
- ``--device`` (default ``cpu``, as SG hard-coded) and ``--threads``.

Like SG, the ``*_NUM_THREADS`` variables are set only after torch is imported, so
they do not cap torch's intra-op threads; ``--threads N`` sets them explicitly.

Python 3.6: no ``from __future__ import annotations``, dataclasses or walrus here.
"""

import argparse
import glob
import json
import os
import shutil
import sys


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--lama-dir", required=True, help="LaMa checkout (saicinpainting/ + configs/prediction/)")
    p.add_argument("--model-dir", required=True, help="big-lama folder (config.yaml + models/)")
    p.add_argument("--checkpoint", default=None, help="checkpoint name in <model-dir>/models (default: prediction config)")
    p.add_argument("--anchor-image", required=True, help="image the first mask is inpainted on")
    p.add_argument("--mask", action="append", required=True, help="mask PNG (repeat; processed in sorted order)")
    p.add_argument("--out", required=True, help="background.png to write")
    p.add_argument("--work-dir", required=True, help="where masks, anchor copies and intermediates are staged")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=None, help="torch intra-op threads (default: torch default, as SG)")
    p.add_argument("--summary", default=None, help="write a JSON summary here")
    return p.parse_args(argv)


def stage_inputs(anchor_image, masks, work_dir):
    """Lay out ``work_dir`` exactly like SG's ``input_mask/<s>/`` before LaMa ran.

    SG's segmentation saved the anchor next to every mask as ``image_k.png``; the
    LaMa dataset pairs ``image_k_mask.png`` with ``image_k.png``.
    """
    from PIL import Image

    if os.path.isdir(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir)
    names = [os.path.basename(m) for m in masks]
    if len(set(names)) != len(names):
        raise SystemExit("mask file names must be unique: %s" % names)
    anchor = Image.open(anchor_image)
    staged = []
    for src, name in zip(masks, names):
        if "_mask" not in name or not name.endswith(".png"):
            raise SystemExit("mask files must be named *_mask*.png like SG's input_mask: %s" % name)
        dst = os.path.join(work_dir, name)
        shutil.copyfile(src, dst)
        image_name = name.rsplit("_mask", 1)[0] + ".png"
        anchor.save(os.path.join(work_dir, image_name), "PNG")
        staged.append(dst)
    return staged


def main(argv=None):
    args = parse_args(argv)
    sys.path.insert(0, os.path.abspath(args.lama_dir))

    import logging

    from saicinpainting.evaluation.utils import move_to_device  # imports torch, like SG

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    import cv2
    import numpy as np
    import torch
    import tqdm
    import yaml
    from omegaconf import OmegaConf
    from torch.utils.data._utils.collate import default_collate

    from saicinpainting.training.data.datasets import make_default_val_dataset
    from saicinpainting.training.trainers import load_checkpoint

    logging.basicConfig(level=logging.WARNING)
    if args.threads is not None:
        torch.set_num_threads(args.threads)

    predict_config = OmegaConf.load(os.path.join(args.lama_dir, "configs", "prediction", "default.yaml"))
    checkpoint = args.checkpoint or predict_config.model.checkpoint
    device = torch.device(args.device)

    train_config_path = os.path.join(args.model_dir, "config.yaml")
    with open(train_config_path, "r") as f:
        train_config = OmegaConf.create(yaml.safe_load(f))
    train_config.training_model.predict_only = True
    train_config.visualizer.kind = "noop"

    out_ext = predict_config.get("out_ext", ".png")
    assert out_ext == ".png", out_ext
    assert not predict_config.get("refine", False), "refinement is not ported (SG never enabled it)"

    checkpoint_path = os.path.join(args.model_dir, "models", checkpoint)
    model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location="cpu")
    model.freeze()
    model.to(device)

    masks = sorted(os.path.abspath(m) for m in args.mask)
    work_dir = os.path.abspath(args.work_dir)
    stage_inputs(os.path.abspath(args.anchor_image), masks, work_dir)
    indir = work_dir + "/"
    dataset_kwargs = dict(predict_config.dataset)
    print("Processing images from ", indir, "torch threads:", torch.get_num_threads())

    dataset = make_default_val_dataset(indir, **dataset_kwargs)
    order = [os.path.basename(f) for f in dataset.mask_filenames]
    cur_res = None
    steps = []
    for img_i in tqdm.trange(len(dataset)):
        mask_fname = dataset.mask_filenames[img_i]
        batch = default_collate([dataset[img_i]])
        with torch.no_grad():
            batch = move_to_device(batch, device)
            batch["mask"] = (batch["mask"] > 0) * 1
            batch = model(batch)
            cur_res = batch[predict_config.out_key][0].permute(1, 2, 0).detach().cpu().numpy()
            unpad_to_size = batch.get("unpad_to_size", None)
            if unpad_to_size is not None:
                orig_height, orig_width = unpad_to_size
                cur_res = cur_res[:orig_height, :orig_width]

        cur_res = np.clip(cur_res * 255, 0, 255).astype("uint8")
        cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)

        next_image = os.path.join(indir, "image_%04d.png" % (img_i + 1))
        cv2.imwrite(next_image, cur_res)
        steps.append({"mask": os.path.basename(mask_fname), "image": os.path.basename(dataset.img_filenames[img_i]),
                      "result": os.path.basename(next_image)})
        dataset = make_default_val_dataset(indir, **dataset_kwargs)

    if cur_res is None:
        raise SystemExit("no masks to inpaint")
    out = os.path.abspath(args.out)
    if os.path.islink(out):
        os.unlink(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cv2.imwrite(out, cur_res)
    print("background:", out)
    if args.summary:
        with open(args.summary, "w") as f:
            json.dump({"order": order, "steps": steps, "out": out, "device": str(device),
                       "torch_threads": torch.get_num_threads(), "checkpoint": checkpoint_path}, f, indent=2)


if __name__ == "__main__":
    main()
