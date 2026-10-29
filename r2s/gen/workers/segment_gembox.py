#!/usr/bin/env python3
"""Step 0c-1 worker: Gemini boxes + SAM3 masks for one DROID scene (env ``gemini``).

The SG "gembox" variant (``src/pipeline/segment_object_gembox_updated.py``):
Gemini (``models/gemini-2.5-flash``) gets the anchor image center-padded to a
square three times (robot parts, human parts, foreground objects with
mass/friction/surface), each box + label prompts SAM3 (``facebook/sam3``), the
masks are persisted in SG's layout and every object mask becomes a point cloud.

Live mode needs ``$GEMINI_API_KEY`` and saves the raw replies to
``masks/gemini_answers.json``. ``--replay-gemini`` re-runs SAM3 and persistence
from saved answers without calling Gemini (see ``sg_glue/segment_replay.py``).

``--stage segment`` (Gemini + SAM3 + persistence) and ``--stage pcds`` (point clouds
from the persisted mask JSON) can run separately; r2s runs the point clouds with
``OMP_NUM_THREADS=1`` because Open3D 0.18's multi-threaded radius-outlier filter is
not deterministic (it drops a few random points per run).

Writes under ``--meta-dir``: ``masks/`` (image_k_mask.png, cropped/, result.json,
gemini_boxes_raw.json, gemini_boxes_debug.png), ``bounding_boxes_and_masks/``,
``pcds/``; and the LaMa inputs into ``--input-mask-dir``.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# sg_glue lives in r2s/gen; append (not prepend) so it cannot shadow env packages.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from sg_glue import gembox, pointcloud, segment_replay  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--meta-dir", required=True, help="<scene_data>/DROID/<s> to read the episode from and write into")
    p.add_argument("--input-mask-dir", required=True, help="where the LaMa inputs go (<scene_data>/input_mask/<s>)")
    p.add_argument("--anchor", default=None, help="anchor image (default: images0/super.png if present, else images0/1.jpg)")
    p.add_argument("--replay-gemini", default=None, help="gemini_answers.json or gemini_boxes_raw.json to replay")
    p.add_argument("--replay-result", default=None, help="result.json with the physics fields (gemini_boxes_raw.json replay)")
    p.add_argument("--model-name", default="models/gemini-2.5-flash")
    p.add_argument("--sam3-model", default="facebook/sam3")
    p.add_argument("--device", default=None, help="SAM3 device (default: cuda if available, else cpu)")
    p.add_argument("--frame-idx", type=int, default=1, help="frame of images0/ and depth/ for the point clouds")
    p.add_argument("--stage", choices=("all", "segment", "pcds"), default="all")
    p.add_argument("--summary", default=None, help="write (segment) or update (pcds) a JSON summary here")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    meta = os.path.abspath(args.meta_dir)
    mask_dir = os.path.join(meta, "masks")
    save_dict = os.path.join(meta, "bounding_boxes_and_masks")
    for d in (mask_dir, save_dict, os.path.join(meta, "pcds"), args.input_mask_dir):
        os.makedirs(d, exist_ok=True)
    if args.stage == "pcds":
        summary = {}
        if args.summary and os.path.exists(args.summary):
            with open(args.summary) as f:
                summary = json.load(f)
        summary["pcd_points"] = pointcloud.mask_to_point_cloud(meta, args.frame_idx)
        summary["pcd_omp_threads"] = os.environ.get("OMP_NUM_THREADS")
        if args.summary:
            with open(args.summary, "w") as f:
                json.dump(summary, f, indent=2)
        return
    anchor = args.anchor or gembox._pick_image(os.path.join(meta, "images0"))
    print(f"[gembox] anchor image: {anchor}")

    summary = {"anchor": anchor}
    answers = None
    if args.replay_gemini:
        prompts, notes, kind = segment_replay.load_replay(args.replay_gemini, args.replay_result)
        summary.update(mode="replay", replay_kind=kind, replay_notes=notes)
        for n in notes:
            print(f"[gembox:replay] {n}")
    else:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise SystemExit("live mode needs GEMINI_API_KEY in the environment (or pass --replay-gemini)")
        prompts, answers = gembox.gemini_prompts(anchor, api_key, args.model_name)
        answers["anchor_image"] = anchor
        summary.update(mode="live", model_name=args.model_name)

    seg = gembox.segment_from_prompts(
        anchor, prompts, mask_dir, save_dict, args.input_mask_dir,
        render_mask_dir=os.path.join(meta, "masks_render"),
        sam3_model_id=args.sam3_model, device=args.device,
    )
    summary.update(seg)
    if answers is not None:
        with open(os.path.join(mask_dir, "gemini_answers.json"), "w") as f:
            json.dump(answers, f, indent=2)

    if args.stage == "all":
        summary["pcd_points"] = pointcloud.mask_to_point_cloud(meta, args.frame_idx)
    if args.summary:
        with open(args.summary, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
