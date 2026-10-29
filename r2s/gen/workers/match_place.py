"""Worker for ``inputs.place`` (env ``match``): place ONE segmented mesh in the scene.

Port of one ``python src/pipeline/MatchFindRot.py --mesh_cnt NNNN [--background True]``
call of the old driver (Genesis can be initialised once per process, so the
step runs this worker once per object). Writes the object's
``transformations.json`` entry to ``--out-json`` (``{"key": ..., "entry": ... | null}``;
``null`` = no keypoint survived filtering, the old script then wrote nothing)
and leaves the old debug files in ``<assets>/<NNNN>/``.

Randomness that was unseeded in the old runs (trimesh surface sampling via
``np.random``, Open3D plane RANSAC, OpenCV homography RANSAC, torch) is seeded
from ``--seed`` once at start-up.
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

# sg_glue lives in r2s/gen; append (not prepend) so it cannot shadow env packages.
sys.path.append(str(Path(__file__).resolve().parents[1]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene-dir", required=True, help="DROID/<s> (images0, depth, masks, pcds, intrinsics/extrinsics)")
    ap.add_argument("--assets-dir", required=True, help="assets/<s> (the segmented GLBs; debug folders go here)")
    ap.add_argument("--mesh-cnt", required=True, help="four-digit object index, e.g. 0003")
    ap.add_argument("--config", required=True, help="sg_glue/match_config.yaml")
    ap.add_argument("--dataset", default="droid", help="config 'dataset' value (droid / bridge)")
    ap.add_argument("--checkpoint-dir", required=True, help="<scene_generation>/MINIMA/weights")
    ap.add_argument("--scene-generation", required=True, help="scene_generation root (provides third_party/LightGlue)")
    ap.add_argument("--background", action="store_true", help="similarity fit + table.glb export (old --background True)")
    ap.add_argument("--background-depth", default=None, help="depth .npy used in background mode (default: depth/1.npy)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    # third_party.LightGlue.lightglue, imported exactly as production did.
    sys.path.append(str(Path(args.scene_generation).resolve()))

    import cv2
    import numpy as np
    import open3d as o3d
    import torch
    import yaml

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cv2.setRNGSeed(args.seed)
    o3d.utility.random.seed(args.seed)

    from sg_glue.match_find_rot import process_mesh

    config = yaml.safe_load(Path(args.config).read_text())
    config["dataset"] = args.dataset
    mesh_glb = f"image_{args.mesh_cnt}_segmented.glb"
    key, entry = process_mesh(
        args.scene_dir, args.assets_dir, mesh_glb, args.mesh_cnt, args.checkpoint_dir, config,
        background=args.background, background_depth_path=args.background_depth,
    )
    Path(args.out_json).write_text(json.dumps({"key": key, "entry": entry}, indent=2) + "\n")
    print(f"[match_place] {key}: {'no keypoints' if entry is None else 'ok'}")


if __name__ == "__main__":
    main()
