"""Worker for ``inputs.table_depth`` (env ``moge``): metric depth of the inpainted background.

Port of scene_generation ``src/pipeline/align_inpaint_updated.py`` ``__main__``:
Depth Anything V2 Metric Indoor on the inpainted background and on the real
frame, RANSAC affine fit (random_state 0) on pixels outside the union of the
inpaint masks, trimmed-LS scale to the sensor depth, then the whole-image
"background object" mask, its masked RGB and its unprojected point cloud.

Writes ``--out-depth`` (inpainted_depth.npy), ``<scene>/whole_mask.png``,
``<masks>/image_<index>_mask.png`` and ``<pcds>/<index>_frame1.ply``.
"""
import argparse
import json
import sys
from pathlib import Path

# sg_glue lives in r2s/gen; append (not prepend) so it cannot shadow env packages.
sys.path.append(str(Path(__file__).resolve().parents[1]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene-dir", required=True, help="DROID/<s> (intrinsics.npy, extrinsics.npy; whole_mask.png goes here)")
    ap.add_argument("--background", required=True, help="scene_background/<s>/background.png (LaMa output)")
    ap.add_argument("--image", required=True, help="DROID/<s>/images0/1.jpg")
    ap.add_argument("--depth", required=True, help="DROID/<s>/depth/1.npy")
    ap.add_argument("--inpaint-mask-dir", required=True, help="input_mask/<s> (union of *_mask.png)")
    ap.add_argument("--index", type=int, required=True, help="index of the background object")
    ap.add_argument("--masks-dir", required=True)
    ap.add_argument("--pcds-dir", required=True)
    ap.add_argument("--out-depth", required=True)
    ap.add_argument("--out-json", default=None, help="fit summary")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import numpy as np

    from sg_glue.align_inpaint import combine_masks, segment_and_unproject, transfer_metric_depth_two_stage_regression

    inpainted_mask = combine_masks(args.inpaint_mask_dir)
    metric_depth, scale, shift = transfer_metric_depth_two_stage_regression(
        args.depth, args.background, args.image, inpainted_mask, device=args.device
    )
    print(f"[align_inpaint_updated] depth-anything-v2 fit  scale={scale:.4f}  shift={shift:.4f}")
    written = segment_and_unproject(args.scene_dir, args.background, metric_depth, args.index, args.masks_dir, args.pcds_dir)
    np.save(args.out_depth, metric_depth)
    if args.out_json:
        Path(args.out_json).write_text(json.dumps({
            "scale": scale, "shift": shift, "index": args.index,
            "depth_min": float(np.min(metric_depth)), "depth_max": float(np.max(metric_depth)),
            "written": written + [args.out_depth],
        }, indent=2) + "\n")


if __name__ == "__main__":
    main()
