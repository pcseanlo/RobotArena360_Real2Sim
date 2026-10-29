"""Compare regenerated step-0 inputs (0a, 0c-1, 0c-2, 0c-4) with the old scene_generation files.

    R2S_WORK=... python tools/legacy/gen_a_compare.py --scene droid820 [--scene droid2]

Old = ``layout.inputs_src`` (``paths.scene_data``, read only), new = ``layout.inputs_out``.
Old folders also hold files later sub-steps added (the table/background mask
``masks/image_0005_mask.png``, ``pcds/0005_frame1.ply``, a ``table`` entry appended to
``result.json``); only what segmentation writes is compared, extras are listed.
The old ``input_mask/<s>/image_k.png`` for k >= 1 were overwritten by SG's LaMa run,
so only ``image_0000.png`` (the anchor) is compared.

Prints a markdown table and writes ``<work>/verify/gen_a_compare_<scene>.json``.
"""

from __future__ import annotations

import argparse
import filecmp
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from r2s.common.io import write_json  # noqa: E402
from r2s.config import Paths  # noqa: E402
from r2s.gen.episode import REQUIRED_FILES, input_mask_dir  # noqa: E402
from r2s.layout import SceneLayout  # noqa: E402


def _img(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path))


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(np.logical_and(a, b).sum() / union)


def _nonzero(img: np.ndarray) -> np.ndarray:
    return img.any(axis=-1) if img.ndim == 3 else img > 0


def ply_points(path: Path) -> np.ndarray:
    from plyfile import PlyData

    v = PlyData.read(str(path))["vertex"].data
    return np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)


def compare_scene(layout: SceneLayout) -> list[dict]:
    old, new = layout.inputs_src, layout.inputs_out
    rows: list[dict] = []

    def row(item, value, ok, note=""):
        rows.append({"item": item, "value": value, "ok": bool(ok), "note": note})

    # 0a episode ---------------------------------------------------------------------
    same = [n for n in REQUIRED_FILES if filecmp.cmp(old.meta_dir / n, new.meta_dir / n, shallow=False)]
    row("episode: camera/joints files identical", f"{len(same)}/{len(REQUIRED_FILES)}", len(same) == len(REQUIRED_FILES))
    n_old = len([p for p in old.images_dir.glob("*.jpg")])
    n_new = len([p for p in new.images_dir.glob("*.jpg")])
    row("episode: images0 frames", f"{n_new} (old {n_old})", n_new == n_old)

    # result.json ------------------------------------------------------------------------
    res_old, res_new = json.loads(old.mask_result.read_text()), json.loads(new.mask_result.read_text())
    n = len(res_new)
    extra = [e["object_name"] for e in res_old[n:]]
    row("result.json == old[:n]", f"n={n}", res_old[:n] == res_new,
        f"old extra entries from later steps: {extra}" if extra else "")
    row("result.json byte-identical", "", filecmp.cmp(old.mask_result, new.mask_result, shallow=False),
        "differs only by the later-appended entries" if extra and res_old[:n] == res_new else "")

    raw_old = json.loads((old.masks_dir / "gemini_boxes_raw.json").read_text())
    raw_new = json.loads((new.masks_dir / "gemini_boxes_raw.json").read_text())
    keys = [k for k in raw_old]
    same_raw = all(raw_old[k] == raw_new.get(k) for k in keys)
    missing = sorted(set(raw_new) - set(raw_old))
    row("gemini_boxes_raw.json (old keys)", f"{len(keys)} keys", same_raw,
        f"new-only keys {missing} (old file predates them)" if missing else "")
    row("gemini_boxes_debug.png pixels", "", np.array_equal(_img(old.masks_dir / "gemini_boxes_debug.png"),
                                                            _img(new.masks_dir / "gemini_boxes_debug.png")))

    # masks/image_k_mask.png (RGB-masked anchor) + crops --------------------------------
    for k in range(n):
        name = f"image_{k:04d}_mask.png"
        a, b = _img(old.masks_dir / name), _img(new.masks_dir / name)
        eq = a.shape == b.shape and np.array_equal(a, b)
        row(f"masks/{name} IoU", f"{iou(_nonzero(a), _nonzero(b)):.4f}", eq, "pixel-identical" if eq else
            f"{int((a != b).any(-1).sum())} px differ")
        c_old = _img(old.masks_dir / "cropped" / f"image_{k:04d}_segmented.png")
        c_new = _img(new.masks_dir / "cropped" / f"image_{k:04d}_segmented.png")
        row(f"masks/cropped/image_{k:04d}_segmented.png", f"{c_new.shape[:2]} (old {c_old.shape[:2]})",
            c_old.shape == c_new.shape and np.array_equal(c_old, c_new))
    later = sorted(p.name for p in old.masks_dir.glob("image_*_mask.png") if int(p.name.split("_")[1]) >= n)
    if later:
        row("old masks/ extra files", ", ".join(later), True, "written by later sub-steps; not compared")

    # bounding_boxes_and_masks -------------------------------------------------------------
    bb = "frame_00001_bounding_boxes_and_masks.json"
    bo, bn = json.loads((old.bbox_masks_dir / bb).read_text()), json.loads((new.bbox_masks_dir / bb).read_text())
    names_ok = [(o["object_name"], o["count"]) for o in bo] == [(o["object_name"], o["count"]) for o in bn]
    row("bbox json names/counts", f"{len(bn)} objects (old {len(bo)})", names_ok)
    for o, m in zip(bo, bn):
        ma, mb = np.asarray(o["mask"], bool), np.asarray(m["mask"], bool)
        row(f"bbox json mask {o['count']} ({o['object_name']}) IoU", f"{iou(ma, mb):.4f}", np.array_equal(ma, mb),
            "" if np.array_equal(ma, mb) else f"{int((ma != mb).sum())} px differ")
    same_bytes = filecmp.cmp(old.bbox_masks_dir / bb, new.bbox_masks_dir / bb, shallow=False)
    row("bbox json content equal", "byte-identical" if same_bytes else "", bo == bn,
        "" if same_bytes or bo != bn else "key order differs (pre-May-14 SG wrote count before mask)")

    # pcds -----------------------------------------------------------------------------------
    for p in sorted((new.meta_dir / "pcds").glob("*_frame1.ply")):
        q = old.meta_dir / "pcds" / p.name
        a, b = ply_points(q), ply_points(p)
        same = len(a) == len(b) and np.array_equal(a, b)
        sa, sb = {tuple(x) for x in a}, {tuple(x) for x in b}
        if same:
            note = "identical"
        elif sa <= sb:
            note = f"new = old + {len(sb - sa)} points (old lost them to Open3D's threaded filter)"
        else:
            note = f"{len(sa - sb)} old points missing, {len(sb - sa)} new points"
        row(f"pcds/{p.name} points", f"{len(b)} (old {len(a)})", same, note)

    # input_mask ---------------------------------------------------------------------------------
    im_old, im_new = input_mask_dir(old), input_mask_dir(new)
    masks_new = sorted(p.name for p in im_new.glob("*_mask.png"))
    masks_old = sorted(p.name for p in im_old.glob("*_mask.png"))
    row("input_mask mask files", f"{len(masks_new)} (old {len(masks_old)})", masks_new == masks_old)
    for name in masks_new:
        a, b = _img(im_old / name), _img(im_new / name)
        eq = a.shape == b.shape and np.array_equal(a, b)
        row(f"input_mask/{name} IoU", f"{iou(a > 0, b > 0):.4f}", eq, "pixel-identical" if eq else
            f"{int((a != b).sum())} px differ")
    row("input_mask/image_0000.png (anchor)", "", np.array_equal(_img(im_old / "image_0000.png"),
                                                                  _img(im_new / "image_0000.png")))

    # physics + background -------------------------------------------------------------------------
    pp_new = new.masks_dir / "physical_properties.json"
    if pp_new.exists():
        row("physical_properties.json byte-identical", "", filecmp.cmp(old.physical_properties, pp_new, shallow=False))
    if new.background_png.exists():
        a = _img(old.background_png).astype(int)
        b = _img(new.background_png).astype(int)
        d = np.abs(a - b)
        row("background.png", f"max |d| {int(d.max())}, {int((d > 0).any(-1).sum())} px differ",
            int(d.max()) == 0, "byte-identical" if filecmp.cmp(old.background_png, new.background_png, shallow=False)
            else "")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scene", action="append", required=True)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    paths = Paths.load(args.config)
    bad = 0
    for scene in args.scene:
        layout = SceneLayout.create(scene, paths)
        rows = compare_scene(layout)
        print(f"\n### {scene}\n\n| check | value | match | note |\n|---|---|---|---|")
        for r in rows:
            print(f"| {r['item']} | {r['value']} | {'yes' if r['ok'] else 'NO'} | {r['note']} |")
        bad += sum(not r["ok"] for r in rows)
        out = write_json(paths.work / "verify" / f"gen_a_compare_{scene}.json", rows)
        print(f"\n(written {out})")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
