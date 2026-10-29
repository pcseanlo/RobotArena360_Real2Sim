"""Seed ``paths.scene_data_out`` with the OLD upstream inputs of steps 0c-5 .. 0c-8.

Each step is then verified on exactly the inputs production used, so the
comparison isolates that step:

    inputs.place        old assets/<s>/image_<k>_segmented.glb (symlinked), object
                        masks + pcds, result.json, episode frame/depth/calibration
                        (+ old inpainted_depth.npy with --with-inpainted-depth)
    inputs.table_depth  scene_background/<s>/background.png, input_mask/<s>/
    inputs.mesh_sheet   chained on the new table_depth outputs
    inputs.meshes       object masks + images0/1.jpg (run it LAST: it replaces the
                        seeded GLB symlinks with new meshes)

Small files are copied; big episode files and the old GLBs are symlinked (every
step writes through ``prepare_output``, which unlinks a symlink first). The
background mask/cloud, transformations.json, inpainted depth and meshes are NOT
seeded -- they are what the steps produce.

    R2S_WORK=... python tools/legacy/gen_b_seed.py --scene droid820
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from r2s.config import Paths, SceneRegistry  # noqa: E402
from r2s.gen.meshes import object_counts, object_mask, segmented_name  # noqa: E402
from r2s.gen.table_depth import input_mask_dir  # noqa: E402
from r2s.layout import SceneLayout  # noqa: E402


def _copy(src: Path, dst: Path, seeded: list[Path]) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    shutil.copy2(src, dst)
    seeded.append(dst)


def _link(src: Path, dst: Path, seeded: list[Path]) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())
    seeded.append(dst)


def seed(layout: SceneLayout, with_inpainted_depth: bool = False) -> list[Path]:
    old, new = layout.inputs_src, layout.inputs_out
    if old.root.resolve() == new.root.resolve():
        raise SystemExit("scene_data_out is the read-only scene_data folder; set R2S_WORK or R2S_SCENE_DATA_OUT")
    seeded: list[Path] = []
    counts = object_counts(old.mask_result)

    # episode (read only by every step)
    _link(old.images_dir / "1.jpg", new.images_dir / "1.jpg", seeded)
    _link(old.depth_dir / "1.npy", new.depth_dir / "1.npy", seeded)
    for p in (old.intrinsics, old.extrinsics, old.lang):
        if p.exists():
            _copy(p, new.meta_dir / p.name, seeded)

    # segmentation (0c-1) and physics (0c-2) outputs; super-resolved masks (0c-3)
    _copy(old.mask_result, new.mask_result, seeded)
    for name in ("physical_properties.json",):
        if (old.masks_dir / name).exists():
            _copy(old.masks_dir / name, new.masks_dir / name, seeded)
    for k in counts:
        _copy(object_mask(old, k), object_mask(new, k), seeded)
        _copy(old.meta_dir / "pcds" / f"{k:04d}_frame1.ply", new.meta_dir / "pcds" / f"{k:04d}_frame1.ply", seeded)

    # inpainting (0c-4) outputs
    _copy(old.background_png, new.background_png, seeded)
    for p in sorted(input_mask_dir(old).iterdir()):
        if p.is_file():
            _copy(p, input_mask_dir(new) / p.name, seeded)

    # old meshes (0c-5) so inputs.place sees production's GLBs
    for k in counts:
        _link(old.assets_dir / f"{segmented_name(k)}.glb", new.assets_dir / f"{segmented_name(k)}.glb", seeded)

    if with_inpainted_depth:
        _copy(old.inpainted_depth, new.inpainted_depth, seeded)
    return seeded


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--with-inpainted-depth", action="store_true",
                    help="also copy the old inpainted_depth.npy (for inputs.place --background-depth inpainted)")
    args = ap.parse_args()
    layout = SceneLayout.create(args.scene, Paths.load(), SceneRegistry())
    for p in seed(layout, args.with_inpainted_depth):
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
