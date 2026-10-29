"""Step 0a ``inputs.episode``: bring a raw DROID episode into the step-0 output tree.

Links (default) or copies the raw episode files from a source folder into
``layout.inputs_out.meta_dir`` (``<scene_data_out>/DROID/<s>/``), so every later
sub-step (0c-1 ... 0c-8) reads and writes one self-contained folder.

Source: ``layout.inputs_src.meta_dir`` (``<scene_data>/DROID/<s>/``) by default, or
``--src`` (e.g. ``.../droid_selected_seiredata/scene_<N>``).

Taken: ``images0/<n>.jpg`` and ``depth/<n>.npy`` (frame files only, linked one by
one into real folders, so ``images0/super.png`` from an earlier superres run is
never picked up), ``intrinsics.npy``, ``extrinsics.npy``, ``joints.npy``,
``grippers.npy``, ``tcp.npy`` (required), ``lang.txt``, ``joints_old.npy``,
``metadata*.json``, ``language_annotations*.json`` (when present; the seiredata
sources have no ``lang.txt``). Never taken: anything a later sub-step writes
(``masks/``, ``pcds/``, ``bounding_boxes_and_masks/``, ``inpainted_depth.npy``,
``mesh_sheet.glb``, ``whole_mask.png``, ``super.png``) and the calibration
by-products no step reads (``extrinsics0.npy``, ``results.pkl``,
``tracker_data.pth``, ``depth_vis/``, videos). ``--extra GLOB`` adds more.

SG's segmentation rewrote ``intrinsics.npy`` as ``intrinsics[0]`` when it was not
3x3; that normalisation happens here (a real file is written instead of a link).
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path

import numpy as np

from r2s.common.io import prepare_output, save_npy
from r2s.layout import InputLayout, SceneLayout
from r2s.steps import Step

FRAME_DIRS = {"images0": re.compile(r"^\d+\.jpg$"), "depth": re.compile(r"^\d+\.npy$")}
REQUIRED_FILES = ("intrinsics.npy", "extrinsics.npy", "joints.npy", "grippers.npy", "tcp.npy")
OPTIONAL_FILES = ("lang.txt", "joints_old.npy")
OPTIONAL_GLOBS = ("metadata*.json", "language_annotations*.json")


# --------------------------------------------------------------------------- shared by the 0c steps
def input_mask_dir(inputs: InputLayout) -> Path:
    """``<root>/input_mask/<s>/``: LaMa inputs (SG's final location after the driver's move)."""
    return inputs.root / "input_mask" / inputs.scene


def output_inputs(layout: SceneLayout) -> InputLayout:
    """``layout.inputs_out``, refusing to write into scene_generation or its data folder."""
    out = layout.inputs_out
    root = out.root.resolve()
    sg = layout.paths.scene_generation.resolve()
    protected = {sg, (sg / "data").resolve()}
    for p in protected:
        if root == p or p in root.parents:
            raise SystemExit(f"step-0 output root {out.root} is inside read-only {p}; set R2S_SCENE_DATA_OUT / R2S_WORK")
    return out


def clear_dir(path: Path) -> None:
    """Remove a folder this step owns (a symlink is unlinked, never followed)."""
    path = Path(path)
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


# --------------------------------------------------------------------------- the step
def source_dir(layout: SceneLayout, args: argparse.Namespace) -> Path:
    return Path(args.src) if args.src else layout.inputs_src.meta_dir


def plan(src: Path, extra: list[str]) -> tuple[list[tuple[Path, str]], list[str]]:
    """(source file, relative destination) pairs, and the optional files that are missing."""
    pairs: list[tuple[Path, str]] = []
    for sub, pattern in FRAME_DIRS.items():
        d = src / sub
        if not d.is_dir():
            raise FileNotFoundError(f"episode source has no {sub}/: {d}")
        names = sorted((n for n in os.listdir(d) if pattern.match(n)), key=lambda n: int(n.split(".")[0]))
        if not names:
            raise FileNotFoundError(f"no frame files in {d}")
        pairs += [(d / n, f"{sub}/{n}") for n in names]
    for name in REQUIRED_FILES:
        if not (src / name).is_file():
            raise FileNotFoundError(f"episode source is missing {name}: {src}")
        pairs.append((src / name, name))
    missing = []
    for name in OPTIONAL_FILES:
        if (src / name).is_file():
            pairs.append((src / name, name))
        else:
            missing.append(name)
    for pattern in (*OPTIONAL_GLOBS, *extra):
        pairs += [(p, p.name) for p in sorted(src.glob(pattern)) if p.is_file()]
    seen: set[str] = set()
    unique = []
    for s, rel in pairs:
        if rel not in seen:
            seen.add(rel)
            unique.append((s, rel))
    return unique, missing


def _place(src: Path, dst: Path, copy: bool) -> None:
    dst = prepare_output(dst)
    if dst.exists():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def import_episode(src: Path, dst_dir: Path, *, copy: bool = False, extra: list[str] = ()) -> dict:
    src, dst_dir = Path(src), Path(dst_dir)
    if src.resolve() == dst_dir.resolve():
        raise SystemExit(f"episode source and destination are the same folder: {src}")
    pairs, missing = plan(src, list(extra))
    for sub, pattern in FRAME_DIRS.items():
        d = dst_dir / sub
        if d.is_symlink():
            d.unlink()
        d.mkdir(parents=True, exist_ok=True)
        wanted = {rel.split("/", 1)[1] for _, rel in pairs if rel.startswith(sub + "/")}
        for n in os.listdir(d):  # frames of an earlier import that the source no longer has
            if pattern.match(n) and n not in wanted:
                (d / n).unlink()
    for s, rel in pairs:
        _place(s, dst_dir / rel, copy)

    stats = {
        "source": str(src.resolve()),
        "mode": "copy" if copy else "link",
        "frames": sum(1 for _, rel in pairs if rel.startswith("images0/")),
        "depth_frames": sum(1 for _, rel in pairs if rel.startswith("depth/")),
        "files": sorted(rel for _, rel in pairs if "/" not in rel),
        "missing_optional": missing,
    }
    intr = np.load(dst_dir / "intrinsics.npy", allow_pickle=True)
    if intr.shape != (3, 3):
        save_npy(dst_dir / "intrinsics.npy", intr[0])
        stats["intrinsics_normalised_from"] = list(intr.shape)
    left = [n for n in os.listdir(dst_dir / "images0") if not FRAME_DIRS["images0"].match(n)]
    if left:
        stats["images0_other_files"] = sorted(left)  # e.g. super.png of an earlier superres run
    return stats


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--src", default=None, help="episode folder (default: <scene_data>/DROID/<s>)")
    parser.add_argument("--copy", action="store_true", help="copy instead of symlinking")
    parser.add_argument("--extra", action="append", default=[], metavar="GLOB",
                        help="also take files matching GLOB from the source (repeatable)")


def run(layout: SceneLayout, args: argparse.Namespace) -> dict:
    out = output_inputs(layout)
    return import_episode(source_dir(layout, args), out.meta_dir, copy=args.copy, extra=args.extra)


def _inputs(layout: SceneLayout, args: argparse.Namespace) -> list[Path]:
    src = source_dir(layout, args)
    return [src / "images0", src / "depth", *(src / n for n in REQUIRED_FILES)]


def _outputs(layout: SceneLayout, args: argparse.Namespace) -> list[Path]:
    out = layout.inputs_out
    return [out.images_dir, out.depth_dir, *(out.meta_dir / n for n in REQUIRED_FILES)]


STEP = Step(
    name="inputs.episode",
    help="Link or copy a raw DROID episode (frames, depth, camera, joints) into <scene_data_out>/DROID/<s>",
    add_arguments=add_arguments,
    run=run,
    inputs=_inputs,
    outputs=_outputs,
)
