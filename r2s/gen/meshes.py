"""Step 0c-5 ``inputs.meshes``: one SAM 3D Objects mesh per segmented object.

Port of scene_generation ``src/pipeline/sam3d.py`` (env ``sam3d-objects``,
worker ``r2s/gen/workers/sam3d_meshes.py``). For every object ``k`` of
``masks/result.json`` it runs ``Inference(<sam3d_objects>/checkpoints/hf/pipeline.yaml)``
on ``images0/1.jpg`` with the mask ``masks/image_<k>_mask.png`` (blue channel > 0
of that RGB-masked image) and ``seed=42``, with mesh post-processing and texture
baking, and writes to ``assets/<s>/``::

    image_<k>_segmented_mask.png  image_<k>_segmented.ply
    image_<k>_segmented.glb       image_<k>_segmented_scaled.glb
    asset_info.json

Reads from and writes to ``paths.scene_data_out``. MoGe (``Ruicheng/moge-vitl``)
loads from the Hugging Face cache.

Deviations from the old script:
  - Only the object masks are used. The old script took every ``*.png`` in
    ``masks/``, so it also made a mesh of ``gemini_boxes_debug.png`` (a debug
    overlay) and, when run after the table-depth step, of the background mask.
  - ``asset_info.json`` is keyed ``image_<k>_segmented_<i>`` with ``i`` counting
    objects only (the old file started at ``gemini_boxes_debug_00``) and its
    ``mesh`` paths are relative to ``assets/<s>/`` (the old ones were absolute
    paths into the old user's home folder).
  - ``numpy`` / ``random`` are seeded with ``--seed`` (the pipeline already seeds
    torch with it on every call).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from r2s.common.io import prepare_output, write_json
from r2s.gen.conda import run_in_env, worker
from r2s.layout import InputLayout, SceneLayout
from r2s.steps import Step

# The entry the old align_inpaint.py appended to result.json for the background
# ("table") object; scenes generated with it (droid2, droid34) still carry it.
_LEGACY_TABLE_ENTRY = {"object_name": "table", "mass": 0.5, "friction": 0.5, "surface": "Rough"}


def _is_legacy_table_entry(entry: dict, position: int, n_entries: int) -> bool:
    return position == n_entries - 1 and all(entry.get(k) == v for k, v in _LEGACY_TABLE_ENTRY.items())


def object_counts(result_json: str | Path) -> list[int]:
    """Object indices ``k`` listed in ``masks/result.json`` (``count`` fields), sorted.

    The trailing background entry appended by the old ``align_inpaint.py``
    (``object_name == "table"``, mass/friction 0.5, ``Rough``) is not an object
    and is skipped.
    """
    entries = json.loads(Path(result_json).read_text())
    counts: list[int] = []
    for i, entry in enumerate(entries):
        if _is_legacy_table_entry(entry, i, len(entries)):
            continue
        for c in entry["count"]:
            counts.append(int(c))
    if len(set(counts)) != len(counts):
        raise ValueError(f"{result_json}: duplicate object counts {counts}")
    return sorted(counts)


def background_index(inputs: InputLayout) -> int:
    """Index of the background ("table") object written by ``inputs.table_depth``:
    the number of objects in ``result.json`` (objects must be ``0..n-1``)."""
    counts = object_counts(inputs.mask_result)
    if counts != list(range(len(counts))):
        raise ValueError(f"{inputs.mask_result}: object counts {counts} are not 0..{len(counts) - 1}")
    return len(counts)


def object_mask(inputs: InputLayout, k: int) -> Path:
    return inputs.masks_dir / f"image_{k:04d}_mask.png"


def segmented_name(k: int) -> str:
    return f"image_{k:04d}_segmented"


def mesh_outputs(inputs: InputLayout, k: int) -> list[Path]:
    name = segmented_name(k)
    d = inputs.assets_dir
    return [d / f"{name}_mask.png", d / f"{name}.ply", d / f"{name}.glb", d / f"{name}_scaled.glb"]


def prepare_worker_dir(directory: Path, root: Path) -> Path:
    """Create ``directory`` and check it really lies under ``root``.

    Workers write with their own libraries (not ``prepare_output``), so a folder
    that is a symlink into read-only data would be written through. Refuse that.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    real, real_root = directory.resolve(), Path(root).resolve()
    if real != real_root and real_root not in real.parents:
        raise RuntimeError(f"refusing to let a worker write into {directory}: it resolves to {real}, outside {real_root}")
    return directory


def _counts_or_empty(inputs: InputLayout) -> list[int]:
    return object_counts(inputs.mask_result) if inputs.mask_result.exists() else []


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=42, help="seed of every Inference call (old: 42)")
    parser.add_argument("--pipeline-config", default=None,
                        help="SAM 3D pipeline.yaml (default: <sam3d_objects>/checkpoints/hf/pipeline.yaml)")
    parser.add_argument("--resize-h", type=int, default=-1, help="resize the saved masks (old default: off)")
    parser.add_argument("--resize-w", type=int, default=-1)


def _pipeline_config(layout: SceneLayout, args: argparse.Namespace) -> Path:
    if args.pipeline_config:
        return Path(args.pipeline_config)
    return layout.paths.sam3d_objects / "checkpoints" / "hf" / "pipeline.yaml"


def inputs(layout: SceneLayout, args: argparse.Namespace) -> list[Path]:
    io = layout.inputs_out
    files = [io.mask_result, io.images_dir / "1.jpg", _pipeline_config(layout, args)]
    return files + [object_mask(io, k) for k in _counts_or_empty(io)]


def outputs(layout: SceneLayout, args: argparse.Namespace) -> list[Path]:
    io = layout.inputs_out
    files = [p for k in _counts_or_empty(io) for p in mesh_outputs(io, k)]
    return files + [io.assets_dir / "asset_info.json"]


def generate_meshes(layout: SceneLayout, args: argparse.Namespace) -> dict:
    io = layout.inputs_out
    counts = object_counts(io.mask_result)
    if not counts:
        raise ValueError(f"{io.mask_result} lists no objects")
    prepare_worker_dir(io.assets_dir, layout.paths.scene_data_out)
    for k in counts:
        for p in mesh_outputs(io, k):
            prepare_output(p)

    run_in_env(
        layout.paths,
        "sam3d-objects",
        [
            worker("sam3d_meshes.py"),
            "--sam3d-root", layout.paths.sam3d_objects,
            "--config", _pipeline_config(layout, args),
            "--image", io.images_dir / "1.jpg",
            "--masks", *[object_mask(io, k) for k in counts],
            "--output-dir", io.assets_dir,
            "--seed", str(args.seed),
            "--resize-h", str(args.resize_h),
            "--resize-w", str(args.resize_w),
        ],
    )

    asset_info = {
        f"{segmented_name(k)}_{i:02d}": {"mesh": f"{segmented_name(k)}.ply", "name": segmented_name(k)}
        for i, k in enumerate(counts)
    }
    write_json(io.assets_dir / "asset_info.json", asset_info)
    missing = [str(p) for k in counts for p in mesh_outputs(io, k) if not p.exists()]
    if missing:
        raise RuntimeError(f"sam3d worker did not write: {missing}")
    return {"objects": counts, "seed": args.seed}


STEP = Step(
    name="inputs.meshes",
    help="SAM 3D Objects mesh (ply, textured glb, scaled glb) for every object mask",
    add_arguments=add_arguments,
    run=generate_meshes,
    inputs=inputs,
    outputs=outputs,
    needs="gpu+conda:sam3d-objects",
)
