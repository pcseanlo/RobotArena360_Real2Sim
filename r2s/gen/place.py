"""Step 0c-6 ``inputs.place``: scale / position / rotation of every object mesh in the scene.

Port of the old driver loop around scene_generation ``src/pipeline/MatchFindRot.py``
(the ORIGINAL file, not ``_updated``; env ``match``, Genesis 0.2.1). For each
object ``k`` (``masks/result.json``), one worker process
(``r2s/gen/workers/match_place.py``):

  1. samples the textured ``assets/<s>/image_<k>_segmented.glb``, scales it to the
     bbox diagonal of the denoised object cloud ``pcds/<k>_frame1.ply``, centres
     it and runs point-to-point / point-to-plane ICP (debug only);
  2. renders the mesh from 240 views in Genesis, matches every view against the
     cropped ``masks/image_<k>_mask.png`` with SuperPoint+LightGlue (MINIMA
     weights) and keeps the view with the most matches;
  3. lifts the matches to 3D (rendered depth / ``depth/1.npy``) and fits a rigid
     transform -- or, for the background object, a similarity transform;
  4. composes ``{scale, translation, rotation, original_rotation, quat_rotation}``
     (identity rotation when no view matched; no entry when no keypoint survived).

The step merges the entries into ``masks/transformations.json`` (written fresh,
in object order). Debug files stay in ``assets/<s>/<kkkk>/``.

Production behaviour kept on purpose (defaults):
  - ``--background-object last``: the old driver passed ``--background True`` to
    the HIGHEST-index segmented mesh, which is always a real object (the table
    mesh never had a ``*_segmented.glb``). That object therefore gets a
    similarity fit -- its ``rotation`` is ``s * R`` (not orthonormal) -- and a
    copy of its posed mesh is exported as ``assets/<s>/table.glb``.
  - ``--background-depth raw``: background mode read ``inpainted_depth.npy`` if it
    already existed and ``depth/1.npy`` otherwise. In pipeline order
    (0c-6 before 0c-7) it does not exist yet, so ``raw`` is what a fresh run used
    (e.g. droid820); re-runs after 0c-7 (e.g. droid2's 2026-05-18 rerun) used
    ``inpainted``. Pick explicitly instead of depending on file existence.

Not reproducible bit-for-bit against old outputs: the old runs did not seed
trimesh surface sampling / Open3D RANSAC (``--seed`` now does), matched views in
``os.listdir`` order (now view-index order) and ran on GPU. Compare with
tolerances (``tools/legacy/gen_b_compare.py``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from r2s.common.io import prepare_output, write_json
from r2s.gen.conda import run_in_env, worker
from r2s.gen.meshes import object_counts, object_mask, prepare_worker_dir, segmented_name
from r2s.layout import InputLayout, SceneLayout
from r2s.steps import Step

VENDOR_DIR = Path(__file__).resolve().parent / "sg_glue"
MATCH_CONFIG = VENDOR_DIR / "match_config.yaml"
_DATASET = {"DROID": "droid", "bridge": "bridge"}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=0,
                        help="seeds numpy/random/torch/cv2/Open3D in each worker (old runs were unseeded)")
    parser.add_argument("--background-object", choices=["last", "none"], default="last",
                        help="'last' = production: the highest-index object is fitted in background mode")
    parser.add_argument("--background-depth", choices=["raw", "inpainted"], default="raw",
                        help="depth for the background object: depth/1.npy or inpainted_depth.npy")
    parser.add_argument("--dataset", default=None, help="config 'dataset' (default: droid / bridge from scene_type)")
    parser.add_argument("--checkpoint-dir", default=None,
                        help="MINIMA weights (default: <scene_generation>/MINIMA/weights)")


def _checkpoint_dir(layout: SceneLayout, args: argparse.Namespace) -> Path:
    return Path(args.checkpoint_dir) if args.checkpoint_dir else layout.paths.scene_generation / "MINIMA" / "weights"


def _counts(io: InputLayout) -> list[int]:
    return object_counts(io.mask_result) if io.mask_result.exists() else []


def _background_count(counts: list[int], args: argparse.Namespace) -> int | None:
    return max(counts) if counts and args.background_object == "last" else None


def object_inputs(io: InputLayout, k: int) -> list[Path]:
    return [io.assets_dir / f"{segmented_name(k)}.glb", io.meta_dir / "pcds" / f"{k:04d}_frame1.ply", object_mask(io, k)]


def inputs(layout: SceneLayout, args: argparse.Namespace) -> list[Path]:
    io = layout.inputs_out
    files = [io.mask_result, io.images_dir / "1.jpg", io.depth_dir / "1.npy", io.intrinsics, io.extrinsics,
             MATCH_CONFIG, _checkpoint_dir(layout, args) / "minima_lightglue.pth",
             layout.paths.scene_generation / "third_party" / "LightGlue" / "lightglue" / "__init__.py"]
    counts = _counts(io)
    if args.background_depth == "inpainted" and _background_count(counts, args) is not None:
        files.append(io.inpainted_depth)
    return files + [p for k in counts for p in object_inputs(io, k)]


def outputs(layout: SceneLayout, args: argparse.Namespace) -> list[Path]:
    return [layout.inputs_out.transformations]


def place_objects(layout: SceneLayout, args: argparse.Namespace) -> dict:
    io = layout.inputs_out
    counts = object_counts(io.mask_result)
    if not counts:
        raise ValueError(f"{io.mask_result} lists no objects")
    background = _background_count(counts, args)
    dataset = args.dataset or _DATASET[layout.config.scene_type]
    root = layout.paths.scene_data_out

    transformations: dict[str, dict] = {}
    status: dict[str, str] = {}
    for k in counts:
        cnt = f"{k:04d}"
        debug_dir = prepare_worker_dir(io.assets_dir / cnt, root)
        prepare_worker_dir(debug_dir / "segmented_images", root)
        for name in ("mesh.ply", "demo.png", "cropped_image.png", "keypoints3d0.npy", "keypoints3d0.ply"):
            prepare_output(debug_dir / name)
        result_json = prepare_output(debug_dir / "placement.json")
        is_bg = k == background
        argv = [
            worker("match_place.py"),
            "--scene-dir", io.meta_dir,
            "--assets-dir", io.assets_dir,
            "--mesh-cnt", cnt,
            "--config", MATCH_CONFIG,
            "--dataset", dataset,
            "--checkpoint-dir", _checkpoint_dir(layout, args),
            "--scene-generation", layout.paths.scene_generation,
            "--seed", str(args.seed),
            "--out-json", result_json,
        ]
        if is_bg:
            prepare_output(io.assets_dir / "table.glb")
            argv.append("--background")
            if args.background_depth == "inpainted":
                argv += ["--background-depth", io.inpainted_depth]
        run_in_env(layout.paths, "match", argv)

        result = json.loads(result_json.read_text())
        if result["entry"] is None:
            status[result["key"]] = "no_keypoints"
            continue
        transformations[result["key"]] = result["entry"]
        status[result["key"]] = "identity" if "original_rotation" not in result["entry"] else "matched"

    write_json(io.transformations, transformations, indent=4)
    return {
        "objects": counts,
        "background_object": background,
        "background_depth": args.background_depth if background is not None else None,
        "status": status,
        "seed": args.seed,
    }


STEP = Step(
    name="inputs.place",
    help="Fit scale/translation/rotation of each object mesh (Genesis render + LightGlue match)",
    add_arguments=add_arguments,
    run=place_objects,
    inputs=inputs,
    outputs=outputs,
    needs="gpu+conda:match",
)
