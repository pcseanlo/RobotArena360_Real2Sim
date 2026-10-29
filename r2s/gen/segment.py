"""Step 0c-1 ``inputs.segment``: Gemini boxes + SAM3 masks, point clouds and LaMa inputs.

Port of SG ``src/pipeline/segment_object_gembox_updated.py`` (the "gembox" variant
of ``bash_scripts/pipeline_droid_test.sh``, which made every reference scene). The
worker ``r2s/gen/workers/segment_gembox.py`` runs in the ``gemini`` conda env (SAM3
via ``transformers``; a GPU is used when visible).

Reads (``<scene_data_out>/DROID/<s>/``): ``images0/super.png`` if present else
``images0/1.jpg`` (anchor), ``images0/1.jpg`` + ``depth/1.npy`` + camera (point clouds).
Writes:
- ``masks/``: ``image_k_mask.png`` (anchor RGB, black outside object k),
  ``cropped/image_k_segmented.png``, ``result.json`` (name/count/mass/friction/
  surface), ``gemini_boxes_raw.json``, ``gemini_boxes_debug.png`` and, in live mode,
  ``gemini_answers.json`` (the raw Gemini replies, for later replays);
- ``bounding_boxes_and_masks/frame_00001_bounding_boxes_and_masks.json``;
- ``pcds/{k:04d}_frame1.ply`` (Open3D run with one OpenMP thread: SG's threaded run
  of ``remove_radius_outlier`` is nondeterministic and dropped 0-4 random points per
  object; the single-thread clouds are supersets of the old ones);
- ``<scene_data_out>/input_mask/<s>/``: binary dilated ``image_k_mask.png`` +
  anchor ``image_k.png`` per object, plus one pair for the union of the robot/human
  (inpaint-only) masks -- the LaMa inputs of step 0c-4.

Gemini: live mode needs ``$GEMINI_API_KEY`` (environment only). ``--replay-gemini``
re-runs SAM3 and persistence from saved answers without calling Gemini: give a
``gemini_answers.json`` (r2s live run) or SG's ``gemini_boxes_raw.json`` (+
``result.json`` for mass/friction/surface, default: next to it). With no value it
replays ``<scene_data>/DROID/<s>/masks/gemini_boxes_raw.json`` (``layout.inputs_src``).

Like SG's driver, ``masks/`` and ``input_mask/<s>/`` are removed first; so are
``pcds/`` and ``bounding_boxes_and_masks/`` (SG left stale files there). The Hugging
Face hub is used offline (``HF_HUB_OFFLINE=1``) so the cached SAM3 weights are the
ones that load; ``--hf-online`` allows downloads.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from r2s.common.io import prepare_output, read_json
from r2s.gen.conda import run_in_env, worker
from r2s.gen.episode import clear_dir, input_mask_dir, output_inputs
from r2s.layout import SceneLayout
from r2s.steps import Step

AUTO = "auto"
GEMINI_MODEL = "models/gemini-2.5-flash"
SAM3_MODEL = "facebook/sam3"


def work_dir(layout: SceneLayout) -> Path:
    return layout.root / "gen" / "segment"


def replay_paths(layout: SceneLayout, args: argparse.Namespace) -> tuple[Path | None, Path | None]:
    if not args.replay_gemini:
        return None, None
    gem = layout.inputs_src.masks_dir / "gemini_boxes_raw.json" if args.replay_gemini == AUTO else Path(args.replay_gemini)
    res = Path(args.replay_result) if args.replay_result else gem.with_name("result.json")
    return gem, res


def stage_replay(layout: SceneLayout, gem: Path, res: Path) -> tuple[Path, Path | None, dict]:
    """Copy the replay answers out of the folders this step is about to clear; validate them."""
    from r2s.gen.sg_glue import segment_replay  # the worker's replay code (needs only numpy/PIL/cv2)

    dst = work_dir(layout) / "replay"
    data = read_json(gem)
    staged_gem = prepare_output(dst / gem.name)
    shutil.copyfile(gem, staged_gem)
    staged_res = None
    if not segment_replay.is_answers_file(data):
        staged_res = prepare_output(dst / ("source_" + res.name))
        shutil.copyfile(res, staged_res)
    prompts, notes, kind = segment_replay.load_replay(str(staged_gem), str(staged_res) if staged_res else None)
    info = {
        "replay_source": str(gem),
        "replay_result_source": str(res) if staged_res else None,
        "replay_kind": kind,
        "replay_prompts": {k: len(prompts[k]) for k in ("robot", "human", "objects")},
        "replay_notes": notes,
    }
    return staged_gem, staged_res, info


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--replay-gemini", nargs="?", const=AUTO, default=None, metavar="JSON",
                        help="replay saved Gemini answers instead of calling Gemini "
                             "(no value: <scene_data>/DROID/<s>/masks/gemini_boxes_raw.json)")
    parser.add_argument("--replay-result", default=None, metavar="JSON",
                        help="result.json with mass/friction/surface for a gemini_boxes_raw.json replay "
                             "(default: next to it)")
    parser.add_argument("--model-name", default=GEMINI_MODEL, help="Gemini model (live mode)")
    parser.add_argument("--sam3-model", default=SAM3_MODEL, help="SAM3 model id (Hugging Face cache)")
    parser.add_argument("--device", default=None, help="SAM3 device (default: cuda if visible, else cpu)")
    parser.add_argument("--hf-online", action="store_true", help="let transformers reach the Hugging Face hub")


def run(layout: SceneLayout, args: argparse.Namespace) -> dict:
    out = output_inputs(layout)
    meta = out.meta_dir
    stats: dict = {}
    extra_env = {} if args.hf_online else {"HF_HUB_OFFLINE": "1"}

    cmd = [worker("segment_gembox.py"), "--meta-dir", meta, "--input-mask-dir", input_mask_dir(out),
           "--model-name", args.model_name, "--sam3-model", args.sam3_model]
    gem, res = replay_paths(layout, args)
    if gem is not None:
        staged_gem, staged_res, info = stage_replay(layout, gem, res)
        stats.update(info)
        cmd += ["--replay-gemini", staged_gem]
        if staged_res is not None:
            cmd += ["--replay-result", staged_res]
        extra_env["GEMINI_API_KEY"] = ""  # replay never talks to Gemini
    elif not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit("inputs.segment live mode needs GEMINI_API_KEY in the environment "
                         "(or --replay-gemini to reuse saved answers)")
    if args.device:
        cmd += ["--device", args.device]

    # SG's driver: rm -rf masks input_mask; r2s also drops the stale point clouds / mask JSON.
    for d in (out.masks_dir, out.bbox_masks_dir, meta / "pcds", input_mask_dir(out), meta / "input_mask"):
        clear_dir(d)
    summary = work_dir(layout) / "summary.json"
    log = work_dir(layout) / "worker.log"
    prepare_output(summary)
    if summary.exists():
        summary.unlink()
    cmd += ["--summary", summary]
    run_in_env(layout.paths, "gemini", [*cmd, "--stage", "segment"], extra_env=extra_env, log_file=log)
    # Open3D's threaded radius-outlier filter drops random points; one thread is deterministic.
    run_in_env(layout.paths, "gemini", [*cmd, "--stage", "pcds"], extra_env={**extra_env, "OMP_NUM_THREADS": "1"},
               log_file=log)
    s = json.loads(summary.read_text())
    stats.update({k: s.get(k) for k in ("mode", "anchor", "sam3_device", "persisted", "pcd_points")})
    stats["sam3_dropped"] = [o["label"] for o in s.get("objects", []) if not o["sam3"]]
    stats["inpaint_only"] = s.get("inpaint_only")
    stats["log"] = str(log)
    return stats


def _inputs(layout: SceneLayout, args: argparse.Namespace) -> list[Path]:
    out = layout.inputs_out
    paths = [out.images_dir / "1.jpg", out.depth_dir / "1.npy", out.intrinsics, out.extrinsics]
    gem, res = replay_paths(layout, args)
    if gem is not None:
        paths.append(gem)
    return paths


def _outputs(layout: SceneLayout, args: argparse.Namespace) -> list[Path]:
    out = layout.inputs_out
    return [
        out.mask_result,
        out.masks_dir / "gemini_boxes_raw.json",
        out.masks_dir / "gemini_boxes_debug.png",
        out.bbox_masks_dir / "frame_00001_bounding_boxes_and_masks.json",
        out.meta_dir / "pcds",
        input_mask_dir(out),
    ]


STEP = Step(
    name="inputs.segment",
    help="Gemini boxes + SAM3 masks -> masks/, point clouds, LaMa inputs (--replay-gemini: no Gemini call)",
    add_arguments=add_arguments,
    run=run,
    inputs=_inputs,
    outputs=_outputs,
    needs="gpu+conda:gemini",
)
