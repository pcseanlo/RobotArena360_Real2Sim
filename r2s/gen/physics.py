"""Step 0c-2 ``inputs.physics``: per-object size, mass, friction and surface from Gemini.

Port of SG ``src/pipeline/prompt_physics_updated.py`` ->
``src/utils/physics_utils.py::query_physical_properties_with_images``: for every
object of ``masks/result.json``, Gemini (``models/gemini-2.5-flash``) sees the
cropped object (``masks/cropped/image_k_segmented.png``) and its binary mask
(``<scene_data_out>/input_mask/<s>/image_k_mask.png``) and answers
``width, height, length, mass, friction, surface`` -> ``masks/physical_properties.json``.

Live mode needs ``$GEMINI_API_KEY`` (environment only) and runs the worker in the
``gemini`` env; the raw replies go to ``masks/physical_properties_answers.json``.
``--replay [JSON]`` copies a saved ``physical_properties.json`` byte for byte after
checking it lists exactly the objects of the current ``result.json`` (no value:
``<scene_data>/DROID/<s>/masks/physical_properties.json``, ``layout.inputs_src``).
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from r2s.common.io import prepare_output, read_json
from r2s.gen.conda import run_in_env, worker
from r2s.gen.episode import input_mask_dir, output_inputs
from r2s.layout import SceneLayout
from r2s.steps import Step

AUTO = "auto"
GEMINI_MODEL = "models/gemini-2.5-flash"
FIELDS = ("object_name", "count", "width", "height", "length", "mass", "friction", "surface")


def replay_path(layout: SceneLayout, args: argparse.Namespace) -> Path | None:
    if not args.replay:
        return None
    return layout.inputs_src.physical_properties if args.replay == AUTO else Path(args.replay)


def output_path(layout: SceneLayout) -> Path:
    return layout.inputs_out.masks_dir / "physical_properties.json"


def check_physics(physics: list, result: list) -> None:
    """Fail unless ``physics`` has SG's schema and lists the objects of ``result.json`` in order."""
    if not isinstance(physics, list):
        raise ValueError("physical_properties.json must be a list")
    if len(physics) != len(result):
        raise ValueError(f"physical_properties.json has {len(physics)} objects, result.json {len(result)}: "
                         f"{[p.get('object_name') for p in physics]} vs {[r.get('object_name') for r in result]}")
    for i, (p, r) in enumerate(zip(physics, result)):
        missing = [f for f in FIELDS if f not in p]
        if missing:
            raise ValueError(f"physical_properties.json entry {i} lacks {missing}")
        if p["object_name"] != str(r["object_name"]).strip() or p["count"] != r["count"]:
            raise ValueError(f"entry {i}: physics {p['object_name']!r}/{p['count']} vs result "
                             f"{r['object_name']!r}/{r['count']}")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--replay", nargs="?", const=AUTO, default=None, metavar="JSON",
                        help="copy a saved physical_properties.json instead of asking Gemini "
                             "(no value: <scene_data>/DROID/<s>/masks/physical_properties.json)")
    parser.add_argument("--model-name", default=GEMINI_MODEL)
    parser.add_argument("--delay", type=float, default=1.0, help="seconds between Gemini requests")


def run(layout: SceneLayout, args: argparse.Namespace) -> dict:
    out = output_inputs(layout)
    result = read_json(out.mask_result)
    dst = prepare_output(output_path(layout))
    src = replay_path(layout, args)
    if src is not None:
        check_physics(read_json(src), result)
        if src.resolve() != dst.resolve():
            shutil.copyfile(src, dst)
        return {"mode": "replay", "source": str(src), "objects": len(result)}

    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit("inputs.physics live mode needs GEMINI_API_KEY in the environment (or --replay)")
    log = layout.root / "gen" / "physics" / "worker.log"
    run_in_env(layout.paths, "gemini", [
        worker("physics_gemini.py"), "--mask-dir", out.masks_dir, "--input-mask-dir", input_mask_dir(out),
        "--out", dst, "--answers-out", out.masks_dir / "physical_properties_answers.json",
        "--model-name", args.model_name, "--delay", str(args.delay),
    ], log_file=log)
    physics = read_json(dst)
    check_physics(physics, result)
    return {"mode": "live", "objects": len(physics), "unparsed": [p["object_name"] for p in physics if p["mass"] is None],
            "log": str(log)}


def _inputs(layout: SceneLayout, args: argparse.Namespace) -> list[Path]:
    out = layout.inputs_out
    paths = [out.mask_result, out.masks_dir / "cropped", input_mask_dir(out)]
    src = replay_path(layout, args)
    if src is not None:
        paths.append(src)
    return paths


STEP = Step(
    name="inputs.physics",
    help="Gemini size/mass/friction/surface per object -> masks/physical_properties.json (--replay: copy a saved one)",
    add_arguments=add_arguments,
    run=run,
    inputs=_inputs,
    outputs=lambda layout, args: [output_path(layout)],
    needs="conda:gemini",
)
