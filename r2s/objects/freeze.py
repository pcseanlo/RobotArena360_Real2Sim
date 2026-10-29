"""Step 10 ``objects.freeze``: pin the object layout the evaluator will use.

Copies one ``candidate.json`` (schema v2, the format of steps 9 and the layout
refiner) byte for byte into ``objects/layout/candidate.json`` and records where it
came from with its sha256, so a later change in the source tree cannot silently
change an eval. Exactly one source:

* ``--refiner-round ROUND``  ``<layout_refiner>/runs/<s>/<ROUND>/candidate.json``
  (the agent-reviewed refiner; production used this, rounds per scene are in
  configs/scenes.yaml from the old submit_generic_home_1200_gpt6_yaw45.sh ROUND table)
* ``--tag TAG``              ``objects/refine/<TAG>/candidate.json`` (Method B, step 9)
* ``--from PATH``            any candidate file

Like the old freeze, a candidate is only accepted next to the ``settle.json`` of
its own render (skip with ``--no-settle-check``).

Outputs: objects/layout/candidate.json, objects/layout/freeze.json, and one row in
the run-wide ``<work>/layout_manifest.tsv`` (scene, source id, sha256[:16], path),
the successor of the old ``layout_eval/<run>/manifest.tsv``.

The evaluator consumes it as before:
``--object-pose-override=<cand> --object-pose-override-mode=absolute
--object-pose-override-max-displacement=0.05 --object-scale-override=<cand>``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import shutil
from pathlib import Path

from r2s.common.io import prepare_output, write_json
from r2s.layout import SceneLayout
from r2s.manifest import file_fingerprint
from r2s.steps import Step


def source_candidate(layout: SceneLayout, args: argparse.Namespace) -> tuple[str, str, Path]:
    """``(kind, id, path)`` of the candidate selected by the options."""
    chosen = [n for n in ("refiner_round", "tag", "from_path") if getattr(args, n)]
    if len(chosen) != 1:
        raise SystemExit(
            "objects.freeze needs exactly one of --refiner-round / --tag / --from "
            f"(got {chosen or 'none'}; scenes.yaml can set refiner_round per scene)"
        )
    if args.refiner_round:
        return "refiner", args.refiner_round, layout.paths.layout_refiner / "runs" / layout.scene / args.refiner_round / "candidate.json"
    if args.tag:
        return "method_b", args.tag, layout.refine_dir(args.tag) / "candidate.json"
    return "file", str(args.from_path), Path(args.from_path)


def validate_candidate(candidate: dict, dump_path: Path | None) -> list[str]:
    """Schema check; returns warnings (objects the dump has but the candidate lacks)."""
    if int(candidate.get("version", 0)) != 2:
        raise ValueError(f"candidate schema version {candidate.get('version')!r}, expected 2")
    objects = candidate.get("objects")
    if not isinstance(objects, list) or not objects:
        raise ValueError("candidate has no objects")
    keys = []
    for obj in objects:
        if "safe_key" not in obj or "pos" not in obj or "quat_wxyz" not in obj:
            raise ValueError(f"candidate object without safe_key/pos/quat_wxyz: {sorted(obj)}")
        keys.append(obj["safe_key"])
    warnings = []
    if dump_path is not None and dump_path.exists():
        dump_keys = {o["safe_key"] for o in json.loads(dump_path.read_text()).get("objects", [])}
        unknown = sorted(set(keys) - dump_keys)
        if unknown:
            raise ValueError(f"candidate objects not in this scene's object dump: {unknown}")
        missing = sorted(dump_keys - set(keys))
        if missing:
            warnings.append(f"objects keep their load pose (not in candidate): {missing}")
    return warnings


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--refiner-round", dest="refiner_round", default=None,
                        help="round folder under <layout_refiner>/runs/<scene>/")
    parser.add_argument("--tag", default=None, help="Method B tag under objects/refine/")
    parser.add_argument("--from", dest="from_path", default=None, help="any candidate.json")
    parser.add_argument("--no-settle-check", dest="settle_check", action="store_false",
                        help="accept a candidate without the settle.json of its render")
    parser.add_argument("--note", default="", help="free text stored with the freeze record")


def _inputs(layout: SceneLayout, args: argparse.Namespace) -> list[Path]:
    _kind, _id, src = source_candidate(layout, args)
    paths = [src]
    if args.settle_check:
        paths.append(src.with_name("settle.json"))
    return paths


def run(layout: SceneLayout, args: argparse.Namespace) -> dict:
    kind, source_id, src = source_candidate(layout, args)
    candidate = json.loads(src.read_text())
    warnings = validate_candidate(candidate, layout.object_dump)
    digest = hashlib.sha256(src.read_bytes()).hexdigest()

    dst = prepare_output(layout.layout_candidate)
    shutil.copyfile(src, dst)
    record = {
        "scene": layout.scene,
        "source_kind": kind,
        "source_id": source_id,
        "source": str(src.resolve()),
        "sha256": digest,
        "sha256_16": digest[:16],
        "settle": file_fingerprint(src.with_name("settle.json")),
        "frozen_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "note": args.note,
        "warnings": warnings,
    }
    write_json(layout.layout_dir / "freeze.json", record)

    manifest = layout.paths.work / "layout_manifest.tsv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    if not manifest.exists():
        manifest.write_text("scene\tsource\tsha256\tpath\tnote\n")
    with manifest.open("a") as f:
        f.write(f"{layout.scene}\t{kind}:{source_id}\t{digest[:16]}\t{src.resolve()}\t{args.note}\n")
    return {"source": f"{kind}:{source_id}", "sha256_16": digest[:16], "warnings": warnings}


STEP = Step(
    name="objects.freeze",
    help="Pin the chosen object layout (candidate.json) with its sha256 for the evaluator",
    add_arguments=add_arguments,
    run=run,
    inputs=_inputs,
    outputs=lambda layout, args: [layout.layout_candidate, layout.layout_dir / "freeze.json"],
    optional_inputs=lambda layout, args: [layout.object_dump],
)
