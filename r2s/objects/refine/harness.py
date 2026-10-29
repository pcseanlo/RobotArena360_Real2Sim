"""Step 9 scoring: offline (+ settle, + optional replay) scores for candidates of one scene.

Port of the old ``asset_refine_harness.py`` ``score`` / ``summarize``. The GPU
launchers (``preview``, ``replay``, ``compare``) are not here: the settle renders
are ``r2s.objects.refine.pipeline.run_settle`` (sim side), replays belong to the
evaluator.

Layout per tag: ``layout.refine_dir(tag)/{candidate.json, object_poses.json,
settle.json, scores.json}``. ``stock`` needs no candidate (the evaluator's own
placement); ``posefit`` falls back to ``layout.refined_poses``. The pose dump scored
against is the tag's own ``object_poses.json`` (rendered at the candidate's scale by
candidate-settle), else the stock dump ``layout.object_dump``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from r2s.common.io import prepare_output
from r2s.objects.refine.common import (
    Candidate,
    RefinePaths,
    build_models,
    candidate_from_refined_poses,
    candidate_stock,
    load_object_pose_dump,
    rest_plane,
)
from r2s.objects.refine.metrics import SceneEvidence, score_candidate, summarize_scores
from r2s.objects.settle import load_settle_dump

DEFAULT_LEAD = 5


def candidate_path_for(paths: RefinePaths, tag: str, explicit=None) -> str | None:
    """candidate.json for a tag; ``stock`` -> None; ``posefit`` -> ``refined_poses.json``."""
    if explicit:
        return os.path.abspath(explicit)
    if tag == "stock":
        return None
    own = paths.candidate_dir(tag) / "candidate.json"
    if own.is_file():
        return str(own)
    if tag == "posefit" and Path(paths.refined_poses).is_file():
        return str(paths.refined_poses)
    return None


def load_candidate_for(paths: RefinePaths, tag: str, dump: dict, camera, explicit=None) -> tuple[Candidate, dict | None]:
    path = candidate_path_for(paths, tag, explicit)
    if tag == "stock" and path is None:
        cand, _ = candidate_stock(dump, camera, scene=paths.scene)
        return cand, None
    if path is None:
        raise FileNotFoundError(f"no candidate for tag {tag!r} under {paths.candidate_dir(tag)}")
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if "candidate_id" in payload or payload.get("version", 0) >= 2:
        cand = Candidate.load(path)
        cand.candidate_id = tag
    else:
        cand = candidate_from_refined_poses(path, scene=paths.scene, candidate_id=tag)
    return cand, payload


def dump_path_for(paths: RefinePaths, tag: str, explicit=None) -> str:
    """The pose dump to score against: the tag's own (rendered at its scale), else the stock dump."""
    for p in (explicit, paths.candidate_dir(tag) / "object_poses.json", paths.object_dump):
        if p and os.path.isfile(p):
            return str(p)
    raise FileNotFoundError(f"no object_poses.json for {paths.scene}; run objects.dump first")


def score_tag(
    paths: RefinePaths,
    tag: str,
    *,
    candidate=None,
    dump=None,
    lead: int = DEFAULT_LEAD,
    weights: dict | None = None,
    use_settled: bool = True,
) -> dict[str, Any]:
    dump_file = dump_path_for(paths, tag, dump)
    dump_payload = load_object_pose_dump(dump_file)
    z_plane = rest_plane(dump_payload)
    cal_plane = dump_payload.get("z_plane")
    evidence = SceneEvidence.load(paths, z_plane=float(cal_plane if cal_plane is not None else 0.02))
    cand, _ = load_candidate_for(paths, tag, dump_payload, evidence.camera, candidate)
    settle_path = paths.candidate_dir(tag) / "settle.json"
    settle = load_settle_dump(settle_path) if settle_path.is_file() else None
    own_dump = paths.candidate_dir(tag) / "object_poses.json"
    if tag != "stock" and own_dump.is_file() and os.path.abspath(dump_file) == os.path.abspath(own_dump):
        # The tag's own dump was rendered at the candidate's scale, so the scale multiplier is
        # already inside the loaded poses: do not apply it twice.
        cand = Candidate(**{**cand.__dict__, "objects": [{**o, "scale_multiplier": 1.0, "scale_apply": False} for o in cand.objects]})
    models = build_models(dump_payload, evidence.camera, candidate=cand, settle=settle, z_plane=z_plane, use_settled=use_settled)
    replay_dir = paths.replay_dir(None if tag == "stock" else tag, lead=lead)
    has_replay = replay_dir is not None and replay_dir.is_dir()
    score = score_candidate(cand, models, evidence, settle=settle, replay_dir=str(replay_dir) if has_replay else None, weights=weights)
    score["inputs"] = {
        "dump": dump_file,
        "candidate": candidate_path_for(paths, tag, candidate),
        "settle": str(settle_path) if settle else None,
        "replay_dir": str(replay_dir) if has_replay else None,
    }
    return score


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def score_tags(
    paths: RefinePaths,
    tags: list[str],
    *,
    candidate=None,
    dump=None,
    lead: int = DEFAULT_LEAD,
    weights: dict | None = None,
    use_settled: bool = True,
    log=print,
) -> list[dict[str, Any]]:
    """Old ``score``: write ``<tag>/scores.json`` for each tag and log the summary table.

    A tag without a candidate / dump is skipped with a message (old behaviour).
    """
    scores = []
    for tag in tags:
        try:
            score = score_tag(
                paths,
                tag,
                candidate=candidate if len(tags) == 1 else None,
                dump=dump,
                lead=lead,
                weights=weights,
                use_settled=use_settled,
            )
        except FileNotFoundError as exc:
            log(f"[score] {tag}: {exc}")
            continue
        out = prepare_output(paths.candidate_dir(tag) / "scores.json")
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(_jsonable(score), handle, indent=2)
        log(f"[score] {tag}: wrote {out}")
        scores.append(score)
    log(summarize_scores(scores))
    return scores


def summarize(score_files: dict[str, list], out_path, log=print) -> str:
    """Old ``summarize``: one markdown table per scene from ``{scene: [scores.json paths]}``."""
    rows = []
    for scene, files in score_files.items():
        scores = []
        for path in files:
            if os.path.isfile(path):
                with open(path, encoding="utf-8") as handle:
                    scores.append(json.load(handle))
        if scores:
            rows.append(f"## {scene}\n\n" + summarize_scores(scores) + "\n")
    text = "# asset refinement comparison\n\n" + "\n".join(rows)
    with open(prepare_output(out_path), "w", encoding="utf-8") as handle:
        handle.write(text)
    log(text)
    return text
