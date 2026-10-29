"""Step 9 ``objects.refine``: the asset-refine cycle for one scene (old ``asset_refine_batch.sh``).

Production sequence (tag ``B_v1``), each stage writing under ``layout.refine_dir(<tag>)``:

1. ``stock-settle`` (GPU, only when ``refine/stock/settle.json`` is missing): load the
   scene with no override, settle ``mode="free"`` for 150 control steps, write
   ``refine/stock/settle.json`` (the stock dump itself is ``layout.object_dump``).
2. ``fit`` (CPU): joint scale + XY + yaw fit of every object
   (``r2s.objects.refine.fit.fit_joint``) from the stock dump, seeded by
   ``layout.refined_poses`` (step 8b), with ``<tag>/weights.json`` /
   ``<tag>/decisions.json`` when present -> ``<tag>/candidate.json``, ``fit.log``,
   ``review/``.
3. ``candidate-settle`` (GPU): load the scene with the candidate's scale multipliers,
   apply its poses (absolute, max displacement 0.05 m), settle ``mode="upright"``,
   150 steps, hold 20 -> ``<tag>/settle.json`` and ``<tag>/object_poses.json``.
4. ``score`` (CPU): ``scores.json`` for ``stock`` and ``<tag>``
   (``r2s.objects.refine.harness.score_tags``).

``--stage all`` runs 1-4. The settle stages need the GPU sim and go through
:func:`run_settle`, which the sim side implements; until then use ``--stage fit``
and ``--stage score`` (CPU only).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from r2s.layout import SceneLayout
from r2s.objects.refine.common import UPRIGHT_CLASSES, RefinePaths
from r2s.objects.refine.fit import FitJointSettings, fit_joint
from r2s.objects.refine.harness import DEFAULT_LEAD, score_tags
from r2s.objects.settle import CANDIDATE_SETTLE_MODE, DEFAULT_HOLD_STEPS, REFINE_SETTLE_STEPS, STOCK_SETTLE_MODE
from r2s.steps import Step

DEFAULT_TAG = "B_v1"
STAGES = ("fit", "score", "all", "stock-settle", "candidate-settle")
# Candidate-settle pose override settings (asset_refine_scene.sh defaults).
CANDIDATE_POSE_MODE = "absolute"
CANDIDATE_POSE_MAX_DISPLACEMENT = 0.05


# ---------------------------------------------------------------------------
# GPU hook
# ---------------------------------------------------------------------------
def run_settle(
    layout: SceneLayout,
    candidate_path: Path | None,
    out_dir: Path,
    mode: str,
    steps: int,
    hold_steps: int,
) -> dict[str, Any]:
    """Settle the scene in the sim and write ``out_dir/settle.json`` (+ ``object_poses.json``).

    To be implemented by the sim side (``r2s.sim``). Contract, mirroring the old
    ``asset_refine_scene.sh`` -> evaluator ``--preview-views-out`` run:

    * build the production scene (splat, cropped collider, objects with the
      collision guard) with ``candidate_path``'s overrides applied at load time:
      ``load_object_scale_multipliers`` (scale) and, only when the candidate
      carries them, ``load_object_asset_overrides`` / ``load_object_collision_overrides``
      (the batch never set ASSET_OVERRIDE / COLLISION_OVERRIDE, so B_v1 ran without);
    * write ``out_dir/object_poses.json`` with ``write_object_pose_dump`` right after
      loading (load-time poses at the candidate's scale, before the pose override);
    * reset; with a candidate: ``resolve_override_for_run(load_object_pose_override(c),
      placement_specs, mode="absolute")`` then ``apply_object_pose_override(env, ...,
      settle_steps=5, max_displacement=0.05)``;
    * ``probe_settle(entities, step_fn, safe_keys, steps=steps, mode=mode,
      hold_steps=hold_steps, object_names=...)`` and ``write_settle_dump(out_dir /
      "settle.json", report, scene_name=layout.scene, source=str(candidate_path or ""))``;
      optional ``settle.mp4`` via ``write_settle_video``.

    Stock: ``candidate_path=None``, ``mode="free"``, ``steps=150``. Candidate:
    ``mode="upright"``, ``steps=150``, ``hold_steps=20``.
    """
    from r2s.sim.settle import run_settle as _sim_run_settle  # GPU sim side (lazy: keeps this module CPU-only)

    return _sim_run_settle(layout, candidate_path, out_dir, mode, steps, hold_steps)


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def tag_dir(layout: SceneLayout, args: argparse.Namespace) -> Path:
    return layout.refine_dir(args.tag)


def _optional_file(path: Path) -> Path | None:
    return path if path.is_file() else None


def stage_fit(layout: SceneLayout, args: argparse.Namespace) -> dict[str, Any]:
    paths = RefinePaths.from_layout(layout, replay_root=args.replay_root)
    out_dir = tag_dir(layout, args)
    lines: list[str] = []

    def log(line: str) -> None:
        print(line)
        lines.append(line)

    try:
        cand = fit_joint(
            paths,
            dump_path=layout.object_dump,
            out_path=out_dir / "candidate.json",
            seed_path=None if args.no_seed else layout.refined_poses,
            settle_path=args.settle,
            weights_path=_optional_file(out_dir / "weights.json"),
            decisions_path=_optional_file(out_dir / "decisions.json"),
            settings=FitJointSettings.from_args(args),
            review_dir=None if args.no_review else out_dir / "review",
            log=log,
        )
    finally:
        from r2s.common.io import prepare_output

        prepare_output(out_dir / "fit.log").write_text("\n".join(lines) + ("\n" if lines else ""))
    return {
        "mean_iou": cand.scores.get("mean_iou"),
        "objective_total": cand.scores.get("objective_total"),
        "statuses": sorted({o.get("status") for o in cand.objects}),
    }


def stage_score(layout: SceneLayout, args: argparse.Namespace) -> dict[str, Any]:
    paths = RefinePaths.from_layout(layout, replay_root=args.replay_root)
    tags = args.score_tags or ["stock", args.tag]
    scores = score_tags(paths, tags, lead=args.lead)
    return {s["candidate_id"]: {"mean_iou": s["mean_iou"], "objective_total": s["objective_total"], "G": s["grasp"]["G"]} for s in scores}


def stage_stock_settle(layout: SceneLayout, args: argparse.Namespace) -> dict[str, Any]:
    return run_settle(layout, None, layout.refine_dir("stock"), STOCK_SETTLE_MODE, args.settle_steps, 0)


def stage_candidate_settle(layout: SceneLayout, args: argparse.Namespace) -> dict[str, Any]:
    out_dir = tag_dir(layout, args)
    return run_settle(layout, out_dir / "candidate.json", out_dir, CANDIDATE_SETTLE_MODE, args.settle_steps, args.settle_hold_steps)


def run(layout: SceneLayout, args: argparse.Namespace) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    if args.stage == "all":
        if not (layout.refine_dir("stock") / "settle.json").is_file():
            stats["stock-settle"] = stage_stock_settle(layout, args)
        stats["fit"] = stage_fit(layout, args)
        stats["candidate-settle"] = stage_candidate_settle(layout, args)
        stats["score"] = stage_score(layout, args)
    elif args.stage == "fit":
        stats["fit"] = stage_fit(layout, args)
    elif args.stage == "score":
        stats["score"] = stage_score(layout, args)
    elif args.stage == "stock-settle":
        stats["stock-settle"] = stage_stock_settle(layout, args)
    elif args.stage == "candidate-settle":
        stats["candidate-settle"] = stage_candidate_settle(layout, args)
    else:  # argparse choices guard this
        raise ValueError(args.stage)
    return stats


# ---------------------------------------------------------------------------
# Step definition
# ---------------------------------------------------------------------------
def add_arguments(parser: argparse.ArgumentParser) -> None:
    d = FitJointSettings()
    parser.add_argument("--stage", choices=STAGES, default="all", help="fit / score are CPU-only; all and *-settle need the GPU sim")
    parser.add_argument("--tag", default=DEFAULT_TAG, help="candidate folder under objects/refine/")
    # fit-joint
    parser.add_argument("--no-seed", action="store_true", help="do not start from objects.posefit's refined_poses.json")
    parser.add_argument("--settle", type=Path, default=None, help="settle.json of a previous round (stability term + settle-then-refit); off in production")
    parser.add_argument("--scales", nargs="*", type=float, default=d.scales, help="scale multipliers to try (default 0.70..1.40 step 0.05)")
    parser.add_argument("--yaw", choices=["off", "sweep", "c4"], default=d.yaw)
    parser.add_argument("--yaw-step-deg", type=float, default=d.yaw_step_deg)
    parser.add_argument(
        "--orientations",
        choices=["yaw", "principal"],
        default=d.orientations,
        help="principal: also try the mesh's six principal-axis-up rest orientations (needs a human check; off in production)",
    )
    parser.add_argument("--orientation-top-k", type=int, default=d.orientation_top_k)
    parser.add_argument("--upright-classes", nargs="*", default=list(UPRIGHT_CLASSES), help="object-name words whose symmetry axis is made vertical before fitting")
    parser.add_argument("--no-upright", action="store_true")
    parser.add_argument("--no-collider-rest", action="store_true", help="do not re-rest objects on the cropped collider under their fitted footprint")
    parser.add_argument("--allow-overlap", action="store_true")
    parser.add_argument("--max-overlap", type=float, default=d.max_overlap)
    parser.add_argument("--no-level", action="store_true")
    parser.add_argument("--level-max-deg", type=float, default=d.level_max_deg)
    parser.add_argument("--no-review", action="store_true")
    # settle
    parser.add_argument("--settle-steps", type=int, default=REFINE_SETTLE_STEPS)
    parser.add_argument("--settle-hold-steps", type=int, default=DEFAULT_HOLD_STEPS)
    # score
    parser.add_argument("--score-tags", nargs="*", default=None, help="tags to score (default: stock and --tag)")
    parser.add_argument("--lead", type=int, default=DEFAULT_LEAD, help="replay lead (only with --replay-root)")
    parser.add_argument(
        "--replay-root",
        type=Path,
        default=None,
        help="folder with replay runs (<root>/<scene>/lead5_wrist1.0[_<tag>]) for the replay grasp/policy terms; "
        "the old harness used its replay_results/ when present",
    )


def _inputs(layout: SceneLayout, args: argparse.Namespace) -> list[Path]:
    inputs = layout.inputs
    out = [layout.object_dump, inputs.intrinsics, inputs.extrinsics, inputs.bbox_masks_dir]
    if args.stage in ("fit", "all"):
        if not args.no_seed:
            out.append(layout.refined_poses)
        if not args.no_collider_rest:
            out.append(layout.cropped_collider_glb)
        if args.settle is not None:
            out.append(args.settle)
    if args.stage == "score":
        for tag in args.score_tags or ["stock", args.tag]:
            if tag not in ("stock", "posefit"):
                out.append(layout.refine_dir(tag) / "candidate.json")
    return out


def _optional_inputs(layout: SceneLayout, args: argparse.Namespace) -> list[Path]:
    d = layout.refine_dir(args.tag)
    return [d / "weights.json", d / "decisions.json"]


def _outputs(layout: SceneLayout, args: argparse.Namespace) -> list[Path]:
    d = layout.refine_dir(args.tag)
    out: list[Path] = []
    if args.stage in ("stock-settle", "all"):
        out.append(layout.refine_dir("stock") / "settle.json")
    if args.stage in ("fit", "all"):
        out += [d / "candidate.json", d / "fit.log"]
        if not args.no_review:
            out.append(d / "review" / "residual_table.md")
    if args.stage in ("candidate-settle", "all"):
        out += [d / "settle.json", d / "object_poses.json"]
    if args.stage in ("score", "all"):
        out += [layout.refine_dir(t) / "scores.json" for t in (args.score_tags or ["stock", args.tag])]
    return out


STEP = Step(
    name="objects.refine",
    help="asset refinement (scale + XY + yaw per object, settle-checked): --stage fit|score on CPU, all/settle on GPU",
    add_arguments=add_arguments,
    run=run,
    inputs=_inputs,
    outputs=_outputs,
    needs="gpu",
    optional_inputs=_optional_inputs,
)
