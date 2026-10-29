"""Step contracts of objects.posefit / objects.refine: production defaults, inputs/outputs, GPU hook."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from r2s.collision.crop import STEP as CROP_STEP
from r2s.config import Paths, SceneConfig
from r2s.layout import CROP_GLB_NAME, SceneLayout
from r2s.objects.pose_fit import STEP as POSEFIT_STEP
from r2s.objects.pose_fit import PoseFitSettings
from r2s.objects.refine import pipeline
from r2s.objects.refine.common import UPRIGHT_CLASSES
from r2s.objects.refine.fit import DEFAULT_SCALES, FitJointSettings


def _args(step, argv=()):
    parser = argparse.ArgumentParser()
    step.add_arguments(parser)
    return parser.parse_args(list(argv))


def _layout(tmp_path) -> SceneLayout:
    p = {k: tmp_path / k for k in Paths.__dataclass_fields__}
    return SceneLayout("droidX", Paths(**p), SceneConfig("droidX", {"scene_type": "DROID"}))


def test_crop_defaults_are_the_evaluator_defaults(tmp_path):
    args = _args(CROP_STEP)
    assert (args.radius, args.z_min, args.z_max, args.sides, args.min_faces) == (1.10, -0.15, 1.45, 24, 4)
    assert args.surface_mask is False
    layout = _layout(tmp_path)
    assert CROP_STEP.outputs(layout, args)[0] == layout.cropped_collider_glb
    assert layout.cropped_collider_glb.name == CROP_GLB_NAME
    assert CROP_STEP.inputs(layout, args) == [layout.collider_glb, layout.refined_pose, layout.refined_scale]


def test_posefit_defaults_are_the_old_script_defaults(tmp_path):
    args = _args(POSEFIT_STEP)
    s = PoseFitSettings.from_args(args)
    assert s == PoseFitSettings()
    assert (s.yaw, s.yaw_step_deg, s.yaw_min_gain, s.min_area_ratio, s.max_area_ratio, s.min_iou) == ("sweep", 15.0, 0.02, 0.5, 2.0, 0.3)
    assert (s.level_max_deg, s.clearance, s.max_overlap, s.height, s.width) == (30.0, 0.0005, 0.25, 360, 640)
    assert not (s.no_polish or s.no_level or s.no_drop or s.allow_overlap)
    assert args.no_overlays is False
    layout = _layout(tmp_path)
    assert POSEFIT_STEP.inputs(layout, args)[0] == layout.object_dump
    assert POSEFIT_STEP.outputs(layout, args)[0] == layout.refined_poses


def test_refine_defaults_are_the_batch_defaults(tmp_path):
    args = _args(pipeline.STEP)
    assert args.stage == "all" and args.tag == "B_v1"
    assert not args.no_seed and args.settle is None and not args.no_review
    s = FitJointSettings.from_args(args)
    assert s.scales is None and s.yaw == "sweep" and s.yaw_step_deg == 15.0
    assert s.orientations == "yaw" and s.orientation_top_k == 2
    assert s.upright_classes == UPRIGHT_CLASSES and not s.no_upright and not s.no_collider_rest
    assert not s.allow_overlap and s.max_overlap == 0.25 and not s.no_level and s.level_max_deg == 30.0
    assert DEFAULT_SCALES[0] == 0.7 and DEFAULT_SCALES[-1] == 1.4 and len(DEFAULT_SCALES) == 15
    assert args.settle_steps == 150 and args.settle_hold_steps == 20 and args.replay_root is None
    layout = _layout(tmp_path)
    fit_args = _args(pipeline.STEP, ["--stage", "fit"])
    ins = pipeline.STEP.inputs(layout, fit_args)
    assert layout.object_dump in ins and layout.refined_poses in ins and layout.cropped_collider_glb in ins
    outs = pipeline.STEP.outputs(layout, fit_args)
    assert layout.refine_dir("B_v1") / "candidate.json" in outs
    score_outs = pipeline.STEP.outputs(layout, _args(pipeline.STEP, ["--stage", "score"]))
    assert score_outs == [layout.refine_dir("stock") / "scores.json", layout.refine_dir("B_v1") / "scores.json"]


def test_settle_stages_delegate_to_the_sim_and_fail_loudly_without_a_dump(tmp_path):
    """run_settle is wired to r2s.sim.settle; without objects.dump's output it must refuse (no GPU needed)."""
    layout = _layout(tmp_path)
    with pytest.raises(FileNotFoundError):
        pipeline.run_settle(layout, None, layout.refine_dir("stock"), "free", 150, 0)
    with pytest.raises(FileNotFoundError):
        pipeline.run(layout, _args(pipeline.STEP, ["--stage", "stock-settle"]))
    assert isinstance(layout.refine_dir("stock"), Path)
