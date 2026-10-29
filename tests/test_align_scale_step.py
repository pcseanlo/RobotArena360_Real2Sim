"""``align.scale`` step wiring: defaults are production, scenes.yaml keys parse, paths come from the layout."""

from __future__ import annotations

import argparse

from r2s.align import scale_fit
from r2s.config import SceneRegistry
from r2s.layout import SceneLayout
from r2s.steps import load_step


def _parser():
    step = load_step("align.scale")
    parser = argparse.ArgumentParser()
    step.add_arguments(parser)
    return step, parser


def test_defaults_are_production():
    _, parser = _parser()
    args = parser.parse_args([])
    s = scale_fit.ScaleFitSettings.from_args(args)
    # old argparse defaults + the old launcher's --truncate_w 0.1 --truncate_h 0.3
    assert (s.n_steps, s.lr, s.s_min_ratio, s.s_max_ratio, s.trim_quantile) == (400, 5e-3, 0.3, 2.0, 0.8)
    assert (s.n_coll_samples, s.n_metric_samples, s.occlusion_cell_px, s.occlusion_depth_tol_rel) == (60000, 8192, 1, 0.02)
    assert (s.outlier_k, s.outlier_std_ratio, s.truncate_w, s.truncate_h) == (60, 0.5, 0.1, 0.3)
    assert s.truncate_left is s.truncate_right is s.truncate_top is s.truncate_bottom is None
    assert not s.match_metric_z and s.coll_z_pad == 0.05 and s.sdf_cell_size == 0.05
    assert not args.viz_mask and not args.viewer


def test_scenes_yaml_options_are_known():
    step, parser = _parser()
    dests = {a.dest for a in parser._actions}
    reg = SceneRegistry()
    for name in reg.names():
        opts = reg.get(name).step_options(step.name)
        assert set(opts) <= dests, (name, opts)


def test_paths_from_layout():
    step, parser = _parser()
    layout = SceneLayout.create("droid820")
    args = parser.parse_args(["--viz-mask"])
    ins = step.inputs(layout, args)
    assert layout.aligned_pose in ins and layout.collider_glb in ins and layout.gaussian_ply in ins
    outs = step.outputs(layout, args)
    assert outs[:2] == [layout.refined_scale, layout.refined_pose]
    assert layout.align_dir / scale_fit.MASK_PLY in outs
    assert step.needs == "gpu"
