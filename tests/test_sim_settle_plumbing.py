"""CPU checks of the settle-run plumbing: which dump a run loads and how a candidate's poses resolve."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from legacy_paths import old_file

from r2s.layout import SceneLayout
from r2s.sim import settle as sim_settle


def _old(scene: str, tag: str, name: str) -> Path:
    return old_file("asset_refine", scene, tag, name)


def test_stock_loads_layout_dump(tmp_path):
    layout = SceneLayout.create("droid820")
    loads = sim_settle.candidate_load_overrides(None)
    assert sim_settle.resolve_dump(layout, None, loads, None, tmp_path) == layout.object_dump


def test_candidate_with_scale_needs_its_own_dump(tmp_path):
    layout = SceneLayout.create("droid820")
    cand = _old("droid820", "B_v1", "candidate.json")
    loads = sim_settle.candidate_load_overrides(cand)
    assert loads["scale"] == {"image_0000_segmented": 0.92, "image_0001_segmented": 0.78,
                              "image_0003_segmented": 0.93, "image_0004_segmented": 0.9}
    explicit = _old("droid820", "B_v1", "object_poses.json")
    assert sim_settle.resolve_dump(layout, cand, loads, explicit, tmp_path) == explicit


@pytest.mark.parametrize("scene", ["droid2", "droid820"])
def test_candidate_replacement_matches_old_dump(scene, tmp_path):
    """objects.dump at the candidate's scale == the old B_v1 run's object_poses.json (needs a seeded run)."""
    import json

    layout = SceneLayout.create(scene)
    needed = [layout.plane_fit_json, layout.receiver_json, layout.cropped_collider_glb, layout.refined_pose]
    if not all(p.exists() for p in needed):
        pytest.skip("run not seeded (tools/legacy/old_repo.py seed --through objects.dump)")
    cand = _old(scene, "B_v1", "candidate.json")
    out = sim_settle.resolve_dump(layout, cand, sim_settle.candidate_load_overrides(cand), None, tmp_path)
    new = json.loads(Path(out).read_text())
    old = json.loads(_old(scene, "B_v1", "object_poses.json").read_text())
    assert new["z_plane"] == old["z_plane"] and new["physics_top"] == old["physics_top"]
    for a, b in zip(new["objects"], old["objects"]):
        assert a["safe_key"] == b["safe_key"] and a["scale"] == b["scale"]
        assert a["pos"] == b["pos"] and a["quat_wxyz"] == b["quat_wxyz"]


def test_delta_override_reproduces_old_log():
    """droid820 B_v1 preview.log: '[object pose override] ... pos=(0.666,-0.041,0.067)' etc. (delta mode)."""
    from r2s.objects.overrides import load_object_pose_override, resolve_override_for_run
    from r2s.sim.scene import load_object_dump

    cand = _old("droid820", "B_v1", "candidate.json")
    _, objects = load_object_dump(_old("droid820", "B_v1", "object_poses.json"))
    specs = {o.safe_key: sim_settle._LoadSpec(o.pos, o.quat_wxyz) for o in objects}
    resolved, _ = resolve_override_for_run(load_object_pose_override(cand), specs, mode="delta")
    logged = {
        "obj_image_0000_segmented": (0.666, -0.041, 0.067),
        "obj_image_0001_segmented": (0.596, 0.429, 0.060),
        "obj_image_0002_segmented": (0.698, 0.172, 0.052),
        "obj_image_0003_segmented": (0.581, 0.175, 0.049),
        "obj_image_0004_segmented": (0.646, -0.287, 0.055),
    }
    got = {p.safe_key: p.pos for p in resolved.placed}
    for key, pos in logged.items():
        assert np.allclose(got[key], pos, atol=6e-4), (key, got[key], pos)
    # absolute mode keeps the file's poses
    absolute, _ = resolve_override_for_run(load_object_pose_override(cand), specs, mode="absolute")
    assert not math.isclose(absolute.placed[0].pos[2], got["obj_image_0000_segmented"][2], abs_tol=1e-4)
