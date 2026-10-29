"""Ported from the old test_object_{collision,scale,asset}_override.py (r2s.objects.overrides).

The evaluator argparse glue (``add_evaluator_arguments`` / ``validate_arguments``)
is gone; its ``{scene}`` substitution and missing-file check are ``scene_path`` and
``require_file``.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from r2s.collision.profiles import MeshCollisionQuality, ObjectCollisionSelection
from r2s.objects import overrides as ov

trimesh = pytest.importorskip("trimesh")


def _write(path, objects):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"objects": objects}))
    return str(path)


def _selection():
    return ObjectCollisionSelection(
        requested_profile="detailed",
        selected_profile="detailed",
        decimate=False,
        decimate_face_num=1000,
        decimate_aggressiveness=2,
        decompose_object_error_threshold=0.05,
        coacd_threshold=0.03,
        quality=MeshCollisionQuality(face_count=1, vertex_count=1, component_count=1, watertight=False, degenerate_face_fraction=0.0, min_extent_ratio=0.1, risk_score=0, reasons=()),
    )


# --------------------------------------------------------------- path helpers
def test_scene_placeholder_and_require_file(tmp_path):
    assert ov.scene_path(None, "droid2") is None
    assert ov.scene_path("/x/{scene}.json", "droid2") == "/x/droid2.json"
    assert ov.scene_path("/x/{scene}.json", None) == "/x/{scene}.json"
    good = _write(tmp_path / "droid9" / "cand.json", [])
    assert ov.require_file(ov.scene_path(str(tmp_path / "{scene}" / "cand.json"), "droid9"), "scale") == good
    with pytest.raises(FileNotFoundError):
        ov.require_file(ov.scene_path(str(tmp_path / "{scene}" / "cand.json"), "droid8"), "scale")
    assert ov.require_file(None, "x") is None


# --------------------------------------------------------------- collision
def test_collision_loader_and_apply(tmp_path):
    path = tmp_path / "cand.json"
    path.write_text(
        json.dumps(
            {
                "objects": [
                    {"safe_key": "obj_image_0001_segmented", "collision": {"convex_hull": True, "reason": "open cup"}},
                    {"safe_key": "obj_image_0002_segmented", "collision": {"coacd_threshold": 0.1, "decimate": True, "decimate_face_num": 500}},
                    {"safe_key": "obj_image_0003_segmented", "collision": {}},
                    {"safe_key": "obj_image_0004_segmented", "collision": {"convex_hull": True, "collision_apply": False}},
                    {"safe_key": "obj_image_0000_segmented", "pos": [0, 0, 0]},
                ]
            }
        )
    )
    overrides = ov.load_object_collision_overrides(str(path))
    assert set(overrides) == {"image_0001_segmented", "image_0002_segmented"}
    assert overrides["image_0001_segmented"] == {"decompose_object_error_threshold": ov.CONVEX_HULL_ERROR_THRESHOLD}
    sel = ov.apply_collision_override(_selection(), ov.collision_override_for(overrides, "image_0001_segmented"))
    assert sel.decompose_object_error_threshold == 1.0 and sel.coacd_threshold == 0.03 and sel.selected_profile == "detailed+override"
    sel2 = ov.apply_collision_override(_selection(), ov.collision_override_for(overrides, "obj_image_0002_segmented"))
    assert sel2.coacd_threshold == 0.1 and sel2.decimate is True and sel2.decimate_face_num == 500
    assert ov.collision_override_for(overrides, "image_0003_segmented") is None and ov.collision_override_for(None, "x") is None
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"safe_key": "obj_a", "collision": {"bogus": 1}}]))
    with pytest.raises(ValueError):
        ov.load_object_collision_overrides(str(bad))


# --------------------------------------------------------------- scale
def test_scale_loader_skips_unity_and_disabled(tmp_path):
    path = _write(
        tmp_path / "cand.json",
        [
            {"safe_key": "obj_image_0000_segmented", "scale_multiplier": 1.15},
            {"safe_key": "obj_image_0001_segmented", "scale_multiplier": 1.0},
            {"safe_key": "obj_image_0002_segmented", "scale_multiplier": 0.8, "scale_apply": False},
            {"safe_key": "obj_image_0003_segmented", "pos": [0, 0, 0]},  # no multiplier at all
            {"safe_key": "image_0004_segmented", "scale_multiplier": 0.5},  # raw key form
        ],
    )
    assert ov.load_object_scale_multipliers(path) == {"image_0000_segmented": 1.15, "image_0004_segmented": 0.5}


def test_scale_loader_rejects_out_of_range(tmp_path):
    with pytest.raises(ValueError):
        ov.load_object_scale_multipliers(_write(tmp_path / "bad.json", [{"safe_key": "obj_a", "scale_multiplier": 9.0}]))
    with pytest.raises(ValueError):
        ov.load_object_scale_multipliers(_write(tmp_path / "bad2.json", [{"safe_key": "obj_a", "scale_multiplier": 0.05}]))


def test_multiplier_for_accepts_both_key_forms():
    mults = {"image_0000_segmented": 1.3}
    assert ov.multiplier_for(mults, "image_0000_segmented") == 1.3
    assert ov.multiplier_for(mults, "obj_image_0000_segmented") == 1.3
    assert ov.multiplier_for(mults, "image_0001_segmented") == 1.0
    assert ov.multiplier_for(None, "image_0000_segmented") == 1.0
    assert ov.multiplier_for({}, "image_0000_segmented") == 1.0


def test_scale_list_payload_accepted(tmp_path):
    path = tmp_path / "list.json"
    path.write_text(json.dumps([{"safe_key": "obj_x", "scale_multiplier": 2.0}]))
    assert ov.load_object_scale_multipliers(str(path)) == {"x": 2.0}


# --------------------------------------------------------------- asset
def test_asset_loader_resolves_relative_paths_and_checks_existence(tmp_path):
    glb = tmp_path / "cup_flat.glb"
    trimesh.creation.box(extents=(0.05, 0.05, 0.08)).export(str(glb))
    cand = tmp_path / "candidate.json"
    cand.write_text(json.dumps({"objects": [
        {"safe_key": "obj_image_0001_segmented", "asset_override": "cup_flat.glb"},
        {"safe_key": "obj_image_0002_segmented", "asset_override": str(glb), "asset_apply": False},
        {"safe_key": "obj_image_0003_segmented", "pos": [0, 0, 0]},
    ]}))
    overrides = ov.load_object_asset_overrides(str(cand))
    assert overrides == {"image_0001_segmented": str(glb)}
    assert ov.asset_for(overrides, "obj_image_0001_segmented") == str(glb) and ov.asset_for(overrides, "image_0002_segmented") is None
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"safe_key": "obj_a", "asset_override": "missing.glb"}]))
    with pytest.raises(FileNotFoundError):
        ov.load_object_asset_overrides(str(bad))


def test_flatten_bottom_fill_and_clip(tmp_path):
    from r2s.collision.support import DEFAULT_GLB_FRONT, DEFAULT_GLB_UP, align_up_and_front, load_combined_mesh, transform_vertices
    from r2s.objects.refine.common import flatten_bottom_asset

    sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.04)
    src = tmp_path / "ball.glb"
    sphere.export(str(src))
    import_rot = align_up_and_front(np.asarray(DEFAULT_GLB_UP), np.asarray(DEFAULT_GLB_FRONT))

    def world(path):
        mesh = load_combined_mesh(path)
        return transform_vertices(np.asarray(mesh.vertices), scale=1.0, root_pos=(0, 0, 0), root_quat_wxyz=(1, 0, 0, 0), mesh_import_rot=import_rot)

    orig = world(str(src))
    # fill (default): the dome is pushed down to the lowest point; height unchanged, flat patch appears
    out = tmp_path / "ball_fill.glb"
    info = flatten_bottom_asset(str(src), str(out), scale=1.0, root_quat_wxyz=(1, 0, 0, 0), flatten_m=0.012)
    assert info["mode"] == "fill" and info["n_vertices_moved"] > 0 and info["flat_patch_area_cm2"] > 5.0
    w = world(str(out))
    assert w[:, 2].min() == pytest.approx(orig[:, 2].min(), abs=1e-6)
    assert w[:, 2].max() == pytest.approx(orig[:, 2].max(), abs=1e-6)
    assert (np.abs(w[:, 2] - w[:, 2].min()) < 1e-6).sum() >= 20
    assert np.allclose(np.sort(orig[orig[:, 2] >= -0.028][:, 2]), np.sort(w[w[:, 2] >= -0.028 - 1e-9][:, 2]), atol=1e-6)
    # clip: the slab is cut off, so the object is 12 mm shorter
    out2 = tmp_path / "ball_clip.glb"
    info2 = flatten_bottom_asset(str(src), str(out2), scale=1.0, root_quat_wxyz=(1, 0, 0, 0), flatten_m=0.012, mode="clip")
    w2 = world(str(out2))
    assert info2["mode"] == "clip" and w2[:, 2].min() == pytest.approx(-0.04 + 0.012, abs=1e-6)
    with pytest.raises(ValueError):
        flatten_bottom_asset(str(src), str(tmp_path / "x.glb"), scale=1.0, root_quat_wxyz=(1, 0, 0, 0), mode="bogus")
