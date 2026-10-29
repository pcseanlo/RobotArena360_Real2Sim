"""objects.dump (CPU reproduction of the old evaluator's object loading) and InputLayout.recorded_joints."""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
import pytest

from r2s.config import Paths, SceneConfig
from r2s.layout import InputLayout, SceneLayout
from r2s.objects import dump as od

trimesh = pytest.importorskip("trimesh")


def _camera():
    """Camera 0.8 m above the table looking down 50 deg along +x (world->camera, OpenCV axes)."""
    K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 180.0], [0.0, 0.0, 1.0]])
    tilt = math.radians(50.0)
    z_c = np.array([math.cos(tilt), 0.0, -math.sin(tilt)])
    x_c = np.array([0.0, -1.0, 0.0])
    y_c = np.cross(z_c, x_c)
    R_c2w = np.stack([x_c, y_c, z_c], axis=1)
    c = np.array([-0.3, 0.0, 0.8])
    T = np.eye(4)
    T[:3, :3] = R_c2w.T
    T[:3, 3] = -R_c2w.T @ c
    return K, T


def test_reproject_to_plane_keeps_points_on_the_plane():
    K, T = _camera()
    X = np.array([0.5, 0.1, 0.02])
    X_new, s = od.reproject_to_plane(X, K, T, 0.02)
    assert np.allclose(X_new, X) and s == pytest.approx(1.0)
    # a point above the plane slides along its pixel ray down to it, and the scale follows the depth ratio
    X_new, s = od.reproject_to_plane(np.array([0.5, 0.1, 0.10]), K, T, 0.02)
    assert X_new[2] == pytest.approx(0.02) and s > 1.0


def test_support_plane_is_max_of_fallback_reference_and_physics(tmp_path):
    plane = tmp_path / "plane.json"
    plane.write_text(json.dumps({"selected": {"center": [0.5, 0, -0.02], "size": [0.4, 0.6, 0.02], "quat_wxyz": [1, 0, 0, 0]}}))
    recv = tmp_path / "recv.json"
    recv.write_text(json.dumps({"selected": {"footprints": {"physics": {"vertices_world": [[0, 0, 0.012], [1, 0, 0.015], [1, 1, float("nan")]]}}}}))
    z, ref, phys = od.object_support_plane_z(plane, recv)
    assert ref == pytest.approx(-0.01) and phys == pytest.approx(0.015)
    assert z == pytest.approx(0.02)  # fallback wins over -0.01 + 0.01 and 0.015
    z, _, _ = od.object_support_plane_z(plane, recv, fallback=0.0, reference_offset=0.04)
    assert z == pytest.approx(0.03)
    assert od.object_support_plane_z(None, None) == (0.02, None, None)


def test_metadata_identity_from_result_json_and_richer_fields_by_name(tmp_path):
    (tmp_path / "masks").mkdir()
    (tmp_path / "masks" / "result.json").write_text(json.dumps([{"count": [0], "object_name": "black bowl"}, {"count": 1, "object_name": "cup target"}]))
    # root-level enrichment reorders counts: joined by normalized name ("cup target" -> "cup")
    (tmp_path / "physical_properties.json").write_text(json.dumps([
        {"count": [0], "object_name": "cup", "mass": 0.2, "friction": 0.7, "material": "Plastic"},
        {"count": [1], "object_name": "black bowl", "mass": 0.01},
    ]))
    inputs = InputLayout(tmp_path.parent, tmp_path.name, "")
    srcs = [("masks/result.json", tmp_path / "masks" / "result.json"), ("masks/physical_properties.json", tmp_path / "masks" / "physical_properties.json"), ("physical_properties.json", tmp_path / "physical_properties.json")]
    meta = od.load_scene_object_metadata(srcs)
    assert meta[1]["object_name"] == "cup target" and meta[1]["mass"] == 0.2 and meta[1]["friction"] == 0.7
    assert meta[0]["mass"] == 0.01
    assert od.resolve_object_mass(meta[0]) == (0.03, 0.01)  # mass floor
    m = od.resolve_object_visual_material(meta[1], mode="auto", splat_backend="nyx")
    assert m.status == "metadata" and m.surface_kind == "bsdf" and m.roughness == 0.45
    glass = od.resolve_object_visual_material({"material": "glass"}, mode="auto", splat_backend="nyx")
    assert glass.surface_kind == "bsdf" and glass.fallback_reason
    assert od.resolve_object_visual_material({"material": "glass"}, mode="auto", splat_backend="gsplat").surface_kind == "glass"
    del inputs


def _scene(tmp_path, rotation=None):
    """A box at (0.5, 0, ~0.04) seen by the camera; a flat crop table at z = 0.01."""
    K, T = _camera()
    meta = tmp_path / "DROID" / "s"
    (meta / "masks").mkdir(parents=True)
    np.save(meta / "intrinsics.npy", K)
    np.save(meta / "extrinsics.npy", T)
    assets = tmp_path / "assets" / "s"
    assets.mkdir(parents=True)
    trimesh.creation.box(extents=(0.06, 0.04, 0.05)).export(str(assets / "image_0000_segmented.glb"))
    R = np.eye(3) if rotation is None else rotation
    (meta / "masks" / "transformations.json").write_text(json.dumps({
        "image_0000_segmented": {"translation": [0.5, 0.0, 0.04], "scale": 1.0, "rotation": R.tolist()},
        "image_0001_segmented": {"translation": [0.6, 0.1, 0.04], "scale": 1.0, "rotation": R.tolist()},  # no GLB
    }))
    (meta / "masks" / "result.json").write_text(json.dumps([{"count": [0], "object_name": "box", "mass": 0.2}]))
    table = trimesh.Trimesh(vertices=[[0.0, -0.5, 0.01], [1.0, -0.5, 0.01], [1.0, 0.5, 0.01], [0.0, 0.5, 0.01]], faces=[[0, 1, 2], [0, 2, 3]], process=False)
    crop = tmp_path / "crop.glb"
    table.export(str(crop))
    return meta, assets, crop


def _load(meta, assets, crop, *, z_plane=0.02, rotation="identity", **kw):
    query = od.build_support_query(crop, rotation=rotation)
    return od.load_scene_objects(
        transformations_path=meta / "masks" / "transformations.json",
        intrinsics_path=meta / "intrinsics.npy",
        extrinsics_path=meta / "extrinsics.npy",
        assets_dir=assets,
        metadata=od.load_scene_object_metadata([("masks/result.json", meta / "masks" / "result.json")]),
        z_plane=z_plane,
        support_query=query,
        log=lambda line: None,
        **kw,
    )


def test_load_scene_objects_reprojects_guards_and_rests(tmp_path):
    meta, assets, crop = _scene(tmp_path)
    objects, skipped = _load(meta, assets, crop)
    assert skipped == ["image_0001_segmented"]  # missing GLB: skipped like the evaluator
    o = objects["obj_image_0000_segmented"]
    assert o.count == 0 and o.object_name == "box" and o.mass == 0.2 and o.friction == 0.5
    assert o.reprojected_pos[2] == pytest.approx(0.02)  # translation slid onto z_plane
    # identity rotation: no fix-ups; box 5 cm tall in GLB y -> z after import, rests on z_plane + 0.0005
    assert o.quat_wxyz == pytest.approx((1.0, 0.0, 0.0, 0.0))
    assert o.spec.base_min_z == pytest.approx(0.02 + 0.0005, abs=1e-7)
    assert o.collision.selected_profile == "detailed" and o.collision.decimate is False
    # the dump record round-trips through the writer
    scene = od.SceneObjects("s", str(meta), 0.02, None, None, objects)
    out = od.dump_scene_objects(scene, tmp_path / "out" / "object_poses.json")
    dumped = json.loads(open(out).read())
    assert [e["safe_key"] for e in dumped["objects"]] == ["obj_image_0000_segmented"]
    assert dumped["objects"][0]["pos"] == [float(v) for v in o.pos]


def test_quaternion_fixups_and_scale_override(tmp_path):
    rz = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    meta, assets, crop = _scene(tmp_path, rotation=rz)
    objects, _ = _load(meta, assets, crop, scale_multipliers={"image_0000_segmented": 1.5})
    o = objects["obj_image_0000_segmented"]
    q = od.rot_matrix_to_wxyz(rz)
    from r2s.common.quat import quat_mul_np

    q = quat_mul_np(quat_mul_np((0, 0, 0.7071, 0.7071), quat_mul_np((0.7071067811865476, -0.7071067811865476, 0.0, 0.0), q)), (0, 0, -0.7071, -0.7071))
    assert o.quat_wxyz == tuple(float(v) for v in q)  # not renormalised
    base, _ = _load(meta, assets, crop)
    # scale = transformations scale x reprojection factor x multiplier
    assert o.scale_multiplier == 1.5 and o.scale == 1.5 * base["obj_image_0000_segmented"].scale
    assert base["obj_image_0000_segmented"].scale > 1.0  # the translation slid down its ray onto z_plane


def test_support_query_rotation_legacy_rotates_the_crop(tmp_path):
    _, _, crop = _scene(tmp_path)
    legacy = od.build_support_query(crop, rotation="legacy")
    ident = od.build_support_query(crop, rotation="identity")
    assert np.allclose(ident.vertices_world[:, 2], 0.01)
    assert np.allclose(legacy.vertices_world[:, 1], 0.01)  # -90 deg about x: z -> y
    with pytest.raises(ValueError):
        od.build_support_query(crop, rotation="bogus")


def test_step_defaults_are_production(tmp_path):
    parser = argparse.ArgumentParser()
    od.STEP.add_arguments(parser)
    args = parser.parse_args([])
    s = od.DumpSettings.from_args(args)
    assert (s.z_plane, s.z_plane_offset, s.init_collision_clearance, s.object_support_clearance) == (0.02, 0.01, 0.005, 0.0005)
    assert (s.footprint_margin, s.surface_normal_z, s.bottom_slab, s.support_query_rotation) == (0.015, 0.75, 0.02, "legacy")
    assert s.init_collision and s.scale_override is None and s.asset_override is None
    assert (s.collision_profile, s.mass_floor, s.material_mode, s.splat_backend) == ("detailed", 0.03, "auto", "nyx")
    layout = SceneLayout("s", Paths(**{k: tmp_path / k for k in Paths.__dataclass_fields__}), SceneConfig("s", {}))
    ins = od.STEP.inputs(layout, args)
    assert layout.cropped_collider_glb in ins and layout.receiver_json in ins and layout.plane_fit_json in ins
    assert od.STEP.outputs(layout, args) == [layout.object_dump]
    args = parser.parse_args(["--scale-override", str(tmp_path / "{scene}" / "c.json")])
    assert tmp_path / "s" / "c.json" in od.STEP.inputs(layout, args)


# ---------------------------------------------------------------- recorded_joints
def _video(path, n_frames):
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15, (32, 24))
    for _ in range(n_frames):
        writer.write(np.zeros((24, 32, 3), dtype=np.uint8))
    writer.release()


def test_recorded_joints_follows_the_old_search_order(tmp_path):
    inputs = InputLayout(tmp_path, "droidX", "DROID")
    inputs.meta_dir.mkdir(parents=True)
    orig = inputs.original_meta_dir
    with pytest.raises(FileNotFoundError):
        inputs.recorded_joints()
    np.save(inputs.joints, np.zeros((5, 7)))
    assert inputs.recorded_joints() == inputs.joints
    orig.mkdir(parents=True)
    np.save(orig / "joints.npy", np.zeros((6, 7)))
    assert inputs.recorded_joints() == inputs.joints  # no video: trimmed/unknown -> processed
    _video(orig / "video1.mp4", 6)
    assert inputs.recorded_joints() == orig / "joints.npy"  # frame-aligned
    _video(orig / "video1.mp4", 9)
    assert inputs.recorded_joints() == inputs.joints  # trimmed
    np.save(orig / "joints_old.npy", np.zeros((9, 7)))
    assert inputs.recorded_joints() == orig / "joints_old.npy"
    np.save(orig / "joints_old.npy", np.zeros((9, 6)))
    with pytest.raises(ValueError):
        inputs.recorded_joints()
