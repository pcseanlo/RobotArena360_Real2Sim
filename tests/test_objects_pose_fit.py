"""Ported from the old test_object_pose_refine.py (r2s.objects.pose_fit + overrides; synthetic scene, no simulator)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
trimesh = pytest.importorskip("trimesh")

from r2s.objects import overrides as opo  # noqa: E402
from r2s.objects import pose_fit as opr  # noqa: E402
from r2s.objects.placement_types import ObjectPlacementSpec, quat_mul_wxyz, yaw_quat_wxyz  # noqa: E402


def _look_at_camera(height=360, width=640):
    """Camera 0.8 m above the table, tilted down 50 deg, looking at +x."""
    # Long focal so the 12x5 cm box covers ~2000 px, like the real DROID masks
    # (700-9500 px on droid41); at 250 px IoU is dominated by pixel quantization.
    K = np.array([[900.0, 0.0, width / 2], [0.0, 900.0, height / 2], [0.0, 0.0, 1.0]])
    tilt = math.radians(50.0)
    # camera axes in world: z_c (forward) points down-forward, x_c = -y_w, y_c = down-ish
    z_c = np.array([math.cos(tilt), 0.0, -math.sin(tilt)])
    x_c = np.array([0.0, -1.0, 0.0])
    y_c = np.cross(z_c, x_c)
    R_c2w = np.stack([x_c, y_c, z_c], axis=1)
    c = np.array([-0.3, 0.0, 0.8])
    T = np.eye(4)
    T[:3, :3] = R_c2w.T
    T[:3, 3] = -R_c2w.T @ c
    return opr.PinholeCamera(K=K, T_w2c=T, height=height, width=width)


def _box_spec(safe_key="obj_box", extents=(0.12, 0.05, 0.04), pos=(0.45, 0.05, 0.02), quat=(1.0, 0.0, 0.0, 0.0)):
    mesh = trimesh.creation.box(extents=extents)
    return ObjectPlacementSpec.from_world_vertices(
        safe_key=safe_key,
        count=1,
        object_name="box",
        vertices_world=np.asarray(mesh.vertices) + np.asarray(pos),
        base_pos=np.asarray(pos),
        base_quat_wxyz=np.asarray(quat),
        faces=np.asarray(mesh.faces),
    )


def test_projection_matches_ray_plane_inverse():
    cam = _look_at_camera()
    pts = np.array([[0.5, 0.1, 0.02], [0.3, -0.2, 0.02]])
    uv, depth = cam.project(pts)
    assert (depth > 0).all()
    back = cam.ray_plane_xy(uv, 0.02)
    assert np.allclose(back, pts[:, :2], atol=1e-6)


def test_silhouette_moves_with_object():
    cam = _look_at_camera()
    model = opr.ObjectModel(spec=_box_spec(), camera=cam)
    a = opr.mask_centroid(model.silhouette(np.zeros(2), 0.0))
    b = opr.mask_centroid(model.silhouette(np.array([0.0, 0.05]), 0.0))
    assert a is not None and b is not None
    assert b[0] < a[0] - 5  # +y world is to the image left for this camera


def test_fit_recovers_translation_and_yaw():
    cam = _look_at_camera()
    model = opr.ObjectModel(spec=_box_spec(), camera=cam)
    true_xy = np.array([0.04, -0.03])
    true_yaw = math.radians(35.0)
    real_mask = model.silhouette(true_xy, true_yaw)
    res = opr.fit_object(model, real_mask, yaw_mode="sweep", yaw_step_deg=15.0, yaw_min_gain=0.02)
    assert res.status == "refined"
    assert np.linalg.norm(res.xy_delta - true_xy) < 0.004
    # a box is 180-degree symmetric: accept either solution
    diff = (math.degrees(res.yaw_rad - true_yaw) + 90.0) % 180.0 - 90.0
    assert abs(diff) < 3.0
    assert res.iou_after > 0.9
    assert res.iou_after >= res.iou_before


def test_fit_translation_only_when_yaw_off():
    cam = _look_at_camera()
    model = opr.ObjectModel(spec=_box_spec(), camera=cam)
    real_mask = model.silhouette(np.array([-0.05, 0.02]), 0.0)
    res = opr.fit_object(model, real_mask, yaw_mode="off")
    assert res.yaw_rad == 0.0
    assert np.linalg.norm(res.xy_delta - np.array([-0.05, 0.02])) < 0.004


def test_symmetric_object_keeps_yaw():
    cam = _look_at_camera()
    cyl = trimesh.creation.cylinder(radius=0.04, height=0.05, sections=48)
    spec = ObjectPlacementSpec.from_world_vertices(
        safe_key="obj_cup",
        count=2,
        object_name="cup",
        vertices_world=np.asarray(cyl.vertices) + np.array([0.5, 0.0, 0.025]),
        base_pos=np.array([0.5, 0.0, 0.025]),
        base_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        faces=np.asarray(cyl.faces),
    )
    model = opr.ObjectModel(spec=spec, camera=cam)
    real_mask = model.silhouette(np.array([0.03, 0.03]), 0.0)
    res = opr.fit_object(model, real_mask, yaw_mode="sweep")
    assert res.yaw_rad == 0.0
    assert res.yaw_iou_range is not None and res.yaw_iou_range < 0.05
    assert np.linalg.norm(res.xy_delta - np.array([0.03, 0.03])) < 0.004


def test_footprint_conflict_keeps_original_pose_for_larger_mover():
    cam = _look_at_camera()
    big = opr.ObjectModel(spec=_box_spec("obj_big", extents=(0.3, 0.3, 0.1), pos=(0.45, 0.0, 0.05)), camera=cam)
    small = opr.ObjectModel(spec=_box_spec("obj_small", extents=(0.05, 0.05, 0.05), pos=(0.45, 0.2, 0.025)), camera=cam)
    models = {"obj_big": big, "obj_small": small}
    r_big = opr.FitResult("obj_big", 1, "big", "refined", xy_delta=np.array([0.0, 0.2]))
    r_small = opr.FitResult("obj_small", 2, "small", "refined", xy_delta=np.array([0.0, 0.0]))
    results = {"obj_big": r_big, "obj_small": r_small}
    notes = opr.resolve_footprint_conflicts(models, results)
    assert len(notes) == 1
    assert r_big.status == "conflict_not_applied" and r_small.status == "refined"
    assert r_big.to_json_dict(big)["apply"] is False
    # no conflict when they stay apart
    r_big.status = "refined"
    r_big.xy_delta = np.zeros(2)
    assert opr.resolve_footprint_conflicts(models, results) == []
    # a plate on a placemat is a stack, not a conflict: flat objects are exempt
    mat = opr.ObjectModel(spec=_box_spec("obj_mat", extents=(0.4, 0.3, 0.005), pos=(0.45, 0.0, 0.0025)), camera=cam)
    plate = opr.ObjectModel(spec=_box_spec("obj_plate", extents=(0.2, 0.2, 0.01), pos=(0.45, 0.0, 0.005)), camera=cam)
    stack = {"obj_mat": opr.FitResult("obj_mat", 3, "mat", "refined"), "obj_plate": opr.FitResult("obj_plate", 4, "plate", "refined")}
    assert opr.resolve_footprint_conflicts({"obj_mat": mat, "obj_plate": plate}, stack) == []
    assert all(r.status == "refined" for r in stack.values())


def test_from_dump_entry_levels_and_drops(tmp_path):
    """A bowl-like cylinder tilted 10 deg and floating 4 cm is rested on the plane."""
    from r2s.collision.support import DEFAULT_GLB_FRONT, DEFAULT_GLB_UP, align_up_and_front

    cyl = trimesh.creation.cylinder(radius=0.06, height=0.03, sections=48)
    # The evaluator imports GLBs with +y up: a cylinder whose axis is +y in the
    # file becomes vertical in the sim; then tilt it 10 deg about world x.
    cyl.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [1.0, 0.0, 0.0]))
    path = tmp_path / "bowl.glb"
    cyl.export(str(path))
    import_rot = align_up_and_front(np.asarray(DEFAULT_GLB_UP), np.asarray(DEFAULT_GLB_FRONT))
    tilt = math.radians(10.0)
    q_tilt = opr._axis_angle_quat_wxyz(np.array([1.0, 0.0, 0.0]), tilt)
    entry = {
        "safe_key": "obj_bowl",
        "count": 0,
        "object_name": "bowl",
        "asset_file": str(path),
        "scale": 1.0,
        "pos": [0.5, 0.0, 0.08],
        "quat_wxyz": list(q_tilt),
    }
    cam = _look_at_camera()
    raw = opr.ObjectModel.from_dump_entry(entry, cam, z_plane=None, level_max_deg=0.0)
    _, tilt_before = opr.minor_axis_tilt(raw.spec.centered_vertices)
    assert 9.0 < tilt_before < 11.0
    assert raw.spec.base_min_z > 0.05
    del import_rot

    rested = opr.ObjectModel.from_dump_entry(entry, cam, z_plane=0.02)
    _, tilt_after = opr.minor_axis_tilt(rested.spec.centered_vertices)
    assert tilt_after < 0.5
    assert abs(rested.level_deg - tilt_before) < 0.1
    assert abs(rested.spec.base_min_z - 0.0205) < 1e-6
    assert rested.z_delta < -0.03
    # the override quaternion reproduces the levelled vertices
    from r2s.collision.support import quat_wxyz_to_matrix

    R_new = quat_wxyz_to_matrix(rested.quat_for(0.0))
    R_raw = quat_wxyz_to_matrix(raw.spec.base_quat_wxyz)
    local = raw.spec.centered_vertices @ R_raw
    assert np.allclose(local @ R_new.T, rested.spec.centered_vertices, atol=1e-9)
    # a rested-only object is applied even without an XY change
    res = opr.FitResult("obj_bowl", 0, "bowl", "unchanged")
    assert res.to_json_dict(rested)["apply"] is True

    # an object already resting (within 1 cm of the plane) keeps its original
    # lowest-vertex height after levelling instead of being pushed lower
    entry_low = dict(entry, pos=[0.5, 0.0, 0.03])
    low_raw = opr.ObjectModel.from_dump_entry(entry_low, cam, z_plane=None, level_max_deg=0.0)
    low = opr.ObjectModel.from_dump_entry(entry_low, cam, z_plane=0.02)
    assert low.level_deg > 9.0
    assert abs(low.spec.base_min_z - low_raw.spec.base_min_z) < 1e-6


def test_oversized_reconstruction_is_not_applied():
    """A mesh 2x too large (4x mask area) keeps its original pose and reports the implied scale."""
    cam = _look_at_camera()
    true_model = opr.ObjectModel(spec=_box_spec(extents=(0.12, 0.05, 0.04)), camera=cam)
    real_mask = true_model.silhouette(np.array([0.03, 0.02]), 0.0)
    big = opr.ObjectModel(spec=_box_spec(extents=(0.24, 0.10, 0.08), pos=(0.45, 0.05, 0.04)), camera=cam)
    res = opr.fit_object(big, real_mask, yaw_mode="off")
    assert res.status == "size_mismatch_not_applied"
    assert np.allclose(res.xy_delta, 0.0)
    assert res.suggested_scale is not None and 0.4 < res.suggested_scale < 0.6
    assert res.to_json_dict(big)["apply"] is False
    # the same mesh at the right size is accepted
    ok = opr.fit_object(opr.ObjectModel(spec=_box_spec(), camera=cam), real_mask, yaw_mode="off")
    assert ok.status == "refined"


def test_missing_mask_is_skipped_and_not_applied():
    cam = _look_at_camera()
    model = opr.ObjectModel(spec=_box_spec(), camera=cam)
    res = opr.fit_object(model, None)
    assert res.status == "skipped_no_mask"
    assert res.to_json_dict(model)["apply"] is False


def test_quat_composition_matches_world_vertices():
    """The override quaternion must rotate the mesh exactly like world_vertices(yaw)."""
    from r2s.collision.support import quat_wxyz_to_matrix

    spec = _box_spec(quat=(0.9238795, 0.0, 0.0, 0.3826834))
    cam = _look_at_camera()
    model = opr.ObjectModel(spec=spec, camera=cam)
    yaw = math.radians(20.0)
    q_new = model.quat_for(yaw)
    R_new = quat_wxyz_to_matrix(q_new)
    R_base = quat_wxyz_to_matrix(spec.base_quat_wxyz)
    local = spec.centered_vertices @ R_base  # back to the body frame
    expected = spec.world_vertices(spec.base_pos, yaw)
    got = local @ R_new.T + spec.base_pos
    assert np.allclose(got, expected, atol=1e-9)


def test_override_roundtrip(tmp_path):
    cam = _look_at_camera()
    model = opr.ObjectModel(spec=_box_spec(), camera=cam)
    res = opr.fit_object(model, model.silhouette(np.array([0.02, 0.0]), 0.0), yaw_mode="off")
    payload = {"version": 1, "objects": [res.to_json_dict(model), {"safe_key": "obj_x", "pos": [0, 0, 0], "quat_wxyz": [1, 0, 0, 0], "apply": False}]}
    path = tmp_path / "refined.json"
    path.write_text(json.dumps(payload))
    override = opo.load_object_pose_override(str(path))
    placed = override.placed
    assert [p.safe_key for p in placed] == ["obj_box"]
    assert np.allclose(placed[0].pos, model.pos_for(res.xy_delta))
    assert np.allclose(override.originals["obj_box"][0], model.spec.base_pos)

    class _Entity:
        def __init__(self, drift=0.0):
            self.pos = None
            self.quat = None
            self.relative = None
            self.drift = drift

        def set_pos(self, pos, relative=False):
            self.pos = np.asarray(pos, dtype=np.float64)
            self.relative = relative

        def set_quat(self, quat, relative=False):
            self.quat = quat

        def get_pos(self, relative=False):
            return self.pos + np.array([self.drift, 0.0, 0.0])

    class _Env:
        def __init__(self, drift):
            self.entities = {"obj_box": _Entity(drift)}
            self.steps = 0

        def step(self, action=None):
            self.steps += 1

    env = _Env(drift=0.0)
    lines = opo.apply_object_pose_override(env, override)
    assert np.allclose(env.entities["obj_box"].pos, placed[0].pos)
    assert env.entities["obj_box"].relative is True  # user/morph frame, not the solver frame
    assert len(lines) == 1 and "obj_box" in lines[0]

    # settle check: an object that physics throws 5 cm away is reverted to its load-time pose
    env = _Env(drift=0.05)
    lines = opo.apply_object_pose_override(env, override, settle_steps=3)
    assert env.steps == 6
    assert np.allclose(env.entities["obj_box"].pos, override.originals["obj_box"][0])
    assert any("reverted" in line for line in lines)
    env = _Env(drift=0.002)
    lines = opo.apply_object_pose_override(env, override, settle_steps=3)
    assert np.allclose(env.entities["obj_box"].pos, placed[0].pos)
    assert any(line.startswith("settled") for line in lines)


def test_dump_writer_and_loader(tmp_path):
    spec = _box_spec()
    path = tmp_path / "poses.json"
    opo.write_object_pose_dump(str(path), {"obj_box": spec}, scene_name="synthetic", scene_meta_dir=None, z_plane=0.02)
    dump = opr.load_object_pose_dump(str(path))
    assert dump["objects"][0]["safe_key"] == "obj_box"
    assert np.allclose(dump["objects"][0]["pos"], spec.base_pos)
    assert opr.select_objects(dump, ["box"]) == dump["objects"]
    assert opr.select_objects(dump, ["7"]) == []


def test_delta_mode_reapplies_change_on_this_runs_pose():
    from r2s.collision.support import quat_wxyz_to_matrix

    fit_load_pos = np.array([0.5, 0.0, 0.03])
    fit_load_quat = np.array([1.0, 0.0, 0.0, 0.0])
    yaw = math.radians(30.0)
    fit_pos = fit_load_pos + np.array([0.04, -0.02, -0.01])
    fit_quat = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
    entry = {
        "safe_key": "obj_a",
        "object_name": "a",
        "pos": list(fit_pos),
        "quat_wxyz": list(fit_quat),
        "load_pos": list(fit_load_pos),
        "load_quat_wxyz": list(fit_load_quat),
        "base_pos": list(fit_load_pos),
        "base_quat_wxyz": list(fit_load_quat),
        "xy_delta_m": [0.04, -0.02],
        "yaw_delta_deg": 30.0,
    }
    override = opo.ObjectPoseOverride(
        placed=[
            opo.PlacedPose("obj_a", "a", fit_pos, fit_quat, yaw, np.array([0.04, -0.02]), "refined"),
        ],
        originals={"obj_a": (fit_load_pos, fit_load_quat)},
    )
    del entry
    # this run loaded the object 2 cm higher and pre-rotated 90 deg about z
    own_pos = fit_load_pos + np.array([0.0, 0.0, 0.02])
    own_quat = np.array([math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)])
    spec = _box_spec("obj_a", pos=tuple(own_pos), quat=tuple(own_quat))
    resolved, notes = opo.resolve_override_for_run(override, {"obj_a": spec}, mode="delta")
    p = resolved.placed[0]
    # XY delta re-applied, z kept from this run's own load pose
    assert np.allclose(p.pos, own_pos + np.array([0.04, -0.02, 0.0]))
    R_expected = quat_wxyz_to_matrix(fit_quat) @ quat_wxyz_to_matrix(own_quat)
    assert np.allclose(quat_wxyz_to_matrix(p.quat_wxyz), R_expected, atol=1e-9)
    assert np.allclose(resolved.originals["obj_a"][0], own_pos)
    # absolute mode keeps the file pose but reverts to this run's pose
    resolved_abs, _ = opo.resolve_override_for_run(override, {"obj_a": spec}, mode="absolute")
    assert np.allclose(resolved_abs.placed[0].pos, fit_pos)
    assert np.allclose(resolved_abs.originals["obj_a"][0], own_pos)
    # objects this run did not load are dropped with a note
    resolved_none, notes = opo.resolve_override_for_run(override, {"obj_other": spec}, mode="delta")
    assert resolved_none.placed == [] and notes


def test_fit_scene_poses_end_to_end(tmp_path):
    """The step's library entry point on a synthetic scene: files in, refined_poses.json + overlays out."""
    cam = _look_at_camera()
    box = trimesh.creation.box(extents=(0.12, 0.05, 0.04))
    # GLB frame is +y up; the evaluator's import rotation turns it z-up.
    box.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [1.0, 0.0, 0.0]))
    glb = tmp_path / "assets" / "image_0001_segmented.glb"
    glb.parent.mkdir()
    box.export(str(glb))
    entry = {"safe_key": "obj_image_0001_segmented", "count": 1, "object_name": "box", "asset_file": str(glb), "scale": 1.0, "pos": [0.45, 0.05, 0.0205], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}
    dump = {"version": 1, "scene": "synthetic", "z_plane": 0.0, "physics_top": 0.0, "objects": [entry]}
    dump_path = tmp_path / "object_poses.json"
    dump_path.write_text(json.dumps(dump))
    model = opr.ObjectModel.from_dump_entry(entry, cam, z_plane=0.0)
    real_mask = model.silhouette(np.array([0.03, -0.02]), 0.0)
    masks_dir = tmp_path / "bounding_boxes_and_masks"
    masks_dir.mkdir()
    (masks_dir / "frame_0000_bounding_boxes_and_masks.json").write_text(json.dumps([{"count": [1], "object_name": "box", "mask": real_mask.astype(int).tolist()}]))
    np.save(tmp_path / "intrinsics.npy", cam.K)
    np.save(tmp_path / "extrinsics.npy", cam.T_w2c)
    out = tmp_path / "posefit" / "refined_poses.json"
    payload = opr.fit_scene_poses(
        dump_path,
        scene="synthetic",
        intrinsics=tmp_path / "intrinsics.npy",
        extrinsics=tmp_path / "extrinsics.npy",
        bbox_masks_dir=masks_dir,
        out_path=out,
        settings=opr.PoseFitSettings(yaw="off"),
        overlay_dir=tmp_path / "posefit" / "overlays",
        log=lambda line: None,
    )
    written = json.loads(out.read_text())
    assert written["objects"][0]["status"] == payload["objects"][0]["status"] == "refined"
    assert np.allclose(written["objects"][0]["xy_delta_m"], [0.03, -0.02], atol=0.004)
    assert (tmp_path / "posefit" / "overlays" / "overlay_all.png").is_file()
    assert written["settings"]["yaw"] == "off" and written["settings"]["min_iou"] == 0.3
