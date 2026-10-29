"""Ported from the old test_asset_refine.py (r2s.objects.refine.*; synthetic scene, no simulator)."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
trimesh = pytest.importorskip("trimesh")
pytest.importorskip("scipy")

from r2s.objects import overrides as oso  # noqa: E402
from r2s.objects import pose_fit as opr  # noqa: E402
from r2s.objects.placement_types import ObjectPlacementSpec  # noqa: E402
from r2s.objects.refine import common as arc  # noqa: E402
from r2s.objects.refine import fit as arf  # noqa: E402
from r2s.objects.refine import metrics as arm  # noqa: E402

from test_objects_pose_fit import _box_spec, _look_at_camera  # noqa: E402

Z_PLANE = 0.02


def _model(spec, cam):
    return opr.ObjectModel(spec=spec, camera=cam, load_pos=spec.base_pos.copy(), load_quat=spec.base_quat_wxyz.copy())


def _refine_paths(root, *, scene="synthetic", cropped_collider=None) -> arc.RefinePaths:
    root = Path(root)
    return arc.RefinePaths(
        scene=scene,
        intrinsics=root / "intrinsics.npy",
        extrinsics=root / "extrinsics.npy",
        bbox_masks_dir=root / "bounding_boxes_and_masks",
        images_dir=root / "images0",
        depth_dir=root / "depth",
        tcp=root / "tcp.npy",
        grippers=root / "grippers.npy",
        joints_old=root / "joints_old.npy",
        joints=root / "joints.npy",
        phys_props=root / "masks" / "physical_properties.json",
        task_spec=root / f"{scene}.py",
        cropped_collider=Path(cropped_collider) if cropped_collider else root / "cropped_collision" / "none.glb",
        object_dump=root / "objects" / "stock" / "object_poses.json",
        refined_poses=root / "objects" / "posefit" / "refined_poses.json",
        refine_root=root / "objects" / "refine",
    )


def _evidence(cam, masks, *, tcp=None, grippers=None, task=None, depth=None, phys=None):
    paths = _refine_paths("/nonexistent")
    ev = arm.SceneEvidence(paths=paths, camera=cam, z_plane=Z_PLANE, real_masks=masks, phys=phys or {}, task=task or arc.TaskInfo(scene="synthetic"), tcp=tcp, grippers=grippers, depth=depth)
    if tcp is not None and grippers is not None:
        ev.events = arc.grasp_events(grippers)
        ev.targets = arm.identify_grasp_targets(cam, tcp, ev.events, masks, task_counts=(task.task_counts if task else None) or None)
    return ev


# ---------------------------------------------------------------------------
# common
# ---------------------------------------------------------------------------
def test_scaled_model_scales_about_root_and_rerests():
    cam = _look_at_camera()
    m = _model(_box_spec(pos=(0.45, 0.05, 0.04)), cam)
    lo, hi = m.spec.base_aabb
    s = arc.scaled_model(m, 1.3)
    lo2, hi2 = s.spec.base_aabb
    assert np.allclose(hi2 - lo2, 1.3 * (hi - lo))
    assert s.spec.base_min_z == pytest.approx(m.spec.base_min_z, abs=1e-9)  # lowest vertex height preserved
    assert s.spec.scale == pytest.approx(1.3)
    assert arc.scaled_model(m, 1.0) is m


def test_grasp_events_rising_edges():
    g = np.array([0, 0, 0.2, 0.7, 0.9, 0.9, 0.3, 0, 0, 0, 0.8, 0.8, 0, 0])
    ev = arc.grasp_events(g, min_gap=2)
    assert ev == [{"onset": 3, "release": 6}, {"onset": 10, "release": 12}]
    assert arc.grasp_events(np.zeros(5)) == []
    # two closes 1 frame apart merge
    g2 = np.array([0, 1, 1, 0, 1, 1, 0])
    assert arc.grasp_events(g2, min_gap=5) == [{"onset": 1, "release": 6}]


def test_fk_flange_matches_dh_reference():
    from r2s.objects.refine.franka_fk import _HAND_D, fk_tcp

    q = np.array([0.1, -0.5, 0.2, -2.0, 0.1, 1.8, 0.7])
    pos, R = arc.fk_flange(q)
    tcp_ref = fk_tcp(q)  # flange + hand along z
    assert np.allclose(pos + R[:, 2] * _HAND_D, tcp_ref, atol=1e-9)
    gp = arc.grasp_points_from_joints(np.stack([q, q]))
    assert gp.shape == (2, 3) and np.allclose(gp[0], tcp_ref)


def test_task_info_parsing(tmp_path):
    src = '''"""droidX — "put toy bear into black bowl"."""
from evalkit.spec import TaskSpec
# 0 black bowl  1 white cup (8cm)  2 stuffed tiger
BEAR, BOWL = 2, 0
TASK = TaskSpec(
    scene="droidX",
    instruction="put toy bear into black bowl",
    objects={"bear": BEAR, "bowl": BOWL},
    notes="Bear (11x13x7cm) barely fits the bowl mouth (15x15cm); placement is tight.",
)
'''
    p = tmp_path / "droidX.py"
    p.write_text(src)
    info = arc.load_task_info(str(p), scene="droidX", object_names={0: "black bowl", 1: "white cup", 2: "stuffed tiger"})
    assert info.instruction == "put toy bear into black bowl"
    assert info.objects == {"bear": 2, "bowl": 0}
    assert info.task_counts == [2, 0]
    assert info.dims_cm[2] == [7.0, 11.0, 13.0]  # the note's bear, not the BOWL constant
    assert info.dims_cm[0] == [15.0, 15.0]
    assert info.dims_cm[1] == [8.0]
    assert "barely fits" in info.notes
    assert arc.load_task_info(str(tmp_path / "missing.py")).objects == {}


def test_candidate_roundtrip_through_both_loaders(tmp_path):
    cam = _look_at_camera()
    m = _model(_box_spec(safe_key="obj_image_0000_segmented"), cam)
    entry = arc.candidate_object_entry(arc.scaled_model(m, 1.2), xy_delta=(0.01, -0.02), yaw_rad=math.radians(30), scale_multiplier=1.2, status="refined")
    cand = arc.Candidate(scene="synthetic", candidate_id="B_v1", method="B", objects=[entry])
    path = cand.save(str(tmp_path / "candidate.json"))
    back = arc.Candidate.load(path)
    assert back.candidate_id == "B_v1" and back.entry("obj_image_0000_segmented")["scale_multiplier"] == pytest.approx(1.2)
    from r2s.objects import overrides as opo

    ov = opo.load_object_pose_override(path)
    assert len(ov.placed) == 1 and np.allclose(ov.placed[0].xy_delta, [0.01, -0.02])
    assert "obj_image_0000_segmented" in ov.originals
    assert oso.load_object_scale_multipliers(path) == {"image_0000_segmented": pytest.approx(1.2)}
    # a stock candidate carries no override and no scale change
    dump = {"z_plane": Z_PLANE, "objects": [{"safe_key": "obj_a", "count": 1, "object_name": "box", "asset_file": "", "scale": 1.0, "pos": [0.4, 0.0, 0.04], "quat_wxyz": [1, 0, 0, 0]}]}
    with pytest.raises(Exception):
        arc.candidate_stock(dump, cam, scene="s")  # asset_file "" cannot be loaded; real dumps carry a GLB


def test_model_at_settled_pose_applies_relative_rotation():
    cam = _look_at_camera()
    m = _model(_box_spec(extents=(0.12, 0.05, 0.04)), cam)
    q_tilt = np.array([math.cos(math.radians(10)), math.sin(math.radians(10)), 0.0, 0.0])  # 20 deg about x
    s = arc.model_at_settled_pose(m, [0.5, 0.1, 0.05], q_tilt)
    assert np.allclose(s.spec.base_pos, [0.5, 0.1, 0.05])
    lo, hi = s.spec.base_aabb
    assert (hi - lo)[1] > 0.05 + 1e-3  # tilting about x grows the y extent


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def test_render_depth_splat_matches_projection():
    cam = _look_at_camera()
    m = _model(_box_spec(), cam)
    z = arm.render_depth_splat(m, np.zeros(2), 0.0)
    sil = m.silhouette(np.zeros(2), 0.0)
    assert np.isfinite(z[sil]).mean() > 0.95
    verts = m.spec.world_vertices(m.spec.base_pos, 0.0)
    uv, depth = cam.project(verts)
    top = verts[:, 2].argmax()
    u, v = np.round(uv[top]).astype(int)
    assert abs(z[v, u] - depth[top]) < 0.01


def test_ring_calibration_and_depth_residual():
    cam = _look_at_camera()
    m = _model(_box_spec(pos=(0.45, 0.05, 0.02 + 0.02)), cam)
    # synthetic real depth: table plane everywhere, box front where the box is; measured = 0.9 * true + 0.05
    H, W = cam.height, cam.width
    cue0 = arm.DepthCue(camera=cam, depth=np.zeros((H, W)), z_plane=Z_PLANE)
    true = cue0.plane_depth.copy()
    zbox = arm.render_depth_splat(m, np.zeros(2), 0.0)
    sil = m.silhouette(np.zeros(2), 0.0)
    true[sil] = zbox[sil]
    meas = 0.9 * true + 0.05
    cue = arm.DepthCue(camera=cam, depth=meas, z_plane=Z_PLANE)
    calib = arm.ring_calibration(sil, None, cue)
    assert calib["quality"] and abs(calib["scale"] - 1 / 0.9) < 1e-3 and abs(calib["offset"] + 0.05 / 0.9) < 1e-3
    res = arm.depth_residual(m, np.zeros(2), 0.0, sil, cue, calib)
    assert abs(res["front_residual_m"]) < 0.003
    assert abs(res["height_est_m"] - 0.04) < 0.01
    # a mesh 30 % too big sits closer to the camera on the near face -> negative residual
    big = arc.scaled_model(m, 1.3)
    res_big = arm.depth_residual(big, np.zeros(2), 0.0, sil, cue, calib)
    assert res_big["front_residual_m"] > 0.003 or res_big["height_sim_m"] > res["height_sim_m"]


def test_dims_prior_residual():
    r = arm.dims_prior_residual([0.04, 0.05, 0.12], phys_dims_m=[0.04, 0.05, 0.12])
    assert r["residual"] == pytest.approx(0.0) and r["source"] == "phys" and r["suggested_multiplier"] == pytest.approx(1.0)
    r = arm.dims_prior_residual([0.04, 0.05, 0.12], phys_dims_m=[0.04, 0.05, 0.12], note_dims_cm=[8.0, 10.0, 24.0])
    assert r["source"] == "note" and r["suggested_multiplier"] == pytest.approx(2.0, rel=1e-6)
    r = arm.dims_prior_residual([0.04, 0.05, 0.12], note_dims_cm=[6.0])  # single number = largest extent
    assert r["suggested_multiplier"] == pytest.approx(0.5, rel=1e-6)
    assert arm.dims_prior_residual([0.1, 0.1, 0.1])["source"] is None


def test_grasp_targets_and_footprint_distance():
    cam = _look_at_camera()
    box = _box_spec(safe_key="obj_box", pos=(0.45, 0.05, 0.04))
    m = _model(box, cam)
    other = _model(_box_spec(safe_key="obj_other", pos=(0.30, -0.20, 0.04)), cam)
    masks = {1: ("box", m.silhouette(np.zeros(2), 0.0)), 2: ("other", other.silhouette(np.zeros(2), 0.0))}
    # fingertip path: descends onto the box centre and closes at frame 20
    T = 40
    tcp = np.tile(np.array([0.45, 0.05, 0.30]), (T, 1))
    tcp[10:21, 2] = np.linspace(0.30, 0.05, 11)
    tcp[21:, 2] = 0.05
    grippers = np.zeros(T)
    grippers[20:35] = 1.0
    ev = _evidence(cam, masks, tcp=tcp, grippers=grippers)
    assert len(ev.targets) == 1 and ev.targets[0]["count"] == 1 and ev.targets[0]["onset"] == 20
    assert len(ev.targets[0]["anchor_points"]) == 11
    fd = arm.footprint_distance(m, np.zeros(2), 0.0, tcp[20])
    assert fd["d_foot_m"] == 0.0 and fd["hit"] and fd["z_ok"]
    assert fd["dz_top_m"] == pytest.approx(0.05 - 0.06, abs=1e-6)
    far = arm.footprint_distance(m, np.array([0.10, 0.0]), 0.0, tcp[20])
    assert far["d_foot_m"] == pytest.approx(0.04, abs=1e-6) and not far["hit"]
    off = arm.grasp_consistency_offline({"obj_box": m, "obj_other": other}, {k: (np.zeros(2), 0.0) for k in ("obj_box", "obj_other")}, ev.targets, {1: "obj_box", 2: "obj_other"})
    assert off[0]["hit"] and off[0]["nearest_safe_key"] == "obj_box"
    g = arm.grasp_score(off)
    assert g["G"] == pytest.approx(1.0)
    assert arm.grasp_score(off, unstable_keys={"obj_box"})["G"] == pytest.approx(0.0)
    assert arm.grasp_penalty(fd) < arm.grasp_penalty(far)


def test_grasp_consistency_replay(tmp_path):
    T = 100
    ticks = np.arange(0, 300, 3)
    np.savez(tmp_path / "replay_frames.npz", control_tick=ticks)
    run = tmp_path / "run"
    run.mkdir()
    names = np.array(["obj_box", "obj_other"])
    pos = np.zeros((300, 2, 3))
    pos[:, 0] = [0.45, 0.05, 0.04]
    pos[:, 1] = [0.30, -0.20, 0.04]
    pos[70:, 0, 2] += np.linspace(0, 0.10, 230)  # box lifts after the grasp
    contact = np.zeros((300, 2), dtype=int)
    contact[60:80, 0] = 2  # box touched around the onset tick (frame 20 -> tick 60)
    contact[10:12, 1] = 1  # other object bumped early = false contact
    np.savez(run / "x_sim_stats.npz", object_names=names, object_pos=pos, object_quat_wxyz=np.tile([1.0, 0, 0, 0], (300, 2, 1)), robot_object_contact_count=contact)
    tcp = np.tile(np.array([0.45, 0.05, 0.05]), (T, 1))
    tcp[20:, 2] += np.linspace(0, 0.10, 80)
    targets = [{"onset": 20, "release": 35, "count": 1, "tcp_onset": list(tcp[20])}]
    rep = arm.grasp_consistency_replay(str(tmp_path), targets, tcp, {1: "obj_box", 2: "obj_other"})
    assert rep[0]["contact_onset"] and rep[0]["lifted"] and rep[0]["false_contacts"] == 1
    assert rep[0]["comotion"] > 0.5
    off = [{"onset": 20, "safe_key": "obj_box", "hit": True, "z_ok": True, "d_foot_m": 0.0}]
    g = arm.grasp_score(off, rep)
    assert g["G"] == pytest.approx(1.0 + 0.5 + 0.5 - 0.5)
    stab = arm.stability_from_replay(str(tmp_path))
    assert stab["obj_box"]["z_drop_m"] > 0.01  # lifted within the window counts as motion (expected: warmup only)


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------
def test_fit_object_joint_recovers_scale_shift_yaw():
    cam = _look_at_camera()
    truth = _model(_box_spec(extents=(0.12 * 1.3, 0.05 * 1.3, 0.04 * 1.3), pos=(0.45 + 0.03, 0.05 - 0.02, 0.02 + 0.026), quat=(1, 0, 0, 0)), cam)
    yaw_true = math.radians(25.0)
    mask = truth.silhouette(np.zeros(2), yaw_true)
    start = _model(_box_spec(pos=(0.45, 0.05, 0.04)), cam)
    ev = _evidence(cam, {1: ("box", mask)})
    res = arf.fit_object_joint(start, ev, key="obj_box", weights={**arm.DEFAULT_WEIGHTS, "dims": 0.0}, yaw_step_deg=15.0, orientation_mode="yaw")
    assert res.status == "refined"
    assert abs(res.scale_multiplier - 1.3) <= 0.05
    assert np.allclose(res.xy_delta, [0.03, -0.02], atol=0.006)
    yaw_err = abs(((math.degrees(res.yaw_rad) - 25.0 + 90) % 180) - 90)  # box is 180-symmetric
    assert yaw_err <= 4.0
    assert res.cues["iou"] > 0.85


def test_fit_object_joint_uses_grasp_anchor_and_orientation():
    cam = _look_at_camera()
    # real: a 12x5x4 box standing on its 5x4 face is 12 cm tall; mesh starts lying (4 cm tall)
    lying = _box_spec(extents=(0.12, 0.05, 0.04), pos=(0.45, 0.05, 0.02 + 0.02))
    start = _model(lying, cam)
    standing = _model(_box_spec(extents=(0.04, 0.05, 0.12), pos=(0.45, 0.05, 0.02 + 0.06)), cam)
    mask = standing.silhouette(np.zeros(2), 0.0)
    T = 30
    tcp = np.tile(np.array([0.45, 0.05, 0.02 + 0.10]), (T, 1))  # closes 2 cm below the standing top
    grippers = np.zeros(T)
    grippers[15:] = 1.0
    ev = _evidence(cam, {1: ("box", mask)}, tcp=tcp, grippers=grippers)
    assert ev.targets and ev.targets[0]["count"] == 1
    res = arf.fit_object_joint(start, ev, key="obj_box", target_points=[np.asarray(p) for p in ev.targets[0]["anchor_points"]], orientation_mode="principal", orientation_top_k=3, z_plane=Z_PLANE)
    assert res.status == "refined" and res.orientation != "as-is"
    lo, hi = res.model.spec.world_aabb(res.model.pos_for(res.xy_delta), res.yaw_rad)
    assert (hi - lo)[2] == pytest.approx(0.12 * res.scale_multiplier, abs=0.01)
    assert res.cues["grasp_anchor"]["hit"]


def test_fit_scene_writes_candidate_and_review(tmp_path, monkeypatch):
    cam = _look_at_camera()
    truth = _model(_box_spec(extents=(0.12, 0.05, 0.04), pos=(0.47, 0.03, 0.04)), cam)
    mask = truth.silhouette(np.zeros(2), 0.0)
    mesh = trimesh.creation.box(extents=(0.12, 0.05, 0.04))
    glb = tmp_path / "box.glb"
    mesh.export(str(glb))
    dump = {"version": 1, "scene": "synthetic", "z_plane": Z_PLANE, "objects": [{"safe_key": "obj_box", "count": 1, "object_name": "box", "asset_file": str(glb), "scale": 1.0, "pos": [0.45, 0.05, 0.04], "quat_wxyz": [1, 0, 0, 0]}]}
    paths = _refine_paths(tmp_path)
    (tmp_path / "bounding_boxes_and_masks").mkdir()
    (tmp_path / "bounding_boxes_and_masks" / "frame_00001_bounding_boxes_and_masks.json").write_text(json.dumps([{"object_name": "box", "count": 1, "mask": mask.astype(int).tolist()}]))
    np.save(tmp_path / "intrinsics.npy", cam.K)
    np.save(tmp_path / "extrinsics.npy", cam.T_w2c)
    with pytest.raises(FileNotFoundError):  # resting on the collider needs the collision.crop output
        arf.fit_scene(paths, dump, candidate_id="B_test", orientation_mode="yaw", log=lambda *a, **k: None)
    cand, models, results, evidence = arf.fit_scene(paths, dump, candidate_id="B_test", orientation_mode="yaw", rest_on_collider=False, log=lambda *a, **k: None)
    assert cand.method == "B" and cand.entry("obj_box")["status"] == "refined"
    entry = cand.entry("obj_box")
    assert entry["apply"] and "scale_multiplier" in entry and "terms" in entry
    out = cand.save(str(tmp_path / "candidate.json"))
    written = arf.write_review(str(tmp_path / "review"), paths, evidence, dump, [(cand, models, arm.score_candidate(cand, models, evidence))])
    assert any(p.endswith("residual_table.md") for p in written) and any(p.endswith("plan_view.png") for p in written)
    assert "| box [1] | B_test |" in open(os.path.join(tmp_path, "review", "residual_table.md")).read()
    # the harness accepts the file as a v2 candidate
    back = arc.Candidate.load(out)
    assert back.candidate_id == "B_test"


def test_bottom_dome_profile_separates_domes_from_flat_bottoms():
    box = trimesh.creation.box(extents=(0.08, 0.08, 0.10))
    ball = trimesh.creation.icosphere(subdivisions=3, radius=0.04)
    flat = arm.bottom_dome_profile(np.asarray(box.vertices) + np.array([0.5, 0.0, 0.05]))
    dome = arm.bottom_dome_profile(np.asarray(ball.vertices) + np.array([0.5, 0.0, 0.04]))
    assert not flat["domed_bottom"] and flat["contact_fraction"] > 0.9
    assert dome["domed_bottom"] and dome["contact_fraction"] < 0.3 and dome["half_section_height_m"] > 0.005


def test_upright_by_symmetry_axis_levels_a_leaning_cylinder():
    cam = _look_at_camera()
    cyl = trimesh.creation.cylinder(radius=0.03, height=0.10, sections=48).subdivide().subdivide()  # dense like a reconstruction
    q_lean = np.array([math.cos(math.radians(15)), math.sin(math.radians(15)), 0.0, 0.0])  # 30 deg about x
    from r2s.objects.pose_fit import _quat_to_matrix

    verts = np.asarray(cyl.vertices) @ _quat_to_matrix(q_lean).T
    spec = ObjectPlacementSpec.from_world_vertices(safe_key="obj_cup", count=1, object_name="blue cup", vertices_world=verts + np.array([0.45, 0.05, 0.06]), base_pos=np.array([0.45, 0.05, 0.06]), base_quat_wxyz=q_lean, faces=np.asarray(cyl.faces))
    m = _model(spec, cam)
    axis, cv = arc.symmetry_axis(m.spec.centered_vertices)
    assert abs(math.degrees(math.acos(axis[2])) - 30.0) < 1.0 and cv < 0.3
    up, info = arc.upright_by_symmetry_axis(m, z_plane=Z_PLANE)
    assert info["applied"] and info["axis_tilt_deg_after"] < 0.5
    assert up.spec.base_min_z == pytest.approx(Z_PLANE + 0.0005, abs=1e-6)
    lo, hi = up.spec.base_aabb
    assert (hi - lo)[2] == pytest.approx(0.10, abs=1e-3)  # standing: full height along z
    # lying flat (~90 deg) is left alone; a class check gates it
    q_flat = np.array([math.cos(math.radians(45)), math.sin(math.radians(45)), 0.0, 0.0])
    verts2 = np.asarray(cyl.vertices) @ _quat_to_matrix(q_flat).T
    spec2 = ObjectPlacementSpec.from_world_vertices(safe_key="obj_b", count=2, object_name="dressing bottle", vertices_world=verts2 + np.array([0.4, 0, 0.03]), base_pos=np.array([0.4, 0, 0.03]), base_quat_wxyz=q_flat, faces=np.asarray(cyl.faces))
    _, info2 = arc.upright_by_symmetry_axis(_model(spec2, cam), z_plane=Z_PLANE)
    assert not info2["applied"] and info2["axis_tilt_deg_before"] > 85
    assert arc.is_upright_class("orange cup") and not arc.is_upright_class("stuffed tiger")


def test_fit_joint_rests_on_the_cropped_collider(tmp_path):
    """fit_joint end to end: candidate.json written, object re-rested on a raised collider patch."""
    cam = _look_at_camera()
    truth = _model(_box_spec(extents=(0.12, 0.05, 0.04), pos=(0.47, 0.03, 0.04)), cam)
    mask = truth.silhouette(np.zeros(2), 0.0)
    mesh = trimesh.creation.box(extents=(0.12, 0.05, 0.04))
    glb = tmp_path / "box.glb"
    mesh.export(str(glb))
    dump = {"version": 1, "scene": "synthetic", "z_plane": Z_PLANE, "objects": [{"safe_key": "obj_box", "count": 1, "object_name": "box", "asset_file": str(glb), "scale": 1.0, "pos": [0.45, 0.05, 0.04], "quat_wxyz": [1, 0, 0, 0]}]}
    dump_path = tmp_path / "object_poses.json"
    dump_path.write_text(json.dumps(dump))
    (tmp_path / "bounding_boxes_and_masks").mkdir()
    (tmp_path / "bounding_boxes_and_masks" / "frame_00001_bounding_boxes_and_masks.json").write_text(json.dumps([{"object_name": "box", "count": 1, "mask": mask.astype(int).tolist()}]))
    np.save(tmp_path / "intrinsics.npy", cam.K)
    np.save(tmp_path / "extrinsics.npy", cam.T_w2c)
    # baked world-space crop (z-up): a table patch 3 cm above the plane under the box
    table = trimesh.Trimesh(vertices=[[0.2, -0.3, 0.05], [0.8, -0.3, 0.05], [0.8, 0.3, 0.05], [0.2, 0.3, 0.05]], faces=[[0, 1, 2], [0, 2, 3]], process=False)
    crop = tmp_path / "crop.glb"
    table.export(str(crop))
    paths = _refine_paths(tmp_path, cropped_collider=crop)
    lines = []
    cand = arf.fit_joint(paths, dump_path=dump_path, out_path=tmp_path / "B_test" / "candidate.json", log=lines.append)
    entry = cand.entry("obj_box")
    assert cand.candidate_id == "B_test" and cand.parent == "stock" and cand.source_dump == str(dump_path)
    assert any("rested on the collider" in n for n in entry["notes"])
    # the GLB box is 5 cm tall once imported z-up: its lowest vertex rests 0.5 mm above the patch
    assert entry["scale_multiplier"] == 1.0
    assert entry["pos"][2] - 0.025 == pytest.approx(0.05 + 0.0005, abs=1e-6)
    assert (tmp_path / "B_test" / "candidate.json").is_file() and any("[fit-joint] wrote" in line for line in lines)
