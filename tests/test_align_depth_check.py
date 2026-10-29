"""``align.check`` on a synthetic scene: a table plane at z = 0 seen by a DROID-like camera.

The sensor depth is the exact depth of the plane, so the metrics of the true
alignment are ~0 and the sensor-optimal scale is the fitted one; scaling the
sensor depth by 5 % moves the optimal scale by 5 %. Run once through the CLI to
check the step plumbing (inputs root, manifest, outputs).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from r2s import cli
from r2s.align import depth_check as dc
from r2s.common.transforms import collider_world_transform, glb_import_rotation
from r2s.layout import SceneLayout
from r2s.surface.geometry import DEFAULT_WORKSPACES

trimesh = pytest.importorskip("trimesh")
pytest.importorskip("igl")
pytest.importorskip("embreex")

H, W = 72, 128
K = np.array([[60.0, 0.0, W / 2], [0.0, 60.0, H / 2], [0.0, 0.0, 1.0]])
CAM = np.array([0.35, 0.43, 0.45])
POSE = np.array([0.2, 0.1, 0.4, 0.7])  # [tx, ty, tz, rz]
SCALE = 0.9


def world_to_camera(cam, target) -> np.ndarray:
    f = (target - cam) / np.linalg.norm(target - cam)
    r = np.cross(f, [0.0, 0.0, 1.0])
    r /= np.linalg.norm(r)
    E = np.eye(4)
    E[:3, :3] = np.stack([r, np.cross(f, r), f])
    E[:3, 3] = -E[:3, :3] @ cam
    return E


def write_scene(scene_dir: Path, mask_dir: Path, align_dir: Path, world_dir: Path, depth_factor: float = 1.0) -> None:
    """Episode files, one object mask, the alignment and a collider GLB whose posed copy is the z = 0 plane."""
    E = world_to_camera(CAM, np.array([0.8, 0.0, 0.0]))
    for d in (scene_dir / "depth", scene_dir / "images0", mask_dir, align_dir, world_dir):
        d.mkdir(parents=True, exist_ok=True)
    np.save(scene_dir / "intrinsics.npy", K)
    np.save(scene_dir / "extrinsics.npy", E)
    cam, R, dworld = dc.camera_rays(scene_dir / "intrinsics.npy", scene_dir / "extrinsics.npy", (H, W))
    depth = np.where(dworld[..., 2] < 0, -cam[2] / dworld[..., 2], np.nan)  # z = 0 plane, camera z = t
    np.save(scene_dir / "depth" / "1.npy", (depth * depth_factor).astype(np.float32))
    Image.fromarray(np.full((H, W, 3), 90, np.uint8)).save(scene_dir / "images0" / "1.jpg")
    mask = np.zeros((H, W), np.uint8)
    mask[30:40, 60:70] = 255
    Image.fromarray(mask).save(mask_dir / "image_0000_mask.png")
    np.save(align_dir / "aligned_pose_refined.npy", POSE)
    np.save(align_dir / "aligned_pose_scene_scale_refined.npy", np.array(SCALE))
    # world = (v_imp * s) @ rot.T + pos, v_imp = v_local @ R_imp.T  =>  v_local from the world plane
    world = np.array([[-1.0, -2.0, 0.0], [3.0, -2.0, 0.0], [3.0, 2.0, 0.0], [-1.0, 2.0, 0.0]])
    pos, rot, _ = collider_world_transform(align_dir / "aligned_pose_refined.npy", SCALE)
    local = (((world - pos) @ rot) / SCALE) @ glb_import_rotation()
    trimesh.Trimesh(local, [[0, 1, 2], [0, 2, 3]], process=False).export(world_dir / "collider.glb")


def _check(tmp: Path, depth_factor: float = 1.0):
    write_scene(tmp / "DROID" / "s", tmp / "input_mask" / "s", tmp / "align", tmp / "world", depth_factor)
    return dc.depth_check(
        depth_npy=tmp / "DROID" / "s" / "depth" / "1.npy", intrinsics=tmp / "DROID" / "s" / "intrinsics.npy",
        extrinsics=tmp / "DROID" / "s" / "extrinsics.npy", mask_dir=tmp / "input_mask" / "s",
        collider_glb=tmp / "world" / "collider.glb", pose_path=tmp / "align" / "aligned_pose_refined.npy",
        scale_path=tmp / "align" / "aligned_pose_scene_scale_refined.npy", workspace=DEFAULT_WORKSPACES["franka"])


def test_exact_alignment_scores_zero(tmp_path):
    report, maps = _check(tmp_path)
    assert report["fitted_scale"] == pytest.approx(SCALE)
    assert 0 < report["object_mask_frac"] < 0.1 and maps["excluded"].any()
    for subset in ("all", "workspace", "table"):
        m = report[subset]
        assert m["n_pixels"] > 500, subset
        assert m["trimmed_chamfer"] < 1e-5 and m["sensor_to_collider_median"] < 1e-5
        assert abs(m["depth_error_median"]) < 1e-5
        assert m["sensor_optimal_scale"] == pytest.approx(SCALE, abs=1e-5)
    h = report["table_height"]
    assert h["sensor_z"] == pytest.approx(0.0, abs=1e-5) and h["collider_minus_sensor"] == pytest.approx(0.0, abs=1e-5)
    assert h["matching_scale"] == pytest.approx(SCALE, abs=1e-4)


def test_scaled_sensor_depth_moves_the_optimal_scale(tmp_path):
    report, _ = _check(tmp_path, depth_factor=1.05)
    for subset in ("all", "table"):
        assert report[subset]["sensor_optimal_scale"] == pytest.approx(1.05 * SCALE, rel=1e-5)
        assert report[subset]["depth_error_median"] < -0.02  # the collider is in front of the real surface
        assert report[subset]["trimmed_chamfer"] > 0.005
    assert report["table_height"]["sensor_z"] < -0.01  # the scaled sensor table sinks below z = 0


def test_trim_mean():
    assert dc.trim_mean(np.arange(10.0), 0.8) == pytest.approx(3.5)
    assert dc.trim_mean(np.array([5.0]), 0.1) == 5.0


def test_step_through_the_cli(tmp_path, monkeypatch):
    data, work = tmp_path / "data", tmp_path / "work"
    monkeypatch.setenv("R2S_SCENE_DATA", str(data))  # pins the inputs root
    monkeypatch.setenv("R2S_WORK", str(work))
    S = work / "scenes" / "s"
    write_scene(data / "DROID" / "s", data / "input_mask" / "s", S / "align", S / "world")
    assert cli.main(["run", "align.check", "--scene", "s"]) == 0
    layout = SceneLayout.create("s")
    report = json.loads((S / "align" / "depth_check.json").read_text())
    assert report["scene"] == "s" and report["all"]["trimmed_chamfer"] < 1e-5
    assert Image.open(S / "align" / "depth_check.png").size == (2 * W, H)
    rec = json.loads(layout.manifest.read_text())["steps"]["align.check"]
    assert rec["status"] == "ok" and rec["inputs_root"] == str(data)
    assert rec["sensor_optimal_scale"]["all"] == pytest.approx(SCALE, abs=1e-5)
