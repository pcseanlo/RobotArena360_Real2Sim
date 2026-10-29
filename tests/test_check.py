"""``r2s check``: the episode-folder validation (docs/custom_capture.md §1) on droid820 and synthetic folders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from r2s import check
from r2s.config import Paths
from r2s.surface.geometry import DEFAULT_WORKSPACES

FRANKA = DEFAULT_WORKSPACES["franka"]


def world_to_camera(cam, target) -> np.ndarray:
    """OpenCV world-to-camera matrix of a camera at ``cam`` looking at ``target`` (world z up)."""
    cam, target = np.asarray(cam, float), np.asarray(target, float)
    f = (target - cam) / np.linalg.norm(target - cam)
    r = np.cross(f, [0.0, 0.0, 1.0])
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    E = np.eye(4)
    E[:3, :3] = np.stack([r, d, f])
    E[:3, 3] = -E[:3, :3] @ cam
    return E


def make_episode(root: Path, *, W=64, H=36, T=5, depth=None, K=None, E=None, lang=True, robot=True) -> Path:
    root.mkdir(parents=True)
    (root / "images0").mkdir()
    (root / "depth").mkdir()
    for n in (0, 1):
        Image.fromarray(np.full((H, W, 3), 120, np.uint8)).save(root / "images0" / f"{n}.jpg")
        np.save(root / "depth" / f"{n}.npy", np.full((H, W), 1.2, np.float32) if depth is None else depth)
    np.save(root / "intrinsics.npy", np.array([[30.0, 0, W / 2], [0, 30.0, H / 2], [0, 0, 1]]) if K is None else K)
    np.save(root / "extrinsics.npy", world_to_camera([0.3, 0.4, 0.5], [0.7, 0.0, 0.0]) if E is None else E)
    if robot:
        np.save(root / "joints.npy", np.zeros((T, 7)))
        np.save(root / "grippers.npy", np.zeros(T))
        np.save(root / "tcp.npy", np.tile(np.eye(4), (T, 1, 1)))
    if lang:
        (root / "lang.txt").write_text("put the cup in the bowl\n")
    return root


def levels(findings, item=None) -> list[str]:
    return [f.level for f in findings if item is None or f.item == item]


def test_good_synthetic_episode(tmp_path):
    findings = check.check_episode(make_episode(tmp_path / "ep"), FRANKA)
    assert levels(findings) == ["OK"] * len(findings), findings


def test_bad_synthetic_episode(tmp_path):
    E_c2w = np.linalg.inv(world_to_camera([0.3, -0.5, 0.25], [0.88, 0.27, 0.0]))  # camera-to-world by mistake
    ep = make_episode(tmp_path / "ep", depth=np.full((36, 64), 1200, np.uint16),
                      K=np.array([[30.0, 0, 40.0], [0, 30.0, 18.5], [0, 0, 1]]), E=E_c2w, lang=False, robot=False)
    (ep / "depth" / "0.npy").unlink()
    findings = check.check_episode(ep, FRANKA)
    by_item = {f.item: f for f in findings}
    assert by_item["depth/0.npy"].level == "FAIL" and "missing" in by_item["depth/0.npy"].message
    assert by_item["depth/1.npy"].level == "FAIL" and "uint16" in by_item["depth/1.npy"].message
    pp = [f for f in findings if f.item == "principal pt"]
    assert [f.level for f in pp] == ["FAIL", "WARN"]  # |2cx - W| = 16 px = 25 %, |2cy - H| = 1 px = 2.8 %
    assert by_item["optical axis"].level == "FAIL" and "camera-to-world" in by_item["optical axis"].message
    assert by_item["robot state"].level == "WARN" and by_item["lang.txt"].level == "WARN"
    assert check.main(["--scene", "droidX", "--src", str(ep), "--inputs-only"]) == 1


def test_principal_point_and_depth_thresholds(tmp_path):
    depth = np.full((36, 64), 1.2, np.float32)
    depth[:2] = np.nan  # 5.6 % holes -> WARN
    ep = make_episode(tmp_path / "ep", depth=depth, K=np.array([[30.0, 0, 32.0], [0, 30.0, 18.6], [0, 0, 1]]))
    findings = check.check_episode(ep, FRANKA)
    assert [f.level for f in findings if f.item == "principal pt"] == ["OK", "WARN"]  # 1.2 px = 3.3 % of 36
    assert levels(findings, "depth/1.npy") == ["WARN"]
    bad = np.full((36, 64), 1.2, np.float32)
    bad[:10] = 0.0  # 28 % holes -> FAIL
    findings = check.check_episode(make_episode(tmp_path / "ep2", depth=bad), FRANKA)
    assert levels(findings, "depth/1.npy") == ["FAIL"]
    findings = check.check_episode(make_episode(tmp_path / "ep3", depth=np.zeros((10, 10), np.float32)), FRANKA)
    assert "!= the RGB's (H, W)" in [f for f in findings if f.item == "depth/1.npy"][0].message


def test_optical_axis():
    assert check.optical_axis_hit(world_to_camera([0.3, 0.4, 0.5], [0.7, 0.0, 0.0]), FRANKA)[0]
    assert not check.optical_axis_hit(world_to_camera([0.3, 0.4, 0.5], [0.7, 0.0, 1.0]), FRANKA)[0]  # looks up
    assert not check.optical_axis_hit(world_to_camera([0.3, 0.4, 0.5], [-1.0, 2.0, 0.0]), FRANKA)[0]  # away


def test_robot_state_lengths(tmp_path):
    ep = make_episode(tmp_path / "ep", T=5)
    np.save(ep / "tcp.npy", np.zeros((7, 3)))  # positions only, wrong length
    msg = [f for f in check.check_episode(ep, FRANKA) if f.item == "robot state"]
    assert [f.level for f in msg] == ["WARN"] and "lengths differ" in msg[0].message
    np.save(ep / "joints_old.npy", np.zeros((7, 7)))  # joints_old.npy stands in for joints.npy (as step 9)
    np.save(ep / "grippers.npy", np.zeros(7))
    assert [f.level for f in check.check_episode(ep, FRANKA) if f.item == "robot state"] == ["OK"]


def test_droid820_passes_with_the_cy_warning(capsys):
    meta = Paths.load().scene_data / "DROID" / "droid820"
    if not meta.is_dir():
        pytest.skip("scene data for droid820 not available")
    findings = check.check_episode(meta, FRANKA)
    assert not [f for f in findings if f.level == "FAIL"], findings
    warns = [f for f in findings if f.level == "WARN"]
    assert len(warns) == 1 and warns[0].item == "principal pt" and "2cy - H" in warns[0].message
    assert check.main(["--scene", "droid820", "--src", str(meta)]) == 0
    out = capsys.readouterr().out
    assert "summary: 11 OK, 1 WARN, 0 FAIL" in out and "align.check" in out
