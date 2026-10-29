"""Tests for step 5 (``r2s.surface.trim``). The old repo had none for this script."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from r2s.common.quat import quat_wxyz_to_matrix
from r2s.common.transforms import splat_world_pose
from r2s.surface.trim import read_vertex_ply, trim_gaussians, write_vertex_ply


def _gaussians(local_xyz: np.ndarray, opacity_logit: np.ndarray, log_scale: np.ndarray) -> np.ndarray:
    dtype = np.dtype([(n, "<f4") for n in ("x", "y", "z", "f_dc_0", "opacity", "scale_0", "scale_1", "scale_2")])
    data = np.zeros(len(local_xyz), dtype=dtype)
    data["x"], data["y"], data["z"] = local_xyz.T
    data["f_dc_0"] = np.arange(len(local_xyz))
    data["opacity"] = opacity_logit
    for i in range(3):
        data[f"scale_{i}"] = log_scale
    return data


class TrimTests(unittest.TestCase):
    def test_ply_round_trip_is_lossless(self) -> None:
        data = _gaussians(np.random.default_rng(0).normal(size=(10, 3)), np.zeros(10), np.zeros(10))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "a.ply"
            write_vertex_ply(path, data)
            back, order = read_vertex_ply(path)
        self.assertEqual(order, "<")
        self.assertEqual(back.dtype, data.dtype)
        self.assertEqual(back.tobytes(), data.tobytes())

    def test_only_floaters_inside_the_region_are_dropped(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            pose_path, scale_path = tmp / "pose.npy", tmp / "scale.npy"
            np.save(pose_path, np.array([0.1, 0.0, 0.2, 0.4], dtype=np.float64))
            np.save(scale_path, np.array([0.8], dtype=np.float64))
            plane_json = tmp / "plane.json"
            plane_json.write_text(json.dumps({"selected": {
                "center": [0.5, 0.0, 0.0], "plane_normal": [0.0, 0.0, 1.0], "plane_offset": 0.0,
                "basis_u": [1.0, 0.0, 0.0], "basis_v": [0.0, 1.0, 0.0], "size": [1.0, 1.0, 0.01],
            }}))

            world = np.array([
                [0.5, 0.0, 0.05],   # faint, in region          -> dropped
                [0.5, 0.0, 0.05],   # oversized, in region      -> dropped
                [0.5, 0.0, 0.05],   # healthy, in region        -> kept
                [1.5, 0.0, 0.05],   # faint, outside the radius -> kept
                [0.5, 0.0, 0.30],   # faint, above the band     -> kept
                [0.5, 0.0, -0.10],  # faint, below the band     -> kept
            ])
            splat_pos, splat_quat, scene_scale = splat_world_pose(pose_path, scale_path)
            local = (world / scene_scale - splat_pos[None, :]) @ quat_wxyz_to_matrix(splat_quat)
            faint, solid = -3.0, 3.0
            data = _gaussians(
                local,
                np.array([faint, solid, solid, faint, faint, faint]),
                np.log(np.array([0.01, 0.2, 0.01, 0.01, 0.01, 0.01])),
            )
            src, dst = tmp / "gaussian.ply", tmp / "world" / "gaussian_trimmed.ply"
            write_vertex_ply(src, data)

            stats = trim_gaussians(src, dst, plane_json=plane_json, pose_path=pose_path, scale_path=scale_path)
            out, _ = read_vertex_ply(dst)

            self.assertEqual(stats["in_region"], 3)
            self.assertEqual(stats["dropped_count"], 2)
            self.assertEqual(out["f_dc_0"].tolist(), [2.0, 3.0, 4.0, 5.0])
            self.assertEqual(out.dtype, data.dtype)
            with self.assertRaises(ValueError):
                trim_gaussians(src, src, plane_json=plane_json, pose_path=pose_path, scale_path=scale_path)


if __name__ == "__main__":
    unittest.main()
