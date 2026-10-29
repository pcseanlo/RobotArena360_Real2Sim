"""Tests for step 4 (``r2s.surface.plane_fit``). The old repo had none for this fitter."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from r2s.common.transforms import collider_world_transform, glb_import_rotation
from r2s.surface.plane_fit import (
    compute_plane_fit_candidates,
    default_options,
    detect_height_slabs,
    fit_plane,
    min_area_rectangle,
    weighted_median,
)

FRANKA = (0.20, 0.95, -0.55, 0.55, -0.08, 0.85)


def _plane_grid(x0, x1, y0, y1, z, step, rng):
    xx, yy = np.meshgrid(np.arange(x0, x1, step), np.arange(y0, y1, step), indexing="xy")
    points = np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, z)))
    points[:, :2] += rng.uniform(-0.3 * step, 0.3 * step, size=(len(points), 2))
    points[:, 2] += rng.normal(0.0, 0.002, size=len(points))
    return points


class PlaneFitHelperTests(unittest.TestCase):
    def test_weighted_median_and_slabs(self) -> None:
        self.assertEqual(weighted_median(np.array([0.0, 1.0, 2.0]), np.array([1.0, 1.0, 5.0])), 2.0)
        rng = np.random.default_rng(0)
        # Peaks in the first/last histogram bin are never detected, so keep both
        # surfaces away from the ends of the height range.
        z = np.concatenate((rng.normal(0.10, 0.004, 200), rng.normal(0.40, 0.004, 80), [0.0, 0.5]))
        slabs = detect_height_slabs(z, np.ones_like(z), bin_width=0.01, min_separation=0.04)
        self.assertEqual(len(slabs), 2)
        self.assertGreater(slabs[0]["z"], slabs[1]["z"])  # sorted by descending height

    def test_min_area_rectangle_recovers_rotated_rectangle(self) -> None:
        angle = np.deg2rad(30.0)
        local = np.array([[-0.4, -0.1], [0.4, -0.1], [0.4, 0.1], [-0.4, 0.1]])
        rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        center, axis_u, half = min_area_rectangle(local @ rot.T + np.array([1.0, 2.0]))
        self.assertTrue(np.allclose(center, [1.0, 2.0]))
        self.assertTrue(np.allclose(half, [0.4, 0.1]))
        self.assertAlmostEqual(abs(float(axis_u @ np.array([np.cos(angle), np.sin(angle)]))), 1.0)

    def test_camera_reference_selects_tabletop_slab_over_floor(self) -> None:
        rng = np.random.default_rng(0)
        table = _plane_grid(0.35, 0.80, -0.30, 0.30, 0.10, 0.01, rng)
        floor = _plane_grid(0.20, 0.95, -0.55, 0.55, -0.04, 0.01, rng)
        # Two small planar ledges (too few points to become regions) widen the height
        # range: peaks in the first/last histogram bin are never detected.
        low = _plane_grid(0.21, 0.26, 0.48, 0.53, -0.075, 0.01, rng)
        high = _plane_grid(0.88, 0.93, 0.45, 0.50, 0.30, 0.01, rng)
        points = np.vstack((table, floor, low, high))
        normals = np.tile([0.0, 0.0, 1.0], (len(points), 1))
        reference = np.zeros(len(points), dtype=bool)
        reference[: len(table)] = True

        args = default_options(robot="franka")
        candidates, stats, pca_normals = compute_plane_fit_candidates(
            points, normals, FRANKA, args, reference_mask=reference
        )

        self.assertEqual(pca_normals.shape, points.shape)
        self.assertGreaterEqual(stats["slab_count"], 2)
        self.assertGreaterEqual(len(candidates), 2)
        selected = candidates[0]
        self.assertAlmostEqual(selected.z_band_center, 0.10, places=3)
        self.assertEqual(selected.plane_normal, [0.0, 0.0, 1.0])
        self.assertEqual(selected.plane_offset, -selected.z_band_center)
        # The box bottom sits `clearance` above the snapped plane; extents match the table.
        self.assertAlmostEqual(
            selected.center[2], selected.z_band_center + args.clearance + 0.5 * args.thickness, places=12
        )
        self.assertTrue(np.allclose(sorted(selected.extents), [0.45, 0.60], atol=0.02))


class FitPlaneEndToEndTests(unittest.TestCase):
    def test_fit_plane_recovers_world_tabletop_through_collider_pose(self) -> None:
        import trimesh

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            pose_path, scale_path = tmp / "aligned_pose_refined.npy", tmp / "scale.npy"
            np.save(pose_path, np.array([0.05, -0.02, 0.30, 0.3], dtype=np.float64))
            np.save(scale_path, np.array([0.7], dtype=np.float64))

            # A world-space tabletop block (top at z=0.12, bottom below the workspace),
            # mapped back into GLB-local coordinates with the inverse collider placement.
            table = trimesh.creation.box(extents=(0.5, 0.7, 0.30))
            table.apply_translation((0.55, 0.0, -0.03))
            pos, rot, _ = collider_world_transform(pose_path, 0.7)
            local = ((np.asarray(table.vertices) - pos[None, :]) @ rot) / 0.7 @ glb_import_rotation()
            glb_path = tmp / "collider.glb"
            trimesh.Trimesh(local, np.asarray(table.faces)).export(glb_path)

            out_json = tmp / "surface" / "operation_surface_plane_fit.json"
            descriptor = fit_plane(
                glb_path,
                pose_path,
                scale_path,
                output_json=out_json,
                options=default_options(robot="franka", camera_frustum_crop=False, num_samples=6000),
            )

            self.assertTrue(out_json.exists())
            selected = descriptor["selected"]
            self.assertIsNotNone(selected)
            self.assertAlmostEqual(-selected["plane_offset"], 0.12, places=3)
            # The min-area rectangle hugs the samples, so it sits just inside the true edge.
            extents = np.sort(selected["extents"])
            self.assertTrue(np.all(extents <= [0.5 + 1e-6, 0.7 + 1e-6]), extents)
            self.assertTrue(np.all(extents >= [0.45, 0.65]), extents)
            self.assertEqual(descriptor["source"]["robot"], "franka")
            self.assertEqual(descriptor["parameters"]["num_samples"], 6000)

    def test_default_options_are_the_production_flags(self) -> None:
        args = default_options()
        # plane_fitting_plane_fit.sh
        self.assertTrue(args.camera_frustum_crop)
        self.assertEqual(args.camera_occlusion_cell_px, 8)
        self.assertEqual(args.coverage, "workspace")
        self.assertEqual(args.num_samples, 400000)
        self.assertEqual(args.plane_inlier_tol, 0.02)
        self.assertEqual(args.max_tilt_deg, 15.0)
        self.assertEqual(args.pca_k, 24)
        self.assertEqual(args.min_planarity, 0.25)
        self.assertEqual(args.region_radius, 0.10)
        self.assertEqual(args.region_normal_deg, 20.0)
        self.assertEqual(args.footprint_mode, "minrect")
        self.assertEqual(args.footprint_axis_snap_deg, 8.0)
        self.assertEqual(args.footprint_grid_min_cell_points, 1)
        self.assertEqual(args.viz_max_points, 400000)
        self.assertFalse(args.viewer)


if __name__ == "__main__":
    unittest.main()
