"""Tests for step 6 (``r2s.surface.receiver_fit``).

Ported from the old ``test_operation_surface_render_receiver_fit.py``. Tests of the
removed modes (support-bounds, coverage-rectangles, inner-rectangle, support-obb,
auto basis, reference-edge snap) are gone; the two that only used a rectangle mode to
inspect the basis / reference clipping now run in polygon mode.
"""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from r2s.common.quat import quat_wxyz_to_matrix
from r2s.common.transforms import splat_world_pose
from r2s.surface.geometry import footprint_mask_from_world_points
from r2s.surface.receiver_fit import (
    default_options,
    fit_receiver,
    fit_render_receiver_from_world_points,
    gaussian_surface_normal_confidence,
)


def _selected_surface() -> dict:
    return {
        "center": [0.0, 0.0, 0.015],
        "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
        "size": [1.0, 1.0, 0.01],
        "basis_u": [1.0, 0.0, 0.0],
        "basis_v": [0.0, 1.0, 0.0],
        "plane_normal": [0.0, 0.0, 1.0],
        "plane_offset": 0.0,
    }


def _grid_points(x0: float, x1: float, y0: float, y1: float, step: float = 0.04) -> np.ndarray:
    x = np.arange(x0, x1 + 1e-9, step)
    y = np.arange(y0, y1 + 1e-9, step)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    zz = np.zeros_like(xx)
    return np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))


def _write_ascii_ply(path: Path, xyz: np.ndarray) -> None:
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(xyz)}",
        "property float x",
        "property float y",
        "property float z",
        "end_header",
    ]
    lines.extend(f"{p[0]} {p[1]} {p[2]}" for p in xyz)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _rotated(points: np.ndarray, angle: float) -> np.ndarray:
    rot = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return points @ rot.T


class OperationSurfaceRenderReceiverFitTests(unittest.TestCase):
    def test_normal_confidence_prefers_flat_anisotropic_splat(self) -> None:
        props = {
            "x": np.zeros(2),
            "scale_0": np.log(np.array([0.10, 0.10])),
            "scale_1": np.log(np.array([0.10, 0.10])),
            "scale_2": np.log(np.array([0.01, 0.01])),
            "rot_0": np.array([1.0, np.sqrt(0.5)]),
            "rot_1": np.array([0.0, 0.0]),
            "rot_2": np.array([0.0, np.sqrt(0.5)]),
            "rot_3": np.zeros(2),
        }
        confidence, source = gaussian_surface_normal_confidence(
            props,
            splat_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        self.assertEqual(source, "gaussian_thinnest_axis")
        self.assertGreater(float(confidence[0]), 0.85)
        self.assertLess(float(confidence[1]), 0.05)

    def test_receiver_can_be_smaller_than_collision_surface(self) -> None:
        selected = _selected_surface()
        points = _grid_points(-0.20, 0.20, -0.15, 0.15)

        receiver, stats = fit_render_receiver_from_world_points(
            selected,
            points,
            grid_size=0.05,
            erosion_cells=0,
            min_area=0.001,
        )

        self.assertIsNotNone(receiver)
        self.assertLess(receiver["area"], selected["size"][0] * selected["size"][1])
        self.assertGreater(stats["support_count"], 0)

    def test_disconnected_support_keeps_largest_component(self) -> None:
        selected = _selected_surface()
        large = _grid_points(-0.20, 0.20, -0.15, 0.15)
        small = _grid_points(0.85, 0.95, 0.85, 0.95)
        points = np.vstack((large, small))

        receiver, _ = fit_render_receiver_from_world_points(
            selected,
            points,
            grid_size=0.05,
            erosion_cells=0,
            min_area=0.001,
        )

        self.assertIsNotNone(receiver)
        xs = [box["center"][0] for box in receiver["support_boxes"]]
        self.assertTrue(all(x < 0.5 for x in xs))

    def test_polygon_contours_reject_disconnected_drift_points(self) -> None:
        selected = _selected_surface()
        table = _grid_points(-0.20, 0.20, -0.20, 0.20, step=0.04)
        drift = np.array(
            [
                [0.50, 0.45, 0.0],
                [0.51, 0.45, 0.0],
                [0.50, 0.46, 0.0],
            ],
            dtype=np.float64,
        )

        receiver, stats = fit_render_receiver_from_world_points(
            selected,
            np.vstack((table, drift)),
            receiver_basis_mode="surface",
            grid_size=0.04,
            erosion_cells=0,
            min_area=0.001,
            receiver_grid_scales=(1.0,),
            receiver_physics_boundary_padding=0.0,
            receiver_lighting_boundary_inset=0.0,
        )

        self.assertIsNotNone(receiver)
        self.assertGreater(stats["raw_cell_count"], stats["component_cell_count"])
        self.assertEqual(stats["contour_input_point_count"], len(table) + len(drift))
        self.assertEqual(stats["contour_retained_point_count"], len(table))
        self.assertEqual(stats["contour_rejected_point_count"], len(drift))
        for role in ("physics", "lighting"):
            boundary = np.asarray(receiver["footprints"][role]["boundary_uv"], dtype=np.float64)
            self.assertLess(float(boundary[:, 0].max()), 0.30)
            self.assertLess(float(boundary[:, 1].max()), 0.30)
            self.assertEqual(
                receiver["footprints"][role]["contour"]["selected_unsupported_cell_count"],
                0,
            )

    def test_support_edge_basis_aligns_to_boundary_edge(self) -> None:
        selected = _selected_surface()
        angle = np.deg2rad(23.0)
        points = _rotated(_grid_points(-0.48, 0.48, -0.16, 0.16, step=0.025), angle)

        receiver, stats = fit_render_receiver_from_world_points(
            selected,
            points,
            receiver_basis_mode="support-edge",
            grid_size=0.025,
            erosion_cells=0,
            min_area=0.001,
        )

        self.assertIsNotNone(receiver)
        expected_u = np.array([np.cos(angle), np.sin(angle), 0.0], dtype=np.float64)
        fitted_u = np.asarray(receiver["basis_u"], dtype=np.float64)
        fitted_v = np.asarray(receiver["basis_v"], dtype=np.float64)
        self.assertEqual(stats["receiver_basis_selected_mode"], "support-edge")
        self.assertGreater(stats["receiver_basis_edge_point_count"], 0)
        self.assertGreater(max(abs(float(np.dot(fitted_u, expected_u))), abs(float(np.dot(fitted_v, expected_u)))), 0.96)
        lighting = receiver["footprints"]["lighting"]
        self.assertTrue(np.allclose(lighting["basis_u"], receiver["basis_u"]))

    def test_tightened_receiver_stays_inside_support_bounds(self) -> None:
        selected = _selected_surface()
        points = _grid_points(-0.23, 0.21, -0.17, 0.13, step=0.04)

        receiver, _ = fit_render_receiver_from_world_points(
            selected,
            points,
            receiver_basis_mode="surface",
            grid_size=0.05,
            erosion_cells=0,
            min_area=0.001,
            receiver_tighten_to_support=True,
        )

        self.assertIsNotNone(receiver)
        for box in receiver["support_boxes"]:
            u = np.asarray(box["basis_u"], dtype=np.float64)
            v = np.asarray(box["basis_v"], dtype=np.float64)
            center = np.asarray(box["center"], dtype=np.float64)
            extents = np.asarray(box["extents"], dtype=np.float64)
            support_uv = np.column_stack((points @ u, points @ v))
            box_uv = np.array([center @ u, center @ v], dtype=np.float64)
            box_min = box_uv - 0.5 * extents
            box_max = box_uv + 0.5 * extents
            self.assertGreaterEqual(box_min[0], support_uv[:, 0].min() - 1e-9)
            self.assertGreaterEqual(box_min[1], support_uv[:, 1].min() - 1e-9)
            self.assertLessEqual(box_max[0], support_uv[:, 0].max() + 1e-9)
            self.assertLessEqual(box_max[1], support_uv[:, 1].max() + 1e-9)

    def test_polygon_receiver_preserves_l_shape_without_panel_seams(self) -> None:
        selected = _selected_surface()
        vertical = _grid_points(-0.35, -0.10, -0.35, 0.35, step=0.05)
        horizontal = _grid_points(-0.35, 0.35, -0.35, -0.10, step=0.05)
        points = np.unique(np.vstack((vertical, horizontal)), axis=0)

        receiver, stats = fit_render_receiver_from_world_points(
            selected,
            points,
            receiver_basis_mode="surface",
            grid_size=0.05,
            erosion_cells=0,
            min_area=0.001,
            receiver_grid_scales=(1.0,),
        )

        self.assertIsNotNone(receiver)
        self.assertEqual(receiver["schema_version"], 2)
        lighting = receiver["footprints"]["lighting"]
        self.assertGreater(len(lighting["triangles"]), 2)
        self.assertEqual(stats["receiver_geometry"], "support_aligned_polygon")
        self.assertGreaterEqual(len(lighting["boundary_uv"]), 4)
        self.assertLess(len(lighting["boundary_uv"]), lighting["occupied_cell_count"])
        self.assertGreaterEqual(stats["receiver_point_coverage"], 0.99)
        self.assertGreater(stats["convex_hull_overcoverage_ratio"], 0.05)
        occupied = {tuple(cell) for cell in lighting["occupied_cells_rc"]}
        self.assertLess(len(occupied), 16 * 16)

    def test_polygon_rejects_convex_bridge_over_unsupported_concavity(self) -> None:
        selected = _selected_surface()
        vertical = _grid_points(-0.35, -0.10, -0.35, 0.35, step=0.05)
        horizontal = _grid_points(-0.35, 0.35, -0.35, -0.10, step=0.05)
        points = np.unique(np.vstack((vertical, horizontal)), axis=0)

        receiver, stats = fit_render_receiver_from_world_points(
            selected,
            points,
            receiver_basis_mode="surface",
            grid_size=0.05,
            erosion_cells=0,
            min_area=0.001,
            receiver_grid_scales=(1.0,),
            receiver_physics_boundary_padding=0.0,
            receiver_lighting_boundary_inset=0.0,
        )

        self.assertIsNotNone(receiver)
        contour = receiver["footprints"]["lighting"]["contour"]
        self.assertEqual(contour["selected_mode"], "concave")
        self.assertGreater(contour["convex_overcoverage_ratio"], 0.05)
        self.assertGreater(contour["convex_unsupported_cell_count"], 0)
        self.assertEqual(stats["contour_unsupported_cell_count"], 0)
        self.assertFalse(receiver["footprints"]["convex_hull_diagnostic"]["acceptable_as_fallback"])

    def test_polygon_keeps_convex_edge_when_support_is_rectangular(self) -> None:
        receiver, stats = fit_render_receiver_from_world_points(
            _selected_surface(),
            _grid_points(-0.35, 0.35, -0.25, 0.25, step=0.05),
            receiver_basis_mode="surface",
            grid_size=0.05,
            erosion_cells=0,
            min_area=0.001,
            receiver_grid_scales=(1.0,),
            receiver_physics_boundary_padding=0.0,
            receiver_lighting_boundary_inset=0.0,
        )

        self.assertIsNotNone(receiver)
        contour = receiver["footprints"]["lighting"]["contour"]
        self.assertEqual(contour["selected_mode"], "convex")
        self.assertLessEqual(contour["convex_overcoverage_ratio"], 0.05)
        self.assertEqual(contour["convex_unsupported_cell_count"], 0)
        self.assertEqual(stats["contour_unsupported_cell_count"], 0)
        self.assertTrue(receiver["footprints"]["convex_hull_diagnostic"]["acceptable_as_fallback"])

    def test_polygon_falls_back_to_connected_occupancy_for_unsafe_concave_hull(self) -> None:
        selected = _selected_surface()
        vertical = _grid_points(-0.35, -0.10, -0.35, 0.35, step=0.05)
        horizontal = _grid_points(-0.35, 0.35, -0.35, -0.10, step=0.05)
        points = np.unique(np.vstack((vertical, horizontal)), axis=0)

        receiver, stats = fit_render_receiver_from_world_points(
            selected,
            points,
            receiver_basis_mode="surface",
            grid_size=0.05,
            erosion_cells=0,
            min_area=0.001,
            receiver_grid_scales=(1.0,),
            receiver_contour_concavity_ratio=1.0,
            receiver_contour_convex_max_overcoverage=0.0,
            receiver_physics_boundary_padding=0.0,
            receiver_lighting_boundary_inset=0.0,
        )

        self.assertIsNotNone(receiver)
        contour = receiver["footprints"]["lighting"]["contour"]
        self.assertEqual(contour["selected_mode"], "occupancy_fallback")
        self.assertEqual(contour["selected_unsupported_cell_count"], 0)
        self.assertEqual(stats["contour_selected_mode"], "occupancy_fallback")

    def test_narrow_opening_does_not_fill_large_boundary_air_pocket(self) -> None:
        points = _grid_points(-0.40, 0.40, -0.30, 0.30, step=0.05)
        # A large empty pocket is connected to the right exterior by a narrow
        # channel. Closing before hole filling used to seal the channel and
        # incorrectly fill the entire pocket as receiver support.
        pocket = (
            (points[:, 0] >= 0.05)
            & (points[:, 0] <= 0.30)
            & (np.abs(points[:, 1]) <= 0.10)
        )
        channel = (points[:, 0] > 0.30) & (np.abs(points[:, 1]) < 0.025)
        points = points[~(pocket | channel)]

        receiver, stats = fit_render_receiver_from_world_points(
            _selected_surface(),
            points,
            receiver_basis_mode="surface",
            grid_size=0.05,
            erosion_cells=0,
            min_area=0.001,
            receiver_grid_scales=(1.0,),
            receiver_physics_boundary_padding=0.0,
            receiver_lighting_boundary_inset=0.0,
        )

        self.assertIsNotNone(receiver)
        footprint = receiver["footprints"]["lighting"]
        probes = np.asarray([[0.17, 0.0, 0.0], [-0.20, 0.0, 0.0]])
        covered = footprint_mask_from_world_points(probes, footprint)
        self.assertFalse(bool(covered[0]))
        self.assertTrue(bool(covered[1]))
        self.assertEqual(stats["filled_interior_hole_cell_count"], 0)
        self.assertEqual(footprint["contour"]["selected_unsupported_cell_count"], 0)

    def test_surface_lift_moves_all_receiver_geometry_without_changing_fit(self) -> None:
        points = _grid_points(-0.30, 0.30, -0.20, 0.20, step=0.05)
        common = {
            "receiver_basis_mode": "surface",
            "grid_size": 0.05,
            "erosion_cells": 0,
            "min_area": 0.001,
            "receiver_grid_scales": (1.0,),
            "receiver_physics_boundary_padding": 0.003,
            "receiver_lighting_boundary_inset": 0.006,
        }
        base, base_stats = fit_render_receiver_from_world_points(
            _selected_surface(),
            points,
            receiver_surface_lift=0.0,
            **common,
        )
        lifted, lifted_stats = fit_render_receiver_from_world_points(
            _selected_surface(),
            points,
            receiver_surface_lift=0.003,
            **common,
        )

        self.assertIsNotNone(base)
        self.assertIsNotNone(lifted)
        self.assertEqual(base["plane_offset"], lifted["plane_offset"])
        self.assertEqual(base_stats["plane_stabilized_offset"], lifted_stats["plane_stabilized_offset"])
        self.assertEqual(base["area"], lifted["area"])
        for role, boundary_offset in (("physics", 0.003), ("lighting", -0.006)):
            base_footprint = base["footprints"][role]
            lifted_footprint = lifted["footprints"][role]
            self.assertTrue(
                np.allclose(base_footprint["vertices_uv"], lifted_footprint["vertices_uv"])
            )
            delta = np.asarray(lifted_footprint["vertices_world"]) - np.asarray(
                base_footprint["vertices_world"]
            )
            self.assertTrue(np.allclose(delta, [0.0, 0.0, 0.003]))
            self.assertEqual(lifted_footprint["contour"]["boundary_offset"], boundary_offset)
        for base_box, lifted_box in zip(base["support_boxes"], lifted["support_boxes"]):
            delta = np.asarray(lifted_box["center"]) - np.asarray(base_box["center"])
            self.assertTrue(np.allclose(delta, [0.0, 0.0, 0.003]))
        self.assertEqual(lifted_stats["receiver_surface_lift"], 0.003)
        self.assertEqual(lifted_stats["receiver_physics_boundary_padding"], 0.003)
        self.assertEqual(lifted_stats["receiver_lighting_boundary_inset"], 0.006)

    def test_reference_clip_rejects_same_height_outside_table_clutter(self) -> None:
        selected = _selected_surface()
        table = _grid_points(-0.25, 0.25, -0.25, 0.25, step=0.04)
        clutter = _grid_points(0.10, 0.30, 1.10, 1.50, step=0.04)
        points = np.vstack((table, clutter))

        receiver, stats = fit_render_receiver_from_world_points(
            selected,
            points,
            receiver_basis_mode="surface",
            grid_size=0.05,
            erosion_cells=0,
            min_area=0.001,
            receiver_clip_to_reference=True,
        )

        self.assertIsNotNone(receiver)
        self.assertLess(stats["support_count"], stats["near_plane_count"])
        for box in receiver["support_boxes"]:
            center_y = float(box["center"][1])
            half_y = 0.5 * float(box["size"][1])
            self.assertLessEqual(center_y + half_y, 0.5 + 1e-9)
        boundary = np.asarray(receiver["footprints"]["lighting"]["boundary_uv"], dtype=np.float64)
        self.assertLess(float(boundary[:, 1].max()), 0.5)

    def test_too_small_support_emits_no_receiver(self) -> None:
        receiver, stats = fit_render_receiver_from_world_points(
            _selected_surface(),
            np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
            grid_size=0.05,
            erosion_cells=0,
            min_area=0.10,
        )

        self.assertIsNone(receiver)
        self.assertLess(stats["receiver_area"], 0.10)

    def test_fit_receiver_writes_descriptor_without_mutating_plane_fit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            op_path = tmp / "operation_surface.json"
            op_descriptor = {"selected": _selected_surface()}
            op_path.write_text(json.dumps(op_descriptor, indent=2), encoding="utf-8")

            result_path = tmp / "aligned_pose.npy"
            scale_path = tmp / "aligned_pose_scene_scale.npy"
            np.save(result_path, np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64))
            np.save(scale_path, np.array([1.0], dtype=np.float64))

            desired_world = _grid_points(-0.20, 0.20, -0.15, 0.15)
            splat_pos, splat_quat, scene_scale = splat_world_pose(result_path, scale_path)
            rot = quat_wxyz_to_matrix(splat_quat)
            local = (desired_world / scene_scale - splat_pos[None, :]) @ rot
            gaussian_path = tmp / "gaussian.ply"
            _write_ascii_ply(gaussian_path, local)

            out_path = tmp / "out" / "operation_surface_shadow_v2.json"
            descriptor = fit_receiver(
                op_path,
                gaussian_path,
                result_path,
                scale_path,
                out_path,
                options=default_options(grid_size=0.05, erosion_cells=0, min_area=0.001),
            )

            self.assertEqual(json.loads(op_path.read_text(encoding="utf-8")), op_descriptor)
            output = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(output, json.loads(json.dumps(descriptor)))
            self.assertIsNotNone(output["selected"])
            self.assertLess(output["selected"]["area"], 1.0)
            self.assertEqual(output["parameters"]["receiver_mode"], "polygon")
            self.assertEqual(output["parameters"]["receiver_surface_lift"], 0.015)
            self.assertIn("lighting", output["selected"]["footprints"])

    def test_default_options_are_the_production_flags(self) -> None:
        args = default_options()
        # evaluate_pi_shadow_visual.sh
        self.assertEqual(args.receiver_contour_concavity_ratio, 0.12)
        self.assertEqual(args.receiver_contour_convex_max_overcoverage, 0.05)
        self.assertEqual(args.receiver_contour_simplify_tolerance, 0.012)
        self.assertEqual(args.receiver_physics_boundary_padding, 0.003)
        self.assertEqual(args.receiver_lighting_boundary_inset, 0.006)
        self.assertEqual(args.receiver_surface_lift, 0.015)
        self.assertEqual(args.erosion_cells, 1)
        self.assertFalse(args.viewer)
        with self.assertRaises(TypeError):
            default_options(receiver_mode="support-bounds")

    def test_fit_does_not_mutate_selected_dict(self) -> None:
        selected = _selected_surface()
        before = copy.deepcopy(selected)
        fit_render_receiver_from_world_points(
            selected,
            _grid_points(-0.20, 0.20, -0.15, 0.15),
            grid_size=0.05,
            erosion_cells=0,
            min_area=0.001,
        )
        self.assertEqual(selected, before)


if __name__ == "__main__":
    unittest.main()
