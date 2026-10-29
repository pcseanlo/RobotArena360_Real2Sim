"""Tests for ``r2s.surface.geometry``.

Ported from the old ``test_operation_surface_precompute.py`` (oriented boxes, largest
rectangles) and ``test_gaussian_shadow.py`` (``footprint_mask_from_world_points``).
"""

import unittest

import numpy as np

from r2s.surface.geometry import (
    camera_frustum_mask,
    decompose_true_mask_into_rectangles,
    footprint_mask_from_world_points,
    largest_axis_aligned_true_rectangle,
    oriented_box_from_plane,
    selected_plane,
)


def _xy_grid(x_values: np.ndarray, y_values: np.ndarray) -> np.ndarray:
    xx, yy = np.meshgrid(x_values, y_values, indexing="xy")
    zz = np.zeros_like(xx)
    return np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))


def _box(points: np.ndarray, **overrides):
    kwargs = dict(
        normal=np.array([0.0, 0.0, 1.0]),
        offset=0.0,
        shrink=0.0,
        footprint_trim_quantile=0.0,
        footprint_padding=0.0,
        footprint_grid_filter=False,
        footprint_grid_size=0.05,
        footprint_grid_min_cell_points=1,
        thickness=0.01,
        clearance=0.0,
        min_side=0.01,
    )
    kwargs.update(overrides)
    return oriented_box_from_plane(points, **kwargs)


class OrientedBoxFromPlaneTests(unittest.TestCase):
    def test_grid_filtered_footprint_recomputes_frame_from_kept_component(self) -> None:
        main_tabletop = _xy_grid(np.linspace(-1.0, 1.0, 21), np.linspace(-0.4, 0.4, 9))
        disconnected_stray_patch = _xy_grid(np.linspace(3.6, 4.0, 5), np.linspace(3.6, 4.0, 5))
        points = np.vstack((main_tabletop, disconnected_stray_patch))

        box = _box(points, footprint_grid_filter=True, footprint_grid_size=0.15)

        self.assertIsNotNone(box)
        assert box is not None
        self.assertTrue(box["footprint_stats"]["footprint_grid_used"])
        self.assertEqual(box["footprint_stats"]["footprint_grid_filtered_count"], len(main_tabletop))
        self.assertGreater(abs(float(np.dot(box["basis_u"], np.array([1.0, 0.0, 0.0])))), 0.98)
        self.assertTrue(np.allclose(np.sort(box["extents"]), np.array([0.8, 2.0]), atol=1e-6))

    def test_inner_rectangle_stays_conservative_for_tapered_support(self) -> None:
        points = []
        for x in np.linspace(-1.0, 1.0, 41):
            half_width = 1.0 - 0.2 * (x + 1.0)
            points.append(_xy_grid(np.array([x]), np.linspace(-half_width, half_width, 25)))
        tapered_tabletop = np.vstack(points)

        outer_box = _box(tapered_tabletop)
        inner_box = _box(
            tapered_tabletop,
            footprint_inner_rectangle=True,
            footprint_inner_support_radius=0.055,
            footprint_inner_erosion_cells=1,
        )

        self.assertIsNotNone(outer_box)
        self.assertIsNotNone(inner_box)
        assert outer_box is not None and inner_box is not None
        self.assertLess(inner_box["area"], outer_box["area"])
        self.assertLess(max(inner_box["extents"]), max(outer_box["extents"]))
        self.assertIn("footprint_inner_rectangle_used", inner_box["footprint_stats"])

    def test_broader_footprint_points_expand_inner_rectangle_beyond_fit_core(self) -> None:
        fit_core = _xy_grid(np.linspace(-0.3, 0.3, 13), np.linspace(-0.25, 0.25, 11))
        connected_support = _xy_grid(np.linspace(-0.9, 0.9, 31), np.linspace(-0.55, 0.55, 21))
        common = dict(
            footprint_inner_rectangle=True,
            footprint_inner_support_radius=0.06,
            footprint_inner_erosion_cells=0,
        )

        core_box = _box(fit_core, **common)
        support_box = _box(fit_core, footprint_points=connected_support, **common)

        self.assertIsNotNone(core_box)
        self.assertIsNotNone(support_box)
        assert core_box is not None and support_box is not None
        self.assertGreater(support_box["area"], core_box["area"] * 3.0)
        self.assertTrue(np.allclose(np.sort(support_box["extents"]), np.array([1.2, 1.9]), atol=0.11))

    def test_multi_rectangles_follow_non_rectangular_support(self) -> None:
        left_block = _xy_grid(np.linspace(-1.0, -0.1, 19), np.linspace(-0.5, 0.5, 21))
        right_block = _xy_grid(np.linspace(0.0, 0.8, 17), np.linspace(-0.2, 0.2, 9))
        support = np.vstack((left_block, right_block))

        box = _box(
            support,
            footprint_points=support,
            footprint_multi_rectangles=True,
            footprint_inner_support_radius=0.06,
            footprint_max_rectangles=4,
            footprint_min_rectangle_cells=4,
        )

        self.assertIsNotNone(box)
        assert box is not None
        self.assertGreaterEqual(len(box["support_boxes"]), 2)
        self.assertGreater(box["area"], 1.0)
        self.assertLess(box["area"], 1.8)


class LargestRectangleTests(unittest.TestCase):
    def test_largest_axis_aligned_true_rectangle(self) -> None:
        mask = np.array(
            [
                [True, True, True, False],
                [True, True, True, False],
                [True, True, False, False],
            ],
            dtype=bool,
        )
        self.assertEqual(largest_axis_aligned_true_rectangle(mask), (0, 1, 0, 2))

    def test_decompose_true_mask_into_rectangles(self) -> None:
        mask = np.array(
            [
                [True, True, True, False],
                [True, True, True, False],
                [False, False, True, True],
                [False, False, True, True],
            ],
            dtype=bool,
        )
        rects = decompose_true_mask_into_rectangles(mask, max_rectangles=3, min_rectangle_cells=2)
        self.assertGreaterEqual(len(rects), 2)


class FootprintMaskTests(unittest.TestCase):
    def test_concave_grid_footprint_rejects_missing_corner(self) -> None:
        footprint = {
            "plane_origin": [0.0, 0.0, 0.0],
            "basis_u": [1.0, 0.0, 0.0],
            "basis_v": [0.0, 1.0, 0.0],
            "uv_lo": [0.0, 0.0],
            "grid_size": 1.0,
            "occupied_cells_rc": [[0, 0], [1, 0], [1, 1]],
        }
        points = np.array([[[0.5, 0.5, 0.0], [1.5, 0.5, 0.0], [1.5, 1.5, 0.0]]])
        mask = footprint_mask_from_world_points(points, footprint)
        self.assertTrue(mask[0, 0])
        self.assertFalse(mask[0, 1])
        self.assertTrue(mask[0, 2])

    def test_polygon_footprint_respects_holes(self) -> None:
        footprint = {
            "plane_origin": [0.0, 0.0, 0.0],
            "basis_u": [1.0, 0.0, 0.0],
            "basis_v": [0.0, 1.0, 0.0],
            "boundary_uv": [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]],
            "holes_uv": [[[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0]]],
        }
        points = np.array([[0.5, 0.5, 0.2], [2.0, 2.0, 0.0], [5.0, 1.0, 0.0]])
        self.assertEqual(footprint_mask_from_world_points(points, footprint).tolist(), [True, False, False])


class SelectedPlaneTests(unittest.TestCase):
    def test_selected_plane_returns_right_handed_basis(self) -> None:
        normal, offset, u, v = selected_plane(
            {"center": [0.0, 0.0, 0.2], "plane_normal": [0.0, 0.0, 2.0], "plane_offset": -0.2,
             "basis_u": [1.0, 0.0, 0.0], "basis_v": [0.0, -1.0, 0.0]}
        )
        self.assertTrue(np.allclose(normal, [0.0, 0.0, 1.0]))
        self.assertEqual(offset, -0.2)
        self.assertGreater(float(np.linalg.det(np.column_stack((u, v, normal)))), 0.0)

    def test_camera_frustum_mask_keeps_points_in_front_of_camera(self) -> None:
        k = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
        points = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [10.0, 0.0, 1.0]])
        mask = camera_frustum_mask(points, k, 100, 100, np.eye(4), near=0.01, far=50.0, truncate_w=0.02, truncate_h=0.02)
        self.assertEqual(mask.tolist(), [True, False, False])


if __name__ == "__main__":
    unittest.main()
