"""Ported from the old test_init_collision_guard.py (r2s.collision.support)."""
import unittest
from unittest import mock

import numpy as np

from r2s.collision.support import BackgroundSupportQuery, corrected_init_position


def _upward_square(z: float, *, x0: float = 0.0, x1: float = 1.0, y0: float = 0.0, y1: float = 1.0):
    return np.array(
        [
            [[x0, y0, z], [x1, y0, z], [x0, y1, z]],
            [[x1, y0, z], [x1, y1, z], [x0, y1, z]],
        ],
        dtype=np.float64,
    )


def _downward_square(z: float):
    return np.array(
        [
            [[0.0, 0.0, z], [0.0, 1.0, z], [1.0, 0.0, z]],
            [[1.0, 0.0, z], [0.0, 1.0, z], [1.0, 1.0, z]],
        ],
        dtype=np.float64,
    )


def _vertical_square(x: float):
    return np.array(
        [
            [[x, 0.0, 0.0], [x, 1.0, 0.0], [x, 0.0, 1.0]],
            [[x, 1.0, 0.0], [x, 1.0, 1.0], [x, 0.0, 1.0]],
        ],
        dtype=np.float64,
    )


class InitCollisionGuardTests(unittest.TestCase):
    def test_support_and_penetration_clearances_are_distinct(self) -> None:
        support_query = mock.Mock()
        support_query.support_for_vertices.return_value = (0.0, 1, 0.0, None, "background")
        vertices = np.array([[0.0, 0.0, -0.01], [0.0, 0.0, 0.01]])
        support = corrected_init_position(
            pos=np.zeros(3),
            aabb_min=vertices.min(axis=0),
            aabb_max=vertices.max(axis=0),
            background_query=support_query,
            fallback_support_z=0.0,
            clearance=0.0005,
            penetration_clearance=0.005,
            object_vertices_world=vertices,
        )
        self.assertAlmostEqual(support.lift, 0.0105)

        penetration_query = mock.Mock()
        penetration_query.support_for_vertices.return_value = (
            0.0,
            1,
            0.0,
            0.01,
            "background_collision",
        )
        penetration = corrected_init_position(
            pos=np.zeros(3),
            aabb_min=vertices.min(axis=0),
            aabb_max=vertices.max(axis=0),
            background_query=penetration_query,
            fallback_support_z=0.0,
            clearance=0.0005,
            penetration_clearance=0.005,
            object_vertices_world=vertices,
        )
        self.assertAlmostEqual(penetration.lift, 0.015)

    def test_background_support_lifts_above_local_surface(self) -> None:
        query = BackgroundSupportQuery.from_arrays(_upward_square(0.25), footprint_margin=0.0)
        correction = corrected_init_position(
            pos=np.array([0.5, 0.5, 0.0]),
            aabb_min=np.array([0.25, 0.25, 0.10]),
            aabb_max=np.array([0.75, 0.75, 0.20]),
            background_query=query,
            fallback_support_z=0.05,
            clearance=0.002,
        )

        self.assertEqual(correction.support_source, "background")
        self.assertAlmostEqual(correction.support_z, 0.25)
        self.assertAlmostEqual(correction.lift, 0.152)
        self.assertAlmostEqual(correction.pos[2], 0.152)

    def test_fallback_support_used_when_background_does_not_overlap(self) -> None:
        query = BackgroundSupportQuery.from_arrays(_upward_square(0.25), footprint_margin=0.0)
        correction = corrected_init_position(
            pos=np.array([2.0, 2.0, 0.0]),
            aabb_min=np.array([2.0, 2.0, 0.01]),
            aabb_max=np.array([2.2, 2.2, 0.08]),
            background_query=query,
            fallback_support_z=0.04,
            clearance=0.003,
        )

        self.assertEqual(correction.support_source, "fallback")
        self.assertAlmostEqual(correction.support_z, 0.04)
        self.assertAlmostEqual(correction.lift, 0.033)

    def test_background_support_wins_over_higher_fallback(self) -> None:
        query = BackgroundSupportQuery.from_arrays(_upward_square(0.02), footprint_margin=0.0)
        correction = corrected_init_position(
            pos=np.array([0.5, 0.5, 0.0]),
            aabb_min=np.array([0.25, 0.25, 0.04]),
            aabb_max=np.array([0.75, 0.75, 0.14]),
            background_query=query,
            fallback_support_z=0.08,
            clearance=0.002,
        )

        self.assertEqual(correction.support_source, "background")
        self.assertAlmostEqual(correction.support_z, 0.02)
        self.assertAlmostEqual(correction.lift, 0.0)

    def test_object_already_above_support_is_unchanged(self) -> None:
        query = BackgroundSupportQuery.from_arrays(_upward_square(0.10), footprint_margin=0.0)
        correction = corrected_init_position(
            pos=np.array([0.5, 0.5, 0.4]),
            aabb_min=np.array([0.25, 0.25, 0.20]),
            aabb_max=np.array([0.75, 0.75, 0.30]),
            background_query=query,
            fallback_support_z=None,
            clearance=0.002,
        )

        self.assertEqual(correction.support_source, "background")
        self.assertAlmostEqual(correction.lift, 0.0)
        self.assertTrue(np.allclose(correction.pos, [0.5, 0.5, 0.4]))

    def test_downward_triangles_are_ignored(self) -> None:
        query = BackgroundSupportQuery.from_arrays(_downward_square(0.30), footprint_margin=0.0)
        correction = corrected_init_position(
            pos=np.array([0.5, 0.5, 0.0]),
            aabb_min=np.array([0.25, 0.25, 0.10]),
            aabb_max=np.array([0.75, 0.75, 0.20]),
            background_query=query,
            fallback_support_z=0.04,
            clearance=0.002,
        )

        self.assertEqual(correction.support_source, "fallback")
        self.assertAlmostEqual(correction.support_z, 0.04)
        self.assertAlmostEqual(correction.lift, 0.0)

    def test_steep_triangles_are_ignored_as_support(self) -> None:
        # support_z_for_aabb still filters by upward normal, so the vertical
        # square should not be picked as the support surface.
        triangles = np.concatenate([_upward_square(0.02), _vertical_square(0.5)], axis=0)
        query = BackgroundSupportQuery.from_arrays(
            triangles,
            footprint_margin=0.0,
            surface_normal_z=0.5,
        )
        self.assertAlmostEqual(
            query.support_z_for_aabb(
                np.array([0.25, 0.25, 0.03]),
                np.array([0.75, 0.75, 0.13]),
            ),
            0.02,
        )

    def test_steep_triangles_are_caught_as_collision(self) -> None:
        # A tall vertical wall through the object's footprint must register as
        # a collision (previously it was silently dropped by the z_limit
        # filter). The reported collision_top is clipped to z_limit so the
        # lift doesn't try to clear an unbounded wall/ceiling.
        triangles = np.concatenate([_upward_square(0.02), _vertical_square(0.5)], axis=0)
        query = BackgroundSupportQuery.from_arrays(
            triangles,
            footprint_margin=0.0,
            surface_normal_z=0.5,
            max_surface_above_aabb=0.05,
        )
        collision_top = query.collision_top_z_for_aabb(
            np.array([0.25, 0.25, 0.03]),
            np.array([0.75, 0.75, 0.13]),
        )
        self.assertIsNotNone(collision_top)
        # z_limit = 0.13 + 0.05 = 0.18
        self.assertAlmostEqual(collision_top, 0.18)

    def test_high_surfaces_above_object_span_are_ignored(self) -> None:
        triangles = np.concatenate([_upward_square(0.10), _upward_square(1.50)], axis=0)
        query = BackgroundSupportQuery.from_arrays(triangles, footprint_margin=0.0)
        correction = corrected_init_position(
            pos=np.array([0.5, 0.5, 0.0]),
            aabb_min=np.array([0.25, 0.25, 0.02]),
            aabb_max=np.array([0.75, 0.75, 0.12]),
            background_query=query,
            fallback_support_z=None,
            clearance=0.002,
        )

        self.assertEqual(correction.support_source, "background")
        self.assertAlmostEqual(correction.support_z, 0.10)
        self.assertAlmostEqual(correction.lift, 0.082)

    def test_support_ceiling_rejects_high_background_as_tabletop(self) -> None:
        query = BackgroundSupportQuery.from_arrays(
            np.concatenate([_upward_square(0.02), _upward_square(0.12)], axis=0),
            footprint_margin=0.0,
        )

        self.assertAlmostEqual(
            query.support_z_for_aabb(
                np.array([0.25, 0.25, 0.03]),
                np.array([0.75, 0.75, 0.15]),
                max_support_z=0.05,
            ),
            0.02,
        )

    def test_overlapping_background_above_support_ceiling_is_collision(self) -> None:
        query = BackgroundSupportQuery.from_arrays(
            np.concatenate([_upward_square(0.02), _upward_square(0.12)], axis=0),
            footprint_margin=0.0,
        )
        correction = corrected_init_position(
            pos=np.array([0.5, 0.5, 0.0]),
            aabb_min=np.array([0.25, 0.25, 0.03]),
            aabb_max=np.array([0.75, 0.75, 0.15]),
            background_query=query,
            fallback_support_z=0.01,
            clearance=0.002,
            background_support_max_z=0.05,
        )

        self.assertEqual(correction.support_source, "background_collision")
        self.assertAlmostEqual(correction.support_z, 0.12)
        self.assertAlmostEqual(correction.lift, 0.092)


class VertexRaycastTests(unittest.TestCase):
    def test_surface_samples_clear_raised_collision_even_with_valid_support(self) -> None:
        ground = _upward_square(0.0)
        raised_top = _upward_square(0.15, x0=0.45, x1=0.55, y0=0.0, y1=1.0)
        query = BackgroundSupportQuery.from_arrays(
            np.concatenate([ground, raised_top], axis=0),
            footprint_margin=0.0,
        )

        object_vertices = np.array(
            [
                [0.70, 0.50, 0.05],  # valid support on ground
                [0.50, 0.50, 0.05],  # inside/under raised collision top
                [0.70, 0.50, 0.25],
                [0.50, 0.50, 0.25],
            ],
            dtype=np.float64,
        )
        correction = corrected_init_position(
            pos=np.array([0.60, 0.50, 0.0]),
            aabb_min=object_vertices.min(axis=0),
            aabb_max=object_vertices.max(axis=0),
            background_query=query,
            fallback_support_z=None,
            clearance=0.002,
            object_vertices_world=object_vertices,
            object_contact_points_world=np.array([[0.70, 0.50, 0.05]], dtype=np.float64),
            object_collision_points_world=object_vertices,
            bottom_slab=0.02,
        )

        self.assertEqual(correction.support_source, "background_collision")
        self.assertAlmostEqual(correction.support_z, 0.15, places=4)
        self.assertAlmostEqual(correction.lift, 0.102, places=4)

    def test_raised_collision_is_not_hidden_by_support_ceiling(self) -> None:
        query = BackgroundSupportQuery.from_arrays(
            np.concatenate([_upward_square(0.0), _upward_square(0.12)], axis=0),
            footprint_margin=0.0,
        )
        object_vertices = np.array(
            [
                [0.50, 0.50, 0.03],
                [0.60, 0.50, 0.03],
                [0.50, 0.50, 0.20],
                [0.60, 0.50, 0.20],
            ],
            dtype=np.float64,
        )

        correction = corrected_init_position(
            pos=np.array([0.55, 0.50, 0.0]),
            aabb_min=object_vertices.min(axis=0),
            aabb_max=object_vertices.max(axis=0),
            background_query=query,
            fallback_support_z=None,
            clearance=0.002,
            background_support_max_z=0.05,
            object_vertices_world=object_vertices,
            object_contact_points_world=object_vertices[:2],
            object_collision_points_world=object_vertices,
            bottom_slab=0.02,
        )

        self.assertEqual(correction.support_source, "background_collision")
        self.assertAlmostEqual(correction.support_z, 0.12, places=4)
        self.assertAlmostEqual(correction.lift, 0.092, places=4)

    def test_same_collision_gives_object_wise_lifts_for_different_heights(self) -> None:
        query = BackgroundSupportQuery.from_arrays(_upward_square(0.15), footprint_margin=0.0)

        low_object = np.array(
            [[0.50, 0.50, 0.05], [0.60, 0.50, 0.05], [0.50, 0.50, 0.25]],
            dtype=np.float64,
        )
        high_object = low_object + np.array([0.0, 0.0, 0.05], dtype=np.float64)

        low = corrected_init_position(
            pos=np.array([0.55, 0.50, 0.0]),
            aabb_min=low_object.min(axis=0),
            aabb_max=low_object.max(axis=0),
            background_query=query,
            fallback_support_z=None,
            clearance=0.002,
            object_vertices_world=low_object,
            object_contact_points_world=low_object[:2],
            object_collision_points_world=low_object,
        )
        high = corrected_init_position(
            pos=np.array([0.55, 0.50, 0.0]),
            aabb_min=high_object.min(axis=0),
            aabb_max=high_object.max(axis=0),
            background_query=query,
            fallback_support_z=None,
            clearance=0.002,
            object_vertices_world=high_object,
            object_contact_points_world=high_object[:2],
            object_collision_points_world=high_object,
        )

        self.assertAlmostEqual(low.lift, 0.102, places=4)
        self.assertAlmostEqual(high.lift, 0.052, places=4)
        self.assertGreater(low.lift, high.lift)

    def test_surface_samples_avoid_aabb_overlift_from_unrelated_high_patch(self) -> None:
        low = _upward_square(0.02, x0=0.0, x1=0.5)
        high = _upward_square(0.40, x0=0.5, x1=1.0)
        query = BackgroundSupportQuery.from_arrays(np.concatenate([low, high], axis=0), footprint_margin=0.0)

        object_vertices = np.array(
            [
                [0.10, 0.20, 0.10],
                [0.40, 0.20, 0.10],
                [0.10, 0.80, 0.10],
                [0.40, 0.80, 0.10],
                [0.80, 0.50, 0.50],  # widens AABB, but is not under the high patch
            ],
            dtype=np.float64,
        )
        correction = corrected_init_position(
            pos=np.array([0.25, 0.50, 0.0]),
            aabb_min=object_vertices.min(axis=0),
            aabb_max=object_vertices.max(axis=0),
            background_query=query,
            fallback_support_z=None,
            clearance=0.002,
            object_vertices_world=object_vertices,
            object_contact_points_world=object_vertices[:4],
            object_collision_points_world=object_vertices,
            bottom_slab=0.02,
        )

        self.assertEqual(correction.support_source, "background")
        self.assertAlmostEqual(correction.support_z, 0.02, places=4)
        self.assertAlmostEqual(correction.lift, 0.0, places=4)

    def test_vertex_path_uses_actual_contact_footprint(self) -> None:
        # Two upward squares side by side: one low at z=0.02 covering x<0.5,
        # one high at z=0.40 covering x>0.5. An object whose AABB spans both
        # but whose bottom vertices all sit in the low x<0.5 region should be
        # lifted to ~0.02, NOT to 0.40 (which is what the AABB path would do).
        low = _upward_square(0.02, x0=0.0, x1=0.5)
        high = _upward_square(0.40, x0=0.5, x1=1.0)
        triangles = np.concatenate([low, high], axis=0)
        query = BackgroundSupportQuery.from_arrays(triangles, footprint_margin=0.0)

        # Object: a thin slab at the bottom (z in [0.10, 0.11]) sitting only
        # on the low side, plus one upper vertex at x>0.5 that extends the
        # AABB. Bottom slab should pick up only the low-side vertices.
        object_vertices = np.array(
            [
                [0.10, 0.20, 0.10],
                [0.40, 0.20, 0.10],
                [0.10, 0.80, 0.10],
                [0.40, 0.80, 0.10],
                [0.10, 0.20, 0.11],
                [0.40, 0.80, 0.11],
                # An outlier at the top, on the high side — extends AABB only
                [0.80, 0.50, 0.50],
            ],
            dtype=np.float64,
        )
        aabb_min = object_vertices.min(axis=0)
        aabb_max = object_vertices.max(axis=0)

        correction = corrected_init_position(
            pos=np.array([0.25, 0.5, 0.10]),
            aabb_min=aabb_min,
            aabb_max=aabb_max,
            background_query=query,
            fallback_support_z=None,
            clearance=0.002,
            object_vertices_world=object_vertices,
            bottom_slab=0.02,
        )

        self.assertEqual(correction.support_source, "background")
        self.assertAlmostEqual(correction.support_z, 0.02, places=4)
        # bottom_z=0.10, support_z=0.02, clearance=0.002 -> no lift needed
        self.assertAlmostEqual(correction.lift, 0.0, places=4)
        self.assertGreater(correction.contact_vertex_count, 0)

    def test_vertex_path_detects_wall_penetration(self) -> None:
        # A box-shaped wall obstacle spanning x in [0.45, 0.55], z in [0, 0.30],
        # full y. An object bottom vertex placed *inside* the wall block
        # should have its upward raycast hit the wall's top -> lateral
        # penetration reported.
        ground = _upward_square(0.0)
        wall_front = _vertical_square(0.45)
        wall_back = _vertical_square(0.55)
        # Top of the wall at z=0.15 — within the object's z-span (object goes
        # up to 0.30), so the z-band penetration heuristic fires.
        wall_top = _upward_square(0.15, x0=0.45, x1=0.55, y0=0.0, y1=1.0)
        triangles = np.concatenate([ground, wall_front, wall_back, wall_top], axis=0)
        query = BackgroundSupportQuery.from_arrays(triangles, footprint_margin=0.0)

        # Object bottom vertices: one inside the wall (x=0.50), one outside.
        object_vertices = np.array(
            [
                [0.50, 0.50, 0.05],  # inside wall -> upward ray hits wall_top at z=0.15
                [0.70, 0.50, 0.05],  # outside wall -> downward ray hits ground
                [0.50, 0.50, 0.30],
                [0.70, 0.50, 0.30],
            ],
            dtype=np.float64,
        )
        aabb_min = object_vertices.min(axis=0)
        aabb_max = object_vertices.max(axis=0)

        correction = corrected_init_position(
            pos=np.array([0.6, 0.5, 0.10]),
            aabb_min=aabb_min,
            aabb_max=aabb_max,
            background_query=query,
            fallback_support_z=None,
            clearance=0.002,
            object_vertices_world=object_vertices,
            bottom_slab=0.02,
        )

        self.assertIsNotNone(correction.lateral_penetration)
        self.assertGreater(correction.lateral_penetration, 0.0)
        self.assertEqual(correction.support_source, "background_collision")


    def test_vertex_path_does_not_overlift_object_with_legs(self) -> None:
        # Object whose AABB extends down to leg_z, but only a small central
        # bottom feature actually penetrates a floor patch. The legs hang in
        # free space (no floor under them). The fix should lift by ~pen +
        # clearance, NOT by (floor_top - leg_z), since the legs don't need
        # to clear anything.
        # Floor patch only covers center (x in [-0.02, 0.02]).
        floor = _upward_square(0.00, x0=-0.02, x1=0.02, y0=-0.02, y1=0.02)
        query = BackgroundSupportQuery.from_arrays(floor, footprint_margin=0.0)

        leg_z = -0.025  # legs hang 25mm below floor top, but no floor under them
        pen_z = -0.005  # central bottom penetrates floor by 5mm
        object_vertices = np.array(
            [
                [-0.10, -0.10, leg_z],
                [0.10, -0.10, leg_z],
                [-0.10, 0.10, leg_z],
                [0.10, 0.10, leg_z],
                [0.00, 0.00, pen_z],
                # an upper vertex so the object has some height
                [0.00, 0.00, 0.20],
            ],
            dtype=np.float64,
        )
        aabb_min = object_vertices.min(axis=0)
        aabb_max = object_vertices.max(axis=0)
        correction = corrected_init_position(
            pos=np.array([0.0, 0.0, 0.0]),
            aabb_min=aabb_min,
            aabb_max=aabb_max,
            background_query=query,
            fallback_support_z=None,
            clearance=0.002,
            object_vertices_world=object_vertices,
            bottom_slab=0.05,
        )

        # Required lift = penetration depth (0.005) + clearance (0.002) = 0.007.
        # The buggy formula would have produced 0.005 + 0.002 - (-0.030) = 0.037.
        self.assertAlmostEqual(correction.lift, 0.007, places=4)


if __name__ == "__main__":
    unittest.main()
