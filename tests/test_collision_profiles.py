"""Ported from the old test_object_collision_profiles.py (r2s.collision.profiles)."""
import json
import os
import tempfile
import unittest
from unittest import mock

import trimesh

from r2s.collision.profiles import (
    inspect_mesh_collision_quality,
    select_object_collision_profile,
    write_collision_profile_manifest,
)


class ObjectCollisionProfileTests(unittest.TestCase):
    def test_clean_watertight_mesh_keeps_detailed_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "box.glb")
            trimesh.creation.box(extents=(1.0, 0.5, 0.25)).export(path)

            quality = inspect_mesh_collision_quality(path)
            selection = select_object_collision_profile(path, requested_profile="auto")

        self.assertTrue(quality.watertight)
        self.assertEqual(quality.component_count, 1)
        self.assertEqual(selection.selected_profile, "detailed")
        self.assertFalse(selection.decimate)
        self.assertEqual(selection.decompose_object_error_threshold, 0.05)

    def test_fragmented_ring_like_scan_uses_stable_legacy_profile(self) -> None:
        segments = []
        for index in range(12):
            angle = index * 3.141592653589793 * 2.0 / 12.0
            segment = trimesh.creation.box(extents=(0.22, 0.08, 0.10))
            segment.apply_translation((0.55, 0.0, 0.0))
            segment.apply_transform(trimesh.transformations.rotation_matrix(angle, (0.0, 0.0, 1.0)))
            segments.append(segment)
        fragmented = trimesh.util.concatenate(segments)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "fragmented_ring.glb")
            fragmented.export(path)
            selection = select_object_collision_profile(path, requested_profile="auto")

        self.assertGreater(selection.quality.component_count, 4)
        self.assertEqual(selection.selected_profile, "legacy")
        self.assertTrue(selection.decimate)
        self.assertEqual(selection.decimate_face_num, 1000)
        self.assertEqual(selection.decompose_object_error_threshold, 0.15)
        self.assertEqual(selection.coacd_threshold, 0.05)

    def test_explicit_decimation_override_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "box.glb")
            trimesh.creation.box().export(path)
            selection = select_object_collision_profile(
                path,
                requested_profile="detailed",
                decimate_override=True,
            )

        self.assertEqual(selection.selected_profile, "detailed")
        self.assertTrue(selection.decimate)

    def test_explicit_profile_can_skip_slow_topology_inspection(self) -> None:
        with mock.patch(
            "r2s.collision.profiles.inspect_mesh_collision_quality",
            side_effect=AssertionError("inspection should be skipped"),
        ):
            selection = select_object_collision_profile(
                "/unused/object.glb",
                requested_profile="detailed",
                inspect_quality=False,
            )

        self.assertEqual(selection.selected_profile, "detailed")
        self.assertEqual(selection.quality.face_count, -1)
        self.assertIn("not-inspected", selection.quality.reasons[0])

    def test_manifest_records_actual_per_object_choice(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mesh_path = os.path.join(tmp, "box.glb")
            output_path = os.path.join(tmp, "profiles.json")
            trimesh.creation.box().export(mesh_path)
            selection = select_object_collision_profile(mesh_path)
            write_collision_profile_manifest(output_path, {"obj_box": selection})
            with open(output_path, encoding="utf-8") as handle:
                payload = json.load(handle)

        self.assertEqual(payload["version"], 1)
        self.assertEqual(payload["objects"]["obj_box"]["selected_profile"], "detailed")


if __name__ == "__main__":
    unittest.main()
