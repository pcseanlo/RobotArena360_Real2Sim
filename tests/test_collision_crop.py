"""Ported from the old test_collision_mesh_crop.py (r2s.collision.crop)."""
import math
import os
import tempfile
import unittest

import numpy as np
import trimesh

from r2s.collision.crop import (
    GLB_WORLDSPACE_LOAD_QUAT_WXYZ,
    IDENTITY_FRONT,
    IDENTITY_QUAT_WXYZ,
    IDENTITY_UP,
    build_reachable_collision_mesh,
    crop_mesh_to_reachable_cylinder,
    mask_support_faces_under_footprint,
    transform_mesh_to_world,
)


def _write_mesh(path: str, mesh: trimesh.Trimesh) -> str:
    mesh.export(path)
    return path


class CollisionMeshCropTests(unittest.TestCase):
    def test_surface_mask_removes_only_covered_upward_faces(self) -> None:
        mesh = trimesh.Trimesh(
            vertices=np.array(
                [
                    [-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0],
                    [-1, -1, 0], [-1, -1, 1], [-1, 1, 1], [-1, 1, 0],
                ],
                dtype=np.float64,
            ),
            faces=np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]], dtype=np.int64),
            process=False,
        )
        footprint = {
            "plane_origin": [0.0, 0.0, 0.0],
            "plane_normal": [0.0, 0.0, 1.0],
            "basis_u": [1.0, 0.0, 0.0],
            "basis_v": [0.0, 1.0, 0.0],
            "vertices_uv": [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]],
        }

        masked, removed = mask_support_faces_under_footprint(mesh, footprint)

        self.assertEqual(removed, 2)
        self.assertEqual(len(masked.faces), 2)
        self.assertTrue(np.all(np.abs(masked.face_normals[:, 2]) < 0.1))

    def test_transform_then_crop_uses_world_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mesh_path = _write_mesh(
                os.path.join(tmp, "box.ply"),
                trimesh.creation.box(extents=(0.4, 0.4, 0.4)),
            )
            root_quat_z_90 = (
                math.cos(math.pi / 4.0),
                0.0,
                0.0,
                math.sin(math.pi / 4.0),
            )

            world_mesh = transform_mesh_to_world(
                mesh_path,
                scene_scale=2.0,
                root_pos=(0.25, 0.0, 0.0),
                root_quat_wxyz=root_quat_z_90,
                mesh_up=IDENTITY_UP,
                mesh_front=IDENTITY_FRONT,
            )
            cropped = crop_mesh_to_reachable_cylinder(
                world_mesh,
                radius=0.55,
                z_min=-0.25,
                z_max=0.25,
                sides=16,
            )

            self.assertIsNotNone(cropped)
            vertices = np.asarray(cropped.vertices)
            self.assertLessEqual(float(np.linalg.norm(vertices[:, :2], axis=1).max()), 0.55 + 1e-6)
            self.assertGreaterEqual(float(vertices[:, 2].min()), -0.25 - 1e-6)
            self.assertLessEqual(float(vertices[:, 2].max()), 0.25 + 1e-6)

    def test_large_crossing_triangles_are_clipped_not_dropped(self) -> None:
        mesh = trimesh.Trimesh(
            vertices=np.array(
                [
                    [-3.0, -3.0, 0.0],
                    [3.0, -3.0, 0.0],
                    [-3.0, 3.0, 0.0],
                    [3.0, 3.0, 0.0],
                ],
                dtype=np.float64,
            ),
            faces=np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64),
            process=False,
        )

        cropped = crop_mesh_to_reachable_cylinder(
            mesh,
            radius=1.0,
            z_min=-0.1,
            z_max=0.1,
            sides=16,
        )

        self.assertIsNotNone(cropped)
        vertices = np.asarray(cropped.vertices)
        self.assertGreater(len(cropped.faces), 0)
        self.assertLessEqual(float(np.linalg.norm(vertices[:, :2], axis=1).max()), 1.0 + 1e-6)

    def test_build_exports_cropped_mesh_with_bounded_vertices(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mesh_path = _write_mesh(
                os.path.join(tmp, "wide_box.ply"),
                trimesh.creation.box(extents=(3.0, 3.0, 1.0)),
            )
            result = build_reachable_collision_mesh(
                mesh_path,
                scene_scale=0.5,
                root_pos=(0.2, 0.3, 0.4),
                root_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
                source_mesh_up=(0.0, 1.0, 0.0),
                source_mesh_front=(1.0, 0.0, 0.0),
                cache_dir=tmp,
                radius=0.8,
                z_min=-0.2,
                z_max=0.2,
                sides=16,
            )

            self.assertTrue(result.used_crop, result.reason)
            self.assertTrue(result.mesh_path.endswith(".glb"))
            self.assertEqual(result.load_scale, 1.0)
            self.assertEqual(result.load_pos, (0.0, 0.0, 0.0))
            self.assertEqual(result.load_quat_wxyz, GLB_WORLDSPACE_LOAD_QUAT_WXYZ)
            self.assertEqual(result.load_up, IDENTITY_UP)
            self.assertEqual(result.load_front, IDENTITY_FRONT)
            self.assertTrue(result.baked_world_space)
            self.assertTrue(os.path.exists(result.mesh_path))
            cropped = trimesh.load(result.mesh_path, force="mesh", process=False)
            vertices = np.asarray(cropped.vertices)
            self.assertLessEqual(float(np.linalg.norm(vertices[:, :2], axis=1).max()), 0.8 + 1e-6)
            self.assertGreaterEqual(float(vertices[:, 2].min()), -0.2 - 1e-6)
            self.assertLessEqual(float(vertices[:, 2].max()), 0.2 + 1e-6)

    def test_cached_crop_keeps_identity_loader_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mesh_path = _write_mesh(
                os.path.join(tmp, "wide_box.glb"),
                trimesh.creation.box(extents=(3.0, 3.0, 1.0)),
            )
            kwargs = dict(
                scene_scale=1.0,
                root_pos=(0.1, 0.2, 0.3),
                root_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
                cache_dir=tmp,
                radius=0.8,
                z_min=-0.2,
                z_max=0.6,
                sides=16,
            )
            built = build_reachable_collision_mesh(mesh_path, **kwargs)
            cached = build_reachable_collision_mesh(mesh_path, **kwargs)

            self.assertTrue(built.used_crop, built.reason)
            self.assertTrue(cached.used_crop, cached.reason)
            self.assertEqual(cached.reason, "cache")
            self.assertEqual(cached.mesh_path, built.mesh_path)
            self.assertEqual(cached.load_scale, 1.0)
            self.assertEqual(cached.load_pos, (0.0, 0.0, 0.0))
            self.assertEqual(cached.load_quat_wxyz, GLB_WORLDSPACE_LOAD_QUAT_WXYZ)
            self.assertEqual(cached.load_up, IDENTITY_UP)
            self.assertEqual(cached.load_front, IDENTITY_FRONT)
            self.assertTrue(cached.baked_world_space)

    def test_empty_crop_falls_back_to_original_mesh(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mesh = trimesh.creation.box(extents=(0.2, 0.2, 0.2))
            mesh.apply_translation((5.0, 0.0, 0.0))
            mesh_path = _write_mesh(os.path.join(tmp, "far_box.ply"), mesh)

            result = build_reachable_collision_mesh(
                mesh_path,
                scene_scale=0.5,
                root_pos=(0.2, 0.3, 0.4),
                root_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
                source_mesh_up=(0.0, 1.0, 0.0),
                source_mesh_front=(1.0, 0.0, 0.0),
                cache_dir=tmp,
                radius=0.8,
                z_min=-0.2,
                z_max=0.2,
                sides=16,
            )

            self.assertFalse(result.used_crop)
            self.assertEqual(result.mesh_path, mesh_path)
            self.assertEqual(result.reason, "empty_crop")
            self.assertEqual(result.load_scale, 0.5)
            self.assertEqual(result.load_pos, (0.2, 0.3, 0.4))
            self.assertEqual(result.load_quat_wxyz, IDENTITY_QUAT_WXYZ)
            self.assertEqual(result.load_up, (0.0, 1.0, 0.0))
            self.assertEqual(result.load_front, (1.0, 0.0, 0.0))
            self.assertFalse(result.baked_world_space)


class EvaluatorPoseTests(unittest.TestCase):
    """The crop must place the collider with the old evaluator's exact quaternion."""

    def test_orig_collider_quat_is_genesis_product_not_adjustment(self) -> None:
        from r2s.collision.crop import evaluator_orig_collider_quat

        q = evaluator_orig_collider_quat()
        # transform_quat_by_quat(identity, ADJUSTMENT) under Genesis 1.4.4
        self.assertEqual(q.tolist(), [0.017452406437283522, -6.93889390390723e-18, 0.9998476951563913, 0.0])

    @unittest.skipUnless(
        os.environ.get("R2S_TEST_GENESIS") == "1",
        "set R2S_TEST_GENESIS=1 to compare against Genesis itself (its numba cache writes into the Eden venv "
        "unless NUMBA_CACHE_DIR is set)",
    )
    def test_genesis_quat_mul_matches_genesis(self) -> None:
        from genesis.utils.geom import transform_quat_by_quat

        from r2s.collision.crop import genesis_quat_mul

        rng = np.random.default_rng(0)
        for _ in range(200):
            a, b = rng.normal(size=4), rng.normal(size=4)
            self.assertTrue((genesis_quat_mul(a, b) == transform_quat_by_quat(b, a)).all())

    def test_evaluator_collider_pose_droid2_sidecar(self) -> None:
        from r2s.collision.crop import evaluator_collider_pose

        with tempfile.TemporaryDirectory() as tmp:
            pose = os.path.join(tmp, "pose.npy")
            scale = os.path.join(tmp, "scale.npy")
            # droid2 aligned_pose_refined.npy / aligned_pose_scene_scale_refined.npy
            np.save(pose, np.array([0.35042038187171853, -0.4930719366492398, 0.2820295585285727, 2.5142312049865723]))
            np.save(scale, np.array(0.6971482234052577))
            pos, quat, s = evaluator_collider_pose(pose, scale)
        # values of optimization_results/droid2/cropped_collision/*.glb.json
        self.assertEqual(list(pos), [0.35042038187171853, -0.4930719366492398, 0.2820295585285727])
        self.assertEqual(list(quat), [0.005385146632571119, -0.9510594116922166, 0.3085148439560126, 0.01660080378168007])
        self.assertEqual(s, 0.6971482234052577)

    def test_crop_name_is_production_layout_name(self) -> None:
        from r2s.collision.crop import crop_glb_name
        from r2s.layout import CROP_GLB_NAME

        self.assertEqual(crop_glb_name(), CROP_GLB_NAME)

    def test_rebuild_never_writes_through_a_seeded_symlink(self) -> None:
        from r2s.collision.crop import crop_glb_name

        with tempfile.TemporaryDirectory() as tmp:
            mesh_path = _write_mesh(os.path.join(tmp, "wide_box.glb"), trimesh.creation.box(extents=(3.0, 3.0, 1.0)))
            readonly = os.path.join(tmp, "readonly")
            os.makedirs(readonly)
            target = os.path.join(readonly, "old.glb")
            with open(target, "wb") as handle:
                handle.write(b"old bytes")
            out_dir = os.path.join(tmp, "out")
            os.makedirs(out_dir)
            name = crop_glb_name(0.8, -0.2, 0.6, 16)
            os.symlink(target, os.path.join(out_dir, name))
            result = build_reachable_collision_mesh(
                mesh_path,
                scene_scale=1.0,
                root_pos=(0.1, 0.2, 0.3),
                root_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
                cache_dir=out_dir,
                radius=0.8,
                z_min=-0.2,
                z_max=0.6,
                sides=16,
                use_cache=False,
            )
            self.assertTrue(result.used_crop, result.reason)
            self.assertFalse(os.path.islink(result.mesh_path))
            with open(target, "rb") as handle:
                self.assertEqual(handle.read(), b"old bytes")

    def test_missing_collider_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                build_reachable_collision_mesh(
                    os.path.join(tmp, "nope.glb"),
                    scene_scale=1.0,
                    root_pos=(0.0, 0.0, 0.0),
                    root_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
                    cache_dir=tmp,
                )


if __name__ == "__main__":
    unittest.main()
