"""The scene_generation code vendored for 0c-6 .. 0c-8 must stay a copy of the original.

Every function the vendored modules claim to keep unchanged is compared, as an
AST (comments and whitespace ignored), with the function of the same name in
``paths.scene_generation``. Skipped when that checkout is not available.
Plus a numeric check of the pure-numpy solvers.
"""

import ast
import sys
import types
import unittest
from pathlib import Path

import numpy as np

from r2s.config import Paths

GEN = Path(__file__).resolve().parents[1] / "r2s" / "gen"
VENDOR = GEN / "sg_glue"
WORKERS = GEN / "workers"

try:
    SG = Paths.load().scene_generation
except Exception:  # pragma: no cover - config not loadable
    SG = Path("/nonexistent")


def _defs(path: Path) -> dict[str, ast.AST]:
    """Top-level functions and ``Class.method`` nodes of a file."""
    tree = ast.parse(path.read_text())
    out: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            out[node.name] = node
        elif isinstance(node, ast.ClassDef):
            out[node.name] = node
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef):
                    out[f"{node.name}.{sub.name}"] = sub
    return out


def _same(a: ast.AST, b: ast.AST) -> bool:
    return ast.dump(a, include_attributes=False) == ast.dump(b, include_attributes=False)


# (vendored file, scene_generation file, names claimed unchanged)
UNCHANGED = [
    ("matching_utils.py", "src/utils/matching_utils.py", [
        "Unprojector.__init__", "Unprojector.project_pixel_to_point", "Unprojector.save_point_cloud",
        "apply_transformation", "compute_rigid_transform", "crop_and_resize", "get_uncropped_keypoints",
        "project_pixel_to_point", "umeyama_alignment", "two_stage_alignment",
    ]),
    ("dataloader_utils.py", "src/utils/dataloader_utils.py", [
        "SamGemini.get_rgb", "SamGemini.filter_pointclouds", "SamGemini.get_original",
    ]),
    ("simulator_genesis.py", "src/sim/simulator_genesis.py", [
        "SimulatorGenesis.__init__", "SimulatorGenesis.set_scene", "SimulatorGenesis.gs_transform_by_quat",
        "SimulatorGenesis.transform_point_cloud",
    ]),
    ("match_find_rot.py", "src/pipeline/MatchFindRot.py", [
        "PointCloudProcessor.reset_params", "PointCloudProcessor.load_ply_pointcloud",
        "PointCloudProcessor.remove_noise_neigh", "PointCloudProcessor.remove_noise",
        "PointCloudProcessor.compute_scale_ply", "PointCloudProcessor.extract_face_colors",
        "PointCloudProcessor.register_point_clouds", "PointCloudProcessor.apply_transformation",
        "PointCloudProcessor.render_genesis", "PointCloudProcessor.load_keypoints",
        "start_simulation",
    ]),
    ("minima_sp_lg.py", "src/utils/MINIMA/data_io_sp_lg.py", [
        "DataIOWrapper.__init__", "DataIOWrapper.preprocess_image", "DataIOWrapper.from_cv_imgs",
        "DataIOWrapper.from_paths", "DataIOWrapper.match_images", "DataIOWrapper.pad_bottom_right",
    ]),
    ("align_inpaint.py", "src/pipeline/align_inpaint_updated.py", [
        "_load_depth_model", "_run_depth_inference", "_grid_subsample_indices", "_ransac_affine_fit",
        "_trimmed_ls_scale", "transfer_metric_depth_two_stage_regression", "combine_masks",
        "load_o3d_intrinsic", "load_extrinsic", "unproject_mask_to_pcd", "return_whole_mask",
    ]),
    ("hunyuanworld.py", "src/utils/hunyuanworld.py", [
        "_fill_small_boundary_spikes", "sheet_warping_helper", "get_filtered_mask", "sheet_warping",
    ]),
]


@unittest.skipUnless((SG / "src" / "pipeline" / "MatchFindRot.py").exists(), "scene_generation checkout not available")
class VendoredCopyTests(unittest.TestCase):
    def test_unchanged_functions(self) -> None:
        for vendored, original, names in UNCHANGED:
            new, old = _defs(VENDOR / vendored), _defs(SG / original)
            for name in names:
                with self.subTest(file=vendored, name=name):
                    self.assertIn(name, new)
                    self.assertIn(name, old)
                    self.assertTrue(_same(new[name], old[name]), f"{vendored}:{name} differs from {original}")

    def test_match_find_rot_mostly_unchanged(self) -> None:
        # rigid/similarity fits: the only change is the dropped save_transformation() call
        new, old = _defs(VENDOR / "match_find_rot.py"), _defs(SG / "src/pipeline/MatchFindRot.py")
        for name in ("PointCloudProcessor.apply_rigid_transform", "PointCloudProcessor.apply_similarity_transform"):
            strip = lambda fn: [s for s in fn.body if "save_transformation" not in ast.dump(s)]  # noqa: E731
            self.assertEqual([ast.dump(s) for s in strip(new[name])], [ast.dump(s) for s in strip(old[name])], name)

    def test_sam3d_helpers(self) -> None:
        new, old = _defs(WORKERS / "sam3d_meshes.py"), _defs(SG / "src/pipeline/sam3d.py")
        # convert_sam_to_hunyuan_space: same statements once the local imports are dropped
        def code(fn):  # statements without the docstring and the worker's local imports
            return [ast.dump(s) for s in fn.body[1:] if not isinstance(s, ast.Import)]

        self.assertEqual(code(new["convert_sam_to_hunyuan_space"]), code(old["convert_sam_to_hunyuan_space"]))

    def test_sheet_wrap_rays(self) -> None:
        new, old = _defs(WORKERS / "sheet_wrap.py"), _defs(SG / "src/pipeline/sheet_wraping_updated.py")
        body_new = [ast.dump(s) for s in new["_get_ray_directions"].body if not isinstance(s, ast.Import)]
        self.assertEqual(body_new, [ast.dump(s) for s in old["_get_ray_directions"].body])

    def test_minima_test_config(self) -> None:
        from r2s.gen.sg_glue import __file__ as _  # noqa: F401  (package imports nothing heavy)

        text = (SG / "src/config/default.py").read_text()
        for line in ("_CN.TEST.IMG0_RESIZE = 640", "_CN.TEST.IMG1_RESIZE = 640", "_CN.TEST.DF = 8",
                     "_CN.TEST.PADDING = False", "_CN.TEST.COARSE_SCALE = 0.125"):
            self.assertIn(line, text)


class SolverTests(unittest.TestCase):
    """compute_rigid_transform / two_stage_alignment recover known transforms."""

    @classmethod
    def setUpClass(cls) -> None:
        # matching_utils imports open3d at module level; the solvers only need numpy.
        stub = "open3d" not in sys.modules
        if stub:
            sys.modules["open3d"] = types.ModuleType("open3d")
        try:
            import importlib.util

            spec = importlib.util.spec_from_file_location("_gen_b_matching_utils", VENDOR / "matching_utils.py")
            cls.mu = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.mu)
        finally:
            if stub:
                del sys.modules["open3d"]

    def test_rigid_and_similarity(self) -> None:
        rng = np.random.default_rng(0)
        src = rng.normal(size=(50, 3))
        a = 0.7
        R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        t = np.array([0.3, -0.2, 0.5])
        R_est, t_est = self.mu.compute_rigid_transform(src, src @ R.T + t)
        np.testing.assert_allclose(R_est, R, atol=1e-9)
        np.testing.assert_allclose(t_est, t, atol=1e-9)
        s, R2, t2 = self.mu.two_stage_alignment(src, 1.7 * (src @ R.T) + t)
        np.testing.assert_allclose(s, 1.7, rtol=1e-6)
        np.testing.assert_allclose(s * (src @ R2.T) + t2, 1.7 * (src @ R.T) + t, atol=1e-6)

    def test_crop_and_uncrop(self) -> None:
        img = np.zeros((40, 60, 3), np.uint8)
        img[10:20, 30:35] = 200
        y0, x0, crop = self.mu.crop_and_resize(img, padding=2)
        self.assertEqual((y0, x0), (8, 28))
        self.assertEqual(crop.shape, (13, 8, 3))
        np.testing.assert_array_equal(self.mu.get_uncropped_keypoints([(1.7, 2.2)], y0, x0), [[29, 10]])


if __name__ == "__main__":
    unittest.main()
