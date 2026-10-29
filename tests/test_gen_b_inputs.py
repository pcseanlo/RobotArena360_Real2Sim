"""Tests for the step-0 orchestration of 0c-5 .. 0c-8 (``inputs.meshes`` / ``place`` /
``table_depth`` / ``mesh_sheet``): object bookkeeping, CLI defaults, file lists.

The model workers themselves run in conda envs and are verified against the old
outputs with ``tools/legacy/gen_b_compare.py``.
"""

import argparse
import json
import os
import tempfile
import unittest
from pathlib import Path

from r2s.config import Paths, SceneConfig
from r2s.gen import mesh_sheet, meshes, place, table_depth
from r2s.gen.meshes import background_index, object_counts, prepare_worker_dir
from r2s.layout import SceneLayout
from r2s.steps import load_step

LEGACY_TABLE = {"object_name": "table", "count": [2], "mass": 0.5, "friction": 0.5, "surface": "Rough"}


def _entry(name, k):
    return {"object_name": name, "count": [k], "mass": 0.1, "friction": 0.4, "surface": "Plastic"}


def _layout(tmp: Path, scene="droidX") -> SceneLayout:
    fields = {f: tmp / f for f in Paths.__dataclass_fields__}
    fields["scene_data"] = tmp / "old"
    fields["scene_data_out"] = tmp / "out"
    return SceneLayout(scene, Paths(**fields), SceneConfig(scene, {"scene_type": "DROID"}))


def _defaults(step_name: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    load_step(step_name).add_arguments(parser)
    return parser.parse_args([])


class ObjectCountTests(unittest.TestCase):
    def _write(self, tmp, entries) -> Path:
        p = Path(tmp) / "result.json"
        p.write_text(json.dumps(entries))
        return p

    def test_counts_sorted_and_legacy_table_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write(tmp, [_entry("cup", 1), _entry("bowl", 0), LEGACY_TABLE])
            self.assertEqual(object_counts(p), [0, 1])

    def test_table_entry_not_last_is_an_object(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write(tmp, [dict(LEGACY_TABLE, count=[0]), _entry("cup", 1)])
            self.assertEqual(object_counts(p), [0, 1])

    def test_duplicate_counts_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write(tmp, [_entry("a", 0), _entry("b", 0)])
            with self.assertRaises(ValueError):
                object_counts(p)

    def test_background_index_is_number_of_objects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            layout = _layout(Path(tmp))
            io = layout.inputs_out
            io.masks_dir.mkdir(parents=True)
            io.mask_result.write_text(json.dumps([_entry("a", 0), _entry("b", 1), LEGACY_TABLE]))
            self.assertEqual(background_index(io), 2)
            io.mask_result.write_text(json.dumps([_entry("a", 0), _entry("b", 2)]))
            with self.assertRaises(ValueError):
                background_index(io)


class WorkerDirTests(unittest.TestCase):
    def test_refuses_symlinked_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "readonly").mkdir()
            (tmp / "out" / "assets").mkdir(parents=True)
            os.symlink(tmp / "readonly", tmp / "out" / "assets" / "droidX")
            with self.assertRaises(RuntimeError):
                prepare_worker_dir(tmp / "out" / "assets" / "droidX", tmp / "out")
            self.assertEqual(prepare_worker_dir(tmp / "out" / "assets" / "new", tmp / "out"), tmp / "out" / "assets" / "new")


class StepDefaultTests(unittest.TestCase):
    def test_registered_and_envs(self) -> None:
        # all four workers use CUDA (sheet warping: SG's helpers default to device="cuda")
        self.assertEqual(load_step("inputs.meshes").needs, "gpu+conda:sam3d-objects")
        self.assertEqual(load_step("inputs.place").needs, "gpu+conda:match")
        self.assertEqual(load_step("inputs.table_depth").needs, "gpu+conda:moge")
        self.assertEqual(load_step("inputs.mesh_sheet").needs, "gpu+conda:match")

    def test_production_defaults(self) -> None:
        a = _defaults("inputs.meshes")
        self.assertEqual((a.seed, a.resize_h, a.resize_w), (42, -1, -1))
        a = _defaults("inputs.place")
        self.assertEqual((a.background_object, a.background_depth, a.seed), ("last", "raw", 0))
        a = _defaults("inputs.mesh_sheet")
        self.assertEqual((a.target_tris, a.beta, a.alpha_threshold, a.depth_eps), (200_000, 10.0, 0.15, 1e-4))

    def test_match_config_has_no_secrets(self) -> None:
        text = place.MATCH_CONFIG.read_text()
        self.assertNotIn("api_key", text)
        import yaml

        cfg = yaml.safe_load(text)
        self.assertEqual(cfg["noise_removal"]["nb_neighbors"], 80)
        self.assertEqual(cfg["icp"]["max_iterations"], 2000)
        self.assertTrue(cfg["gemini_mask"])
        self.assertFalse(cfg["use_bypass"])


class FileListTests(unittest.TestCase):
    def test_inputs_outputs_follow_result_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            layout = _layout(Path(tmp))
            io = layout.inputs_out
            io.masks_dir.mkdir(parents=True)
            io.mask_result.write_text(json.dumps([_entry("a", 0), _entry("b", 1)]))

            a = _defaults("inputs.meshes")
            ins = meshes.inputs(layout, a)
            self.assertIn(io.masks_dir / "image_0001_mask.png", ins)
            self.assertNotIn(io.masks_dir / "image_0002_mask.png", ins)
            self.assertIn(io.assets_dir / "image_0001_segmented_scaled.glb", meshes.outputs(layout, a))

            a = _defaults("inputs.place")
            ins = place.inputs(layout, a)
            self.assertIn(io.assets_dir / "image_0001_segmented.glb", ins)
            self.assertIn(io.meta_dir / "pcds" / "0001_frame1.ply", ins)
            self.assertNotIn(io.inpainted_depth, ins)  # raw depth by default
            a.background_depth = "inpainted"
            self.assertIn(io.inpainted_depth, place.inputs(layout, a))
            self.assertEqual(place._background_count([0, 1], a), 1)
            a.background_object = "none"
            self.assertIsNone(place._background_count([0, 1], a))

            outs = table_depth.outputs(layout, _defaults("inputs.table_depth"))
            self.assertIn(io.masks_dir / "image_0002_mask.png", outs)
            self.assertIn(io.meta_dir / "pcds" / "0002_frame1.ply", outs)
            self.assertIn(io.root / "input_mask" / "droidX", table_depth.inputs(layout, _defaults("inputs.table_depth")))

            self.assertIn(io.masks_dir / "image_0002_mask.png", mesh_sheet.inputs(layout, _defaults("inputs.mesh_sheet")))
            self.assertEqual(mesh_sheet.outputs(layout, None), [io.mesh_sheet])


if __name__ == "__main__":
    unittest.main()
