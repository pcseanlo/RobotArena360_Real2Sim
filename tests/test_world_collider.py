"""Tests for step 1b ``world.collider``: the mesher and presets are still the ones verified
against the user's ply2glb tool, the step runs in the repo env, and scenes.yaml picks
the right mode."""

import argparse
import ast
import hashlib
import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path

from r2s.config import SceneRegistry
from r2s.steps import load_step
from r2s.world import collider

# Fingerprints taken when both were checked against the original tool (ply2glb.py and
# run.sh of 2026-05-17, 2026-10-28). A deliberate change to the mesher or a preset
# moves every collider: update the fingerprint and record it in docs/changes.md.
MESHER_SHA256 = "83b255c086055e840deab10041ec68bb9ad37e4bd89239f22e1e4984a66f6583"  # code below the docstring
PRESETS_SHA256 = "3a804259793d6f1759b3826c5a6fc3d8cd5b60b8c64a4a7ee78ccfd56723bd6e"
MESHER = Path(collider.__file__).with_name("poisson_mesher.py")


def _code_below_docstring(path: Path) -> str:
    src = path.read_text()
    end = ast.parse(src).body[0].end_lineno
    return "\n".join(src.splitlines()[end:]) + "\n"


class ColliderPresetTests(unittest.TestCase):
    def test_mesher_is_the_verified_code(self) -> None:
        self.assertEqual(hashlib.sha256(_code_below_docstring(MESHER).encode()).hexdigest(), MESHER_SHA256)

    def test_presets_are_the_verified_ones(self) -> None:
        digest = hashlib.sha256(json.dumps(collider.PRESETS, sort_keys=True).encode()).hexdigest()
        self.assertEqual(digest, PRESETS_SHA256)

    def test_argv_and_defaults(self) -> None:
        argv = collider.collider_argv(Path("in.ply"), Path("out.glb"), "planar")
        self.assertEqual(argv[:4], ["-m", "r2s.world.poisson_mesher", "in.ply", "out.glb"])
        self.assertIn("--meshfix", argv)
        parser = argparse.ArgumentParser()
        load_step("world.collider").add_arguments(parser)
        self.assertEqual(parser.parse_args([]).mode, "surface")
        self.assertEqual(load_step("world.collider").needs, "cpu")

    @unittest.skipUnless(importlib.util.find_spec("open3d"), "open3d not installed (scripts/setup_env.sh)")
    def test_mesher_runs_in_this_env(self) -> None:
        proc = subprocess.run([sys.executable, "-m", collider.MESHER, "--help"], capture_output=True, text=True,
                              timeout=300)
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertIn("usage:", proc.stdout)

    def test_planar_scenes(self) -> None:
        reg = SceneRegistry()
        modes = {name: reg.get(name).step_options("world.collider").get("mode") for name in reg.names()}
        self.assertEqual(modes["droid17"], "planar")
        self.assertEqual(modes["droid37"], "planar")
        self.assertEqual(modes["droid820"], "surface")
        self.assertTrue(all(m in collider.PRESETS for m in modes.values()))


if __name__ == "__main__":
    unittest.main()
