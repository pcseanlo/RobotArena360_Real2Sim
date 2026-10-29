"""tools/legacy: imports resolve after the move, and old_repo.py seeds from (fake) old checkouts."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
LEGACY = REPO / "tools" / "legacy"


def _run(args, tmp_path, **env):
    full = dict(os.environ, R2S_WORK=str(tmp_path / "work"), PYTHONDONTWRITEBYTECODE="1", **env)
    return subprocess.run([sys.executable, str(LEGACY / "old_repo.py"), *args], capture_output=True, text=True,
                          env=full, cwd=str(tmp_path))


def test_sibling_imports_resolve():
    """Legacy scripts import each other as siblings (their folder is sys.path[0]), never ``tools.``."""
    siblings = {p.stem for p in LEGACY.glob("*.py")}
    assert not (REPO / "tools" / "__init__.py").exists() and not (REPO / "tools" / "verify").exists()
    for path in sorted(LEGACY.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module:
                top = node.module.split(".")[0]
                assert top != "tools", f"{path.name}: from {node.module} import ..."
                if top not in sys.stdlib_module_names and top not in {"r2s", "numpy", "scipy", "PIL", "trimesh",
                                                                        "torch", "genesis", "eden", "gsplat", "plyfile"}:
                    assert top in siblings, f"{path.name}: unknown module {node.module}"
    assert (LEGACY / "README.md").is_file()


def test_seed_from_fake_old_checkout(tmp_path):
    old, echo2 = tmp_path / "old", tmp_path / "echo2"
    opt = old / "optimization_results" / "droidX"
    opt.mkdir(parents=True)
    for name in ("aligned_pose.npy", "aligned_pose_refined.npy"):
        np.save(opt / name, np.array([0.1, 0.2, 0.3, 0.4]))
    for name in ("aligned_pose_scene_scale.npy", "aligned_pose_scene_scale_refined.npy"):
        np.save(opt / name, np.array(0.9))
    (opt / "operation_surface_plane_fit.json").write_text("{}")  # after --through: not seeded
    (echo2 / "droidX").mkdir(parents=True)
    (echo2 / "droidX" / "gaussian.ply").write_bytes(b"ply\n")
    (echo2 / "droidX" / "collider.glb").write_bytes(b"glTF")
    env = {"R2S_OLD_REPO": str(old), "R2S_OLD_ECHO2": str(echo2)}

    where = _run(["where"], tmp_path, **env)
    assert where.returncode == 0 and where.stdout.split() == [f"OLD={old}", f"ECHO2={echo2}"]
    proc = _run(["seed", "--scene", "droidX", "--through", "align.scale"], tmp_path, **env)
    assert proc.returncode == 0, proc.stderr
    S = tmp_path / "work" / "scenes" / "droidX"
    assert (S / "world" / "collider.glb").is_symlink() and (S / "world" / "collider.glb").resolve() == echo2 / "droidX" / "collider.glb"
    assert not (S / "align" / "aligned_pose_refined.npy").is_symlink()  # small files are copied
    assert np.load(S / "align" / "aligned_pose_scene_scale_refined.npy") == 0.9
    assert not (S / "surface").exists()
    assert "relative_transform" in proc.stderr  # reported as missing in the old repo, not fatal


def test_missing_old_checkout_is_a_clear_error(tmp_path):
    proc = _run(["seed", "--scene", "droidX", "--through", "align.scale"], tmp_path,
                R2S_OLD_REPO=str(tmp_path / "nowhere"))
    assert proc.returncode == 1 and "old repo not found" in proc.stderr
    assert not (tmp_path / "work").exists()
