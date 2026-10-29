"""Run the OLD repo's own code on the production inputs (baseline for steps 7, 8b, 9).

Everything runs in a subprocess with ``PYTHONPATH=<old repo>``,
``PYTHONDONTWRITEBYTECODE=1`` and a cwd under the output root, so nothing is ever
written into the old repo. Outputs per scene (``<out>/<scene>/``):

    cropped_collision/<crop>.glb(.json)        collision_mesh_crop.build_reachable_collision_mesh,
                                               called with the evaluator's own pose math
    object_pose_fit/refined_poses.json         object_pose_refine.py fit (script defaults)
    asset_refine/<scene>/B_v1/candidate.json   asset_refine_fit.py fit-joint (asset_refine_batch.sh args)
    score_disk/<scene>/{stock,B_v1}/scores.json   asset_refine_harness.py score on the on-disk candidate
    score_fit/<scene>/{stock,B_v1}/scores.json    ... on the candidate fitted above

    python tools/legacy/objects_oldcode_baseline.py --scenes droid2 droid820 [--stages crop posefit fit score]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from old_repo import ECHO2, OLD, OLDCODE_BASELINE  # noqa: E402  (also puts the repo on sys.path)
from r2s.common.transforms import SCALE_OPT_ADJUSTMENT  # noqa: E402
from r2s.config import Paths  # noqa: E402

_PATHS = Paths.load()
DATA = _PATHS.scene_data
DEFAULT_OUT = OLDCODE_BASELINE
DEFAULT_PYTHON = str(_PATHS.eden / ".venv" / "bin" / "python")  # the old evaluator ran in Eden's own venv
TAG = "B_v1"

# Evaluator pose math, executed from the evaluator's own source (functions pulled out with ast).
CROP_SCRIPT = r'''
import ast, json, math, os, sys
import numpy as np
from genesis.utils.geom import transform_quat_by_quat
from collision_mesh_crop import build_reachable_collision_mesh

old, scene, echo2, out_dir, adjustment = sys.argv[1:6]
src = open(os.path.join(old, "world_lab_evaluate_with_scene.py")).read()
tree = ast.parse(src)
want = {"quat_mul_np", "quat_z_rot", "_axis_quat_np", "euler_zyx_to_quat", "apply_relative_transform_to_quat"}
ns = {"np": np, "math": math}
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in want:
        exec(compile(ast.Module(body=[node], type_ignores=[]), "<evaluator>", "exec"), ns)
opt = os.path.join(old, "optimization_results", scene)
result_path = os.path.join(opt, "aligned_pose_refined.npy")
scale_path = os.path.join(opt, "aligned_pose_scene_scale_refined.npy")
result = np.load(result_path)
if result.shape[0] == 4:
    tx, ty, tz = float(result[0]), float(result[1]), float(result[2]); rz = float(result[3]); rx, ry = 0.0, 0.0
else:
    tx, ty, tz = float(result[0]), float(result[1]), float(result[2]); rx, ry, rz = float(result[3]), float(result[4]), float(result[5])
SCENE_SCALE = float(np.load(scale_path))
ADJUSTMENT = np.array(json.loads(adjustment))  # r2s.common.transforms.SCALE_OPT_ADJUSTMENT == the evaluator's
orig_collider_quat = transform_quat_by_quat(np.array([1.0, 0.0, 0.0, 0.0]), ADJUSTMENT)
collider_pos = (tx, ty, tz)
q = ns["apply_relative_transform_to_quat"](orig_collider_quat, rz, rx_rad=rx, ry_rad=ry)
res = build_reachable_collision_mesh(
    os.path.join(echo2, scene, "collider.glb"),
    scene_name=scene,
    scene_scale=SCENE_SCALE,
    root_pos=collider_pos,
    root_quat_wxyz=q,
    source_mesh_up=(0, 1, 0),
    source_mesh_front=(1, 0, 0),
    cache_dir=out_dir,
    radius=1.10, z_min=-0.15, z_max=1.45, sides=24,
)
print(json.dumps({k: (list(v) if isinstance(v, tuple) else v) for k, v in res.__dict__.items()}, default=str))
'''


def _run(cmd: list[str], *, cwd: Path, python: str, log_path: Path | None = None) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(OLD)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # Genesis' numba kernels are cache=True: keep their cache out of the (read-only) Eden venv.
    env["NUMBA_CACHE_DIR"] = str(cwd.parent / ".numba_cache")
    cwd.mkdir(parents=True, exist_ok=True)
    full = [python, *cmd]
    print("[baseline]", " ".join(full if len(" ".join(full)) < 400 else full[:2] + ["..."]), flush=True)
    started = time.time()
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as handle:
            rc = subprocess.call(full, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT)
    else:
        rc = subprocess.call(full, cwd=cwd, env=env)
    print(f"[baseline]   rc={rc} ({time.time() - started:.0f} s)", flush=True)
    if rc != 0:
        raise SystemExit(f"old-code command failed (rc={rc}); see {log_path or 'output above'}")


def _copy(src: Path, dst: Path) -> None:
    if not src.is_file():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    shutil.copy2(src, dst)


def stage_crop(scene: str, out: Path, python: str) -> None:
    d = out / "cropped_collision"
    for p in d.glob("reachable_collider_*"):
        p.unlink()
    adjustment = json.dumps(SCALE_OPT_ADJUSTMENT.tolist())
    _run(["-c", CROP_SCRIPT, str(OLD), scene, str(ECHO2), str(d), adjustment], cwd=out, python=python,
         log_path=out / "logs" / "crop.log")


def stage_posefit(scene: str, out: Path, python: str) -> None:
    d = out / "object_pose_fit"
    _run(
        [
            str(OLD / "object_pose_refine.py"), "fit",
            "--scene", scene, "--scene-data-dir", str(DATA),
            "--object-poses", str(OLD / "object_pose_fit" / scene / "object_poses.json"),
            "--out", str(d / "refined_poses.json"),
            "--overlay-dir", str(d / "overlays"),
        ],
        cwd=out, python=python, log_path=d / "fit.log",
    )


def stage_fit(scene: str, out: Path, python: str) -> None:
    root = out / "asset_refine"
    tag_old = OLD / "asset_refine" / scene / TAG
    extra = []
    if (tag_old / "decisions.json").is_file():
        extra += ["--decisions", str(tag_old / "decisions.json")]
    if (tag_old / "weights.json").is_file():
        extra += ["--weights", str(tag_old / "weights.json")]
    seed = OLD / "object_pose_fit" / scene / "refined_poses.json"
    seed_arg = ["--seed", str(seed)] if seed.is_file() else []
    _run(
        [
            str(OLD / "asset_refine_fit.py"), "fit-joint",
            "--scene", scene, "--scene-data-dir", str(DATA), "--refine-root", str(root),
            "--object-poses", str(OLD / "asset_refine" / scene / "stock" / "object_poses.json"),
            "--out", str(root / scene / TAG / "candidate.json"),
            *seed_arg, *extra,
        ],
        cwd=out, python=python, log_path=root / scene / TAG / "fit.log",
    )


def stage_score(scene: str, out: Path, python: str) -> None:
    old_root = OLD / "asset_refine" / scene
    for name, candidate in (("score_disk", old_root / TAG / "candidate.json"), ("score_fit", out / "asset_refine" / scene / TAG / "candidate.json")):
        if not candidate.is_file():
            print(f"[baseline] {name}: no candidate {candidate}; skipped")
            continue
        root = out / name
        _copy(old_root / "stock" / "object_poses.json", root / scene / "stock" / "object_poses.json")
        _copy(old_root / "stock" / "settle.json", root / scene / "stock" / "settle.json")
        _copy(old_root / TAG / "object_poses.json", root / scene / TAG / "object_poses.json")
        _copy(old_root / TAG / "settle.json", root / scene / TAG / "settle.json")
        _copy(candidate, root / scene / TAG / "candidate.json")
        _run(
            [
                str(OLD / "asset_refine_harness.py"), "--scene-data-dir", str(DATA), "--refine-root", str(root),
                "score", "--scene", scene, "--tags", "stock", TAG,
            ],
            cwd=out, python=python, log_path=root / scene / "score.log",
        )


STAGES = {"crop": stage_crop, "posefit": stage_posefit, "fit": stage_fit, "score": stage_score}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--stages", nargs="+", default=list(STAGES), choices=list(STAGES))
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--python", default=DEFAULT_PYTHON)
    args = ap.parse_args()
    out_root = args.out_root.resolve()
    if OLD.resolve() in (out_root, *out_root.parents):
        raise SystemExit("refusing to write inside the old repo")
    for scene in args.scenes:
        for stage in args.stages:
            STAGES[stage](scene, out_root / scene, args.python)
    return 0


if __name__ == "__main__":
    sys.exit(main())
