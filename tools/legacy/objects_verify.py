"""Run the new collision / object steps on a seeded work run and compare with the old-code baseline.

    R2S_WORK=.../runs/verify python tools/legacy/objects_verify.py --scenes droid2 droid820

Per scene:

1. seed the inputs these steps read from the old repo (``tools/legacy/old_repo.py``
   mapping) *only when missing*, so other agents' new outputs in the same run are
   left alone; the stock dump (``objects.dump``) and the production seed
   (``objects.posefit``'s on-disk ``refined_poses.json``, which the old
   ``asset_refine_batch.sh`` passed as ``--seed``) are always copied;
2. ``collision.crop``;
3. ``objects.refine --stage fit`` (seeded by the on-disk refined poses, like the baseline);
4. copy the old settle dumps (+ the candidate render's ``object_poses.json``) into
   the refine folders, ``objects.refine --stage score --replay-root <old>/replay_results``;
5. ``objects.posefit`` (overwrites the seeded refined poses);
6. compare everything with ``tools/legacy/objects_oldcode_baseline.py``'s outputs
   (``objects_compare.py``); the report goes to ``<work>/compare_objects/<scene>.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from old_repo import OLD, OLDCODE_BASELINE, old_outputs  # noqa: E402
from r2s.collision.crop import crop_glb_name  # noqa: E402
from r2s.config import Paths  # noqa: E402
from r2s.layout import SceneLayout  # noqa: E402
from objects_compare import DEFAULT_IGNORE, compare_any, _print  # noqa: E402

TAG = "B_v1"


def _copy(src: Path, dst: Path) -> None:
    if not src.is_file():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    shutil.copy2(src, dst)


def seed(layout: SceneLayout) -> None:
    pairs = old_outputs(layout)
    for step in ("world.collider", "align.scale"):
        for src, dst in pairs[step]:
            if dst.exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.suffix in {".ply", ".glb"}:
                dst.symlink_to(src)
            else:
                shutil.copy2(src, dst)
            print(f"[seed] {dst}")
    for step in ("objects.dump", "objects.posefit"):
        for src, dst in pairs[step]:
            _copy(src, dst)
            print(f"[seed] {dst}")


def r2s_run(step: str, scene: str, *args: str, python: str) -> None:
    cmd = [python, "-m", "r2s", "run", step, "--scene", scene, *args]
    print("[verify]", " ".join(cmd), flush=True)
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(REPO)
    rc = subprocess.call(cmd, cwd=str(REPO), env=env)
    if rc != 0:
        raise SystemExit(f"{step} failed for {scene} (rc={rc})")


def compare_scene(layout: SceneLayout, base: Path) -> list[dict]:
    s = layout.scene
    crop = crop_glb_name()
    rdir = layout.refine_dir(TAG)
    pairs = [
        ("crop mesh", "mesh", layout.collision_dir / crop, base / "cropped_collision" / crop, {}),
        ("crop sidecar", "json", layout.collision_dir / (crop + ".json"), base / "cropped_collision" / (crop + ".json"), {}),
        ("posefit refined_poses", "json", layout.refined_poses, base / "object_pose_fit" / "refined_poses.json", {}),
        ("posefit log", "log", None, base / "object_pose_fit" / "fit.log", {}),
        # 'parent' is the seed's folder name: posefit (new layout) vs <scene> (old object_pose_fit/<scene>/)
        ("refine candidate", "json", rdir / "candidate.json", base / "asset_refine" / s / TAG / "candidate.json", {"ignore": [*DEFAULT_IGNORE, "parent"]}),
        ("refine fit.log", "log", rdir / "fit.log", base / "asset_refine" / s / TAG / "fit.log", {}),
        ("score stock", "json", layout.refine_dir("stock") / "scores.json", base / "score_fit" / s / "stock" / "scores.json", {}),
        ("score B_v1", "json", rdir / "scores.json", base / "score_fit" / s / TAG / "scores.json", {}),
    ]
    results = []
    for label, kind, a, b, kw in pairs:
        if a is None:
            continue
        try:
            r = compare_any(kind, a, b, **kw)
        except FileNotFoundError as exc:
            r = {"kind": kind, "a": str(a), "b": str(b), "equal": False, "first": [f"missing: {exc}"]}
        r["label"] = f"{s} {label}"
        _print(r)
        results.append(r)
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--baseline", type=Path, default=OLDCODE_BASELINE)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--skip-run", action="store_true", help="only compare")
    args = ap.parse_args()
    paths = Paths.load()
    print(f"[verify] work root {paths.work}")
    for scene in args.scenes:
        layout = SceneLayout.create(scene, paths)
        if not args.skip_run:
            seed(layout)
            r2s_run("collision.crop", scene, python=args.python)
            r2s_run("objects.refine", scene, "--stage", "fit", python=args.python)
            old_tag = OLD / "asset_refine" / scene
            _copy(old_tag / "stock" / "settle.json", layout.refine_dir("stock") / "settle.json")
            _copy(old_tag / TAG / "settle.json", layout.refine_dir(TAG) / "settle.json")
            _copy(old_tag / TAG / "object_poses.json", layout.refine_dir(TAG) / "object_poses.json")
            r2s_run("objects.refine", scene, "--stage", "score", "--replay-root", str(OLD / "replay_results"), python=args.python)
            r2s_run("objects.posefit", scene, python=args.python)
        results = compare_scene(layout, args.baseline / scene)
        out = paths.work / "compare_objects" / f"{scene}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2) + "\n")
        print(f"[verify] report -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
