"""Bridge to the old ``scene_gaussian_alignment`` repo, for seeding and regression checks only.

The old repo, Eden's ``echo2`` assets and the old object folders are **read only**.
Seeding copies small files and symlinks big ones; every r2s step writes its outputs
through ``r2s.common.io.prepare_output``, which unlinks a symlink before writing,
so a seeded link can never be written through.

    python tools/legacy/old_repo.py seed --scene droid2 --through surface.plane --run verify
    python tools/legacy/old_repo.py reference --scene droid2      # print old output paths
    eval "$(python tools/legacy/old_repo.py where)"                # OLD=... ECHO2=... for shell scripts

``OLD`` / ``ECHO2`` (``$R2S_OLD_REPO`` / ``$R2S_OLD_ECHO2``) are the one definition the
other legacy scripts and the regression tests use. Importing this module needs only
the standard library (r2s is imported inside the functions).
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

def _env_path(name: str) -> Path:
    """``$name``; unset, a path that never exists, so every message names the variable."""
    value = os.environ.get(name)
    return Path(value) if value else Path(f"<{name} unset>")


# Set in configs/local.env (see tools/legacy/README.md).
OLD = _env_path("R2S_OLD_REPO")
ECHO2 = _env_path("R2S_OLD_ECHO2")
# Old-code baseline written by objects_oldcode_baseline.py, read by objects_*verify.py.
OLDCODE_BASELINE = _env_path("R2S_WORK_ROOT") / "runs" / "verify_oldcode"
# The old runs recorded absolute paths through another mount of the same data:
# R2S_OLD_PATH_ALIASES="OLD_PREFIX=NEW_PREFIX[;...]", applied to both sides of a JSON comparison.
PATH_ALIASES = [tuple(item.split("=", 1)) for item in os.environ.get("R2S_OLD_PATH_ALIASES", "").split(";")
                if "=" in item]
CROP = "reachable_collider_r1p100_zm0p150_1p450_s24.glb"


def require_old() -> Path:
    """``OLD``, or exit with a clear message when the old checkout is not there."""
    if not (OLD / "optimization_results").is_dir():
        raise SystemExit(f"old repo not found at {OLD} (set R2S_OLD_REPO in configs/local.env); "
                         "these tools compare against it")
    return OLD


def old_outputs(layout: SceneLayout) -> dict[str, list[tuple[Path, Path]]]:
    """For each step: (old file, new file) pairs of that step's outputs."""
    s = layout.scene
    opt = OLD / "optimization_results" / s
    return {
        "world.import": [(ECHO2 / s / "gaussian.ply", layout.gaussian_ply)],
        "world.collider": [(ECHO2 / s / "collider.glb", layout.collider_glb)],
        "align.pose": [
            (opt / "aligned_pose.npy", layout.aligned_pose),
            (opt / "aligned_pose_scene_scale.npy", layout.aligned_scale),
            (opt / "aligned_pose_relative_transform.npy", layout.relative_transform),
        ],
        "align.scale": [
            (opt / "aligned_pose_refined.npy", layout.refined_pose),
            (opt / "aligned_pose_scene_scale_refined.npy", layout.refined_scale),
        ],
        "surface.plane": [(opt / "operation_surface_plane_fit.json", layout.plane_fit_json)],
        "surface.trim": [(ECHO2 / s / "gaussian_trimmed.ply", layout.gaussian_trimmed_ply)],
        "surface.receiver": [(opt / "operation_surface_shadow_v2.json", layout.receiver_json)],
        "collision.crop": [
            (opt / "cropped_collision" / CROP, layout.cropped_collider_glb),
            (opt / "cropped_collision" / (CROP + ".json"), layout.cropped_collider_glb.with_name(CROP + ".json")),
        ],
        "objects.dump": [(OLD / "asset_refine" / s / "stock" / "object_poses.json", layout.object_dump)],
        "objects.posefit": [(OLD / "object_pose_fit" / s / "refined_poses.json", layout.refined_poses)],
    }


_LINK_SUFFIXES = {".ply", ".glb"}


def seed(layout: SceneLayout, through: str) -> list[Path]:
    """Populate the run with old outputs of every step up to and including ``through``."""
    from r2s.steps import STEP_MODULES

    require_old()
    order = list(STEP_MODULES)
    if through not in order:
        raise SystemExit(f"unknown step {through}")
    wanted = order[: order.index(through) + 1]
    seeded = []
    for step, pairs in old_outputs(layout).items():
        if step not in wanted:
            continue
        for src, dst in pairs:
            if not src.exists():
                print(f"  (missing in old repo) {src}", file=sys.stderr)
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            if src.suffix in _LINK_SUFFIXES:
                dst.symlink_to(src)
            else:
                shutil.copy2(src, dst)
            seeded.append(dst)
    return seeded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_seed = sub.add_parser("seed")
    p_seed.add_argument("--scene", required=True)
    p_seed.add_argument("--through", required=True, help="last step whose old outputs are seeded")
    p_seed.add_argument("--run", default=None, help="run name under <work>/.. (sets R2S_WORK)")
    p_ref = sub.add_parser("reference")
    p_ref.add_argument("--scene", required=True)
    sub.add_parser("where", help="print OLD=... ECHO2=... (shell-quoted, for eval)")
    args = parser.parse_args()
    if args.cmd == "where":
        print(f"OLD={shlex.quote(str(OLD))} ECHO2={shlex.quote(str(ECHO2))}")
        return 0

    from r2s.config import Paths
    from r2s.layout import SceneLayout

    paths = Paths.load()
    if getattr(args, "run", None):
        os.environ["R2S_WORK"] = str(paths.work.parent / args.run)
        paths = Paths.load()
    layout = SceneLayout.create(args.scene, paths)
    if args.cmd == "seed":
        for p in seed(layout, args.through):
            print(p)
    else:
        for step, pairs in old_outputs(layout).items():
            for src, dst in pairs:
                print(f"{step:<18} {src}  ->  {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
