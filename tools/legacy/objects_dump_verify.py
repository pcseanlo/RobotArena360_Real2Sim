"""Verify ``objects.dump`` against the old-code baseline and the on-disk old dumps.

    R2S_WORK=.../runs/verify_dump python tools/legacy/objects_dump_verify.py --scenes droid2 droid820 droid1034

Uses its own work run (inputs seeded from the old repo: collider symlink, refined
pose/scale, plane fit, shadow_v2 copies), so no other agent's outputs are touched.
Per scene: ``collision.crop`` and ``objects.dump`` through the CLI (stock), then the
library for the B_v1 dump (``--scale-override`` = the on-disk B_v1 candidate) and for
``--support-query-rotation identity``. Compares with
``runs/verify_oldcode/<s>/dump/{stock,B_v1}/object_poses.json`` (old code today) and
the on-disk ``asset_refine/<s>/{stock,B_v1}/object_poses.json`` /
``object_pose_fit/<s>/object_poses.json``; reports per-object z changes for identity.
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

from old_repo import OLD, OLDCODE_BASELINE, PATH_ALIASES, old_outputs  # noqa: E402
from objects_compare import _print, compare_json  # noqa: E402
from r2s.config import Paths  # noqa: E402
from r2s.layout import SceneLayout  # noqa: E402
from r2s.objects.dump import DumpSettings, dump_scene_objects, load_scene_objects_from_layout  # noqa: E402



def seed(layout: SceneLayout) -> None:
    pairs = old_outputs(layout)
    for step in ("world.collider", "align.scale", "surface.plane", "surface.receiver"):
        for src, dst in pairs[step]:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            if src.suffix in {".ply", ".glb"}:
                dst.symlink_to(src)
            else:
                shutil.copy2(src, dst)


def r2s_run(step: str, scene: str, *args: str) -> None:
    cmd = [sys.executable, "-m", "r2s", "run", step, "--scene", scene, *args]
    print("[verify]", " ".join(cmd), flush=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(REPO))
    if subprocess.call(cmd, cwd=str(REPO), env=env) != 0:
        raise SystemExit(f"{step} failed for {scene}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--baseline", type=Path, default=OLDCODE_BASELINE)
    args = ap.parse_args()
    paths = Paths.load()
    if paths.work.name != "verify_dump":
        raise SystemExit(f"set R2S_WORK to a run named verify_dump (got {paths.work})")
    report: dict = {}
    for s in args.scenes:
        layout = SceneLayout.create(s, paths)
        seed(layout)
        r2s_run("collision.crop", s)
        r2s_run("objects.dump", s)
        extra = paths.work / "compare_dump" / s
        quiet = lambda line: None  # noqa: E731
        cand = OLD / "asset_refine" / s / "B_v1" / "candidate.json"
        b_v1 = dump_scene_objects(load_scene_objects_from_layout(layout, DumpSettings(scale_override=str(cand)), log=quiet), extra / "B_v1" / "object_poses.json")
        ident_scene = load_scene_objects_from_layout(layout, DumpSettings(support_query_rotation="identity"), log=quiet)
        ident = dump_scene_objects(ident_scene, extra / "identity" / "object_poses.json")

        results = []
        for label, a, b in [
            ("stock vs old code", layout.object_dump, args.baseline / s / "dump" / "stock" / "object_poses.json"),
            ("stock vs disk asset_refine/stock", layout.object_dump, OLD / "asset_refine" / s / "stock" / "object_poses.json"),
            ("stock vs disk object_pose_fit", layout.object_dump, OLD / "object_pose_fit" / s / "object_poses.json"),
            ("B_v1 vs old code", b_v1, args.baseline / s / "dump" / "B_v1" / "object_poses.json"),
            ("B_v1 vs disk", b_v1, OLD / "asset_refine" / s / "B_v1" / "object_poses.json"),
        ]:
            r = compare_json(a, b, aliases=PATH_ALIASES)
            r["label"] = f"{s} {label}"
            _print(r)
            results.append(r)

        legacy = json.loads(layout.object_dump.read_text())
        identity = json.loads(Path(ident).read_text())
        dz = []
        for lo, io in zip(legacy["objects"], identity["objects"]):
            assert lo["safe_key"] == io["safe_key"]
            o = ident_scene.objects[io["safe_key"]]
            dz.append(
                {
                    "safe_key": lo["safe_key"],
                    "object_name": lo["object_name"],
                    "z_legacy": lo["pos"][2],
                    "z_identity": io["pos"][2],
                    "dz_m": io["pos"][2] - lo["pos"][2],
                    "dxy_m": float(sum((a - b) ** 2 for a, b in zip(io["pos"][:2], lo["pos"][:2])) ** 0.5),
                    "identity_support_source": o.correction.support_source if o.correction else None,
                    "identity_lift": o.correction.lift if o.correction else None,
                }
            )
            print(f"  {s} {lo['object_name'][:22]:22s} z legacy {lo['pos'][2]:.4f} identity {io['pos'][2]:.4f}  dz {100 * dz[-1]['dz_m']:+.2f} cm ({dz[-1]['identity_support_source']})")
        report[s] = {"compare": results, "identity_dz": dz}
    out = paths.work / "compare_dump" / "report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[verify] report -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
