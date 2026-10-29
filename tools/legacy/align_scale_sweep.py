"""Recover the lost per-scene truncation of the old scale fit (GPU).

Builds the scale-fit scene once from a run's ``align/aligned_pose.npy`` (seed it from
the old repo with ``tools/legacy/old_repo.py seed --through align.pose``), samples the
collider / metric surfaces once, then re-runs the (deterministic, given the samples)
Adam Chamfer fit for every truncation in the grid and compares each refined scale
with the old ``optimization_results/<s>/aligned_pose_scene_scale_refined.npy``.

    R2S_WORK=.../runs/verify_sim python tools/legacy/align_scale_sweep.py --scene droid820 \
        --out .../runs/verify_sim/_scale_sweep/droid820

Writes ``<out>/sweep.json`` (one row per setting, sorted by |s - s_old|) and the
init snapshot. ``--metric-mesh`` overrides the mesh (e.g. to test the segmented-GLB
fallback the old script used before ``mesh_sheet.glb`` existed).
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from r2s.align import scale_fit  # noqa: E402
from r2s.common.io import write_json  # noqa: E402
from r2s.layout import SceneLayout  # noqa: E402
from old_repo import OLD  # noqa: E402

DEFAULT_GRID = (0.05, 0.1, 0.2, 0.3, 0.4)


def problem_from_samples(layout: SceneLayout, samples: Path) -> scale_fit.ScaleProblem:
    """Rebuild the fit problem from a saved ``samples.npz`` (same samples, no Genesis)."""
    from r2s.align import scale_core as core
    from r2s.common.transforms import load_pose

    z = np.load(samples)
    pose = np.load(layout.aligned_pose)
    t, rx, ry, rz = load_pose(layout.aligned_pose)
    K, W, H, fov_y, cp, cl, cu = core.load_calibration(str(layout.inputs.intrinsics), str(layout.inputs.extrinsics))
    calib = dict(K=K, W=W, H=H, fov_y=fov_y, cam_pos=np.asarray(cp), cam_lookat=np.asarray(cl), cam_up=np.asarray(cu))
    return scale_fit.ScaleProblem(
        x_coll_0=z["x_coll_0"], x_metric=z["x_metric"], pivot=z["pivot"], s_0=float(np.load(layout.aligned_scale)),
        t_canonical=t, rx=rx, ry=ry, rz=rz, rotate_xy=pose.reshape(-1).shape[0] == 6, pose_dtype=pose.dtype, calib=calib,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--grid", type=float, nargs="+", default=list(DEFAULT_GRID), help="values for both truncate_w and truncate_h")
    ap.add_argument("--extra", type=str, nargs="*", default=["0.1,0.3", "0.4,0.4", "0.2,0.2"], help="extra w,h pairs")
    ap.add_argument("--metric-mesh", type=Path, default=None)
    ap.add_argument("--samples", type=Path, default=None, help="reuse a previous run's samples.npz (no Genesis build)")
    ap.add_argument("--no-tf32", action="store_true", help="with --samples: full-FP32 matmuls (en.init enables TF32)")
    args = ap.parse_args()

    layout = SceneLayout.create(args.scene)
    old_scale_path = OLD / "optimization_results" / args.scene / "aligned_pose_scene_scale_refined.npy"
    old_pose_path = OLD / "optimization_results" / args.scene / "aligned_pose_refined.npy"
    s_old = float(np.load(old_scale_path))
    old_pose = np.load(old_pose_path)
    ns = argparse.Namespace(metric_mesh=args.metric_mesh)
    metric_mesh, source = scale_fit.resolve_metric_mesh(layout, ns)
    print(f"[sweep] {args.scene}: s_old={s_old:.6f} metric={metric_mesh} ({source})")

    base = scale_fit.ScaleFitSettings()
    t0 = time.time()
    if args.samples is not None:
        # en.init's torch setup (TF32 matmuls on, the default since Eden c53dd64d / Nov 2025): torch.cdist runs
        # through matmuls, so the fit differs at the 1e-3 level without it.
        from eden.utils.torch import configure_torch_backends

        configure_torch_backends(allow_tf32=not args.no_tf32)
        problem = problem_from_samples(layout, args.samples)
    else:
        problem = None
    if problem is None:
        problem, env = scale_fit.prepare_problem(
            pose_path=layout.aligned_pose,
            scale_path=layout.aligned_scale,
            collider_glb=layout.collider_glb,
            gaussian_ply=layout.gaussian_ply,
            metric_mesh=metric_mesh,
            intrinsics=layout.inputs.intrinsics,
            extrinsics=layout.inputs.extrinsics,
            log_dir=args.out / "eden_logs",
            settings=base,
            init_snapshot=args.out / scale_fit.SNAPSHOT_INIT,
        )
        np.savez_compressed(args.out / "samples.npz", x_coll_0=problem.x_coll_0, x_metric=problem.x_metric, pivot=problem.pivot)
    build_s = time.time() - t0

    pairs = list(itertools.product(args.grid, args.grid))
    for item in args.extra:
        w, h = (float(v) for v in item.split(","))
        if (w, h) not in pairs:
            pairs.append((w, h))
    rows = []
    for w, h in pairs:
        t1 = time.time()
        settings = replace(base, truncate_w=w, truncate_h=h)
        s_new, loss = scale_fit.solve(problem, settings)
        pose_new = scale_fit.refined_pose_array(problem, s_new)
        rows.append({
            "truncate_w": w, "truncate_h": h, "s_refined": s_new, "loss": loss,
            "abs_diff": abs(s_new - s_old), "rel_diff": abs(s_new - s_old) / s_old,
            "pose_diff_m": float(np.linalg.norm(pose_new[:3] - old_pose[:3])),
            "seconds": round(time.time() - t1, 1),
        })
        print(f"[sweep] w={w:.2f} h={h:.2f}  s={s_new:.6f}  |ds|={abs(s_new - s_old):.6f}  loss={loss:.6f}")
    rows.sort(key=lambda r: r["abs_diff"])
    write_json(args.out / "sweep.json", {
        "scene": args.scene, "s_old": s_old, "metric_mesh": str(metric_mesh), "metric_mesh_source": source,
        "s_init": problem.s_0, "n_coll": int(len(problem.x_coll_0)), "n_metric": int(len(problem.x_metric)),
        "build_seconds": round(build_s, 1), "rows": rows,
    })
    print(f"[sweep] best: {json.dumps(rows[0])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
