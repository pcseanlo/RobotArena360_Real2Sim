"""Compare the new step 0c-5 .. 0c-8 and 1b outputs with the old (production) files.

    R2S_WORK=... python tools/legacy/gen_b_compare.py --scene droid820 [--steps meshes place ...]
                                                      [--old-collider PATH] [--json OUT]

Old files come from ``paths.scene_data`` (read only) and, for the collider,
Eden's ``assets/echo2/<s>/collider.glb``. New files from ``paths.scene_data_out``
and ``<work>/scenes/<s>/world/collider.glb``. Prints one table per step and
writes everything to ``<work>/verify/gen_b_<scene>.json`` (or ``--json``).

Metrics
  meshes       vertex/face counts, bbox extents, symmetric Chamfer (mean of the two
               nearest-neighbour means, 20k surface samples) in the mesh frame and
               after rigid ICP, both relative to the old bbox diagonal, and nearest-
               VERTEX distances (fraction reproduced exactly, p95/p99) -- surface
               sampling alone puts two samplings of the same mesh ~0.5 sample
               spacing apart, so the vertex metric is the one that shows real change
  place        per object: scale ratio new/old, translation distance, angle between
               the rotations (polar part for the scaled background matrix), and the
               ratio of the matrices' own scale (background similarity fit)
  table_depth  max / mean |new - old| of inpainted_depth.npy, exact-equality of the
               background mask image, point count / max point distance of its cloud
  mesh_sheet   vertex/face counts, symmetric Chamfer (m), nearest-vertex distances
  collider     vertex/face counts, symmetric Chamfer (m) on 200k samples, nearest-vertex distances
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from r2s.common.mesh import load_combined_mesh  # noqa: E402
from r2s.config import Paths, SceneRegistry  # noqa: E402
from r2s.gen.meshes import background_index, object_counts, object_mask, segmented_name  # noqa: E402
from r2s.layout import SceneLayout  # noqa: E402
from old_repo import ECHO2  # noqa: E402



# --------------------------------------------------------------------- geometry
def _sample(mesh, n: int, seed: int = 0) -> np.ndarray:
    import trimesh

    if len(mesh.faces) == 0:
        return np.asarray(mesh.vertices, dtype=np.float64)
    pts, _ = trimesh.sample.sample_surface(mesh, n, seed=seed)
    return np.asarray(pts, dtype=np.float64)


def chamfer(a: np.ndarray, b: np.ndarray) -> dict:
    from scipy.spatial import cKDTree

    d_ab = cKDTree(b).query(a, k=1)[0]
    d_ba = cKDTree(a).query(b, k=1)[0]
    return {
        "chamfer": float(0.5 * (d_ab.mean() + d_ba.mean())),
        "p95": float(max(np.quantile(d_ab, 0.95), np.quantile(d_ba, 0.95))),
        "hausdorff": float(max(d_ab.max(), d_ba.max())),
    }


def vertex_nn(old_v: np.ndarray, new_v: np.ndarray, scale: float) -> dict:
    """Nearest-vertex distances in both directions (no sampling noise): how many
    vertices are reproduced exactly (< 1e-6 x scale) and how far the rest moved."""
    from scipy.spatial import cKDTree

    d_no = cKDTree(old_v).query(new_v, k=1)[0]
    d_on = cKDTree(new_v).query(old_v, k=1)[0]
    d = np.concatenate([d_no, d_on])
    return {"exact_frac": float((d < 1e-6 * scale).mean()), "mean": float(d.mean()),
            "p95": float(np.quantile(d, 0.95)), "p99": float(np.quantile(d, 0.99)), "max": float(d.max())}


def _icp(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    import trimesh

    matrix, transformed, _ = trimesh.registration.icp(src, dst, max_iterations=50, reflection=False, scale=False)
    return np.asarray(transformed)


def mesh_stats(path: Path) -> dict:
    mesh = load_combined_mesh(path)
    ext = mesh.bounds[1] - mesh.bounds[0]
    return {"vertices": int(len(mesh.vertices)), "faces": int(len(mesh.faces)), "extents": [float(x) for x in ext]}


def compare_meshes(old: Path, new: Path, n: int = 20000, align: bool = True) -> dict:
    mo, mn = load_combined_mesh(old), load_combined_mesh(new)
    diag = float(np.linalg.norm(mo.bounds[1] - mo.bounds[0])) or 1.0
    # Same sampling seed for both: identical meshes give identical samples (Chamfer 0).
    a, b = _sample(mn, n, 0), _sample(mo, n, 0)
    out = {
        "old": mesh_stats(old), "new": mesh_stats(new), "old_bbox_diag": diag,
        "raw": chamfer(a, b),
    }
    out["vertex_nn"] = vertex_nn(np.asarray(mo.vertices, dtype=np.float64), np.asarray(mn.vertices, dtype=np.float64), diag)
    if len(mo.vertices) == len(mn.vertices) and len(mo.faces) == len(mn.faces):
        out["vertex_max_abs"] = float(np.abs(np.asarray(mo.vertices) - np.asarray(mn.vertices)).max())
        out["faces_equal"] = bool(np.array_equal(np.asarray(mo.faces), np.asarray(mn.faces)))
    out["raw_rel"] = out["raw"]["chamfer"] / diag
    if align:
        out["icp"] = chamfer(_icp(a, b), b)
        out["icp_rel"] = out["icp"]["chamfer"] / diag
    return out


# ---------------------------------------------------------------- rotations
def _polar(m: np.ndarray) -> tuple[np.ndarray, float]:
    u, s, vt = np.linalg.svd(m)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r, float(np.cbrt(abs(np.linalg.det(m))))


def rotation_angle_deg(r_old: np.ndarray, r_new: np.ndarray) -> float:
    r = r_old.T @ r_new
    return float(np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1.0, 1.0))))


def _posed(points: np.ndarray, entry: dict) -> np.ndarray:
    """``R (s v) + t`` -- the composition MatchFindRot uses for table.glb."""
    return (float(entry["scale"]) * points) @ np.array(entry["rotation"], dtype=float).T + np.array(entry["translation"])


def posed_difference(glb: Path, old: dict, new: dict, n: int = 5000) -> dict:
    """Place the same mesh samples with both poses: per-point distance (pose error)
    and Chamfer (ignores symmetric re-parametrisations, e.g. a cube turned 180 deg)."""
    pts = _sample(load_combined_mesh(glb), n, 0)
    a, b = _posed(pts, old), _posed(pts, new)
    size = float(np.linalg.norm(b.max(0) - b.min(0))) or 1.0
    return {"posed_point_dist": float(np.linalg.norm(a - b, axis=1).mean()),
            "posed_chamfer": chamfer(b, a)["chamfer"], "posed_size": size}


def compare_transformations(old_json: Path, new_json: Path, assets_dir: Path | None = None) -> dict:
    old = json.loads(old_json.read_text())
    new = json.loads(new_json.read_text()) if new_json.exists() else {}
    rows = {}
    for key in sorted(set(old) | set(new)):
        if key not in old or key not in new:
            rows[key] = {"missing_in": "old" if key not in old else "new"}
            continue
        o, n = old[key], new[key]
        ro, so = _polar(np.array(o["rotation"], dtype=float))
        rn, sn = _polar(np.array(n["rotation"], dtype=float))
        rows[key] = {
            "scale_old": float(o["scale"]), "scale_new": float(n["scale"]),
            "scale_ratio": float(n["scale"]) / float(o["scale"]),
            "translation_dist": float(np.linalg.norm(np.subtract(n["translation"], o["translation"]))),
            "rotation_deg": rotation_angle_deg(ro, rn),
            "matrix_scale_old": so, "matrix_scale_new": sn,
            "matched_old": "original_rotation" in o, "matched_new": "original_rotation" in n,
        }
        if assets_dir is not None and (assets_dir / f"{key}.glb").exists():
            rows[key].update(posed_difference(assets_dir / f"{key}.glb", o, n))
    return rows


# --------------------------------------------------------------------- steps
def step_meshes(layout: SceneLayout) -> dict:
    old, new = layout.inputs_src, layout.inputs_out
    rows = {}
    for k in object_counts(new.mask_result):
        name = segmented_name(k)
        row = {}
        for suffix, align in ((".glb", True), ("_scaled.glb", True), (".ply", False)):
            po, pn = old.assets_dir / f"{name}{suffix}", new.assets_dir / f"{name}{suffix}"
            if pn.is_symlink() or not pn.exists():
                row[suffix] = {"missing": True}
                continue
            row[suffix] = compare_meshes(po, pn, align=align)
        mo, mn = old.assets_dir / f"{name}_mask.png", new.assets_dir / f"{name}_mask.png"
        if mn.exists():
            import imageio.v2 as imageio

            a, b = imageio.imread(mo), imageio.imread(mn)
            row["_mask.png"] = {"equal": bool(a.shape == b.shape and np.array_equal(a, b)),
                                "differing_px": int((a != b).sum()) if a.shape == b.shape else -1}
        rows[name] = row
    return rows


def step_place(layout: SceneLayout, new_json: Path | None = None) -> dict:
    # posed metrics use the OLD meshes (the ones production placed)
    return compare_transformations(layout.inputs_src.transformations, new_json or layout.inputs_out.transformations,
                                   layout.inputs_src.assets_dir)


def step_table_depth(layout: SceneLayout) -> dict:
    import imageio.v2 as imageio
    import trimesh

    old, new = layout.inputs_src, layout.inputs_out
    do, dn = np.load(old.inpainted_depth), np.load(new.inpainted_depth)
    diff = np.abs(dn.astype(np.float64) - do.astype(np.float64))
    n = background_index(new)
    out = {
        "depth_shape": list(dn.shape), "depth_max_abs": float(diff.max()), "depth_mean_abs": float(diff.mean()),
        "depth_bit_identical": bool(np.array_equal(do, dn)),
        "depth_old_mean": float(do.mean()), "depth_new_mean": float(dn.mean()), "background_index": n,
    }
    # the old background files may sit at a different index (legacy align_inpaint.py)
    old_idx = n
    for name, po, pn in (("whole_mask", old.meta_dir / "whole_mask.png", new.meta_dir / "whole_mask.png"),
                         ("bg_mask", object_mask(old, old_idx), object_mask(new, n))):
        if po.exists() and pn.exists():
            a, b = imageio.imread(po), imageio.imread(pn)
            out[f"{name}_equal"] = bool(a.shape == b.shape and np.array_equal(a, b))
    po, pn = old.meta_dir / "pcds" / f"{old_idx:04d}_frame1.ply", new.meta_dir / "pcds" / f"{n:04d}_frame1.ply"
    if po.exists() and pn.exists():
        a = np.asarray(trimesh.load(str(po)).vertices)
        b = np.asarray(trimesh.load(str(pn)).vertices)
        out["bg_pcd_points_old"], out["bg_pcd_points_new"] = int(len(a)), int(len(b))
        out["bg_pcd_max_dist"] = float(np.abs(a - b).max()) if a.shape == b.shape else None
    return out


def step_mesh_sheet(layout: SceneLayout) -> dict:
    return compare_meshes(layout.inputs_src.mesh_sheet, layout.inputs_out.mesh_sheet, n=100000, align=False)


def step_collider(layout: SceneLayout, old_collider: Path | None) -> dict:
    old = old_collider or ECHO2 / layout.scene / "collider.glb"
    out = compare_meshes(old, layout.collider_glb, n=200000, align=False)
    out["old_path"] = str(old)
    return out


# --------------------------------------------------------------------- report
def _fmt(x, nd=4):
    return f"{x:.{nd}g}" if isinstance(x, float) else str(x)


def print_report(scene: str, res: dict) -> None:
    print(f"\n### {scene}")
    if "meshes" in res:
        print("\ninputs.meshes   (Chamfer relative to the old bbox diagonal)")
        print(f"{'object':<22}{'file':<13}{'V old/new':>18}{'F old/new':>18}{'raw':>10}{'icp':>10}"
              f"{'vtx exact':>10}{'vtx p95':>10}{'vtx p99':>10}")
        for name, row in res["meshes"].items():
            for suffix, r in row.items():
                if suffix == "_mask.png":
                    print(f"{name:<22}{suffix:<13}  equal={r['equal']} differing_px={r['differing_px']}")
                    continue
                if r.get("missing"):
                    print(f"{name:<22}{suffix:<13}  (missing)")
                    continue
                v = f"{r['old']['vertices']}/{r['new']['vertices']}"
                f = f"{r['old']['faces']}/{r['new']['faces']}"
                icp = _fmt(r.get("icp_rel", float("nan")), 3)
                vn = r["vertex_nn"]
                d = r["old_bbox_diag"]
                print(f"{name:<22}{suffix:<13}{v:>18}{f:>18}{_fmt(r['raw_rel'], 3):>10}{icp:>10}"
                      f"{vn['exact_frac']:>10.3f}{vn['p95'] / d:>10.2g}{vn['p99'] / d:>10.2g}")
    if "place" in res:
        print("\ninputs.place")
        print(f"{'object':<22}{'scale o/n':>22}{'ratio':>8}{'dt [m]':>9}{'dR [deg]':>10}{'mat s o/n':>16}"
              f"{'matched o/n':>13}{'pose err [mm]':>15}{'chamfer [mm]':>14}{'size [mm]':>11}")
        for key, r in res["place"].items():
            if "missing_in" in r:
                print(f"{key:<22}  missing in {r['missing_in']}")
                continue
            print(f"{key:<22}{r['scale_old']:>11.4f}/{r['scale_new']:<10.4f}{r['scale_ratio']:>8.3f}"
                  f"{r['translation_dist']:>9.4f}{r['rotation_deg']:>10.2f}"
                  f"{r['matrix_scale_old']:>8.3f}/{r['matrix_scale_new']:<7.3f}{str(r['matched_old'])[0]:>10}/{str(r['matched_new'])[0]}"
                  f"{1e3 * r.get('posed_point_dist', float('nan')):>15.1f}{1e3 * r.get('posed_chamfer', float('nan')):>14.1f}"
                  f"{1e3 * r.get('posed_size', float('nan')):>11.0f}")
    if "table_depth" in res:
        print("\ninputs.table_depth")
        for k, v in res["table_depth"].items():
            print(f"  {k:<22} {_fmt(v, 6)}")
    for step in ("mesh_sheet", "collider"):
        if step in res:
            r = res[step]
            print(f"\n{step}: V {r['old']['vertices']}/{r['new']['vertices']}  F {r['old']['faces']}/{r['new']['faces']}"
                  f"  chamfer {r['raw']['chamfer']:.3g} m  p95 {r['raw']['p95']:.3g} m  hausdorff {r['raw']['hausdorff']:.3g} m"
                  f"  (rel {r['raw_rel']:.3g})"
                  + (f"  vertex_max_abs {r['vertex_max_abs']:.3g} faces_equal {r['faces_equal']}" if "vertex_max_abs" in r else ""))
            vn = r["vertex_nn"]
            print(f"    vertex NN: exact {vn['exact_frac']:.4f}  mean {vn['mean']:.3g} m  p95 {vn['p95']:.3g} m"
                  f"  p99 {vn['p99']:.3g} m  max {vn['max']:.3g} m")


STEPS = ("meshes", "place", "table_depth", "mesh_sheet", "collider")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--steps", nargs="+", choices=STEPS, default=list(STEPS))
    ap.add_argument("--old-collider", type=Path, default=None)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--transformations", type=Path, default=None,
                    help="new transformations.json to compare (default: the scene_data_out one)")
    args = ap.parse_args()
    layout = SceneLayout.create(args.scene, Paths.load(), SceneRegistry())

    res: dict = {}
    fns = {"meshes": step_meshes, "place": lambda lay: step_place(lay, args.transformations), "table_depth": step_table_depth,
           "mesh_sheet": step_mesh_sheet, "collider": lambda lay: step_collider(lay, args.old_collider)}
    for step in args.steps:
        try:
            res[step] = fns[step](layout)
        except FileNotFoundError as exc:
            res[step] = {"error": str(exc)}
            print(f"[{step}] skipped: {exc}", file=sys.stderr)
    ok = {k: v for k, v in res.items() if "error" not in v}
    print_report(args.scene, ok)
    out = args.json or layout.paths.work / "verify" / f"gen_b_{args.scene}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2) + "\n")
    print(f"\n[gen_b_compare] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
