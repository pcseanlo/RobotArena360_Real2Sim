"""Compare an end-to-end run (only new code) with the old repo's outputs for one scene.

Differences compound from step to step (a new pose fit -> new scale -> new plane...),
so this reports physical quantities, not file equality: alignment, table height and
extent, support plane, object load positions, Method B fit, settle verdicts.

    R2S_WORK=<run> python tools/legacy/e2e_compare.py --scene droid820 [--old-fit <candidate.json>]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from r2s.layout import SceneLayout  # noqa: E402
from old_repo import OLD  # noqa: E402


def _load(path: Path):
    return json.loads(Path(path).read_text())


def _objects_by_key(doc: dict) -> dict:
    return {o["safe_key"]: o for o in doc.get("objects", [])}


def _yaw(q) -> float:
    w, x, y, z = q
    return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--old-fit", type=Path, default=None, help="old B_v1 candidate (default: old repo's)")
    args = ap.parse_args()
    lay = SceneLayout.create(args.scene)
    s = args.scene
    opt = OLD / "optimization_results" / s
    rows: list[tuple[str, str, str, str]] = []

    def row(name, old, new, delta):
        rows.append((name, old, new, delta))

    # alignment
    p_old, p_new = np.load(opt / "aligned_pose_refined.npy"), np.load(lay.refined_pose)
    s_old, s_new = float(np.load(opt / "aligned_pose_scene_scale_refined.npy")), float(np.load(lay.refined_scale))
    row("scene scale", f"{s_old:.5f}", f"{s_new:.5f}", f"{100 * (s_new / s_old - 1):+.2f}%")
    row("translation (m)", np.array2string(p_old[:3], precision=4), np.array2string(p_new[:3], precision=4),
        f"{100 * np.linalg.norm(p_new[:3] - p_old[:3]):.2f} cm")
    row("yaw (deg)", f"{math.degrees(p_old[-1]):.2f}", f"{math.degrees(p_new[-1]):.2f}",
        f"{math.degrees(p_new[-1] - p_old[-1]):+.2f}")

    # table
    pf_old, pf_new = _load(opt / "operation_surface_plane_fit.json")["selected"], _load(lay.plane_fit_json)["selected"]
    c_old, c_new = np.array(pf_old["center"]), np.array(pf_new["center"])
    row("table plane centre (m)", np.array2string(c_old, precision=3), np.array2string(c_new, precision=3),
        f"{100 * np.linalg.norm(c_new - c_old):.2f} cm")
    row("table box size (m)", np.array2string(np.array(pf_old["size"]), precision=3),
        np.array2string(np.array(pf_new["size"]), precision=3), "")

    def phys_top(path):
        fp = _load(path)["selected"]["footprints"]["physics"]
        v = np.array(fp["vertices_world"], dtype=float)
        return float(np.nanmax(v[:, 2])), len(v)

    t_old, n_old = phys_top(opt / "operation_surface_shadow_v2.json")
    t_new, n_new = phys_top(lay.receiver_json)
    row("physics table top z (m)", f"{t_old:.4f} ({n_old} verts)", f"{t_new:.4f} ({n_new} verts)", f"{100 * (t_new - t_old):+.2f} cm")

    # objects at load
    d_old, d_new = _load(OLD / "asset_refine" / s / "stock" / "object_poses.json"), _load(lay.object_dump)
    row("z_plane (m)", f"{d_old['z_plane']:.4f}", f"{d_new['z_plane']:.4f}", f"{100 * (d_new['z_plane'] - d_old['z_plane']):+.2f} cm")
    o_old, o_new = _objects_by_key(d_old), _objects_by_key(d_new)
    for k in sorted(set(o_old) & set(o_new)):
        a, b = np.array(o_old[k]["pos"]), np.array(o_new[k]["pos"])
        row(f"load pos {o_old[k].get('object_name', k)}", np.array2string(a, precision=3), np.array2string(b, precision=3),
            f"{100 * np.linalg.norm(b - a):.2f} cm, scale {o_new[k]['scale'] / o_old[k]['scale']:.3f}x")

    # Method B fit
    old_fit = args.old_fit or OLD / "asset_refine" / s / "B_v1" / "candidate.json"
    new_fit = lay.refine_dir("B_v1") / "candidate.json"
    if old_fit.exists() and new_fit.exists():
        f_old, f_new = _objects_by_key(_load(old_fit)), _objects_by_key(_load(new_fit))
        for k in sorted(set(f_old) & set(f_new)):
            a, b = f_old[k], f_new[k]
            row(f"B_v1 {a.get('object_name', k)}",
                f"pos {np.array2string(np.array(a['pos']), precision=3)} x{a.get('scale_multiplier', 1):.2f}",
                f"pos {np.array2string(np.array(b['pos']), precision=3)} x{b.get('scale_multiplier', 1):.2f}",
                f"{100 * np.linalg.norm(np.array(b['pos']) - np.array(a['pos'])):.2f} cm, yaw {(_yaw(b['quat_wxyz']) - _yaw(a['quat_wxyz']) + 180) % 360 - 180:+.1f} deg")

    # settle verdicts
    for tag in ("stock", "B_v1"):
        so, sn = OLD / "asset_refine" / s / tag / "settle.json", lay.refine_dir(tag) / "settle.json"
        if so.exists() and sn.exists():
            vo = {o.get("object_name", o.get("key")): bool(o.get("unstable")) for o in _load(so)["objects"]}
            vn = {o.get("object_name", o.get("key")): bool(o.get("unstable")) for o in _load(sn)["objects"]}
            fmt = lambda v: ", ".join(f"{n}:{'U' if u else 'S'}" for n, u in sorted(v.items()))  # noqa: E731
            row(f"settle {tag} (U=unstable)", fmt(vo), fmt(vn), "same" if vo == vn else "differs")

    width = max(len(r[0]) for r in rows)
    for name, old, new, delta in rows:
        print(f"{name:<{width}}  old {old}\n{'':<{width}}  new {new}   [{delta}]")
    out = lay.root / "e2e_vs_old.json"
    out.write_text(json.dumps([dict(zip(("quantity", "old", "new", "delta"), r)) for r in rows], indent=2) + "\n")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
