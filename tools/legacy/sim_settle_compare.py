"""Compare a new ``settle.json`` with the old repo's (per object: drift, tilt, verdicts, settled pose).

    python tools/legacy/sim_settle_compare.py --new RUN/settle.json \
        --old OLD/asset_refine/droid820/stock/settle.json [--out cmp.json]

Physics is not bit-identical across Genesis versions (old 1.2.3, HEAD 1.4.4) and the
old scene loaded objects with a different root frame (see docs/porting/sim_assembly.md),
so the table reports deltas and flags every object whose ``unstable`` / ``settled`` /
``topples_after_release`` verdict flips.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from r2s.common.io import write_json  # noqa: E402


def _angle_deg(q1, q2) -> float:
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)
    d = abs(float(np.dot(q1 / np.linalg.norm(q1), q2 / np.linalg.norm(q2))))
    return 2.0 * math.degrees(math.acos(min(1.0, d)))


def compare(new_path: Path, old_path: Path) -> dict:
    new = json.loads(Path(new_path).read_text())
    old = json.loads(Path(old_path).read_text())
    nmap = {o["safe_key"]: o for o in new["objects"]}
    rows = []
    for o in old["objects"]:
        n = nmap.get(o["safe_key"])
        if n is None:
            rows.append({"safe_key": o["safe_key"], "missing_in_new": True})
            continue
        row = {
            "safe_key": o["safe_key"],
            "object_name": o.get("object_name", ""),
            "xy_drift_cm": [100 * o["xy_drift_m"], 100 * n["xy_drift_m"]],
            "z_drop_cm": [100 * o["z_drop_m"], 100 * n["z_drop_m"]],
            "tilt_deg": [o["tilt_deg"], n["tilt_deg"]],
            "yaw_drift_deg": [o["yaw_drift_deg"], n["yaw_drift_deg"]],
            "residual_mm": [None if o.get("residual_motion_m") is None else 1000 * o["residual_motion_m"],
                            None if n.get("residual_motion_m") is None else 1000 * n["residual_motion_m"]],
            "settled": [o.get("settled"), n.get("settled")],
            "unstable": [o.get("unstable"), n.get("unstable")],
            "topples_after_release": [o.get("topples_after_release"), n.get("topples_after_release")],
            "commanded_pos_delta_cm": 100 * float(np.linalg.norm(np.subtract(n["commanded_pos"], o["commanded_pos"]))),
            "settled_pos_delta_cm": 100 * float(np.linalg.norm(np.subtract(n["settled_pos"], o["settled_pos"]))),
            "settled_xy_delta_cm": 100 * float(np.linalg.norm(np.subtract(n["settled_pos"][:2], o["settled_pos"][:2]))),
            "settled_quat_delta_deg": _angle_deg(n["settled_quat_wxyz"], o["settled_quat_wxyz"]),
            "commanded_quat_delta_deg": _angle_deg(n["commanded_quat_wxyz"], o["commanded_quat_wxyz"]),
        }
        row["flips"] = [k for k in ("unstable", "settled", "topples_after_release")
                        if row[k][0] is not None and row[k][1] is not None and bool(row[k][0]) != bool(row[k][1])]
        rows.append(row)
    return {"new": str(new_path), "old": str(old_path), "steps": [old.get("steps"), new.get("steps")], "rows": rows,
            "n_flips": sum(bool(r.get("flips")) for r in rows)}


def format_table(result: dict) -> str:
    lines = [f"old {result['old']}\nnew {result['new']}  steps old/new {result['steps']}",
             f"{'object':22s} {'xy cm old/new':>15s} {'tilt deg old/new':>17s} {'z cm old/new':>15s} "
             f"{'unst o/n':>9s} {'d settled cm':>12s} {'d quat deg':>10s} flips"]
    for r in result["rows"]:
        if r.get("missing_in_new"):
            lines.append(f"{r['safe_key']:22s} MISSING")
            continue
        name = (r["object_name"] or r["safe_key"])[:22]
        lines.append(
            f"{name:22s} {r['xy_drift_cm'][0]:7.2f}/{r['xy_drift_cm'][1]:<7.2f} {r['tilt_deg'][0]:8.1f}/{r['tilt_deg'][1]:<8.1f} "
            f"{r['z_drop_cm'][0]:7.2f}/{r['z_drop_cm'][1]:<7.2f} {str(r['unstable'][0])[0]}/{str(r['unstable'][1])[0]:>7s} "
            f"{r['settled_pos_delta_cm']:12.2f} {r['settled_quat_delta_deg']:10.1f} {','.join(r['flips'])}"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new", required=True, type=Path)
    ap.add_argument("--old", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    result = compare(args.new, args.old)
    print(format_table(result))
    if args.out:
        write_json(args.out, result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
