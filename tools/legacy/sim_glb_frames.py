"""Which rotation does Eden HEAD apply to a GLB for a given ``up``/``front``? (GPU, ~2 min)

Loads one small GLB four times as fixed entities (single convex hull, so the
collision vertices are a subset of the file's vertices) and finds, for each, the
axis-aligned rotation ``A`` with ``world = R(quat) (s * A v) + pos`` that maps the
file's vertices onto the built geometry:

* ``up`` default (0,0,1)              -- old scale-fit collider; the cropped collider (+ quat Rx-90)
* ``up=(0,1,0), front=(1,0,0)``       -- old evaluator's uncropped collider
* ``up=(0,1,0), front=(-1,0,0)``      -- old scale-fit metric mesh

    python tools/legacy/sim_glb_frames.py --glb DATA/assets/droid2/image_0000_segmented.glb --out DIR
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from r2s.common.io import write_json  # noqa: E402
from r2s.common.mesh import load_combined_mesh  # noqa: E402

GLB_WORLDSPACE_LOAD_QUAT_WXYZ = (0.7071067811865476, -0.7071067811865475, 0.0, 0.0)


def axis_rotations() -> list[np.ndarray]:
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            m = np.zeros((3, 3))
            for i, (p, sg) in enumerate(zip(perm, signs)):
                m[i, p] = sg
            if np.linalg.det(m) > 0:
                out.append(m)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glb", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    import genesis as gs
    import eden as en
    from scipy.spatial import cKDTree

    en.init(gs.gpu, log_root_path=str(args.out / "eden_logs"))
    from eden.envs.base import EnvBase
    from eden.options import EnvOptions, SceneOptions
    from eden.options.entities import EntityOptions

    common = dict(
        file=str(args.glb.resolve()), scale=1.0, is_fixed_base=True, convexify=True, decimate=False,
        decompose_object_error_threshold=float("inf"),
    )
    cases = {
        "up_default": dict(),
        "up_y_front_x": dict(up=(0, 1, 0), front=(1, 0, 0)),
        "up_y_front_negx": dict(up=(0, 1, 0), front=(-1, 0, 0)),
        "up_default_quat_glb_worldspace": dict(default_root_quat=GLB_WORLDSPACE_LOAD_QUAT_WXYZ),
    }
    scene = SceneOptions(**{name: EntityOptions(**common, **kw) for name, kw in cases.items()})
    env = EnvBase(env_options=EnvOptions(num_envs=1, episode_length_s=1.0), scene_options=scene, show_viewer=False, eval_mode=True)
    env.build()

    verts = np.asarray(load_combined_mesh(args.glb).vertices, dtype=np.float64)
    rots = axis_rotations()
    report = {}
    for name in cases:
        geoms = list(env.entities[name]._entity.geoms)
        built = np.concatenate([np.asarray(g.get_verts().detach().cpu().numpy(), dtype=np.float64).reshape(-1, 3) for g in geoms])
        best = None
        for a in rots:
            d, _ = cKDTree(verts @ a.T).query(built)
            score = float(np.median(d))
            if best is None or score < best[0]:
                best = (score, a, float(d.max()))
        report[name] = {"A": best[1].astype(int).tolist(), "median_m": best[0], "max_m": best[2], "n_built_verts": int(len(built))}
        print(f"{name:32s} A={best[1].astype(int).tolist()} median {best[0]:.2e} max {best[2]:.2e} ({len(built)} verts)")
    write_json(args.out / "glb_frames.json", report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
