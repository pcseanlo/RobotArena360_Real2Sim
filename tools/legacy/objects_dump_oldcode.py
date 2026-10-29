"""Old-code baseline for ``objects.dump``: run the OLD evaluator's own object loading on CPU.

The functions are taken from ``world_lab_evaluate_with_scene.py`` itself (AST, with
their transitive module-level dependencies) and executed with the old repo's local
modules (``init_collision_guard``, ``asset_init_randomization``, ...); the Eden option
classes are stubbed (the dump never touches them). Run with ``PYTHONPATH=<old repo>``
and a cwd outside it:

    python tools/legacy/objects_dump_oldcode.py --scene droid2 --out DIR/object_poses.json \
        [--scale-override cand.json] [--rotation legacy|identity] [--scene-data-dir DATA]

Production call (evaluator 6476 / 6571-6720): z_plane from the plane fit + shadow_v2,
crop = the old-code crop (``--crop-cache-dir``, must be a cache hit), guard query
with the crop's load pose, ``load_scene_objects(..., object_collision_profile="detailed",
splat_backend="nyx")``, then ``write_object_pose_dump``.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import sys
from dataclasses import dataclass, replace

import numpy as np
from scipy.spatial.transform import Rotation as ScipyRotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from old_repo import ECHO2, OLD as _OLD  # noqa: E402  (also puts the repo on sys.path)
from r2s.common.transforms import SCALE_OPT_ADJUSTMENT  # noqa: E402  (the evaluator's ADJUSTMENT)

OLD = str(_OLD)
LOCAL_MODULES = {
    "init_collision_guard",
    "asset_init_randomization",
    "collision_mesh_crop",
    "object_collision_profiles",
    "object_pose_override",
    "object_scale_override",
    "object_asset_override",
    "object_collision_override",
}
ROOTS = ["load_scene_objects", "object_support_plane_z", "load_scene_object_metadata"]


class _Stub:
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs


def evaluator_namespace() -> dict:
    src = open(os.path.join(OLD, "world_lab_evaluate_with_scene.py")).read()
    tree = ast.parse(src)
    import types

    module = types.ModuleType("old_evaluator")  # dataclasses need a registered module
    sys.modules["old_evaluator"] = module
    ns: dict = module.__dict__
    ns.update({
        "np": np, "math": math, "os": os, "json": json, "re": __import__("re"),
        "dataclass": dataclass, "replace": replace, "ScipyRotation": ScipyRotation,
        "EntityOptions": _Stub, "BSDFSurfaceOptions": _Stub, "GlassSurfaceOptions": _Stub, "EmissionSurfaceOptions": _Stub,
    })
    exec("from __future__ import annotations", ns)
    for node in tree.body:  # the evaluator's own imports from the old repo's light modules
        if isinstance(node, ast.ImportFrom) and node.module in LOCAL_MODULES:
            exec(compile(ast.Module(body=[node], type_ignores=[]), "<evaluator-imports>", "exec"), ns)
    defs: dict[str, ast.AST] = {}
    order: list[ast.AST] = []
    for node in tree.body:
        names = []
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names = [node.name]
        elif isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = [node.target.id]
        for n in names:
            defs[n] = node
        if names:
            order.append(node)
    needed: set[str] = set()
    stack = list(ROOTS)
    while stack:
        name = stack.pop()
        if name in needed or name not in defs:
            continue
        needed.add(name)
        for sub in ast.walk(defs[name]):
            if isinstance(sub, ast.Name) and sub.id in defs and sub.id not in ns:
                stack.append(sub.id)
    nodes = [n for n in order if any(defs.get(k) is n for k in needed)]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<evaluator>", "exec"), ns)
    ns["_extracted"] = sorted(needed)
    return ns


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scene-data-dir", default=None,
                    help="scene_generation data (default: paths.scene_data); the old evaluator's own spelling "
                         "of it reproduces the old dumps' path strings")
    ap.add_argument("--opt-root", default=os.path.join(OLD, "optimization_results"))
    ap.add_argument("--echo2", default=str(ECHO2))
    ap.add_argument("--crop-cache-dir", required=True, help="folder with the old-code crop + sidecar (objects_oldcode_baseline.py --stages crop); must be a cache hit")
    ap.add_argument("--scale-override", default=None)
    ap.add_argument("--rotation", choices=["legacy", "identity"], default="legacy")
    args = ap.parse_args()

    ns = evaluator_namespace()
    print(f"[oldcode] extracted {ns['_extracted']}")
    s = args.scene
    if args.scene_data_dir is None:
        from r2s.config import Paths

        args.scene_data_dir = str(Paths.load().scene_data)
    opt = os.path.join(args.opt_root, s)
    scene_meta_dir = os.path.join(args.scene_data_dir, "DROID", s)
    scene_asset_dir = os.path.join(args.scene_data_dir, "assets", s)

    z_plane, reference_top, physics_top = ns["object_support_plane_z"](
        os.path.join(opt, "operation_surface_plane_fit.json"),
        os.path.join(opt, "operation_surface_shadow_v2.json"),
        reference_offset=0.01,
        fallback=0.02,
    )
    print(f"[oldcode] z_plane={z_plane!r} reference_top={reference_top!r} physics_top={physics_top!r}")

    # crop exactly as the evaluator (6488-6520); a matching sidecar makes this a read-only cache hit
    from genesis.utils.geom import transform_quat_by_quat

    result = np.load(os.path.join(opt, "aligned_pose_refined.npy"))
    tx, ty, tz, rz = (float(v) for v in result[:4])
    assert result.shape[0] == 4
    SCENE_SCALE = float(np.load(os.path.join(opt, "aligned_pose_scene_scale_refined.npy")))
    ADJUSTMENT = np.array(SCALE_OPT_ADJUSTMENT)
    orig_collider_quat = transform_quat_by_quat(np.array([1.0, 0.0, 0.0, 0.0]), ADJUSTMENT)
    qz = np.array([math.cos(rz * 0.5), 0.0, 0.0, math.sin(rz * 0.5)])
    w1, x1, y1, z1 = qz
    w2, x2, y2, z2 = orig_collider_quat
    q = tuple(float(v) for v in np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]))
    cache_dir = os.path.abspath(args.crop_cache_dir)
    if os.path.commonpath([cache_dir, os.path.abspath(OLD)]) == os.path.abspath(OLD):
        raise SystemExit("refusing a crop cache inside the old repo (a cache miss would write there)")
    if "NUMBA_CACHE_DIR" not in os.environ:
        raise SystemExit("set NUMBA_CACHE_DIR (Genesis' numba cache would otherwise write into the Eden venv)")
    crop = ns["build_reachable_collision_mesh"](
        os.path.join(args.echo2, s, "collider.glb"), scene_name=s, scene_scale=SCENE_SCALE,
        root_pos=(tx, ty, tz), root_quat_wxyz=q, source_mesh_up=(0, 1, 0), source_mesh_front=(1, 0, 0),
        cache_dir=cache_dir, radius=1.10, z_min=-0.15, z_max=1.45, sides=24,
    )
    if crop.reason != "cache":
        raise SystemExit(f"crop was not a cache hit ({crop.reason}); refusing to rebuild into {cache_dir}")
    quat = crop.load_quat_wxyz if args.rotation == "legacy" else (1.0, 0.0, 0.0, 0.0)
    query = ns["BackgroundSupportQuery"](
        crop.mesh_path, scene_scale=crop.load_scale, root_pos=crop.load_pos, root_quat_wxyz=quat,
        footprint_margin=0.015, surface_normal_z=0.75, mesh_up=crop.load_up, mesh_front=crop.load_front,
    )
    scale_multipliers = ns["load_object_scale_multipliers"](args.scale_override) if args.scale_override else None
    metadata = ns["load_scene_object_metadata"](scene_meta_dir)
    specs: dict = {}
    ns["load_scene_objects"](
        data_dir=scene_meta_dir, asset_dir=scene_asset_dir, z_plane=z_plane,
        init_collision_enabled=True, init_collision_clearance=0.005, object_support_clearance=0.0005,
        init_collision_query=query, init_collision_fallback_support_z=z_plane,
        init_collision_background_support_max_z=z_plane + 0.005, init_collision_bottom_slab=0.02,
        nyx_texture_cache_dir=None, object_contact_profile="default", object_collision_profile="detailed",
        object_collision_decimate=None, object_decompose_error_threshold=0.05, object_coacd_threshold=0.03,
        object_collision_legacy_face_num=1000, object_mass_floor=0.03, collision_selections_out={},
        object_metadata_by_count=metadata, object_material_mode="auto", splat_backend="nyx",
        material_resolutions_out={}, placement_specs_out=specs, scale_multipliers=scale_multipliers,
        collision_overrides=None, asset_overrides=None,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    ns["write_object_pose_dump"](args.out, specs, scene_name=s, scene_meta_dir=scene_meta_dir, z_plane=z_plane,
                                 physics_top=physics_top, reference_top=reference_top)
    print(f"[oldcode] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
