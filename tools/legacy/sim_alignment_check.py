"""Does the HEAD sim scene put the splat, the collider and the objects where r2s says they are? (GPU)

Builds the preview scene (``r2s.sim.scene.build_scene``) for one scene and checks:

1. splat: the env's own gsplat render of the main camera (Eden's ``BatchSplatRenderer``
   with ``PCDOptions(scale=s, pos=t, quat)``) against ``r2s.align.render``'s gsplat render
   of the same PLY at the same pose from the DROID calibration (``x -> q_rel (s * base x) + t``):
   mean |diff|, PSNR, and the sub-pixel shift between the two (phase correlation);
2. collider: world vertices of the built ``background`` collision geoms against the
   uncropped ``collider.glb`` placed with ``r2s.common.transforms.collider_world_transform``
   (distance to a dense surface sampling), and the baked crop GLB's own vertices likewise;
3. objects: each object's post-reset pose (``get_pos/get_quat(relative=True)``) against
   the dump's load pose.

    R2S_WORK=.../runs/verify_sim python tools/legacy/sim_alignment_check.py --scene droid2 --out DIR
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from r2s.common.io import prepare_output, write_json  # noqa: E402
from r2s.common.mesh import load_combined_mesh  # noqa: E402
from r2s.common.quat import axis_quat_np, euler_zyx_to_quat, quat_mul_np  # noqa: E402
from r2s.common.transforms import (  # noqa: E402
    SCALE_OPT_ADJUSTMENT,
    SCALE_OPT_BASE_ROT,
    collider_world_transform,
    glb_import_rotation,
)
from r2s.layout import SceneLayout  # noqa: E402


def phase_shift(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Sub-pixel (dy, dx) such that ``b`` ~ ``a`` shifted by it (grey images)."""
    fa, fb = np.fft.fft2(a), np.fft.fft2(b)
    r = fa.conj() * fb
    r /= np.maximum(np.abs(r), 1e-12)
    corr = np.fft.ifft2(r).real
    iy, ix = np.unravel_index(np.argmax(corr), corr.shape)
    h, w = corr.shape

    def sub(c_m, c_0, c_p):
        d = c_m - 2 * c_0 + c_p
        return 0.0 if abs(d) < 1e-12 else 0.5 * (c_m - c_p) / d

    dy = iy + sub(corr[(iy - 1) % h, ix], corr[iy, ix], corr[(iy + 1) % h, ix])
    dx = ix + sub(corr[iy, (ix - 1) % w], corr[iy, ix], corr[iy, (ix + 1) % w])
    dy = dy - h if dy > h / 2 else dy
    dx = dx - w if dx > w / 2 else dx
    return float(dy), float(dx)


def stats(d: np.ndarray) -> dict:
    return {k: float(v) for k, v in {
        "median": np.median(d), "p90": np.percentile(d, 90), "p99": np.percentile(d, 99), "max": d.max(), "mean": d.mean(),
    }.items()} | {"n": int(len(d))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--surface-samples", type=int, default=3_000_000)
    args = ap.parse_args()

    import imageio.v3 as iio
    import torch
    import trimesh
    from scipy.spatial import cKDTree

    layout = SceneLayout.create(args.scene)
    from r2s.sim.scene import build_scene, rgb_uint8

    scene = build_scene(layout, dump_path=layout.object_dump, run_dir=args.out / "sim")
    env = scene.env
    report: dict = {"scene": args.scene, "pose": {"t": scene.pose.t.tolist(), "scale": scene.pose.scale,
                                                  "splat_quat": scene.pose.splat_quat.tolist()}}

    # ---- 1. splat ----------------------------------------------------------------
    cam = env.cameras["camera"]
    composite = scene.render("camera")
    iio.imwrite(prepare_output(args.out / "camera_env.png"), composite)
    iio.imwrite(prepare_output(args.out / "wrist_camera_env.png"), scene.render("wrist_camera"))
    cam_T = torch.as_tensor(cam._cam.get_transform(), dtype=torch.float32, device=env.device).reshape(-1, 4, 4)[:1]
    width, height = cam.cam_resolution
    splat_eden, _ = env._splat_renderer.render(cam_T[:, :3, 3], cam_T[:, :3, :3], height, width, cam.cam_fov)
    splat_eden = rgb_uint8(splat_eden.mul(255).byte())

    from r2s.align.render import Camera, SplatModel

    device = torch.device("cuda")
    camera = Camera.from_droid(layout.inputs.intrinsics, layout.inputs.extrinsics, device)
    base = quat_mul_np(SCALE_OPT_ADJUSTMENT, SCALE_OPT_BASE_ROT)
    model = SplatModel(layout.gaussian_trimmed_ply, base, device)
    s = float(scene.pose.scale)
    model.xyz = model.xyz * s
    model.scale = model.scale * s
    p = scene.pose
    q_rel = axis_quat_np(p.rz, 2) if (p.rx == 0.0 and p.ry == 0.0) else euler_zyx_to_quat(p.rx, p.ry, p.rz)
    with torch.no_grad():
        ref = model.render(camera, torch.tensor(q_rel, dtype=torch.float32, device=device),
                           torch.tensor(p.t, dtype=torch.float32, device=device), bg=None)
    splat_r2s = (ref.clamp(0, 1) * 255).byte().cpu().numpy()
    iio.imwrite(prepare_output(args.out / "splat_eden.png"), splat_eden)
    iio.imwrite(prepare_output(args.out / "splat_r2s.png"), splat_r2s)
    diff = np.abs(splat_eden.astype(np.int16) - splat_r2s.astype(np.int16)).astype(np.uint8)
    iio.imwrite(prepare_output(args.out / "splat_absdiff_x4.png"), np.clip(diff.astype(np.int32) * 4, 0, 255).astype(np.uint8))
    mse = float(np.mean((splat_eden.astype(np.float64) - splat_r2s.astype(np.float64)) ** 2))
    dy, dx = phase_shift(splat_eden.mean(-1), splat_r2s.mean(-1))
    report["splat"] = {
        "resolution_eden": [int(width), int(height)], "resolution_r2s": [camera.width, camera.height],
        "fov_eden_deg": float(cam.cam_fov), "fov_r2s_deg": camera.fov_y_deg,
        "mean_abs_diff_u8": float(diff.mean()), "max_abs_diff_u8": int(diff.max()),
        "frac_px_diff_gt_8": float((diff.max(-1) > 8).mean()),
        "psnr_db": float(10 * math.log10(255.0 ** 2 / max(mse, 1e-12))), "shift_px_dy_dx": [dy, dx],
    }
    print("[splat]", report["splat"])

    # ---- 2. collider ---------------------------------------------------------------
    pos, rot, _ = collider_world_transform(layout.refined_pose, s)
    col = load_combined_mesh(layout.collider_glb)
    world = trimesh.Trimesh(vertices=(np.asarray(col.vertices) @ glb_import_rotation().T * s) @ rot.T + pos, faces=col.faces, process=False)
    samples = trimesh.sample.sample_surface(world, args.surface_samples, seed=0)[0]
    tree = cKDTree(np.concatenate([samples, world.vertices]))
    built = np.concatenate([
        np.asarray(g.get_verts().detach().cpu().numpy(), dtype=np.float64).reshape(-1, 3)
        for g in env.entities["background"]._entity.geoms
    ])
    crop = load_combined_mesh(layout.cropped_collider_glb)
    crop_v = np.asarray(crop.vertices, dtype=np.float64)
    np.savez_compressed(args.out / "collider_points.npz", built=built, crop=crop_v)
    spacing = math.sqrt(world.area / args.surface_samples)
    collider = {"n_geoms": len(env.entities["background"]._entity.geoms), "reference_sample_spacing_m": spacing}
    try:  # exact point-to-triangle distance
        import igl

        V = np.ascontiguousarray(world.vertices, dtype=np.float64)
        F = np.ascontiguousarray(world.faces, dtype=np.int64)
        for name, pts in (("built_collision_verts", built), ("crop_glb_verts", crop_v)):
            sq, _, closest = igl.point_mesh_squared_distance(np.ascontiguousarray(pts), V, F)
            collider[f"{name}_to_reference_surface_m"] = stats(np.sqrt(sq))
            collider[f"{name}_mean_offset_m"] = (pts - closest).mean(0).tolist()
        collider["method"] = "igl point_mesh_squared_distance"
    except ImportError:
        collider["built_collision_verts_to_reference_samples_m"] = stats(tree.query(built)[0])
        collider["crop_glb_verts_to_reference_samples_m"] = stats(tree.query(crop_v)[0])
        collider["method"] = "nearest surface sample"
    report["collider"] = collider
    print("[collider]", report["collider"])

    # ---- 3. objects ----------------------------------------------------------------
    objs = {}
    for o in scene.objects:
        e = env.entities[o.safe_key]
        gp = e.get_pos(relative=True).detach().cpu().numpy().reshape(-1, 3)[0]
        gq = e.get_quat(relative=True).detach().cpu().numpy().reshape(-1, 4)[0]
        q = np.asarray(o.quat_wxyz) / np.linalg.norm(o.quat_wxyz)
        ang = 2 * math.degrees(math.acos(min(1.0, abs(float(np.dot(gq / np.linalg.norm(gq), q))))))
        objs[o.safe_key] = {"pos_err_m": float(np.linalg.norm(gp - np.asarray(o.pos))), "quat_err_deg": ang}
    report["objects_after_reset"] = objs
    print("[objects]", objs)
    write_json(args.out / "alignment_report.json", report)
    scene.destroy()
    return 0


if __name__ == "__main__":
    sys.exit(main())
