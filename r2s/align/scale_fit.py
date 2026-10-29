"""Step 3 ``align.scale``: refine the scene scale against the metric-depth mesh.

Port of the old scale-fit script. The splat/collider pose from
step 2 is at scale ``s0`` (1.0); this step builds the collider and the metric mesh
as Genesis entities, samples both surfaces from the built geoms, keeps the points
the camera sees (frustum with edge truncation, per-pixel z-buffer on the metric
side), drops z-outliers, and fits ``s`` by Adam on a trimmed Chamfer distance,
scaling the collider about the camera: ``t(s) = s * t_canonical + (1 - s) * cam_pos``.

Inputs : align/aligned_pose.npy, align/aligned_pose_scene_scale.npy, world/collider.glb,
         world/gaussian.ply (snapshots only), DATA/<type>/<s>/{intrinsics,extrinsics}.npy,
         metric mesh = DATA/<type>/<s>/mesh_sheet.glb, else the highest-index
         DATA/assets/<s>/image_<N>_segmented.glb (loud warning, recorded in the manifest)
Outputs: align/aligned_pose_scene_scale_refined.npy (0-d float64), align/aligned_pose_refined.npy,
         align/scale_init_snapshot.png, align/scale_refined_snapshot.png, [align/scale_mask.ply]

Numerics are unchanged (``r2s.align.scale_core``). Geometry comes from the same place
as before: Genesis entities built with the old ``EntityOptions`` -- collider convexified
(coacd 0.01, decimate to 30000 faces, forced decomposition, aggressiveness 2, preprocess
resolution 150) at ``scale=s0, pos=t(s0), quat=q_rel*ADJ`` with the default ``up``; metric
mesh at scale 1, ``up=(0,1,0), front=(-1,0,0)``, at the camera pose
(``lookat_to_quat``). ``tools/legacy/sim_glb_frames.py`` checks that Eden HEAD imports
both GLBs in the same frames as the old run (Rx(+90) and Ry(180) respectively).

Changes forced by Eden HEAD: the non-RL ``SplatEnvBase`` / ``EdenSplatConfig`` (the old
``SplatRLEnvBase`` + ``EdenRLConfig`` with the splat inside the scene options is gone);
the splat entity is ``PCDOptions(scale=s, pos=t, quat)`` (HEAD places ``R (s x) + pos``;
the old renderer placed ``s (R x + pos)`` and took ``pos = t / s``); no ``sh_degree``.
Other changes: the refined snapshot is always written (the old one only with
``--build_scene``) and the viewer is opt-in (``--viewer``) without the busy loop.
Per-scene truncation lives in ``configs/scenes.yaml`` (``align.scale``).
"""

from __future__ import annotations

import argparse
import math
import time
import warnings
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np

from r2s.align import scale_core as core
from r2s.common.io import prepare_output, save_npy
from r2s.common.transforms import load_pose
from r2s.layout import SceneLayout
from r2s.steps import Step

SNAPSHOT_INIT = "scale_init_snapshot.png"
SNAPSHOT_REFINED = "scale_refined_snapshot.png"
MASK_PLY = "scale_mask.ply"


@dataclass
class ScaleFitSettings:
    """``find_scale_chamfer`` / sampling knobs. Defaults = production (old defaults + the old launcher's truncation)."""

    n_steps: int = 400
    lr: float = 5e-3
    s_min_ratio: float = 0.3
    s_max_ratio: float = 2.0
    trim_quantile: float = 0.8
    n_coll_samples: int = 60000
    n_metric_samples: int = 8192
    occlusion_cell_px: int = 1
    occlusion_depth_tol_rel: float = 0.02
    outlier_k: int = 60
    outlier_std_ratio: float = 0.5
    truncate_w: float = 0.1
    truncate_h: float = 0.3
    truncate_left: float | None = None
    truncate_right: float | None = None
    truncate_top: float | None = None
    truncate_bottom: float | None = None
    match_metric_z: bool = False
    coll_z_pad: float = 0.05
    sdf_cell_size: float = 0.05

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ScaleFitSettings":
        return cls(**{f.name: getattr(args, f.name) for f in fields(cls)})


@dataclass
class ScaleProblem:
    """World samples at ``s0`` plus everything needed to solve for ``s`` (no Genesis needed to solve)."""

    x_coll_0: np.ndarray
    x_metric: np.ndarray
    pivot: np.ndarray
    s_0: float
    t_canonical: np.ndarray
    rx: float
    ry: float
    rz: float
    rotate_xy: bool
    pose_dtype: Any
    calib: dict

    def t_of(self, s: float) -> np.ndarray:
        return s * self.t_canonical + (1.0 - s) * self.pivot


# --------------------------------------------------------------------------- Eden
_EDEN_INITED = False


def _ensure_eden_init(log_dir: Path, use_cpu: bool = False) -> None:
    global _EDEN_INITED
    if _EDEN_INITED:
        return
    import genesis as gs
    import eden as en

    en.init(backend=gs.cpu if use_cpu else gs.gpu, log_root_path=str(log_dir))
    _EDEN_INITED = True


def build_env(
    *,
    s: float,
    rz: float,
    t: np.ndarray,
    collider_glb: str | Path,
    gaussian_ply: str | Path,
    metric_mesh: str | Path,
    intrinsics: str | Path,
    extrinsics: str | Path,
    log_dir: Path,
    rx: float = 0.0,
    ry: float = 0.0,
    sdf_cell_size: float = 0.05,
    use_cpu: bool = False,
):
    """Old ``_build_env`` on Eden HEAD: collider + metric mesh entities, splat, one camera."""
    K, W, H, fov_y, cam_pos, cam_lookat, cam_up = core.load_calibration(str(intrinsics), str(extrinsics))
    print(f"[calib] WxH=({W},{H})  fov_y={fov_y:.2f}  pos={cam_pos}  lookat={cam_lookat}")

    _ensure_eden_init(log_dir, use_cpu=use_cpu)
    from genesis.utils.geom import transform_quat_by_quat
    from eden.envs.splat import SplatEnvBase
    from eden.options.camera import CameraOptions, CamerasOptions
    from eden.options.entities import EntityOptions
    from eden.options.envs import EnvOptions
    from eden.options.materials import RigidMaterialOptions
    from eden.options.splat import PCDOptions, SplatOptions
    from eden.utils.configs import EdenSplatConfig
    from r2s.sim.options import ReconSceneOptions

    # Genesis' own quaternion product, as the old file used (it normalises; see r2s.collision.crop.genesis_quat_mul).
    orig_splat_quat = transform_quat_by_quat(core._BASE_ROT, core._ADJUSTMENT)
    orig_collider_quat = transform_quat_by_quat(np.array((1.0, 0.0, 0.0, 0.0)), core._ADJUSTMENT)
    splat_quat = core.apply_relative_transform_to_quat(orig_splat_quat, rz, rx_rad=rx, ry_rad=ry)
    collision_quat = core.apply_relative_transform_to_quat(orig_collider_quat, rz, rx_rad=rx, ry_rad=ry)

    tx, ty, tz = float(t[0]), float(t[1]), float(t[2])
    # HEAD's gsplat renderer places R (s x) + pos, so the splat takes the world translation itself
    # (the old renderer took pos = t / s).
    splat_pos = (tx, ty, tz)
    collider_pos = (tx, ty, tz)
    print(f"[pose] s={s:.5f}  collider_pos={collider_pos}  splat_pos={splat_pos}")

    splat_opts = SplatOptions(
        background=PCDOptions(path=str(gaussian_ply), scale=s, pos=splat_pos, quat=splat_quat),
    )
    collision_entity = EntityOptions(
        file=str(collider_glb),
        scale=s,
        default_root_pos=collider_pos,
        default_root_quat=collision_quat,
        is_fixed_base=True,
        convexify=True,
        coacd_threshold=0.01,
        decimate_face_num=30000,
        decompose_object_error_threshold=0.0,
        decimate_aggressiveness=2,
        coacd_preprocess_resolution=150,
        vis_mode="collision",
        material=RigidMaterialOptions(sdf_cell_size=sdf_cell_size),
    )
    metric_pos = tuple(float(v) for v in cam_pos)
    metric_quat = core.lookat_to_quat(cam_pos, cam_lookat)
    metric_entity = EntityOptions(
        file=str(metric_mesh),
        scale=1.0,
        up=(0, 1, 0),
        front=(-1, 0, 0),
        default_root_pos=metric_pos,
        default_root_quat=metric_quat,
        is_fixed_base=True,
        convexify=False,
        decimate=False,
        vis_mode="visual",
    )
    cfg = EdenSplatConfig(
        env_options=EnvOptions(num_envs=1, episode_length_s=5.0),
        # ReconSceneOptions adds the default ground plane, as the old scene had.
        scene_options=ReconSceneOptions(background=collision_entity, metric_ref=metric_entity),
        splat_options=splat_opts,
        cameras_options=CamerasOptions(
            camera=CameraOptions(
                cam_pos=tuple(float(v) for v in cam_pos),
                cam_lookat=tuple(float(v) for v in cam_lookat),
                cam_up=tuple(float(v) for v in cam_up),
                cam_fov=float(fov_y),
                cam_resolution=(W, H),
                background_entities=["background"],
            ),
        ),
    )
    env = SplatEnvBase.from_config(cfg, show_viewer=False, eval_mode=True)
    env.build()
    env.reset()
    env.step()
    calib = dict(
        K=K, W=W, H=H, fov_y=fov_y,
        cam_pos=np.asarray(cam_pos, dtype=np.float64),
        cam_lookat=np.asarray(cam_lookat, dtype=np.float64),
        cam_up=np.asarray(cam_up, dtype=np.float64),
    )
    return env, calib


def save_camera_snapshot(env, path: str | Path, label: str = "init") -> None:
    """Render the main camera (splat + entities) and save it as a PNG (old ``_save_camera_snapshot``)."""
    import torch
    from PIL import Image

    rgb = env.cameras["camera"].render_rgb()
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.detach().cpu().numpy()
    rgb = np.asarray(rgb)
    if rgb.ndim == 4:
        rgb = rgb[0]
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0.0, 1.0) if rgb.max() <= 1.0 + 1e-6 else rgb / 255.0
        rgb = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
    path = prepare_output(path)
    Image.fromarray(rgb).save(path)
    print(f"[snapshot] {label} -> {path}  shape={rgb.shape}")


def launch_viewer(env, port: int, s: float) -> None:
    """Optional viser viewer on the refined scene; blocks until Ctrl-C."""
    from eden.extensions.visualization.viser import ViserViewer

    vis = ViserViewer(env, port=port)
    vis.build()
    env.step()
    vis.update()
    print(f"[viser] open  http://localhost:{port}  (showing s={s:.5f}); Ctrl-C to stop")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


# --------------------------------------------------------------------------- the fit
def prepare_problem(
    *,
    pose_path: str | Path,
    scale_path: str | Path,
    collider_glb: str | Path,
    gaussian_ply: str | Path,
    metric_mesh: str | Path,
    intrinsics: str | Path,
    extrinsics: str | Path,
    log_dir: Path,
    settings: ScaleFitSettings,
    init_snapshot: str | Path | None = None,
    use_cpu: bool = False,
):
    """Build the scene at ``s0``, snapshot it, and sample both surfaces. Returns ``(problem, env)``."""
    pose = np.load(pose_path)
    s_0 = float(np.load(scale_path))
    t_canonical, rx0, ry0, rz0 = load_pose(pose_path)
    rotate_xy = pose.reshape(-1).shape[0] == 6
    if rotate_xy:
        print(f"[pose]  canonical(s=1.0)  tx={t_canonical[0]:.4f}  ty={t_canonical[1]:.4f}  tz={t_canonical[2]:.4f}  "
              f"rx={math.degrees(rx0):.2f}°  ry={math.degrees(ry0):.2f}°  rz={math.degrees(rz0):.2f}°  s_0(init)={s_0:.5f}")
    else:
        print(f"[pose]  canonical(s=1.0)  tx={t_canonical[0]:.4f}  ty={t_canonical[1]:.4f}  tz={t_canonical[2]:.4f}  "
              f"rz={math.degrees(rz0):.2f}°  s_0(init)={s_0:.5f}")

    _K, _W, _H, _fov, _cam_pos, _cam_lookat, _cam_up = core.load_calibration(str(intrinsics), str(extrinsics))
    pivot = np.asarray(_cam_pos, dtype=np.float64)
    t_init_world = s_0 * t_canonical + (1.0 - s_0) * pivot
    print(f"[init]  t_init_world(@ s_0)={t_init_world}  pivot(cam)={pivot}")

    print(f"[build] constructing Eden scene at s_0={s_0:.5f} for point extraction ...")
    env, calib = build_env(
        s=s_0, rz=rz0, t=t_init_world,
        collider_glb=collider_glb, gaussian_ply=gaussian_ply, metric_mesh=metric_mesh,
        intrinsics=intrinsics, extrinsics=extrinsics, log_dir=log_dir,
        rx=rx0, ry=ry0, sdf_cell_size=settings.sdf_cell_size, use_cpu=use_cpu,
    )
    if init_snapshot is not None:
        save_camera_snapshot(env, init_snapshot, label=f"init s_0={s_0:.4f}")

    x_coll_0, x_metric_world = core._extract_world_samples(
        env, n_coll=settings.n_coll_samples, n_metric=settings.n_metric_samples,
        outlier_k=settings.outlier_k, outlier_std_ratio=settings.outlier_std_ratio,
    )
    pivot = np.asarray(calib["cam_pos"], dtype=np.float64)
    print(f"[metric] world sample bbox: {x_metric_world.min(0)} -> {x_metric_world.max(0)}")
    print(f"[coll]   world sample bbox: {x_coll_0.min(0)} -> {x_coll_0.max(0)}")
    print(f"[pivot] camera position = {pivot}")
    problem = ScaleProblem(
        x_coll_0=x_coll_0, x_metric=x_metric_world, pivot=pivot, s_0=s_0, t_canonical=t_canonical,
        rx=rx0, ry=ry0, rz=rz0, rotate_xy=rotate_xy, pose_dtype=pose.dtype, calib=calib,
    )
    return problem, env


def solve(problem: ScaleProblem, settings: ScaleFitSettings, mask_debug_ply: str | Path | None = None) -> tuple[float, float]:
    """Old Adam chamfer refinement on already-sampled points: ``(s_refined, last_loss)``."""
    c = problem.calib
    print(f"[opt]   running Adam scale refinement (init s={problem.s_0:.5f}) ...")
    return core.find_scale_chamfer(
        x_coll_0=problem.x_coll_0,
        x_metric=problem.x_metric,
        pivot=problem.pivot,
        s_init=problem.s_0,
        K=c["K"], cam_pos=c["cam_pos"], cam_lookat=c["cam_lookat"], cam_up=c["cam_up"],
        W=c["W"], H=c["H"],
        n_steps=settings.n_steps,
        lr=settings.lr,
        s_min_ratio=settings.s_min_ratio,
        s_max_ratio=settings.s_max_ratio,
        trim_quantile=settings.trim_quantile,
        mask_debug_ply=None if mask_debug_ply is None else str(mask_debug_ply),
        occlusion_cell_px=settings.occlusion_cell_px,
        occlusion_depth_tol_rel=settings.occlusion_depth_tol_rel,
        truncate_w=settings.truncate_w,
        truncate_h=settings.truncate_h,
        truncate_left=settings.truncate_left,
        truncate_right=settings.truncate_right,
        truncate_top=settings.truncate_top,
        truncate_bottom=settings.truncate_bottom,
        match_metric_z=settings.match_metric_z,
        coll_z_pad=settings.coll_z_pad,
    )


def refined_pose_array(problem: ScaleProblem, s_refined: float) -> np.ndarray:
    t = problem.t_of(s_refined)
    if problem.rotate_xy:
        return np.array([t[0], t[1], t[2], problem.rx, problem.ry, problem.rz], dtype=problem.pose_dtype)
    return np.array([t[0], t[1], t[2], problem.rz], dtype=problem.pose_dtype)


def fit_scale(
    *,
    pose_path: Path,
    scale_path: Path,
    collider_glb: Path,
    gaussian_ply: Path,
    metric_mesh: Path,
    intrinsics: Path,
    extrinsics: Path,
    out_scale: Path,
    out_pose: Path,
    snapshot_dir: Path,
    log_dir: Path,
    settings: ScaleFitSettings | None = None,
    viz_mask: Path | None = None,
    refined_snapshot: bool = True,
    viewer: bool = False,
    port: int = 8081,
    use_cpu: bool = False,
) -> dict[str, Any]:
    """The whole old ``main()`` for one scene; returns stats for the manifest."""
    settings = settings or ScaleFitSettings()
    trunc = {
        "left": settings.truncate_w if settings.truncate_left is None else settings.truncate_left,
        "right": settings.truncate_w if settings.truncate_right is None else settings.truncate_right,
        "top": settings.truncate_h if settings.truncate_top is None else settings.truncate_top,
        "bottom": settings.truncate_h if settings.truncate_bottom is None else settings.truncate_bottom,
    }
    print("[vis]   truncate fractions " + " ".join(f"{k}={v:.3f}" for k, v in trunc.items()))
    problem, env = prepare_problem(
        pose_path=pose_path, scale_path=scale_path, collider_glb=collider_glb, gaussian_ply=gaussian_ply,
        metric_mesh=metric_mesh, intrinsics=intrinsics, extrinsics=extrinsics, log_dir=log_dir,
        settings=settings, init_snapshot=snapshot_dir / SNAPSHOT_INIT, use_cpu=use_cpu,
    )
    s_refined, loss = solve(problem, settings, mask_debug_ply=viz_mask)
    t_refined = problem.t_of(s_refined)
    print(f"[opt]   s_refined={s_refined:.6f}  chamfer_loss={loss:.6f}"
          f"  t_refined=({t_refined[0]:.4f}, {t_refined[1]:.4f}, {t_refined[2]:.4f})")
    save_npy(out_scale, np.array(s_refined, dtype=np.float64))
    save_npy(out_pose, refined_pose_array(problem, s_refined))
    print(f"[saved] refined scale -> {out_scale}\n[saved] refined pose  -> {out_pose}")

    if refined_snapshot or viewer:
        print(f"[build] rebuilding scene at s_refined={s_refined:.5f} for the refined snapshot ...")
        env.destroy()
        env, _ = build_env(
            s=s_refined, rz=problem.rz, t=t_refined,
            collider_glb=collider_glb, gaussian_ply=gaussian_ply, metric_mesh=metric_mesh,
            intrinsics=intrinsics, extrinsics=extrinsics, log_dir=log_dir,
            rx=problem.rx, ry=problem.ry, sdf_cell_size=settings.sdf_cell_size, use_cpu=use_cpu,
        )
        save_camera_snapshot(env, snapshot_dir / SNAPSHOT_REFINED, label=f"refined s={s_refined:.4f}")
        if viewer:
            launch_viewer(env, port, s_refined)
    return {
        "s_init": problem.s_0,
        "s_refined": float(s_refined),
        "chamfer_loss": float(loss),
        "t_refined": [float(v) for v in t_refined],
        "n_coll_samples": int(len(problem.x_coll_0)),
        "n_metric_samples": int(len(problem.x_metric)),
        "truncate": trunc,
        "settings": asdict(settings),
    }


# --------------------------------------------------------------------------- step
def resolve_metric_mesh(layout: SceneLayout, args: argparse.Namespace) -> tuple[Path, str]:
    if getattr(args, "metric_mesh", None):
        return Path(args.metric_mesh), "explicit"
    try:
        path, source = core.find_metric_mesh(layout.inputs)
    except FileNotFoundError:
        return layout.inputs.mesh_sheet, "missing"
    return Path(path), source


def add_arguments(parser: argparse.ArgumentParser) -> None:
    d = ScaleFitSettings()
    parser.add_argument("--n-steps", type=int, default=d.n_steps)
    parser.add_argument("--lr", type=float, default=d.lr)
    parser.add_argument("--s-min-ratio", type=float, default=d.s_min_ratio)
    parser.add_argument("--s-max-ratio", type=float, default=d.s_max_ratio)
    parser.add_argument("--trim-quantile", type=float, default=d.trim_quantile)
    parser.add_argument("--n-coll-samples", type=int, default=d.n_coll_samples)
    parser.add_argument("--n-metric-samples", type=int, default=d.n_metric_samples)
    parser.add_argument("--occlusion-cell-px", type=int, default=d.occlusion_cell_px,
                        help="pixel cell of the metric-side z-buffer (0 = frustum only)")
    parser.add_argument("--occlusion-depth-tol-rel", type=float, default=d.occlusion_depth_tol_rel)
    parser.add_argument("--outlier-k", type=int, default=d.outlier_k, help="k of the z-only outlier filter (0 = off)")
    parser.add_argument("--outlier-std-ratio", type=float, default=d.outlier_std_ratio)
    parser.add_argument("--truncate-w", type=float, default=d.truncate_w,
                        help="fraction of the image width dropped at each side (production 0.1, the old launcher's)")
    parser.add_argument("--truncate-h", type=float, default=d.truncate_h,
                        help="fraction of the image height dropped at top and bottom (production 0.3)")
    parser.add_argument("--truncate-left", type=float, default=None)
    parser.add_argument("--truncate-right", type=float, default=None)
    parser.add_argument("--truncate-top", type=float, default=None)
    parser.add_argument("--truncate-bottom", type=float, default=None)
    parser.add_argument("--match-metric-z", action="store_true")
    parser.add_argument("--coll-z-pad", type=float, default=d.coll_z_pad)
    parser.add_argument("--sdf-cell-size", type=float, default=d.sdf_cell_size)
    parser.add_argument("--metric-mesh", type=Path, default=None,
                        help="explicit metric GLB (default: mesh_sheet.glb, else the highest-index segmented GLB)")
    parser.add_argument("--viz-mask", action="store_true", help=f"write align/{MASK_PLY} (visibility-mask debug cloud)")
    parser.add_argument("--viewer", action="store_true", help="open a viser viewer on the refined scene (blocks)")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--cpu", action="store_true")


def run(layout: SceneLayout, args: argparse.Namespace) -> dict[str, Any]:
    metric_mesh, source = resolve_metric_mesh(layout, args)
    notes: list[str] = []
    if source == "segmented_fallback":
        msg = (
            f"[align.scale] WARNING: {layout.scene} has no {layout.inputs.mesh_sheet.name}; "
            f"falling back to the highest-index segmented object GLB {metric_mesh} as the metric mesh. "
            "This is what the old script did, but it is a single object, not the scene's depth."
        )
        print("!" * 100 + "\n" + msg + "\n" + "!" * 100)
        warnings.warn(msg)
        notes.append(msg)
    stats = fit_scale(
        pose_path=layout.aligned_pose,
        scale_path=layout.aligned_scale,
        collider_glb=layout.collider_glb,
        gaussian_ply=layout.gaussian_ply,
        metric_mesh=metric_mesh,
        intrinsics=layout.inputs.intrinsics,
        extrinsics=layout.inputs.extrinsics,
        out_scale=layout.refined_scale,
        out_pose=layout.refined_pose,
        snapshot_dir=layout.align_dir,
        log_dir=layout.align_log_dir / "scale_eden",
        settings=ScaleFitSettings.from_args(args),
        viz_mask=layout.align_dir / MASK_PLY if args.viz_mask else None,
        viewer=args.viewer,
        port=args.port,
        use_cpu=args.cpu,
    )
    stats.update(metric_mesh=str(metric_mesh), metric_mesh_source=source, notes=notes)
    return stats


def _inputs(layout: SceneLayout, args: argparse.Namespace) -> list[Path]:
    return [
        layout.aligned_pose,
        layout.aligned_scale,
        layout.collider_glb,
        layout.gaussian_ply,
        layout.inputs.intrinsics,
        layout.inputs.extrinsics,
        resolve_metric_mesh(layout, args)[0],
    ]


def _outputs(layout: SceneLayout, args: argparse.Namespace) -> list[Path]:
    out = [layout.refined_scale, layout.refined_pose, layout.align_dir / SNAPSHOT_INIT, layout.align_dir / SNAPSHOT_REFINED]
    if args.viz_mask:
        out.append(layout.align_dir / MASK_PLY)
    return out


STEP = Step(
    name="align.scale",
    help="refine the scene scale against the metric-depth mesh (trimmed Chamfer, camera pivot)",
    add_arguments=add_arguments,
    run=run,
    inputs=_inputs,
    outputs=_outputs,
    needs="gpu",
)
