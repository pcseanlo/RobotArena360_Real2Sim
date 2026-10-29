"""Step 2 ``align.pose``: rigidly place the splat so it matches the real camera view.

Renders the splat from the fixed DROID camera and fits ``x, y, z, yaw`` (optionally
roll/pitch with ``--rotate-xy``) at scene scale 1, against the LaMa-inpainted
background image (objects removed). Loss = ``w_rgb * RGB-L2 + w_dino * (1 - cosine)``
of DINOv2 ViT-S/14 patch tokens. CMA-ES global search, then Adam with gradients
through gsplat.

Inputs : world/gaussian.ply, DATA/scene_background/<s>/background.png,
         DATA/DROID/<s>/{intrinsics,extrinsics}.npy
Outputs: align/aligned_pose.npy            [tx, ty, tz, rz] (or 6 values), scale-1 frame
         align/aligned_pose_scene_scale.npy  1.0 (the scale fit refines it)
         align/aligned_pose_relative_transform.npy  [tx, ty, tz, qw, qx, qy, qz]
         align/pose_fit_render.png, align/pose_fit_overlay.png  (review)
         align/log/<ts>_<optimizer>/p1_{cma,adam}/progress.gif  (with --log)

Ported from the old repo's pose-fit script (docs/porting/old_to_new.md). Changes: the renderer is
gsplat with Eden HEAD's conventions (``r2s.align.render``) instead of Eden's removed
2DGS renderer, so no Genesis scene is built; the Adam "best" snapshot is taken
before the optimizer step (the old one paired a loss with the next step's params);
the default start is the camera pose (``--init camera``: translation = camera position,
yaw = camera heading + 90 deg). Image-to-world generators (Marble) build the world around the input camera, so
every old result sits within ~0.2 m / ~5 deg of it; the old start (yaw 0) needed many
hand-tuned reruns to reach the right basin.
"""

from __future__ import annotations

import argparse
import datetime
import math
from pathlib import Path

import numpy as np

from r2s.common.io import prepare_output, save_npy
from r2s.common.quat import quat_mul_np
from r2s.common.transforms import SCALE_OPT_ADJUSTMENT, SCALE_OPT_BASE_ROT
from r2s.layout import SceneLayout
from r2s.steps import Step


# --------------------------------------------------------------------- DINOv2
class DINOv2Extractor:
    RESIZE_HW = (420, 560)  # must be divisible by patch_size=14

    def __init__(self, device):
        import torch

        self.torch = torch
        self.device = device
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.patch = 14
        self.Hp = self.RESIZE_HW[0] // self.patch
        self.Wp = self.RESIZE_HW[1] // self.patch
        self.mean = torch.tensor([0.5, 0.5, 0.5], device=device, dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor([0.5, 0.5, 0.5], device=device, dtype=torch.float32).view(1, 3, 1, 1)

        was_enabled = torch.backends.cudnn.enabled
        try:
            torch.backends.cudnn.enabled = True
            with torch.no_grad():
                self.model.forward_features(torch.zeros(1, 3, *self.RESIZE_HW, device=device))
        except RuntimeError:
            torch.backends.cudnn.enabled = False
            self._cudnn_disabled = True
            with torch.no_grad():
                self.model.forward_features(torch.zeros(1, 3, *self.RESIZE_HW, device=device))
        else:
            self._cudnn_disabled = False
            torch.backends.cudnn.enabled = was_enabled

    def extract(self, rgb_hw3):
        torch, F = self.torch, self.torch.nn.functional
        x = rgb_hw3.float().permute(2, 0, 1).unsqueeze(0)
        x = F.interpolate(x, size=self.RESIZE_HW, mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        prev = torch.backends.cudnn.enabled
        if self._cudnn_disabled:
            torch.backends.cudnn.enabled = False
        try:
            with torch.amp.autocast("cuda", enabled=False):
                tokens = self.model.forward_features(x)["x_norm_patchtokens"]
        finally:
            torch.backends.cudnn.enabled = prev
        return tokens[0].reshape(self.Hp, self.Wp, -1)


def rgb_l2_loss(rend, tgt):
    return ((rend - tgt) ** 2).mean()


def cosine_loss(a, b):
    import torch.nn.functional as F

    a2 = a.reshape(-1, a.shape[-1])
    b2 = b.reshape(-1, b.shape[-1])
    return 1.0 - F.cosine_similarity(a2, b2, dim=-1).mean()


def relative_transform(tx: float, ty: float, tz: float, rz: float, rx: float = 0.0, ry: float = 0.0) -> np.ndarray:
    from r2s.common.quat import euler_zyx_to_quat

    if rx == 0.0 and ry == 0.0:
        half = rz * 0.5
        q = (math.cos(half), 0.0, 0.0, math.sin(half))
    else:
        q = tuple(euler_zyx_to_quat(rx, ry, rz))
    return np.array([tx, ty, tz, *q], dtype=np.float64)


# ----------------------------------------------------------------- the fit
def fit_pose(
    *,
    gaussian_ply: Path,
    background_png: Path,
    intrinsics: Path,
    extrinsics: Path,
    optimizer: str = "cma_then_adam",
    max_iters: int = 1500,
    lr_trans: float = 0.025,
    lr_rot: float = 0.01,
    cma_sigma0: float = 0.5,
    cma_max_evals: int = 1000,
    rgb_weight_cma: float = 0.1,
    dino_weight_cma: float = 1.0,
    rgb_weight_adam: float = 10.0,
    dino_weight_adam: float = 1.0,
    rotate_xy: bool = False,
    init: str = "camera",
    bg: float | None = 0.5,
    seed: int = 42,
    log_dir: Path | None = None,
    log_every: int = 50,
) -> dict:
    """Return ``{"pose": np.ndarray, "render": HxWx3 uint8, "target": HxWx3 uint8, ...}``."""
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from tqdm import tqdm

    from r2s.align.render import Camera, SplatModel, axis_quat_t, quat_mul_t

    device = torch.device("cuda")
    torch.manual_seed(seed)
    camera = Camera.from_droid(intrinsics, extrinsics, device)
    base_quat = quat_mul_np(SCALE_OPT_ADJUSTMENT, SCALE_OPT_BASE_ROT)  # = transform_quat_by_quat(BASE, ADJ)
    splat = SplatModel(gaussian_ply, base_quat, device)
    W, H = camera.width, camera.height
    print(f"[calib] WxH=({W},{H}) fov_y={camera.fov_y_deg:.2f} gaussians={splat.xyz.shape[0]}")

    cam_pos = torch.tensor(camera.position(), dtype=torch.float32, device=device)
    forward = torch.linalg.inv(camera.viewmat)[:3, 2]  # OpenCV camera +z = viewing direction
    cam_lookat = cam_pos + forward / forward.norm()  # == old load_calibration's lookat
    init_yaw = 0.0
    if init == "camera":
        # An image-to-world splat is generated around the input camera: the splat origin is the
        # camera and, after the base rotation, the view axis sits 90 deg from the world
        # heading. Every old result lies within ~0.2 m / ~5 deg of this start.
        init_trans = cam_pos.detach()
        init_yaw = math.atan2(float(forward[1]), float(forward[0])) + math.pi / 2.0
    elif init == "cam_pos":  # old default: camera position, yaw 0
        init_trans = cam_pos.detach()
    elif init == "bg_centroid":  # old --start_from_bg_centroid
        init_trans = (cam_lookat - splat.centroid).detach()
    else:
        raise ValueError(f"unknown init {init!r}")
    print(f"[init] {init}: init_trans={init_trans.cpu().numpy()} init_yaw={math.degrees(init_yaw):.1f} deg")

    real_np = np.array(Image.open(background_png).convert("RGB"))
    real_rgb = torch.from_numpy(real_np).to(device).float() / 255.0
    real_rgb = F.interpolate(real_rgb.permute(2, 0, 1)[None], size=(H, W), mode="bilinear", align_corners=False)[0]
    real_rgb = real_rgb.permute(1, 2, 0).contiguous()

    print("[dino] loading dinov2_vits14 ...")
    dino = DINOv2Extractor(device)
    with torch.no_grad():
        real_dino = dino.extract(real_rgb)

    def pose_quat(rz, rx=None, ry=None):
        if rx is None:
            return axis_quat_t(rz, 2)
        return quat_mul_t(axis_quat_t(rz, 2), quat_mul_t(axis_quat_t(ry, 1), axis_quat_t(rx, 0)))

    def scalar(v):
        return torch.tensor([float(v)], device=device)

    def evaluate(x: np.ndarray, rgb_w: float, dino_w: float):
        with torch.no_grad():
            if len(x) == 4:
                q = pose_quat(scalar(x[3]))
            else:
                q = pose_quat(scalar(x[5]), scalar(x[3]), scalar(x[4]))
            rgb = splat.render(camera, q, torch.tensor(x[:3], dtype=torch.float32, device=device), bg=bg)
            loss = rgb_w * rgb_l2_loss(rgb, real_rgb) + dino_w * cosine_loss(dino.extract(rgb), real_dino)
            return float(loss), rgb

    def to_u8(rgb):
        return (rgb.detach().clamp(0, 1) * 255).byte().cpu().numpy()

    def save_gif(frames, sub):
        if log_dir and frames:
            path = prepare_output(log_dir / sub / "progress.gif")
            frames[0].save(path, save_all=True, append_images=frames[1:], loop=0, duration=200)

    stats: dict = {}

    # ---------------------------------------------------------------- CMA-ES
    def run_cma(x0: np.ndarray, S: np.ndarray) -> np.ndarray:
        import cma

        dim = x0.shape[0]
        best = {"loss": float("inf"), "x": x0.copy(), "rgb": None, "evals": 0, "frames": []}

        def objective(x_norm):
            x = x_norm * S + x0
            loss, rgb = evaluate(x, rgb_weight_cma, dino_weight_cma)
            best["evals"] += 1
            if loss < best["loss"]:
                best.update(loss=loss, x=x.copy(), rgb=rgb)
            if log_dir and best["evals"] % log_every == 0:
                best["frames"].append(Image.fromarray(to_u8(best["rgb"])).convert("P", palette=Image.ADAPTIVE))
            return loss

        opts = cma.CMAOptions()
        opts["maxfevals"] = cma_max_evals
        opts["tolx"] = 1e-4
        opts["tolfun"] = 1e-5
        opts["CMA_stds"] = [0.5, 0.5, 0.5, 1.0] if dim == 4 else [0.5, 0.5, 0.5, 0.5, 0.5, 1.0]
        opts["verbose"] = -9
        opts["seed"] = seed
        print(f"[cma] sigma0={cma_sigma0} max_evals={cma_max_evals} dim={dim}")
        cma.CMAEvolutionStrategy(np.zeros(dim), cma_sigma0, opts).optimize(objective)
        save_gif(best["frames"], "p1_cma")
        stats["cma"] = {"best_loss": best["loss"], "evals": best["evals"]}
        print(f"[cma done] best_loss={best['loss']:.5f} x={best['x']}")
        return best["x"]

    # ------------------------------------------------------------------ Adam
    def run_adam(x_init: np.ndarray) -> np.ndarray:
        trans = torch.tensor(x_init[:3], dtype=torch.float32, device=device, requires_grad=True)
        rz = torch.tensor([x_init[-1]], dtype=torch.float32, device=device, requires_grad=True)
        rxy = None
        if rotate_xy:
            rxy = torch.tensor(x_init[3:5], dtype=torch.float32, device=device, requires_grad=True)
        groups = [{"params": [trans], "lr": lr_trans}, {"params": [rz], "lr": lr_rot}]
        if rxy is not None:
            groups.append({"params": [rxy], "lr": lr_rot})
        opt = torch.optim.Adam(groups)
        warmup = max(1, max_iters // 10)
        cosine = max(1, max_iters - warmup)
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, end_factor=1.0, total_iters=warmup),
                torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cosine, eta_min=min(lr_trans, lr_rot) * 0.001),
            ],
            milestones=[warmup],
        )
        best_loss, best_x, frames = float("inf"), x_init.copy(), []
        for step in tqdm(range(max_iters), desc="adam"):
            q = pose_quat(rz) if rxy is None else pose_quat(rz, rxy[0:1], rxy[1:2])
            rgb = splat.render(camera, q, trans, bg=bg)
            loss = rgb_weight_adam * rgb_l2_loss(rgb, real_rgb) + dino_weight_adam * cosine_loss(dino.extract(rgb), real_dino)
            loss_val = float(loss.detach())
            if loss_val < best_loss:  # snapshot the params that produced this loss
                best_loss = loss_val
                t = trans.detach().cpu().numpy().astype(np.float64)
                best_x = np.concatenate([t, rxy.detach().cpu().numpy().astype(np.float64)] if rxy is not None else [t])
                best_x = np.append(best_x, float(rz.detach()))
            opt.zero_grad()
            loss.backward()
            if step == 0 and (trans.grad is None or rz.grad is None):
                raise RuntimeError("pose parameters received no gradient through the renderer")
            torch.nn.utils.clip_grad_norm_([trans, rz] + ([rxy] if rxy is not None else []), max_norm=0.1)
            opt.step()
            sched.step()
            if log_dir and (step % log_every == 0 or step == max_iters - 1):
                frames.append(Image.fromarray(to_u8(rgb)).convert("P", palette=Image.ADAPTIVE))
        save_gif(frames, "p1_adam")
        stats["adam"] = {"best_loss": best_loss, "iters": max_iters}
        print(f"[adam done] best_loss={best_loss:.5f} x={best_x}")
        return best_x

    t0 = init_trans.cpu().numpy().astype(np.float64)
    if rotate_xy:
        S = np.array([2.0, 2.0, 2.0, math.radians(30.0), math.radians(30.0), math.radians(45.0)])
        x0 = np.concatenate([t0, [0.0, 0.0, init_yaw]])
    else:
        S = np.array([2.0, 2.0, 2.0, math.radians(45.0)])
        x0 = np.concatenate([t0, [init_yaw]])

    if optimizer in ("cma", "cma_then_adam"):
        best = run_cma(x0, S)
        if optimizer == "cma_then_adam":
            best = run_adam(best)
    elif optimizer == "adam":
        sweep = np.linspace(-math.pi, math.pi, 37)[:-1]
        losses = [evaluate(np.concatenate([t0, [a]]) if not rotate_xy else np.concatenate([t0, [0.0, 0.0, a]]),
                           rgb_weight_adam, dino_weight_adam)[0] for a in sweep]
        start = x0.copy()
        start[-1] = float(sweep[int(np.argmin(losses))])
        stats["sweep_best_deg"] = math.degrees(start[-1])
        best = run_adam(start)
    else:
        raise ValueError(f"unknown optimizer {optimizer!r}")

    final_loss, final_rgb = evaluate(best, rgb_weight_adam, dino_weight_adam)
    stats["final_loss_adam_weights"] = final_loss
    return {
        "pose": np.asarray(best, dtype=np.float64),
        "render": to_u8(final_rgb),
        "target": to_u8(real_rgb),
        "stats": stats,
    }


# --------------------------------------------------------------------- step
def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--optimizer", default="cma_then_adam", choices=["cma_then_adam", "adam", "cma"])
    parser.add_argument("--max-iters", dest="max_iters", type=int, default=1500)
    parser.add_argument("--lr-trans", dest="lr_trans", type=float, default=0.025)
    parser.add_argument("--lr-rot", dest="lr_rot", type=float, default=0.01)
    parser.add_argument("--cma-sigma0", dest="cma_sigma0", type=float, default=0.5)
    parser.add_argument("--cma-max-evals", dest="cma_max_evals", type=int, default=1000)
    parser.add_argument("--rgb-weight-cma", dest="rgb_weight_cma", type=float, default=0.1)
    parser.add_argument("--dino-weight-cma", dest="dino_weight_cma", type=float, default=1.0)
    parser.add_argument("--rgb-weight-adam", dest="rgb_weight_adam", type=float, default=10.0)
    parser.add_argument("--dino-weight-adam", dest="dino_weight_adam", type=float, default=1.0)
    parser.add_argument("--rotate-xy", dest="rotate_xy", action="store_true", help="also fit roll and pitch")
    parser.add_argument("--init", default="camera", choices=["camera", "cam_pos", "bg_centroid"],
                        help="start pose: camera = camera position + heading-derived yaw (default); "
                             "cam_pos = old default (yaw 0); bg_centroid = old --start_from_bg_centroid")
    parser.add_argument("--bg", type=float, default=0.5, help="render background grey level (old renderer: 0.5)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log", action="store_true", help="write progress GIFs under align/log/")
    parser.add_argument("--log-every", dest="log_every", type=int, default=50)


def run(layout: SceneLayout, args: argparse.Namespace) -> dict:
    from PIL import Image

    log_dir = None
    if args.log:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = layout.align_log_dir / f"{ts}_{args.optimizer}_lrt{args.lr_trans}_lrr{args.lr_rot}"
    result = fit_pose(
        gaussian_ply=layout.gaussian_ply,
        background_png=layout.inputs.background_png,
        intrinsics=layout.inputs.intrinsics,
        extrinsics=layout.inputs.extrinsics,
        optimizer=args.optimizer,
        max_iters=args.max_iters,
        lr_trans=args.lr_trans,
        lr_rot=args.lr_rot,
        cma_sigma0=args.cma_sigma0,
        cma_max_evals=args.cma_max_evals,
        rgb_weight_cma=args.rgb_weight_cma,
        dino_weight_cma=args.dino_weight_cma,
        rgb_weight_adam=args.rgb_weight_adam,
        dino_weight_adam=args.dino_weight_adam,
        rotate_xy=args.rotate_xy,
        init=args.init,
        bg=args.bg,
        seed=args.seed,
        log_dir=log_dir,
        log_every=args.log_every,
    )
    pose = result["pose"]
    save_npy(layout.aligned_pose, pose)
    save_npy(layout.aligned_scale, np.array(1.0, dtype=np.float64))
    if pose.shape[0] == 6:
        rel = relative_transform(*pose[:3], pose[5], rx=pose[3], ry=pose[4])
    else:
        rel = relative_transform(*pose[:3], pose[3])
    save_npy(layout.relative_transform, rel)
    Image.fromarray(result["render"]).save(prepare_output(layout.align_dir / "pose_fit_render.png"))
    overlay = (0.5 * result["render"].astype(np.float32) + 0.5 * result["target"].astype(np.float32)).astype(np.uint8)
    Image.fromarray(overlay).save(prepare_output(layout.align_dir / "pose_fit_overlay.png"))
    return {"pose": pose.tolist(), "yaw_deg": math.degrees(float(pose[-1])), **result["stats"]}


STEP = Step(
    name="align.pose",
    help="Fit the splat's rigid pose to the inpainted DROID background (CMA-ES + Adam through gsplat)",
    add_arguments=add_arguments,
    run=run,
    inputs=lambda layout, args: [layout.gaussian_ply, layout.inputs.background_png,
                                 layout.inputs.intrinsics, layout.inputs.extrinsics],
    outputs=lambda layout, args: [layout.aligned_pose, layout.aligned_scale, layout.relative_transform],
    needs="gpu",
)
