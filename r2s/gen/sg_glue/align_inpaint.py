"""Inpainted-background metric depth (Depth Anything V2 Metric Indoor) for ``inputs.table_depth``.

Vendored from scene_generation ``src/pipeline/align_inpaint_updated.py`` (env
``moge``). Bodies unchanged; ``segment_and_unproject`` was rewritten around the
same operations (see below) and the Gemini table-mask path (dead: production
used the whole-image mask) and its API key are gone.

``segment_and_unproject`` differences:
  - the background index is passed in (the step uses the number of object
    entries in ``masks/result.json``) instead of ``max(existing image_*_mask.png) + 1``,
    so re-running the step overwrites the same files instead of adding
    ``image_0006``, ``image_0007``, ...;
  - output paths are explicit.
"""
import os

import cv2
import numpy as np
import open3d as o3d
import torch
from PIL import Image
from sklearn.linear_model import LinearRegression, RANSACRegressor
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"


_DEPTH_CACHE = {"model": None, "processor": None, "device": None}


def _load_depth_model(device="cuda"):
    if _DEPTH_CACHE["model"] is None or _DEPTH_CACHE["device"] != device:
        processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_ID)
        model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_ID).to(device).eval()
        _DEPTH_CACHE["model"] = model
        _DEPTH_CACHE["processor"] = processor
        _DEPTH_CACHE["device"] = device
    return _DEPTH_CACHE["model"], _DEPTH_CACHE["processor"]


def _run_depth_inference(rgb_image, model, processor, device="cuda"):
    """Run Depth Anything V2 Metric on an RGB image.

    Returns (depth, valid_mask) shaped like the input H x W. Mirrors the
    interface of the original `_run_moge_inference` so callers don't change.
    """
    H, W = rgb_image.shape[:2]
    pil_img = Image.fromarray(rgb_image)
    inputs = processor(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    pred = outputs.predicted_depth  # (1, h, w)
    pred = torch.nn.functional.interpolate(
        pred.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False
    ).squeeze(1)
    depth = pred[0].detach().cpu().numpy().astype(np.float32)
    valid_mask = np.isfinite(depth) & (depth > 0)
    return depth, valid_mask


def _grid_subsample_indices(mask_2d, stride=4):
    """Return a flat boolean array selecting True pixels of `mask_2d` on a stride grid."""
    H, W = mask_2d.shape
    grid = np.zeros_like(mask_2d, dtype=bool)
    grid[::stride, ::stride] = True
    return mask_2d & grid


def _ransac_affine_fit(x, y, residual_threshold):
    """Fit y = a*x + b robustly. Returns (a, b, inlier_mask)."""
    try:
        ransac = RANSACRegressor(
            estimator=LinearRegression(),
            min_samples=max(2, int(0.5 * len(x))),
            residual_threshold=residual_threshold,
            max_trials=200,
            random_state=0,
        )
        ransac.fit(x.reshape(-1, 1), y)
        a = float(ransac.estimator_.coef_[0])
        b = float(ransac.estimator_.intercept_)
        inliers = ransac.inlier_mask_
        return a, b, inliers
    except Exception:
        # Fallback: ordinary least squares.
        reg = LinearRegression().fit(x.reshape(-1, 1), y)
        a = float(reg.coef_[0])
        b = float(reg.intercept_)
        return a, b, np.ones_like(y, dtype=bool)


def _trimmed_ls_scale(x, y, trim_frac=0.2, n_iters=3):
    """Fit y = s * x with iterative trimmed least squares."""
    keep = np.ones_like(x, dtype=bool)
    s = float(np.median(y / np.clip(x, 1e-8, None)))
    for _ in range(n_iters):
        xk, yk = x[keep], y[keep]
        if xk.size < 10:
            break
        s = float(np.sum(xk * yk) / max(np.sum(xk * xk), 1e-12))
        residuals = np.abs(y - s * x)
        lo = np.quantile(residuals, trim_frac)
        hi = np.quantile(residuals, 1.0 - trim_frac)
        keep = (residuals >= lo) & (residuals <= hi)
    return s


def transfer_metric_depth_two_stage_regression(
    original_metric_depth_path,
    inpainted_image_path,
    image_path,  # ORIGINAL, non-inpainted RGB
    inpaint_mask,
    device="cuda",
):
    # --- 0. Load and prepare ---
    orig_depth_gt = np.load(original_metric_depth_path).astype(np.float32)
    inpaint_rgb = np.array(Image.open(inpainted_image_path).convert("RGB"))
    orig_rgb = np.array(Image.open(image_path).convert("RGB"))
    mask = np.array(inpaint_mask).astype(np.uint8)
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = (mask > 0).astype(np.uint8)

    H, W = inpaint_rgb.shape[:2]
    orig_depth_gt = cv2.resize(orig_depth_gt, (W, H), interpolation=cv2.INTER_NEAREST)
    orig_rgb = cv2.resize(orig_rgb, (W, H), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

    # Dilate the inpaint mask so ~mask (= the "background" we sample) is eroded
    # away from the inpaint boundary. Avoids halo pixels biasing the fit.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    bg_excluded = cv2.dilate(mask, kernel, iterations=1).astype(bool)

    # --- 1. Run depth on both RGBs ---
    model, processor = _load_depth_model(device=device)
    depth_inpaint, valid_inpaint = _run_depth_inference(inpaint_rgb, model, processor, device)
    depth_orig, valid_orig = _run_depth_inference(orig_rgb, model, processor, device)

    # --- 2. Stage 1: affine fit  depth_orig ≈ a * depth_inpaint + b on background ---
    bg = (
        (~bg_excluded)
        & valid_inpaint
        & valid_orig
        & np.isfinite(depth_inpaint)
        & np.isfinite(depth_orig)
    )
    bg_grid = _grid_subsample_indices(bg, stride=4)
    if bg_grid.sum() < 100:
        raise RuntimeError("Too few valid background pixels for stage-1 alignment.")

    X = depth_inpaint[bg_grid]
    Y = depth_orig[bg_grid]
    res_thresh = 0.05 * float(np.median(np.abs(Y)) + 1e-6)
    a, b, inliers_s1 = _ransac_affine_fit(X, Y, residual_threshold=res_thresh)

    aligned_depth_inpaint = depth_inpaint * a + b

    # --- 3. Stage 2: scale  orig_depth_gt ≈ s * aligned_depth_inpaint on background ---
    seg2 = (~bg_excluded) & np.isfinite(aligned_depth_inpaint) & np.isfinite(orig_depth_gt)
    seg2_grid = _grid_subsample_indices(seg2, stride=4)
    if seg2_grid.sum() < 100:
        raise RuntimeError("Too few valid background pixels for stage-2 alignment.")

    x2 = aligned_depth_inpaint[seg2_grid]
    y2 = orig_depth_gt[seg2_grid]
    # Drop divide-by-zero / non-positive predictors before trimmed LS.
    keep = (x2 > 1e-6) & np.isfinite(x2) & np.isfinite(y2)
    if keep.sum() < 10:
        scale2 = float(np.median(y2) / max(np.median(x2), 1e-6))
    else:
        scale2 = _trimmed_ls_scale(x2[keep], y2[keep], trim_frac=0.2, n_iters=3)

    metric_depth = aligned_depth_inpaint * scale2
    total_scale = a * scale2
    total_shift = b * scale2
    return metric_depth, float(total_scale), float(total_shift)


def combine_masks(mask_dir):
    """Union of all `*_mask.png` files in `mask_dir` as a uint8 image."""
    mask_files = sorted(f for f in os.listdir(mask_dir) if f.endswith("_mask.png"))
    if not mask_files:
        raise ValueError(f"No *_mask.png files found in {mask_dir}!")

    first_img = Image.open(os.path.join(mask_dir, mask_files[0])).convert("L")
    width, height = first_img.size
    combined_mask = np.zeros((height, width), dtype=np.uint8)
    for mask_file in mask_files:
        img = Image.open(os.path.join(mask_dir, mask_file)).convert("L")
        combined_mask = np.maximum(combined_mask, np.array(img))
    return combined_mask


def load_o3d_intrinsic(intrinsics_path: str, width: int, height: int) -> o3d.camera.PinholeCameraIntrinsic:
    K = np.load(intrinsics_path)
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(width, height, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    return intrinsic


def load_extrinsic(extrinsics_path: str) -> np.ndarray:
    return np.load(extrinsics_path)


def unproject_mask_to_pcd(rgb, depth, mask, intrinsic_o3d, extrinsic):
    mask = mask.astype(bool)
    masked_rgb = np.zeros_like(rgb, dtype=np.uint8)
    masked_depth = np.zeros_like(depth, dtype=np.float32)
    masked_rgb[mask] = rgb[mask]
    masked_depth[mask] = depth[mask]
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(masked_rgb),
        o3d.geometry.Image(masked_depth),
        depth_scale=1.0,
        convert_rgb_to_intensity=False,
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic_o3d, extrinsic)
    return pcd


def return_whole_mask(image_path: str, output_dir: str) -> str:
    im = Image.open(image_path).convert("RGB")
    mask = np.ones((im.height, im.width), dtype=np.uint8) * 255
    mask_path = os.path.join(output_dir, "whole_mask.png")
    Image.fromarray(mask).save(mask_path)
    return mask_path


def segment_and_unproject(scene_dir, rgb_path, metric_depth, index, masks_dir, pcds_dir):
    """Write ``whole_mask.png``, ``masks/image_<index>_mask.png`` (the masked
    background RGB) and ``pcds/<index>_frame1.ply`` (its unprojection)."""
    intrinsics_path = os.path.join(scene_dir, "intrinsics.npy")
    extrinsics_path = os.path.join(scene_dir, "extrinsics.npy")
    rgb = np.array(Image.open(rgb_path).convert("RGB"))
    depth = metric_depth.astype(np.float32)
    H, W = rgb.shape[:2]
    intrinsic_o3d = load_o3d_intrinsic(intrinsics_path, W, H)
    extrinsic = load_extrinsic(extrinsics_path)
    mask_paths = [return_whole_mask(rgb_path, scene_dir)]
    os.makedirs(pcds_dir, exist_ok=True)
    written = []
    for mask_path in mask_paths:
        mask = np.array(Image.open(mask_path))
        masked_rgb = rgb.copy()
        masked_rgb[mask == 0] = [0, 0, 0]
        filename = f"image_{index:04d}_mask.png"
        masked_img_path = os.path.join(masks_dir, filename)
        Image.fromarray(masked_rgb).save(masked_img_path)
        pcd = unproject_mask_to_pcd(rgb, depth, mask, intrinsic_o3d, extrinsic)
        out_path = os.path.join(pcds_dir, f"{index:04d}_frame1.ply")
        o3d.io.write_point_cloud(out_path, pcd)
        written += [mask_path, masked_img_path, out_path]
    return written
