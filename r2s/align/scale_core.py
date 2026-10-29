"""Numerics of step 3 ``align.scale``, moved verbatim from the old scale-fit script.

Everything here is plain numpy / torch: visibility masks (frustum + per-pixel z-buffer
occlusion + edge truncation), the z-only statistical outlier filter, area-weighted
surface sampling of an entity's collision geoms, and the trimmed-Chamfer Adam fit of
the scene scale about the camera pivot. The function bodies, constants and operation
order are unchanged (old line numbers in brackets); only these edits were made:

* ``_save_mask_debug_ply`` writes through ``prepare_output`` (never through a symlink);
* ``find_metric_mesh`` takes the scene's input layout instead of string paths and
  reports which file it picked (``r2s.align.scale_fit`` warns on the fallback);
* ``_BASE_ROT`` / ``_ADJUSTMENT`` are imported from ``r2s.common.transforms`` (same values);
* imports.

The Genesis side (building the collider / metric-mesh entities the samples come from)
lives in ``r2s.align.scale_fit``.
"""

from __future__ import annotations

import math
import re

import numpy as np
from scipy.spatial.transform import Rotation as ScipyRotation

from r2s.common.io import prepare_output
# Identical to the old file's private copies [529-558].
from r2s.common.quat import euler_zyx_to_quat, quat_mul_np
# [49-50] base splat rotation and the shared ~2 deg adjustment (wxyz): the old file's own
# copies of these constants, which have one definition in r2s.common.transforms.
from r2s.common.transforms import SCALE_OPT_ADJUSTMENT as _ADJUSTMENT
from r2s.common.transforms import SCALE_OPT_BASE_ROT as _BASE_ROT


# ── Metric mesh discovery [55-87] ─────────────────────────────────────────────

_SEGMENTED_RE = re.compile(r"^image_(\d+)_segmented\.glb$")


def find_metric_mesh(inputs) -> tuple[str, str]:
    """``(path, source)`` of the metric-depth GLB for one scene (``inputs`` = ``layout.inputs``).

    Preference order (unchanged): ``DATA/<type>/<s>/mesh_sheet.glb``, else the
    highest-index ``DATA/assets/<s>/image_<N>_segmented.glb``. ``source`` is
    ``"mesh_sheet"`` or ``"segmented_fallback"``.
    """
    sheet = inputs.mesh_sheet
    if sheet.is_file():
        return str(sheet), "mesh_sheet"
    asset_dir = inputs.assets_dir
    if not asset_dir.is_dir():
        raise FileNotFoundError(
            f"Scene assets folder not found: {asset_dir} (and no {sheet}).\n"
            "Pass --metric-mesh explicitly."
        )
    candidates: list[tuple[int, str]] = []
    for path in asset_dir.iterdir():
        m = _SEGMENTED_RE.match(path.name)
        if m is not None:
            candidates.append((int(m.group(1)), path.name))
    if not candidates:
        raise FileNotFoundError(
            f"No `image_*_segmented.glb` files in {asset_dir} (and no {sheet}).\n"
            "Pass --metric-mesh explicitly."
        )
    candidates.sort(key=lambda x: x[0])
    _, fn = candidates[-1]
    return str(asset_dir / fn), "segmented_fallback"


# ── Visibility-filtered Adam chamfer optimisation [92-503] ───────────────────

def _save_mask_debug_ply(
    path: str,
    coll_world: np.ndarray,
    coll_mask: np.ndarray,
    metric_world: np.ndarray,
    metric_mask: np.ndarray | None = None,
    cam_pos: np.ndarray | None = None,
    coll_frustum_mask: np.ndarray | None = None,
    metric_frustum_mask: np.ndarray | None = None,
) -> None:
    """
    Colours:
      green  — kept (visible surface)
      yellow — in frustum but occlusion-culled
      red    — outside frustum
      blue   — metric kept, magenta — metric culled (frustum or occlusion)
      orange — metric in frustum but occlusion-culled (only when frustum mask supplied)
    """
    pts_list, col_list = [], []
    if coll_frustum_mask is None:
        coll_frustum_mask = coll_mask
    coll_occluded = coll_frustum_mask & (~coll_mask)
    coll_outside = ~coll_frustum_mask
    pts_list.append(coll_world[coll_mask])
    col_list.append(np.tile(np.array([[0, 255, 0]], dtype=np.uint8),
                            (coll_mask.sum(), 1)))
    pts_list.append(coll_world[coll_occluded])
    col_list.append(np.tile(np.array([[255, 245, 180]], dtype=np.uint8),
                            (coll_occluded.sum(), 1)))
    pts_list.append(coll_world[coll_outside])
    col_list.append(np.tile(np.array([[255, 200, 200]], dtype=np.uint8),
                            (coll_outside.sum(), 1)))

    if metric_mask is None:
        metric_mask = np.ones(len(metric_world), dtype=bool)
    if metric_frustum_mask is None:
        metric_frustum_mask = metric_mask
    metric_occluded = metric_frustum_mask & (~metric_mask)
    metric_outside = ~metric_frustum_mask
    pts_list.append(metric_world[metric_mask])
    col_list.append(np.tile(np.array([[0, 64, 255]], dtype=np.uint8),
                            (metric_mask.sum(), 1)))
    pts_list.append(metric_world[metric_occluded])
    col_list.append(np.tile(np.array([[255, 220, 180]], dtype=np.uint8),
                            (metric_occluded.sum(), 1)))
    pts_list.append(metric_world[metric_outside])
    col_list.append(np.tile(np.array([[245, 200, 245]], dtype=np.uint8),
                            (metric_outside.sum(), 1)))
    if cam_pos is not None:
        pts_list.append(np.asarray(cam_pos, dtype=np.float64).reshape(1, 3))
        col_list.append(np.array([[255, 255, 255]], dtype=np.uint8))

    pts = np.concatenate(pts_list, axis=0)
    cols = np.concatenate(col_list, axis=0)
    import trimesh
    cloud = trimesh.PointCloud(pts, colors=cols)
    cloud.export(str(prepare_output(path)))
    print(f"[viz]   wrote mask debug cloud → {path}  "
          f"({len(pts)} pts: "
          f"{coll_mask.sum()} coll-kept, {coll_occluded.sum()} coll-occluded, "
          f"{coll_outside.sum()} coll-outside, "
          f"{metric_mask.sum()} metric-kept, {metric_occluded.sum()} metric-occluded, "
          f"{metric_outside.sum()} metric-outside)")


def _camera_basis(
    cam_pos: np.ndarray,
    cam_lookat: np.ndarray,
    cam_up: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cam_pos    = np.asarray(cam_pos,    dtype=np.float64)
    cam_lookat = np.asarray(cam_lookat, dtype=np.float64)
    cam_up     = np.asarray(cam_up,     dtype=np.float64)
    forward = cam_lookat - cam_pos
    forward /= np.linalg.norm(forward)
    # `cam_up` (from load_calibration) is the camera's image-up direction in
    # world frame. With OpenCV's Y-down image convention the camera basis
    # {right, image_up, forward} is left-handed, so right = forward × image_up.
    # The historical `cross(cam_up, forward)` produced the negation of right,
    # mirroring u about cx; symmetric frustum margins masked the bug.
    right = np.cross(forward, cam_up)
    right /= np.linalg.norm(right)
    # Re-orthogonalise the image-up against forward (Gram–Schmidt). This must
    # equal cam_up (up to numerical noise) because the projection formula
    #     v = fy * (-yc) / z + cy
    # treats `up_a` as the image-up direction; deriving up_a = forward × right
    # would yield -cam_up and silently mirror v about cy.
    up_a = np.cross(right, forward)
    up_a /= np.linalg.norm(up_a)
    return cam_pos, forward, right, up_a


def _project_to_uvz(
    x_world: np.ndarray,
    cam_pos: np.ndarray,
    forward: np.ndarray,
    right: np.ndarray,
    up_a: np.ndarray,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = x_world - cam_pos
    z  = d @ forward
    xc = d @ right
    yc = d @ up_a
    z_safe = np.maximum(z, 1e-9)
    u = K[0, 0] * xc / z_safe + K[0, 2]
    v = K[1, 1] * (-yc) / z_safe + K[1, 2]
    return u, v, z


def _visible_mask(
    x_world: np.ndarray,
    cam_pos: np.ndarray,
    cam_lookat: np.ndarray,
    cam_up: np.ndarray,
    K: np.ndarray,
    W: int,
    H: int,
    near: float = 0.01,
    far: float = 50.0,
    truncate_w: float = 0.05,
    truncate_h: float = 0.05,
    truncate_left: float | None = None,
    truncate_right: float | None = None,
    truncate_top: float | None = None,
    truncate_bottom: float | None = None,
) -> np.ndarray:
    cam_pos, forward, right, up_a = _camera_basis(cam_pos, cam_lookat, cam_up)
    u, v, z = _project_to_uvz(x_world, cam_pos, forward, right, up_a, K)
    in_depth = (z > near) & (z < far)
    off_l = float(truncate_w if truncate_left is None else truncate_left) * float(W)
    off_r = float(truncate_w if truncate_right is None else truncate_right) * float(W)
    off_t = float(truncate_h if truncate_top is None else truncate_top) * float(H)
    off_b = float(truncate_h if truncate_bottom is None else truncate_bottom) * float(H)
    in_img = ((u >= off_l) & (u < (W - off_r))
              & (v >= off_t) & (v < (H - off_b)))
    return in_depth & in_img


def _visible_surface_mask(
    x_world: np.ndarray,
    cam_pos: np.ndarray,
    cam_lookat: np.ndarray,
    cam_up: np.ndarray,
    K: np.ndarray,
    W: int,
    H: int,
    *,
    cell_px: int = 1,
    depth_tol_rel: float = 0.02,
    depth_tol_abs: float = 0.02,
    near: float = 0.01,
    far: float = 50.0,
    truncate_w: float = 0.05,
    truncate_h: float = 0.05,
    truncate_left: float | None = None,
    truncate_right: float | None = None,
    truncate_top: float | None = None,
    truncate_bottom: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (visible_surface_mask, frustum_mask).

    visible_surface_mask is the AND of the frustum membership and a per-pixel
    z-buffer test that drops points farther than `min_z + max(depth_tol_abs,
    depth_tol_rel * min_z)` within their pixel cell. With cell_px <= 0 the
    occlusion test is skipped and the frustum mask is returned unchanged.

    Falls back to the frustum mask alone if fewer than 100 points survive
    occlusion culling, mirroring the existing safety net.
    """
    frustum = _visible_mask(x_world, cam_pos, cam_lookat, cam_up, K, W, H,
                            near=near, far=far,
                            truncate_w=truncate_w, truncate_h=truncate_h,
                            truncate_left=truncate_left,
                            truncate_right=truncate_right,
                            truncate_top=truncate_top,
                            truncate_bottom=truncate_bottom)
    if cell_px <= 0:
        return frustum, frustum

    idx_in = np.flatnonzero(frustum)
    if idx_in.size == 0:
        return frustum, frustum

    cam_pos_a, forward, right, up_a = _camera_basis(cam_pos, cam_lookat, cam_up)
    u, v, z = _project_to_uvz(x_world[idx_in], cam_pos_a, forward, right, up_a, K)

    cu = np.floor(u / float(cell_px)).astype(np.int64)
    cv = np.floor(v / float(cell_px)).astype(np.int64)
    cell_id = cu * np.int64(10_000_003) + cv  # cheap unique key per cell

    order = np.argsort(cell_id, kind="stable")
    cell_sorted = cell_id[order]
    z_sorted = z[order]
    boundaries = np.concatenate(
        ([0], np.flatnonzero(np.diff(cell_sorted)) + 1, [len(cell_sorted)])
    )

    keep_local = np.zeros(idx_in.size, dtype=bool)
    for s, e in zip(boundaries[:-1], boundaries[1:]):
        zs = z_sorted[s:e]
        z_min = zs.min()
        tol = max(depth_tol_abs, depth_tol_rel * float(max(z_min, 1e-6)))
        keep_local[order[s:e]] = zs <= (z_min + tol)

    surface = np.zeros_like(frustum)
    surface[idx_in[keep_local]] = True
    if surface.sum() < 20:
        print(f"[vis]   z-buffer kept only {surface.sum()} pts; "
              f"falling back to frustum-only mask.")
        return frustum, frustum
    return surface, frustum


def find_scale_chamfer(
    x_coll_0: np.ndarray,
    x_metric: np.ndarray,
    pivot: np.ndarray,
    s_init: float,
    K: np.ndarray,
    cam_pos: np.ndarray,
    cam_lookat: np.ndarray,
    cam_up: np.ndarray,
    W: int,
    H: int,
    n_steps: int = 400,
    lr: float = 5e-3,
    s_min_ratio: float = 0.5,
    s_max_ratio: float = 2.0,
    trim_quantile: float = 0.9,
    log_every: int = 25,
    mask_debug_ply: str | None = None,
    occlusion_cell_px: int = 1,
    occlusion_depth_tol_rel: float = 0.02,
    truncate_w: float = 0.05,
    truncate_h: float = 0.05,
    truncate_left: float | None = None,
    truncate_right: float | None = None,
    truncate_top: float | None = None,
    truncate_bottom: float | None = None,
    match_metric_z: bool = False,
    coll_z_pad: float = 0.05,
) -> tuple[float, float]:
    try:
        import torch  # type: ignore[import]
    except ImportError as e:
        raise RuntimeError("Adam optimisation requires PyTorch.") from e

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    x_init = np.asarray(x_coll_0, dtype=np.float64)
    m = np.asarray(x_metric, dtype=np.float64)
    n_pts = len(x_init)

    # Occlusion filtering is intentionally disabled for the collision cloud:
    # the collision mesh and metric mesh are sampled into the same camera but
    # represent the same physical surface from different sources, so a shared
    # z-buffer would have them spuriously occlude each other. We cannot mix
    # them safely, so we keep occlusion culling only on the metric side and
    # use a frustum-only mask for collision points.
    vis, vis_frustum = _visible_surface_mask(
        x_init, cam_pos, cam_lookat, cam_up, K, W, H,
        cell_px=occlusion_cell_px,
        depth_tol_rel=occlusion_depth_tol_rel,
        truncate_w=truncate_w,
        truncate_h=truncate_h,
        truncate_left=truncate_left,
        truncate_right=truncate_right,
        truncate_top=truncate_top,
        truncate_bottom=truncate_bottom,
    )
    if vis.sum() < 100:
        print(f"[warn]  only {vis.sum()} collision samples visible to camera; "
              f"check extrinsics or increase n_pts.")
    coll_world_before = x_init.copy()
    coll_mask = vis.copy()
    coll_frustum_mask = vis_frustum.copy()

    v_np = (x_init[vis] - pivot) / s_init
    print(f"[vis]   collision samples visible: {len(v_np)} / {n_pts} "
          f"(frustum {vis_frustum.sum()}, occluded {(vis_frustum & ~vis).sum()})")

    metric_world_before = m.copy()
    m_surface_full, m_frustum_full = _visible_surface_mask(
        m, cam_pos, cam_lookat, cam_up, K, W, H,
        cell_px=occlusion_cell_px,
        depth_tol_rel=occlusion_depth_tol_rel,
        truncate_w=truncate_w,
        truncate_h=truncate_h,
        truncate_left=truncate_left,
        truncate_right=truncate_right,
        truncate_top=truncate_top,
        truncate_bottom=truncate_bottom,
    )
    if m_surface_full.sum() < max(50, int(0.01 * len(m))):
        print(f"[vis]   surface filter kept {m_surface_full.sum()}/{len(m)} metric points; "
              f"falling back to all metric points.")
        metric_mask_for_viz = np.ones(len(m), dtype=bool)
        metric_frustum_for_viz = np.ones(len(m), dtype=bool)
    else:
        m = m[m_surface_full]
        metric_mask_for_viz = m_surface_full
        metric_frustum_for_viz = m_frustum_full
        print(f"[vis]   metric points visible:     {len(m)} / {len(x_metric)} "
              f"(frustum {m_frustum_full.sum()}, "
              f"occluded {(m_frustum_full & ~m_surface_full).sum()})")

    # Optional: drop collision points whose world-z lies outside the metric
    # cloud's z-range (with `coll_z_pad` slack). Useful when the metric mesh
    # is the segmented table top only, so the floor below should be excluded
    # from the chamfer even though it is in the camera frustum.
    #
    # The metric mesh is fixed in metres while the collision mesh is at the
    # initial (possibly very wrong) scale, so a raw world-z gate rejects
    # everything when `s_init` is far off. We first estimate a coarse
    # collision→metric scale `ratio` from the ratio of median camera-forward
    # depths over each cloud's in-frustum points, then apply the z-band to
    # where each collision point *would land* at that ratio (pivot rescaled).
    # This isn't a substitute for the chamfer optimisation — it just keeps
    # the gate functional even when init is bad.
    if match_metric_z and len(m) > 0 and vis.any():
        # Use a low percentile of camera-forward depth instead of the median.
        # The median is dominated by whichever surface has more points (the
        # ground, in the collision cloud), so it pulls the predicted ratio
        # away from the table. The closest dominant surface in both clouds
        # should be the table edge facing the camera, so the 10th-percentile
        # depth is a robust common reference.
        _, fwd, _, _ = _camera_basis(cam_pos, cam_lookat, cam_up)
        cam_pos_a = np.asarray(cam_pos, dtype=np.float64)
        d_coll = (x_init[vis] - cam_pos_a) @ fwd
        d_metric = (m - cam_pos_a) @ fwd
        pct = 10.0
        ref_coll = float(np.percentile(d_coll, pct))
        ref_metric = float(np.percentile(d_metric, pct))
        ratio = ref_metric / max(ref_coll, 1e-6)
        x_pred_z = pivot[2] + ratio * (x_init[:, 2] - pivot[2])
        z_lo = float(m[:, 2].min()) - float(coll_z_pad)
        z_hi = float(m[:, 2].max()) + float(coll_z_pad)
        z_band = (x_pred_z >= z_lo) & (x_pred_z <= z_hi)
        n_before = int(vis.sum())
        vis = vis & z_band
        coll_mask = vis.copy()
        coll_frustum_mask = coll_frustum_mask & z_band
        v_np = (x_init[vis] - pivot) / s_init
        print(f"[vis]   z-band gate: depth-ratio={ratio:.3f} "
              f"(metric_p{int(pct)}={ref_metric:.3f}, coll_p{int(pct)}={ref_coll:.3f});  "
              f"kept {int(vis.sum())} / {n_before} collision points within "
              f"metric z ∈ [{z_lo:.3f}, {z_hi:.3f}] (pad={coll_z_pad}).")

    if mask_debug_ply is not None:
        _save_mask_debug_ply(
            mask_debug_ply,
            coll_world=coll_world_before,
            coll_mask=coll_mask,
            metric_world=metric_world_before,
            metric_mask=metric_mask_for_viz,
            cam_pos=pivot,
            coll_frustum_mask=coll_frustum_mask,
            metric_frustum_mask=metric_frustum_for_viz,
        )

    v_t     = torch.as_tensor(v_np,  dtype=dtype, device=device)
    m_t     = torch.as_tensor(m,     dtype=dtype, device=device)
    pivot_t = torch.as_tensor(pivot, dtype=dtype, device=device)

    log_ratio = torch.zeros((), dtype=dtype, device=device, requires_grad=True)
    opt = torch.optim.Adam([log_ratio], lr=lr)

    log_lo = float(np.log(s_min_ratio))
    log_hi = float(np.log(s_max_ratio))

    k_m = max(int(trim_quantile * len(m_t)),   1)
    k_c = max(int(trim_quantile * len(v_t)),   1)

    last_loss = float("inf")
    for step in range(n_steps):
        opt.zero_grad()
        s = float(s_init) * torch.exp(log_ratio)
        x = pivot_t + s * v_t

        d = torch.cdist(m_t, x)
        d_m2c = d.min(dim=1).values
        d_c2m = d.min(dim=0).values

        d_m2c_k = torch.topk(d_m2c, k_m, largest=False).values
        d_c2m_k = torch.topk(d_c2m, k_c, largest=False).values
        loss_m = (d_m2c_k ** 2).mean()
        loss_c = (d_c2m_k ** 2).mean()
        loss = loss_m + loss_c

        loss.backward()
        opt.step()

        with torch.no_grad():
            log_ratio.clamp_(log_lo, log_hi)

        if step % log_every == 0 or step == n_steps - 1:
            print(f"[adam]  step {step:04d}  s={s.item():.6f}  "
                  f"loss={loss.item():.6f} "
                  f"(m→c={loss_m.item():.6f}, c→m={loss_c.item():.6f})  "
                  f"log_ratio={log_ratio.item():+.4f}")
        last_loss = float(loss.item())

    s_final = float(s_init * math.exp(float(log_ratio.item())))
    lr_val = float(log_ratio.item())
    if lr_val <= log_lo + 1e-4 or lr_val >= log_hi - 1e-4:
        print(f"[warn]  s={s_final:.5f} hit the clamp range "
              f"[{s_min_ratio*s_init:.3f}, {s_max_ratio*s_init:.3f}] — "
              f"result may be unreliable.")
    return s_final, last_loss


# ── Calibration + quaternion helpers [508-598; quat_mul_np/euler_zyx_to_quat from r2s.common.quat] ───────────────────────────────

def load_calibration(intrinsics_path: str, extrinsics_path: str):
    K = np.load(intrinsics_path).astype(np.float64)
    T_w2c = np.load(extrinsics_path).astype(np.float64).reshape(4, 4)
    T_c2w = np.linalg.inv(T_w2c)
    correction = np.array([[1.0,  0.0,  0.0],
                           [0.0, -1.0,  0.0],
                           [0.0,  0.0, -1.0]])
    T_c2w[:3, :3] = T_c2w[:3, :3] @ correction

    fy = float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    W, H   = int(round(2 * cx)), int(round(2 * cy))
    fov_y  = 2.0 * math.atan(H / (2.0 * fy)) * 180.0 / math.pi

    pos     = T_c2w[:3, 3].copy()
    forward = -T_c2w[:3, 2]
    up      = T_c2w[:3, 1].copy()
    lookat  = pos + forward
    return K, W, H, fov_y, pos, lookat, up


def quat_z_rot(rz_rad: float) -> np.ndarray:
    half = rz_rad * 0.5
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64)


def normalize(v, eps=1e-8):
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Zero-length vector cannot be normalized.")
    return v / n


def rot_matrix_to_wxyz(R: np.ndarray) -> tuple[float, float, float, float]:
    x, y, z, w = ScipyRotation.from_matrix(R).as_quat()
    q = np.array([w, x, y, z], dtype=np.float64)
    if q[0] < 0:
        q = -q
    return tuple(map(float, q))


def lookat_to_quat(pos, lookat, up=np.array([0.0, 0.0, 1.0])):
    pos = np.asarray(pos, dtype=np.float64)
    lookat = np.asarray(lookat, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    forward = normalize(lookat - pos)
    z_cam_world = -forward
    x_cam_world = normalize(np.cross(up, z_cam_world))
    y_cam_world = normalize(np.cross(z_cam_world, x_cam_world))

    R = np.stack([x_cam_world, y_cam_world, z_cam_world], axis=1)
    return rot_matrix_to_wxyz(R)


def apply_relative_transform_to_quat(orig_quat: np.ndarray, rz_rad: float,
                                     rx_rad: float = 0.0, ry_rad: float = 0.0) -> tuple:
    """Compose `q_z · q_y · q_x · orig_quat`. With rx=ry=0 this collapses to a pure Z rotation."""
    if rx_rad == 0.0 and ry_rad == 0.0:
        q_rel = quat_z_rot(rz_rad)
    else:
        q_rel = euler_zyx_to_quat(rx_rad, ry_rad, rz_rad)
    q_new = quat_mul_np(q_rel, np.asarray(orig_quat, dtype=np.float64))
    return tuple(float(v) for v in q_new)


# ── Entity sampling [740-943] (duck-typed: ``entity._entity.geoms`` / ``get_verts`` / ``init_faces``) ──

def _get_world_verts(entity, n_sub: int, rng_seed: int = 0) -> np.ndarray:
    import torch
    v = entity._entity.get_verts()
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu().numpy()
    v = np.asarray(v, dtype=np.float64)
    if v.ndim == 3:
        v = v[0]
    if len(v) > n_sub:
        idx = np.random.default_rng(rng_seed).choice(len(v), size=n_sub, replace=False)
        v = v[idx]
    return v


def _sample_triangles_area_weighted(
    verts: np.ndarray,
    faces: np.ndarray,
    n_sample: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Area-weighted uniform sampling on a triangle mesh."""
    if len(faces) == 0 or n_sample <= 0:
        return np.empty((0, 3), dtype=np.float64)
    tri = verts[faces]                      # (F, 3, 3)
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    areas = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    total = float(areas.sum())
    if total <= 0.0:
        return np.empty((0, 3), dtype=np.float64)
    probs = areas / total
    face_idx = rng.choice(len(faces), size=n_sample, replace=True, p=probs)
    u = rng.random(n_sample)
    v = rng.random(n_sample)
    flip = (u + v) > 1.0
    u[flip] = 1.0 - u[flip]
    v[flip] = 1.0 - v[flip]
    a = tri[face_idx, 0]
    b = tri[face_idx, 1]
    c = tri[face_idx, 2]
    pts = a + u[:, None] * (b - a) + v[:, None] * (c - a)
    return pts.astype(np.float64)


def _get_world_surface_samples(
    entity,
    n_sample: int,
    rng_seed: int = 0,
    *,
    triangle_filter=None,
) -> np.ndarray:
    """
    Area-weighted surface sampling in world frame.

    Iterates over the entity's collision geoms (each is a convex piece after
    coacd decomposition). For each piece, fetches world-frame vertices via
    Genesis' `get_verts()` and the face index list via `init_faces`, then
    samples points uniformly over the triangulated surface so a large flat
    face (e.g. a tabletop) contributes proportionally to its area rather
    than to its (few) vertices. Falls back to vertex sampling on any error.

    If `triangle_filter` is given, it is called with the per-triangle
    centroids (Nx3 array, world frame) and must return a boolean mask of
    triangles to keep before area-weighted sampling. This is used to focus
    collision samples inside the camera frustum so the in-frustum count is
    proportional to the visible region rather than to the room's full area.
    """
    import torch
    rng = np.random.default_rng(rng_seed)
    try:
        geoms = list(entity._entity.geoms)
        if not geoms:
            raise RuntimeError("entity has no collision geoms")

        per_geom_verts: list[np.ndarray] = []
        per_geom_faces: list[np.ndarray] = []
        per_geom_area:  list[float] = []
        for g in geoms:
            v = g.get_verts()
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().numpy()
            v = np.asarray(v, dtype=np.float64)
            if v.ndim == 3:
                v = v[0]
            f = np.asarray(g.init_faces, dtype=np.int64)
            if len(f) == 0:
                continue
            tri = v[f]
            if triangle_filter is not None:
                centroids = tri.mean(axis=1)
                keep = np.asarray(triangle_filter(centroids), dtype=bool)
                if keep.shape != (len(f),):
                    raise RuntimeError(
                        f"triangle_filter returned shape {keep.shape}, expected ({len(f)},)"
                    )
                if not keep.any():
                    continue
                f = f[keep]
                tri = tri[keep]
            a = 0.5 * np.linalg.norm(
                np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
            ).sum()
            per_geom_verts.append(v)
            per_geom_faces.append(f)
            per_geom_area.append(float(a))

        total_area = float(np.sum(per_geom_area))
        if not per_geom_verts or total_area <= 0.0:
            raise RuntimeError("no usable triangles found")

        weights = np.asarray(per_geom_area) / total_area
        alloc = np.maximum(np.round(weights * n_sample).astype(int), 0)
        diff = n_sample - int(alloc.sum())
        if diff != 0:
            alloc[int(np.argmax(weights))] += diff
        alloc = np.maximum(alloc, 0)

        chunks: list[np.ndarray] = []
        for v, f, n in zip(per_geom_verts, per_geom_faces, alloc):
            if n > 0:
                chunks.append(_sample_triangles_area_weighted(v, f, int(n), rng))
        pts = np.concatenate(chunks, axis=0) if chunks else np.empty((0, 3))
        if len(pts) > n_sample:
            idx = rng.choice(len(pts), size=n_sample, replace=False)
            pts = pts[idx]
        return pts
    except Exception as e:
        print(f"[sample] surface sampling failed ({e}); falling back to verts.")
        return _get_world_verts(entity, n_sample, rng_seed=rng_seed)


def _statistical_outlier_filter(
    pts: np.ndarray,
    k: int = 20,
    std_ratio: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Z-axis statistical outlier removal: for each point, find its `k` nearest
    neighbours in 3D and compute the mean |Δz| to those neighbours. Points
    whose score exceeds `mean + std_ratio * std` over the cloud are dropped.

    Depth-projection noise on the metric mesh appears almost entirely along
    the camera viewing ray (≈ world z for top-down views), so a 1-D z-only
    score removes floaters without penalising legitimate horizontal spread
    such as a wide tabletop.
    """
    n = len(pts)
    if n <= k + 1 or k <= 0:
        return pts, np.ones(n, dtype=bool)
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return pts, np.ones(n, dtype=bool)
    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=k + 1)      # column 0 is self
    nbr_idx = idx[:, 1:]
    z = pts[:, 2]
    nbr_z = z[nbr_idx]                      # (n, k)
    score = np.abs(nbr_z - z[:, None]).mean(axis=1)
    thresh = float(score.mean() + std_ratio * score.std())
    keep = score <= thresh
    return pts[keep], keep


def _extract_world_samples(
    env,
    n_coll: int,
    n_metric: int,
    *,
    calib: dict | None = None,
    truncate_w: float = 0.05,
    truncate_h: float = 0.05,
    outlier_k: int = 20,
    outlier_std_ratio: float = 2.0,
):
    """
    Sample world-frame surface points and remove statistical outliers.

    `calib`/`truncate_w`/`truncate_h` are accepted for forward compatibility
    (e.g. so callers may grow into frustum-aware sampling later) but are not
    used to pre-filter triangles here — the visibility-mask debug PLY is
    more informative when the full clouds are passed downstream.

    Outlier removal: points whose mean distance to their `outlier_k`
    nearest neighbours exceeds `mean + outlier_std_ratio * std` are
    dropped. This catches stray points spawned by noisy depth projection
    in the metric mesh and rare detached fragments in the collision mesh.
    Set `outlier_k <= 0` to skip.
    """
    del calib, truncate_w, truncate_h  # reserved; unused

    v_coll = _get_world_surface_samples(env.entities["background"], n_coll, rng_seed=0)
    v_metric = _get_world_surface_samples(env.entities["metric_ref"], n_metric, rng_seed=1)

    n_coll_raw, n_metric_raw = len(v_coll), len(v_metric)
    if outlier_k > 0:
        v_coll, _   = _statistical_outlier_filter(v_coll,   k=outlier_k, std_ratio=outlier_std_ratio)
        v_metric, _ = _statistical_outlier_filter(v_metric, k=outlier_k, std_ratio=outlier_std_ratio)

    print(f"[genesis] collision surface samples (world): {len(v_coll)}"
          f" (filtered {n_coll_raw - len(v_coll)} outliers from {n_coll_raw})")
    print(f"[genesis] metric surface samples   (world): {len(v_metric)}"
          f" (filtered {n_metric_raw - len(v_metric)} outliers from {n_metric_raw})")
    return v_coll, v_metric
