"""Step 4 ``surface.plane``: fit the horizontal operation surface (tabletop) box.

Inputs:  ``world/collider.glb`` (step 1b), ``align/aligned_pose_refined.npy`` and
         ``align/aligned_pose_scene_scale_refined.npy`` (step 3), and the DROID camera
         ``intrinsics.npy`` / ``extrinsics.npy`` (step-0 scene folder).
Output:  ``surface/operation_surface_plane_fit.json`` (``selected`` = the support box the
         evaluator and steps 5-6 use); with ``--viz-ply`` also a colored
         ``surface/operation_surface_plane_fit.ply``.

Ported from the old ``operation_surface_plane_fit.py``; defaults are the flags of
``plane_fitting_plane_fit.sh``. The pipeline is a robust planar-region segmentation:

  1. recompute per-point local normals + planarity via PCA over k-NN
     (mesh face normals from a convex-decomposition collider are noisy)
  2. keep points that are upward-facing, planar, and inside the workspace
  3. detect horizontal slabs as modes of a planarity-weighted height histogram
  4. grow contiguous, normal-consistent planar regions inside each slab band
  5. refit each region with a Tukey-biweight total-least-squares plane
  6. snap the accepted surface to exact +Z at the robust median height
  7. emit a ranked list of conservative supported rectangles

The robust fit is used only to choose inliers; the emitted surface is snapped
to exact +Z so the collider stays horizontal and objects rest stably. The camera
frustum only picks which slab is the operation surface; it never crops the fit.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from r2s.common.camera import load_camera_model, load_camera_pose_from_extrinsics
from r2s.common.io import write_json
from r2s.common.mesh import mesh_world_geometry
from r2s.common.quat import matrix_to_quat_wxyz
from r2s.common.transforms import (
    DEFAULT_GLB_FRONT,
    DEFAULT_GLB_UP,
    PLACEMENT_LINEAGE,
    SCALE_OPT_ADJUSTMENT,
    align_up_and_front,
    collider_world_transform,
    load_scene_scale,
)
from r2s.layout import SceneLayout
from r2s.steps import Step
from r2s.surface.geometry import (
    DEFAULT_WORKSPACES,
    PlaneCandidate,
    auto_workspace_bounds,
    bounds_args,
    camera_visible_surface_mask,
    connected_components_radius,
    expanded_bounds,
    format_bounds,
    oriented_box_from_plane,
    point_bounds,
    sample_surface_world,
    upward_normal_mask,
    workspace_mask,
)

EPS = 1e-9


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Return the weighted median of `values` (robust snap height)."""
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if len(values) == 0:
        return 0.0
    if not np.any(weights > 0):
        return float(np.median(values))
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumw = np.cumsum(weights)
    cutoff = 0.5 * cumw[-1]
    idx = int(np.searchsorted(cumw, cutoff))
    return float(values[min(idx, len(values) - 1)])


def compute_pca_normals(
    points: np.ndarray,
    k: int,
    mesh_normals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate per-point normals and planarity from a local PCA over k-NN.

    Mesh face normals on a convex-decomposition collider are facet-quantized
    and noisy; a covariance over each point's k nearest neighbors yields a
    smoother surface normal plus a planarity score that says how flat the
    neighborhood is. The face normal is used only to disambiguate the sign.

    Returns `(normals (N,3) unit, planarity (N,) in [0,1])`.
    """
    points = np.asarray(points, dtype=np.float64)
    mesh_normals = np.asarray(mesh_normals, dtype=np.float64)
    n = len(points)
    normals = mesh_normals.copy()
    planarity = np.zeros(n, dtype=np.float64)
    if n < 4:
        return normals, planarity

    k = int(min(max(k, 4), n))
    tree = cKDTree(points)
    _, nbr_idx = tree.query(points, k=k)
    nbr = points[nbr_idx]                       # (N, k, 3)
    centered = nbr - nbr.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", centered, centered) / float(k - 1)

    finite = np.all(np.isfinite(cov), axis=(1, 2))
    eigvals = np.zeros((n, 3), dtype=np.float64)
    eigvecs = np.zeros((n, 3, 3), dtype=np.float64)
    if np.any(finite):
        ev, evec = np.linalg.eigh(cov[finite])  # ascending eigenvalues
        eigvals[finite] = ev
        eigvecs[finite] = evec

    l0, l1, l2 = eigvals[:, 0], eigvals[:, 1], eigvals[:, 2]
    pca_normal = eigvecs[:, :, 0]               # smallest-eigenvalue direction
    planarity = np.clip((l1 - l0) / (l2 + EPS), 0.0, 1.0)

    # Degenerate neighborhoods (collinear / coincident, non-finite) fall back
    # to the mesh face normal with zero planarity.
    degenerate = (~finite) | (l2 <= EPS) | ~np.all(np.isfinite(pca_normal), axis=1)
    pca_normal[degenerate] = mesh_normals[degenerate]
    planarity[degenerate] = 0.0

    norms = np.linalg.norm(pca_normal, axis=1, keepdims=True)
    pca_normal = pca_normal / np.maximum(norms, EPS)

    # Sign-disambiguate against the mesh face normal (sign only, not direction).
    sign = np.where(np.einsum("ij,ij->i", pca_normal, mesh_normals) < 0.0, -1.0, 1.0)
    pca_normal = pca_normal * sign[:, None]
    return pca_normal, planarity


def upward_pca_mask(
    pca_normals: np.ndarray,
    max_tilt_rad: float,
    normal_abs: bool,
) -> np.ndarray:
    """Keep points whose PCA normal lies within the max-tilt cone of +Z."""
    nz = pca_normals[:, 2]
    cos_tilt = np.abs(nz) if normal_abs else nz
    return cos_tilt >= math.cos(max_tilt_rad)


def _smooth(values: np.ndarray, sigma_bins: float) -> np.ndarray:
    """Gaussian-smooth a 1-D array with a small truncated kernel."""
    if sigma_bins <= 0 or len(values) < 3:
        return values
    radius = max(1, int(math.ceil(3.0 * sigma_bins)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma_bins) ** 2)
    kernel /= kernel.sum()
    return np.convolve(values, kernel, mode="same")


def detect_height_slabs(
    z: np.ndarray,
    weights: np.ndarray,
    bin_width: float,
    min_separation: float,
    rel_floor: float = 0.15,
    band_half: float = 0.06,
) -> list[dict[str, Any]]:
    """Find horizontal slabs as modes of a planarity-weighted height histogram.

    Weighting by planarity lets flat tabletop points dominate while clutter
    scattered over many heights is suppressed, so each real horizontal surface
    (countertop, shelf, floor) shows up as one clean peak.

    Returns a list of `{"z", "strength", "lo", "hi"}` sorted by descending z.
    """
    z = np.asarray(z, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if len(z) < 8 or float(np.ptp(z)) < max(min_separation, bin_width):
        zc = weighted_median(z, weights)
        return [{"z": zc, "strength": 1.0, "lo": zc - band_half, "hi": zc + band_half}]

    nbins = int(max(16, math.ceil(float(np.ptp(z)) / max(bin_width, EPS))))
    hist, edges = np.histogram(z, bins=nbins, weights=np.maximum(weights, 0.0))
    centers = 0.5 * (edges[:-1] + edges[1:])
    dens = _smooth(hist.astype(np.float64), sigma_bins=1.5)
    total = float(dens.sum()) + EPS

    peak_idx = [
        i
        for i in range(1, len(dens) - 1)
        if dens[i] >= dens[i - 1] and dens[i] >= dens[i + 1] and dens[i] > 0.0
    ]
    if not peak_idx:
        zc = weighted_median(z, weights)
        return [{"z": zc, "strength": 1.0, "lo": zc - band_half, "hi": zc + band_half}]

    peak_max = max(dens[i] for i in peak_idx)
    peak_idx = [i for i in peak_idx if dens[i] >= rel_floor * peak_max]
    bin_step = float(centers[1] - centers[0])

    slabs: list[dict[str, Any]] = []
    for i in peak_idx:
        half = 0.5 * dens[i]
        lo_i = i
        while lo_i > 0 and dens[lo_i] >= half:
            lo_i -= 1
        hi_i = i
        while hi_i < len(dens) - 1 and dens[hi_i] >= half:
            hi_i += 1
        zc = float(centers[i])
        lo = max(zc - band_half, float(centers[lo_i]) - 0.5 * bin_step)
        hi = min(zc + band_half, float(centers[hi_i]) + 0.5 * bin_step)
        slabs.append({"z": zc, "strength": float(dens[i] / total), "lo": lo, "hi": hi})

    # Merge slabs whose centers are closer than `min_separation` (keep stronger).
    slabs.sort(key=lambda s: s["z"])
    merged: list[dict[str, Any]] = []
    for slab in slabs:
        if merged and slab["z"] - merged[-1]["z"] < min_separation:
            if slab["strength"] > merged[-1]["strength"]:
                merged[-1] = slab
        else:
            merged.append(slab)
    merged.sort(key=lambda s: s["z"], reverse=True)
    return merged


def grow_planar_regions(
    points: np.ndarray,
    normals: np.ndarray,
    radius: float,
    min_points: int,
    normal_cos: float,
) -> list[np.ndarray]:
    """Region-grow contiguous, normal-consistent planar patches.

    Like a connected-components flood fill, but an edge is only followed when
    the two PCA normals agree within `normal_cos`. This separates the counter
    from an adjacent wall or a tilted neighbor, while gaps from objects sitting
    on the counter merely stop growth without splitting the surface.
    """
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    if n == 0:
        return []
    tree = cKDTree(points)
    visited = np.zeros(n, dtype=bool)
    regions: list[np.ndarray] = []
    for seed in range(n):
        if visited[seed]:
            continue
        stack = [seed]
        visited[seed] = True
        component = [seed]
        while stack:
            cur = stack.pop()
            for nb in tree.query_ball_point(points[cur], radius):
                if visited[nb]:
                    continue
                if float(np.dot(normals[cur], normals[nb])) < normal_cos:
                    continue
                visited[nb] = True
                stack.append(nb)
                component.append(nb)
        if len(component) >= min_points:
            regions.append(np.asarray(component, dtype=np.int64))
    regions.sort(key=len, reverse=True)
    return regions


def robust_tls_plane(
    points: np.ndarray,
    weights: np.ndarray,
    inlier_tol: float,
    max_tilt_rad: float,
    iters: int = 10,
) -> dict[str, Any] | None:
    """Tukey-biweight total-least-squares plane fit via IRLS.

    A weighted covariance gives a total-least-squares plane (errors in all
    axes, unlike a z-residual lstsq). Tukey biweights fully reject outliers
    rather than just down-weighting them, so a few stray points cannot drag
    the plane. Used only to select inliers; geometry is snapped afterwards.

    Returns `{"normal", "centroid", "tilt_rad", "inlier_mask"}` or None.
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n < 3:
        return None
    w = np.maximum(np.asarray(weights, dtype=np.float64), EPS).copy()
    base = w.copy()
    normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    centroid = pts.mean(axis=0)
    for _ in range(int(iters)):
        wsum = float(w.sum()) + EPS
        centroid = (w[:, None] * pts).sum(axis=0) / wsum
        d = pts - centroid
        cov = (d * w[:, None]).T @ d / wsum
        if not np.all(np.isfinite(cov)):
            return None
        eigvals, eigvecs = np.linalg.eigh(cov)
        if eigvals[2] <= EPS:
            return None
        normal = eigvecs[:, 0]
        resid = d @ normal
        scale = 1.4826 * float(np.median(np.abs(resid))) + EPS
        u = resid / (4.685 * scale)
        tukey = np.where(np.abs(u) < 1.0, (1.0 - u * u) ** 2, 0.0)
        w = base * tukey
        if float(w.sum()) <= EPS:
            return None

    if normal[2] < 0.0:
        normal = -normal
    nz = float(np.clip(abs(normal[2]), 0.0, 1.0))
    tilt_rad = math.acos(nz)
    if tilt_rad > max_tilt_rad:
        return None
    resid = (pts - centroid) @ normal
    inlier_mask = np.abs(resid) <= inlier_tol
    if int(inlier_mask.sum()) < 3:
        return None
    return {
        "normal": normal,
        "centroid": centroid,
        "tilt_rad": float(tilt_rad),
        "inlier_mask": inlier_mask,
    }


def largest_grid_component(
    uv: np.ndarray,
    grid_size: float,
    min_cell_points: int,
) -> np.ndarray:
    """Keep points in the largest 4-connected occupied cell component.

    Drops disconnected stray blobs (so the box does not stretch to reach
    them) while keeping support points that border the main cluster.
    """
    n = len(uv)
    if grid_size <= 0 or n == 0:
        return np.ones(n, dtype=bool)
    ij = np.floor((uv - uv.min(axis=0)) / grid_size).astype(np.int64)
    counts: dict[tuple[int, int], int] = {}
    for a, b in ij:
        counts[(int(a), int(b))] = counts.get((int(a), int(b)), 0) + 1
    occupied = {c for c, k in counts.items() if k >= int(min_cell_points)}
    if not occupied:
        return np.ones(n, dtype=bool)
    seen: set[tuple[int, int]] = set()
    best: set[tuple[int, int]] = set()
    for start in occupied:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        comp = [start]
        while stack:
            ca, cb = stack.pop()
            for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = (ca + da, cb + db)
                if nb in occupied and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
                    comp.append(nb)
        if len(comp) > len(best):
            best = set(comp)
    keep = np.array([(int(a), int(b)) in best for a, b in ij], dtype=bool)
    return keep if keep.any() else np.ones(n, dtype=bool)


def min_area_rectangle(uv: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Minimum-area enclosing rectangle of 2-D points (rotating calipers).

    The returned axes lie along an actual edge of the point cloud's convex
    hull, so the box is tight against the table edge instead of skewed by a
    PCA frame. Returns `(center (2,), axis_u (2,) unit, half_extents (2,))`.
    """
    uv = np.asarray(uv, dtype=np.float64)
    if len(uv) < 3:
        lo, hi = uv.min(axis=0), uv.max(axis=0)
        return 0.5 * (lo + hi), np.array([1.0, 0.0]), 0.5 * (hi - lo)
    try:
        from scipy.spatial import ConvexHull

        hull = uv[ConvexHull(uv).vertices]
    except Exception:
        lo, hi = uv.min(axis=0), uv.max(axis=0)
        return 0.5 * (lo + hi), np.array([1.0, 0.0]), 0.5 * (hi - lo)

    best = None
    m = len(hull)
    for i in range(m):
        edge = hull[(i + 1) % m] - hull[i]
        length = float(np.linalg.norm(edge))
        if length < EPS:
            continue
        ex = edge / length
        ey = np.array([-ex[1], ex[0]])
        px = uv @ ex
        py = uv @ ey
        area = (px.max() - px.min()) * (py.max() - py.min())
        if best is None or area < best[0]:
            best = (area, ex, ey, px.min(), px.max(), py.min(), py.max())
    _, ex, ey, xlo, xhi, ylo, yhi = best
    center = 0.5 * (xlo + xhi) * ex + 0.5 * (ylo + yhi) * ey
    half = np.array([0.5 * (xhi - xlo), 0.5 * (yhi - ylo)])
    # Make axis_u the longer side for a stable, readable orientation.
    if half[1] > half[0]:
        ex, ey = ey, -ex
        half = half[::-1].copy()
    return center, ex, half


def min_area_horizontal_box(
    points_flat: np.ndarray,
    support_z: float,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    """Tight, cleanly-oriented horizontal collider box for a support cloud.

    Prunes disconnected strays, fits the minimum-area oriented rectangle, and
    snaps the orientation to the world axes when it is nearly axis-aligned.
    """
    uv = np.asarray(points_flat, dtype=np.float64)[:, :2]
    if len(uv) < int(args.min_band_points):
        return None
    if bool(args.footprint_grid_filter):
        keep = largest_grid_component(
            uv, float(args.footprint_grid_size), int(args.footprint_grid_min_cell_points)
        )
        if keep.sum() >= int(args.min_band_points):
            uv = uv[keep]

    center2, axis_u, half = min_area_rectangle(uv)

    # Snap orientation to world axes when the rectangle is nearly aligned.
    theta = math.atan2(axis_u[1], axis_u[0])
    theta_mod = ((theta + math.pi / 4.0) % (math.pi / 2.0)) - math.pi / 4.0
    if abs(math.degrees(theta_mod)) <= float(args.footprint_axis_snap_deg):
        axis_u = np.array([1.0, 0.0])
        lo, hi = uv.min(axis=0), uv.max(axis=0)
        center2 = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)

    half = half - float(args.shrink) + max(0.0, float(args.footprint_padding))
    extents = 2.0 * half
    if np.any(extents < float(args.min_side)):
        return None

    u3 = np.array([axis_u[0], axis_u[1], 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    v3 = np.cross(normal, u3)
    plane_center = np.array([center2[0], center2[1], support_z])
    center = plane_center + normal * (float(args.clearance) + 0.5 * float(args.thickness))
    rot = np.column_stack((u3, v3, normal))
    quat = matrix_to_quat_wxyz(rot)
    size = np.array([extents[0], extents[1], float(args.thickness)])
    box = {
        "center": center,
        "quat_wxyz": quat,
        "size": size,
        "basis_u": u3,
        "basis_v": v3,
        "extents": extents,
        "area": float(extents[0] * extents[1]),
    }
    return {**box, "support_boxes": [dict(box)], "footprint_stats": {"footprint_minrect": True}}


def build_region_candidate(
    region_points: np.ndarray,
    region_planarity: np.ndarray,
    region_id: int,
    slab: dict[str, Any],
    bounds: tuple[float, float, float, float, float, float],
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    """Fit, snap-to-horizontal, and box a single planar region."""
    region_points = np.asarray(region_points, dtype=np.float64)
    if len(region_points) < int(args.min_band_points):
        return None
    fit = robust_tls_plane(
        region_points,
        region_planarity,
        inlier_tol=float(args.plane_inlier_tol),
        max_tilt_rad=math.radians(float(args.max_tilt_deg)),
        iters=int(args.tls_iters),
    )
    if fit is None:
        return None
    inlier_mask = fit["inlier_mask"]
    inliers = region_points[inlier_mask]
    inlier_planarity = np.asarray(region_planarity, dtype=np.float64)[inlier_mask]
    if len(inliers) < int(args.min_band_points):
        return None

    # Snap to an exactly horizontal +Z surface at the robust median height.
    support_z = weighted_median(inliers[:, 2], inlier_planarity)
    flattened = inliers.copy()
    flattened[:, 2] = support_z

    if args.footprint_mode == "minrect":
        box = min_area_horizontal_box(flattened, support_z, args)
    else:
        box = oriented_box_from_plane(
            flattened,
            normal=np.array([0.0, 0.0, 1.0], dtype=np.float64),
            offset=-support_z,
            shrink=float(args.shrink),
            footprint_trim_quantile=float(args.footprint_trim_quantile),
            footprint_padding=float(args.footprint_padding),
            footprint_grid_filter=bool(args.footprint_grid_filter),
            footprint_grid_size=float(args.footprint_grid_size),
            footprint_grid_min_cell_points=int(args.footprint_grid_min_cell_points),
            thickness=float(args.thickness),
            clearance=float(args.clearance),
            min_side=float(args.min_side),
            footprint_inner_rectangle=(args.footprint_mode == "inner-rectangle"),
            footprint_inner_support_radius=args.footprint_support_radius,
            footprint_inner_erosion_cells=int(args.footprint_erosion_cells),
            footprint_points=flattened,
            footprint_multi_rectangles=False,
        )
    if box is None:
        return None

    signed = inliers[:, 2] - support_z
    reachable = workspace_mask(inliers, bounds)
    return {
        "cluster_id": int(region_id),
        "cluster_count": int(len(region_points)),
        "support_points": inliers,
        "support_z": float(support_z),
        "band_count": int(len(inliers)),
        "band_ratio": float(len(inliers) / max(len(region_points), 1)),
        "z_span": float(np.ptp(inliers[:, 2])),
        "rms": float(np.sqrt(np.mean(np.square(signed)))) if len(signed) else 0.0,
        "tilt_deg": float(math.degrees(fit["tilt_rad"])),
        "level_z": float(slab["z"]),
        "mean_planarity": float(np.mean(inlier_planarity)) if len(inlier_planarity) else 0.0,
        "reachability": float(reachable.mean()) if len(reachable) else 0.0,
        "slab_strength": float(slab["strength"]),
        "box": box,
    }


def score_region_candidate(
    raw: dict[str, Any],
    *,
    max_area: float,
    args: argparse.Namespace,
) -> float:
    """Score a region without over-weighting raw point count.

    Point count enters only indirectly through area and slab strength; the
    dominant signals are how flat the surface is, how large and reachable its
    footprint is, and how cleanly it stands out as a height mode.
    """
    w = args.score_weights
    planarity_score = float(raw["mean_planarity"])
    area_ref = max(float(args.level_min_area), float(max_area), EPS)
    area_score = float(np.clip(float(raw["box"]["area"]) / area_ref, 0.0, 1.0))
    reach_score = float(raw["reachability"])
    slab_score = float(np.clip(raw["slab_strength"], 0.0, 1.0))
    tilt_score = max(0.0, 1.0 - float(raw["tilt_deg"]) / max(float(args.max_tilt_deg), EPS))
    return float(
        w[0] * planarity_score
        + w[1] * area_score
        + w[2] * reach_score
        + w[3] * slab_score
        + w[4] * tilt_score
    )


def compute_plane_fit_candidates(
    points: np.ndarray,
    normals: np.ndarray,
    bounds: tuple[float, float, float, float, float, float],
    args: argparse.Namespace,
    reference_mask: np.ndarray | None = None,
) -> tuple[list[PlaneCandidate], dict[str, Any], np.ndarray]:
    """Create ranked candidates via robust planar-region segmentation.

    `reference_mask` (e.g. camera-frustum-visible points) does not gate the
    fit: the tabletop is fitted from the full mesh within the workspace. The
    reference is used only to identify which horizontal slab is the operation
    surface, so the box can grow past what the camera saw.
    """
    pca_normals, planarity = compute_pca_normals(points, int(args.pca_k), normals)
    max_tilt_rad = math.radians(float(args.max_tilt_deg))
    reachable = workspace_mask(points, bounds)
    upward = upward_pca_mask(pca_normals, max_tilt_rad, bool(args.normal_abs))
    planar = planarity >= float(args.min_planarity)
    support_mask = reachable & upward & planar

    stats: dict[str, Any] = {
        "sample_count": int(len(points)),
        "reachable_count": int(reachable.sum()),
        "upward_count": int(upward.sum()),
        "planar_count": int(planar.sum()),
        "support_point_count": int(support_mask.sum()),
        "reference_support_count": (
            int((support_mask & reference_mask).sum()) if reference_mask is not None else 0
        ),
        "slab_count": 0,
        "cluster_count": 0,
    }
    support_points = points[support_mask]
    support_normals = pca_normals[support_mask]
    support_planarity = planarity[support_mask]
    support_reference = (
        reference_mask[support_mask]
        if reference_mask is not None
        else np.zeros(len(support_points), dtype=bool)
    )
    if len(support_points) == 0:
        return [], stats, pca_normals

    height_window = float(args.height_window)
    band_half = float(args.level_band_scale) * height_window
    slabs = detect_height_slabs(
        support_points[:, 2],
        support_planarity,
        bin_width=float(args.slab_bin_width),
        min_separation=max(height_window, float(args.plane_inlier_tol) * 2.0),
        rel_floor=float(args.slab_rel_floor),
        band_half=band_half,
    )
    stats["slab_count"] = int(len(slabs))
    stats["slabs"] = [
        {"z": float(s["z"]), "strength": float(s["strength"]),
         "lo": float(s["lo"]), "hi": float(s["hi"])}
        for s in slabs
    ]

    normal_cos = math.cos(math.radians(float(args.region_normal_deg)))
    raw_candidates: list[dict[str, Any]] = []
    grown_region_total = 0
    for slab_id, slab in enumerate(slabs):
        band = (support_points[:, 2] >= slab["lo"]) & (support_points[:, 2] <= slab["hi"])
        band_pts = support_points[band]
        band_norm = support_normals[band]
        band_plan = support_planarity[band]
        if len(band_pts) < int(args.min_region_points):
            continue
        regions = grow_planar_regions(
            band_pts,
            band_norm,
            radius=float(args.region_radius),
            min_points=int(args.min_region_points),
            normal_cos=normal_cos,
        )
        if not regions:
            # Fallback: ignore the normal-consistency edge test.
            regions = connected_components_radius(
                band_pts,
                radius=float(args.region_radius),
                min_points=int(args.min_region_points),
                z_thresh=None,
            )
        grown_region_total += len(regions)
        if not regions:
            continue
        # All grown regions in one slab belong to the same horizontal surface
        # (same height band); merge them so a counter that fragmented around
        # objects is not split. Spatially disconnected blobs are still pruned
        # downstream by `oriented_box_from_plane`'s footprint grid filter.
        merged_idx = np.concatenate(regions)
        raw = build_region_candidate(
            band_pts[merged_idx],
            band_plan[merged_idx],
            region_id=slab_id,
            slab=slab,
            bounds=bounds,
            args=args,
        )
        if raw is not None:
            raw["grown_region_count"] = int(len(regions))
            # How many reference (camera-visible) support points fall on this
            # slab band -- used only to pick the operation surface, not to fit.
            raw["reference_count"] = int(support_reference[band].sum())
            raw_candidates.append(raw)

    stats["grown_region_count"] = int(grown_region_total)
    stats["cluster_count"] = int(len(raw_candidates))
    stats["valid_cluster_count"] = int(len(raw_candidates))
    if not raw_candidates:
        return [], stats, pca_normals

    max_area = max(float(raw["box"]["area"]) for raw in raw_candidates)
    candidates: list[PlaneCandidate] = []
    ranking: list[dict[str, Any]] = []
    for raw in raw_candidates:
        box = raw["box"]
        score = score_region_candidate(raw, max_area=max_area, args=args)
        ranking.append(
            {
                "region_id": int(raw["cluster_id"]),
                "score": float(score),
                "region_count": int(raw["cluster_count"]),
                "inlier_count": int(raw["band_count"]),
                "band_ratio": float(raw["band_ratio"]),
                "area": float(box["area"]),
                "z_span": float(raw["z_span"]),
                "support_z": float(raw["support_z"]),
                "tilt_deg": float(raw["tilt_deg"]),
                "mean_planarity": float(raw["mean_planarity"]),
                "reachability": float(raw["reachability"]),
                "slab_strength": float(raw["slab_strength"]),
                "reference_count": int(raw.get("reference_count", 0)),
            }
        )
        candidates.append(
            PlaneCandidate(
                index=0,
                cluster_id=int(raw["cluster_id"]),
                score=float(score),
                area=float(box["area"]),
                rms_residual=float(raw["rms"]),
                max_abs_residual=float(
                    np.max(np.abs(raw["support_points"][:, 2] - raw["support_z"]))
                ),
                inlier_count=int(raw["band_count"]),
                seed_inlier_count=int(raw["band_count"]),
                cluster_count=int(raw["cluster_count"]),
                z_band_count=int(raw["band_count"]),
                z_band_center=float(raw["support_z"]),
                z_band_width=float(args.height_window),
                z_span=float(raw["z_span"]),
                reachability=float(raw["reachability"]),
                selection_distance=0.0,
                normal_z=1.0,
                center=box["center"].tolist(),
                quat_wxyz=box["quat_wxyz"].tolist(),
                size=box["size"].tolist(),
                plane_normal=[0.0, 0.0, 1.0],
                plane_offset=-float(raw["support_z"]),
                basis_u=box["basis_u"].tolist(),
                basis_v=box["basis_v"].tolist(),
                extents=box["extents"].tolist(),
                support_boxes=[
                    {
                        "center": support_box["center"].tolist(),
                        "quat_wxyz": support_box["quat_wxyz"].tolist(),
                        "size": support_box["size"].tolist(),
                        "basis_u": support_box["basis_u"].tolist(),
                        "basis_v": support_box["basis_v"].tolist(),
                        "extents": support_box["extents"].tolist(),
                        "area": float(support_box["area"]),
                    }
                    for support_box in box["support_boxes"]
                ],
            )
        )

    candidates.sort(key=lambda candidate: candidate.score, reverse=True)

    # Frustum reference: the operation surface is the slab carrying the most
    # camera-visible support points. Promote that candidate to the front so it
    # is selected, while the rest stay as ranked alternatives.
    target_slab_id: int | None = None
    if reference_mask is not None:
        ref_counts = {int(raw["cluster_id"]): int(raw.get("reference_count", 0))
                      for raw in raw_candidates}
        if ref_counts and max(ref_counts.values()) > 0:
            target_slab_id = max(ref_counts, key=ref_counts.get)
            candidates.sort(key=lambda c: (c.cluster_id != target_slab_id, -c.score))
    stats["target_slab_id"] = target_slab_id

    for index, candidate in enumerate(candidates):
        candidate.index = int(index)
    ranking.sort(key=lambda item: (item["region_id"] != target_slab_id, -item["score"]))
    stats["region_ranking"] = ranking
    return candidates, stats, pca_normals


def build_descriptor(
    args: argparse.Namespace,
    candidates: list[PlaneCandidate],
    stats: dict[str, Any],
    transform_meta: dict[str, Any],
    bounds: tuple[float, float, float, float, float, float],
    *,
    collider_glb: str | Path,
    pose_path: str | Path,
    scale_path: str | Path,
    intrinsics: str | Path | None,
    extrinsics: str | Path | None,
) -> dict[str, Any]:
    selected_index = int(args.select_index) if args.select_index is not None and candidates else 0
    if candidates and not (0 <= selected_index < len(candidates)):
        raise ValueError(
            f"--select-index {selected_index} is out of range for {len(candidates)} candidates."
        )
    return {
        "version": 1,
        "source": {
            "collider_glb": os.path.abspath(collider_glb),
            "result": os.path.abspath(pose_path),
            "scene_scale": os.path.abspath(scale_path),
            "robot": args.robot,
        },
        "transform": transform_meta,
        "parameters": {
            "fit_mode": "plane_fit",
            "mesh_up": list(args.mesh_up),
            "mesh_front": list(args.mesh_front),
            "camera_frustum_crop_enabled": bool(args.camera_frustum_crop),
            "camera_intrinsics": os.path.abspath(intrinsics) if intrinsics else None,
            "camera_extrinsics": os.path.abspath(extrinsics) if extrinsics else None,
            "camera_near_far": [float(args.camera_near), float(args.camera_far)],
            "camera_truncate_wh": [float(args.camera_truncate_w), float(args.camera_truncate_h)],
            "camera_occlusion_cell_px": int(args.camera_occlusion_cell_px),
            "workspace_bounds": list(bounds),
            "coverage": str(args.coverage),
            "workspace_crop_enabled": not bool(args.no_workspace_crop),
            "auto_workspace_crop_enabled": bool(args.auto_workspace_crop),
            "auto_workspace_crop_stats": dict(getattr(args, "auto_crop_stats", {})),
            "selected_candidate_index": selected_index if candidates else None,
            "num_samples": int(args.num_samples),
            "normal_thresh": float(args.normal_thresh),
            "normal_abs": bool(args.normal_abs),
            "pca_k": int(args.pca_k),
            "min_planarity": float(args.min_planarity),
            "slab_bin_width": float(args.slab_bin_width),
            "slab_rel_floor": float(args.slab_rel_floor),
            "region_radius": float(args.region_radius),
            "region_normal_deg": float(args.region_normal_deg),
            "min_region_points": int(args.min_region_points),
            "tls_iters": int(args.tls_iters),
            "score_weights": [float(x) for x in args.score_weights],
            "height_window": float(args.height_window),
            "min_band_points": int(args.min_band_points),
            "plane_inlier_tol": float(args.plane_inlier_tol),
            "max_tilt_deg": float(args.max_tilt_deg),
            "level_band_scale": float(args.level_band_scale),
            "level_min_area": float(args.level_min_area),
            "footprint_mode": str(args.footprint_mode),
            "footprint_axis_snap_deg": float(args.footprint_axis_snap_deg),
            "footprint_grid_filter": bool(args.footprint_grid_filter),
            "footprint_grid_size": float(args.footprint_grid_size),
            "footprint_grid_min_cell_points": int(args.footprint_grid_min_cell_points),
            "footprint_support_radius": (
                float(args.footprint_support_radius) if args.footprint_support_radius is not None else None
            ),
            "footprint_erosion_cells": int(args.footprint_erosion_cells),
            "footprint_trim_quantile": float(args.footprint_trim_quantile),
            "footprint_padding": float(args.footprint_padding),
            "shrink": float(args.shrink),
            "thickness": float(args.thickness),
            "clearance": float(args.clearance),
            "min_side": float(args.min_side),
            "seed": int(args.seed),
        },
        "stats": stats,
        "selected": asdict(candidates[selected_index]) if candidates else None,
        "candidates": [asdict(candidate) for candidate in candidates],
    }


# ---------------------------------------------------------------------------
# Plain function + step
# ---------------------------------------------------------------------------

def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Plane-fit options. Defaults are the ``plane_fitting_plane_fit.sh`` production values."""
    parser.add_argument("--camera-frustum-crop", action=argparse.BooleanOptionalAction, default=True,
                        help="Use the camera-visible samples to pick the operation slab (never crops the fit).")
    parser.add_argument("--camera-near", type=float, default=0.01)
    parser.add_argument("--camera-far", type=float, default=50.0)
    parser.add_argument("--camera-truncate-w", type=float, default=0.02)
    parser.add_argument("--camera-truncate-h", type=float, default=0.02)
    parser.add_argument("--camera-occlusion-cell-px", type=int, default=8)
    parser.add_argument("--camera-depth-tol-rel", type=float, default=0.02)
    parser.add_argument("--camera-depth-tol-abs", type=float, default=0.02)
    parser.add_argument("--mesh-up", type=float, nargs=3, default=DEFAULT_GLB_UP)
    parser.add_argument("--mesh-front", type=float, nargs=3, default=DEFAULT_GLB_FRONT)
    parser.add_argument("--robot", choices=sorted(DEFAULT_WORKSPACES.keys()), default=None,
                        help="Workspace preset (default: the scene's robot in scenes.yaml).")
    parser.add_argument("--workspace-bounds", type=float, nargs=6, default=None)
    parser.add_argument("--no-workspace-crop", action="store_true")
    parser.add_argument("--auto-workspace-crop", action="store_true")
    parser.add_argument("--coverage", choices=["workspace", "auto-crop"], default="workspace",
                        help="'workspace': crop to the robot's fixed reachable workspace bounds "
                             "(box covers at least the reachable area); 'auto-crop': auto-crop "
                             "the table cluster from the full mesh (tighter, may shrink).")
    parser.add_argument("--auto-crop-cluster-radius", type=float, default=0.12)
    parser.add_argument("--auto-crop-min-points", type=int, default=50)
    parser.add_argument("--auto-crop-margin", type=float, default=0.08)
    parser.add_argument("--auto-crop-target", type=float, nargs=3, default=None)
    parser.add_argument("--auto-crop-target-falloff", type=float, default=1.0)
    parser.add_argument("--auto-crop-z-span-falloff", type=float, default=0.35)
    parser.add_argument("--select-index", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=400000)
    parser.add_argument("--normal-thresh", type=float, default=0.90)
    parser.add_argument("--normal-abs", action=argparse.BooleanOptionalAction, default=True)
    # Robust planar-region segmentation controls.
    parser.add_argument("--pca-k", type=int, default=24,
                        help="Neighbor count for the local-PCA normal/planarity estimate.")
    parser.add_argument("--min-planarity", type=float, default=0.25,
                        help="Drop points whose local neighborhood is less planar than this.")
    parser.add_argument("--slab-bin-width", type=float, default=0.01,
                        help="Bin width (m) of the planarity-weighted height histogram.")
    parser.add_argument("--slab-rel-floor", type=float, default=0.15,
                        help="Keep height-histogram peaks at least this fraction of the max peak.")
    parser.add_argument("--region-radius", type=float, default=0.10,
                        help="Neighbor radius (m) for normal-consistent region growing.")
    parser.add_argument("--region-normal-deg", type=float, default=20.0,
                        help="Max normal disagreement (deg) for a region-growing edge.")
    parser.add_argument("--min-region-points", type=int, default=40,
                        help="Discard grown regions smaller than this.")
    parser.add_argument("--tls-iters", type=int, default=10,
                        help="IRLS iterations for the Tukey-biweight TLS plane fit.")
    parser.add_argument("--score-weights", type=float, nargs=5,
                        default=[0.30, 0.25, 0.20, 0.15, 0.10],
                        help="Score weights: planarity area reachability slab tilt.")
    parser.add_argument("--height-window", type=float, default=0.03)
    parser.add_argument("--min-band-points", type=int, default=30)
    parser.add_argument("--plane-inlier-tol", type=float, default=0.02,
                        help="Max orthogonal distance (m) for a point to count as a plane inlier.")
    parser.add_argument("--max-tilt-deg", type=float, default=15.0,
                        help="Reject fitted planes tilted more than this from horizontal.")
    parser.add_argument("--level-band-scale", type=float, default=2.0,
                        help="Slab band half-width cap, in height-window units.")
    parser.add_argument("--level-min-area", type=float, default=0.02,
                        help="Reference footprint area (m^2) used by the area score term.")
    parser.add_argument("--footprint-mode", choices=["minrect", "extent", "inner-rectangle"],
                        default="minrect",
                        help="'minrect': tight minimum-area oriented rectangle (world-axis "
                             "snapped) around the connected support cluster; 'extent': trimmed "
                             "min/max box; 'inner-rectangle': conservative largest inscribed rectangle.")
    parser.add_argument("--footprint-axis-snap-deg", type=float, default=8.0,
                        help="Snap the minrect box to world X/Y axes when within this many "
                             "degrees of axis-aligned.")
    parser.add_argument("--footprint-grid-filter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--footprint-grid-size", type=float, default=0.05)
    parser.add_argument("--footprint-grid-min-cell-points", type=int, default=1)
    parser.add_argument("--footprint-support-radius", type=float, default=None)
    parser.add_argument("--footprint-erosion-cells", type=int, default=1)
    parser.add_argument("--footprint-trim-quantile", type=float, default=0.0,
                        help="Clip this fraction of the most extreme footprint points along each axis.")
    parser.add_argument("--footprint-padding", type=float, default=0.0,
                        help="Inward (negative) / outward padding applied to the footprint rectangle, in meters.")
    parser.add_argument("--shrink", type=float, default=0.0)
    parser.add_argument("--thickness", type=float, default=0.01)
    parser.add_argument("--clearance", type=float, default=0.003)
    parser.add_argument("--min-side", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--viz-ply", action="store_true",
                        help="Also write the colored point/plane PLY (surface/operation_surface_plane_fit.ply).")
    parser.add_argument("--viz-max-points", type=int, default=400000)
    parser.add_argument("--viz-max-candidates", type=int, default=12)
    parser.add_argument("--viz-selected-only", action="store_true")
    parser.add_argument("--viz-plane-lift", type=float, default=0.004)
    parser.add_argument("--viz-selected-point-thresh", type=float, default=None)
    parser.add_argument("--viz-selected-point-padding", type=float, default=0.03)
    parser.add_argument("--viewer", action="store_true", help="Serve a viser view of the fit and block until Ctrl-C.")
    parser.add_argument("--viser-port", type=int, default=8081)
    parser.add_argument("--viser-point-size", type=float, default=0.006)
    parser.add_argument("--viser-plane-opacity", type=float, default=0.45)
    parser.add_argument("--viser-frame-length", type=float, default=0.25)
    parser.add_argument("--viser-full-mesh", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--viser-mesh-opacity", type=float, default=0.16)
    parser.add_argument("--viser-mesh-max-faces", type=int, default=150000)


def default_options(**overrides: Any) -> argparse.Namespace:
    """The step's default (production) options, with ``overrides`` applied."""
    parser = argparse.ArgumentParser(add_help=False)
    add_arguments(parser)
    args = parser.parse_args([])
    for key, value in overrides.items():
        if not hasattr(args, key):
            raise TypeError(f"unknown plane-fit option {key!r}")
        setattr(args, key, value)
    return args


def fit_plane(
    collider_glb: str | Path,
    pose_path: str | Path,
    scale_path: str | Path,
    *,
    intrinsics: str | Path | None = None,
    extrinsics: str | Path | None = None,
    output_json: str | Path | None = None,
    viz_ply: str | Path | None = None,
    options: argparse.Namespace | None = None,
) -> dict[str, Any]:
    """Fit the operation surface; returns the descriptor (and writes it when ``output_json`` is set)."""
    # Work on a copy: the CLI records the caller's namespace in the manifest.
    args = argparse.Namespace(**vars(options if options is not None else default_options()))
    if args.robot is None:
        args.robot = "franka"
    # `auto_workspace_bounds()` is reused from the original script and still
    # reads the old generic clustering name.
    args.cluster_z_thresh = None
    # The visualization helpers are reused from the original script; they only
    # need these values to decide which selected-plane points to color yellow.
    args.inlier_thresh = float(args.plane_inlier_tol)
    args.expand_inlier_thresh = None
    args.extrinsics = str(extrinsics) if extrinsics is not None else None
    requested_bounds = (
        tuple(args.workspace_bounds)
        if args.workspace_bounds is not None
        else DEFAULT_WORKSPACES[args.robot]
    )
    collider_path = os.path.abspath(collider_glb)
    for label, path in (
        ("Collider mesh", collider_glb),
        ("Pose result", pose_path),
        ("Scene scale file", scale_path),
        ("Camera extrinsics file", extrinsics),
        ("Camera intrinsics file", intrinsics),
    ):
        if path is not None and not os.path.exists(path):
            raise FileNotFoundError(f"{label} not found: {path}")
    if args.camera_frustum_crop and (intrinsics is None or extrinsics is None):
        raise ValueError("--camera-frustum-crop requires both intrinsics and extrinsics.")

    rng = np.random.default_rng(args.seed)
    # `sample_surface_world` -> trimesh `mesh.sample` draws from numpy's global
    # RNG, not the passed Generator; seed it so a scene fits reproducibly.
    np.random.seed(int(args.seed))
    scene_scale = load_scene_scale(scale_path)
    world_pos, world_rot, transform_meta = collider_world_transform(pose_path, scene_scale)
    mesh_import_rot = align_up_and_front(
        np.asarray(args.mesh_up, dtype=np.float64), np.asarray(args.mesh_front, dtype=np.float64)
    )
    transform_meta["mesh_up"] = list(args.mesh_up)
    transform_meta["mesh_front"] = list(args.mesh_front)
    transform_meta["mesh_import_quat_wxyz"] = matrix_to_quat_wxyz(mesh_import_rot).tolist()
    args.collider_root_pos = transform_meta["collider_pos"]
    args.collider_root_quat_wxyz = transform_meta["collider_quat_wxyz"]

    print(f"[load] collider_glb={collider_path}")
    print("[mode] fit_mode=plane_fit")
    print(f"[lineage] placement={PLACEMENT_LINEAGE}")
    print(f"[scale] scene_scale={scene_scale:.6f} path={scale_path}")
    print(f"[pose] collider_pos={world_pos.tolist()} adjustment={SCALE_OPT_ADJUSTMENT.tolist()}")
    if extrinsics:
        cam_pos, cam_quat = load_camera_pose_from_extrinsics(extrinsics)
        print(f"[camera] pos={cam_pos.tolist()} quat_wxyz={cam_quat.tolist()}")

    print(f"[sample] drawing {args.num_samples} surface samples")
    points, normals = sample_surface_world(
        mesh_path=collider_path,
        n_samples=args.num_samples,
        scene_scale=scene_scale,
        world_pos=world_pos,
        world_rot=world_rot,
        mesh_import_rot=mesh_import_rot,
        rng=rng,
    )
    # The frustum is a *reference* only: it does not crop the points used to
    # fit. The full mesh is kept so the tabletop box can grow past what the
    # camera saw, out to the robot-reachable area.
    reference_mask = None
    if args.camera_frustum_crop:
        k, width, height, t_world_to_camera, _, _, _ = load_camera_model(intrinsics, extrinsics)
        camera_mask, frustum_mask = camera_visible_surface_mask(
            points, k, width, height, t_world_to_camera, args
        )
        kept = int(camera_mask.sum())
        print(
            f"[camera] frustum reference: {kept}/{len(points)} samples visible "
            f"(frustum={int(frustum_mask.sum())}, occluded={int((frustum_mask & ~camera_mask).sum())}); "
            "used to pick the operation slab, not to crop the fit"
        )
        if kept >= max(50, int(0.005 * len(points))):
            reference_mask = camera_mask

    sample_bounds = point_bounds(points)
    upward_bounds = point_bounds(points[upward_normal_mask(normals, args)])
    auto_crop_stats: dict[str, Any] = {}
    auto_bounds = None
    if args.coverage == "auto-crop" and args.auto_workspace_crop:
        auto_bounds, auto_crop_stats = auto_workspace_bounds(points, normals, requested_bounds, args)
        if auto_bounds is None:
            print("[bounds] auto workspace crop failed; falling back to requested workspace")
        else:
            print(f"[bounds] auto crop {format_bounds(auto_bounds)}")
    args.auto_crop_stats = auto_crop_stats
    if auto_bounds is not None:
        bounds = auto_bounds
    elif args.no_workspace_crop and sample_bounds:
        bounds = expanded_bounds(sample_bounds, margin=1e-6)
    else:
        bounds = requested_bounds
    print(f"[coverage] mode={args.coverage}")
    print(f"[bounds] sample    {format_bounds(sample_bounds)}")
    print(f"[bounds] upward    {format_bounds(upward_bounds)}")
    print(f"[bounds] workspace {format_bounds(bounds)}")

    candidates, stats, pca_normals = compute_plane_fit_candidates(
        points, normals, bounds, args, reference_mask=reference_mask
    )
    # Color the viewer by the smoothed PCA normals the fitter actually used,
    # not the facet-quantized mesh normals, so the box's real support shows.
    viz_normals = pca_normals
    print(
        "[filter] "
        f"sample={stats.get('sample_count', 0)} reachable={stats.get('reachable_count', 0)} "
        f"upward={stats.get('upward_count', 0)} planar={stats.get('planar_count', 0)} "
        f"support={stats.get('support_point_count', 0)} "
        f"reference={stats.get('reference_support_count', 0)} "
        f"slabs={stats.get('slab_count', 0)} "
        f"grown_regions={stats.get('grown_region_count', 0)} "
        f"candidates={stats.get('cluster_count', 0)} valid={stats.get('valid_cluster_count', 0)}"
    )
    if stats.get("target_slab_id") is not None:
        print(f"[reference] operation slab = id {stats['target_slab_id']} (most camera-visible points)")
    for slab in stats.get("slabs", []):
        print(f"  [slab] z={slab['z']:.4f} strength={slab['strength']:.3f} "
              f"band=[{slab['lo']:.4f},{slab['hi']:.4f}]")
    for item in stats.get("region_ranking", []):
        print(
            f"  [region] id={item['region_id']:02d} score={item['score']:.3f} "
            f"count={item['region_count']} inliers={item['inlier_count']} "
            f"area={item['area']:.4f} z={item['support_z']:.4f} "
            f"tilt={item['tilt_deg']:.2f}deg planarity={item['mean_planarity']:.3f} "
            f"reach={item['reachability']:.3f} slab={item['slab_strength']:.3f} "
            f"ref={item.get('reference_count', 0)}"
        )
    for candidate in candidates[:8]:
        print(
            f"  #{candidate.index:02d} region={candidate.cluster_id:02d} score={candidate.score:.3f} "
            f"area={candidate.area:.4f} inliers={candidate.inlier_count} "
            f"z={candidate.z_band_center:.4f} size={np.round(candidate.size, 4).tolist()}"
        )
    if stats.get("support_point_count", 0) == 0:
        suggested = expanded_bounds(upward_bounds, margin=0.05) if upward_bounds else None
        print(
            "[hint] no support points were selected. Try lowering --min-planarity or "
            "--normal-thresh, raising --max-tilt-deg, or widening the workspace crop such as "
            f"`--workspace-bounds {bounds_args(suggested)}`."
        )
    elif not candidates:
        print(
            "[hint] support points were found, but no region produced a valid tabletop. "
            "Try lowering --min-region-points / --min-band-points, raising --region-radius, "
            "or raising --max-tilt-deg."
        )

    descriptor = build_descriptor(
        args,
        candidates,
        stats,
        transform_meta,
        bounds,
        collider_glb=collider_path,
        pose_path=pose_path,
        scale_path=scale_path,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
    )
    if output_json is not None:
        write_json(output_json, descriptor)
    if candidates:
        selected_index = int(args.select_index) if args.select_index is not None else 0
        selected = candidates[selected_index]
        print(
            f"[write] selected region #{selected.cluster_id:02d} surface saved to {output_json} "
            f"center={np.round(selected.center, 4).tolist()}"
        )
    else:
        print(f"[write] no valid surface found; diagnostics saved to {output_json}")

    if viz_ply is not None:
        from r2s.surface.viz import write_surface_viz_ply

        drawn = write_surface_viz_ply(viz_ply, points, viz_normals, bounds, candidates, args, rng)
        print(f"[viz] wrote {viz_ply} (surfaces={drawn})")
    if args.viewer:
        from r2s.surface.viz import downsample_mesh_faces, launch_surface_viz_viewer

        mesh_viz = None
        if args.viser_full_mesh:
            mesh_viz = mesh_world_geometry(collider_path, scene_scale, world_pos, world_rot, mesh_import_rot)
            mesh_viz = downsample_mesh_faces(mesh_viz[0], mesh_viz[1], int(args.viser_mesh_max_faces), rng)
        launch_surface_viz_viewer(points, viz_normals, bounds, candidates, args, rng, mesh_viz=mesh_viz)
    return descriptor


def run(layout: SceneLayout, args: argparse.Namespace) -> dict[str, Any]:
    options = argparse.Namespace(**vars(args))
    if options.robot is None:
        options.robot = layout.config.robot
    descriptor = fit_plane(
        layout.collider_glb,
        layout.refined_pose,
        layout.refined_scale,
        intrinsics=layout.inputs.intrinsics,
        extrinsics=layout.inputs.extrinsics,
        output_json=layout.plane_fit_json,
        viz_ply=layout.plane_fit_ply if args.viz_ply else None,
        options=options,
    )
    selected = descriptor["selected"]
    if selected is None:
        raise RuntimeError(
            f"no operation surface found for {layout.scene}; diagnostics in {layout.plane_fit_json}"
        )
    return {
        "robot": options.robot,
        "candidate_count": len(descriptor["candidates"]),
        "selected_center": selected["center"],
        "selected_size": selected["size"],
        "target_slab_id": descriptor["stats"].get("target_slab_id"),
    }


STEP = Step(
    name="surface.plane",
    help="Fit the horizontal operation-surface box from the background collider",
    add_arguments=add_arguments,
    run=run,
    inputs=lambda layout, args: [
        layout.collider_glb,
        layout.refined_pose,
        layout.refined_scale,
        layout.inputs.intrinsics,
        layout.inputs.extrinsics,
    ],
    outputs=lambda layout, args: [layout.plane_fit_json] + ([layout.plane_fit_ply] if args.viz_ply else []),
)
