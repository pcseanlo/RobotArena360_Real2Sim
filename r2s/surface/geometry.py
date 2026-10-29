"""Geometry helpers shared by the support-surface steps (4-6).

Moved verbatim from the old ``operation_surface_precompute.py`` (scene sampling,
camera/workspace crops, footprint boxes and occupancy-mask utilities), plus
``selected_plane`` from ``operation_surface_render_receiver_fit.py`` and
``footprint_mask_from_world_points`` from ``gaussian_shadow.py``. Pose, scale,
camera and mesh loading live in ``r2s.common``.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from r2s.common.mesh import load_combined_mesh
from r2s.common.quat import matrix_to_quat_wxyz

DEFAULT_WORKSPACES = {
    "widowx": (0.15, 0.80, -0.45, 0.45, -0.08, 0.55),
    "franka": (0.20, 0.95, -0.55, 0.55, -0.08, 0.85),
}


@dataclass
class PlaneCandidate:
    index: int
    cluster_id: int
    score: float
    area: float
    rms_residual: float
    max_abs_residual: float
    inlier_count: int
    seed_inlier_count: int
    cluster_count: int
    z_band_count: int
    z_band_center: float
    z_band_width: float
    z_span: float
    reachability: float
    selection_distance: float
    normal_z: float
    center: list[float]
    quat_wxyz: list[float]
    size: list[float]
    plane_normal: list[float]
    plane_offset: float
    basis_u: list[float]
    basis_v: list[float]
    extents: list[float]
    support_boxes: list[dict[str, Any]]


# ------------------------------------------------------------------ camera crop
def project_to_uvz_matrix(
    points_world: np.ndarray,
    k: np.ndarray,
    t_world_to_camera: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points_world, dtype=np.float64)
    homog = np.concatenate((points, np.ones((len(points), 1), dtype=np.float64)), axis=1)
    points_camera = (np.asarray(t_world_to_camera, dtype=np.float64).reshape(4, 4) @ homog.T).T[:, :3]
    x_cam = points_camera[:, 0]
    y_cam = points_camera[:, 1]
    z_cam = points_camera[:, 2]
    z_safe = np.maximum(z_cam, 1e-9)
    u = float(k[0, 0]) * (x_cam / z_safe) + float(k[0, 2])
    v = float(k[1, 1]) * (y_cam / z_safe) + float(k[1, 2])
    return u, v, z_cam


def camera_frustum_mask(
    points_world: np.ndarray,
    k: np.ndarray,
    width: int,
    height: int,
    t_world_to_camera: np.ndarray,
    near: float,
    far: float,
    truncate_w: float,
    truncate_h: float,
    truncate_left: float | None = None,
    truncate_right: float | None = None,
    truncate_top: float | None = None,
    truncate_bottom: float | None = None,
) -> np.ndarray:
    u, v, z = project_to_uvz_matrix(points_world, k, t_world_to_camera)
    off_l = float(truncate_w if truncate_left is None else truncate_left) * float(width)
    off_r = float(truncate_w if truncate_right is None else truncate_right) * float(width)
    off_t = float(truncate_h if truncate_top is None else truncate_top) * float(height)
    off_b = float(truncate_h if truncate_bottom is None else truncate_bottom) * float(height)
    return (
        (z > float(near))
        & (z < float(far))
        & (u >= off_l)
        & (u < (float(width) - off_r))
        & (v >= off_t)
        & (v < (float(height) - off_b))
    )


def camera_visible_surface_mask(
    points_world: np.ndarray,
    k: np.ndarray,
    width: int,
    height: int,
    t_world_to_camera: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    frustum = camera_frustum_mask(
        points_world,
        k,
        width,
        height,
        t_world_to_camera,
        near=args.camera_near,
        far=args.camera_far,
        truncate_w=args.camera_truncate_w,
        truncate_h=args.camera_truncate_h,
        truncate_left=getattr(args, "camera_truncate_left", None),
        truncate_right=getattr(args, "camera_truncate_right", None),
        truncate_top=getattr(args, "camera_truncate_top", None),
        truncate_bottom=getattr(args, "camera_truncate_bottom", None),
    )
    cell_px = int(args.camera_occlusion_cell_px)
    if cell_px <= 0 or int(frustum.sum()) == 0:
        return frustum, frustum

    idx_in = np.flatnonzero(frustum)
    u, v, z = project_to_uvz_matrix(points_world[idx_in], k, t_world_to_camera)
    cu = np.floor(u / float(cell_px)).astype(np.int64)
    cv = np.floor(v / float(cell_px)).astype(np.int64)
    cell_id = cu * np.int64(10_000_003) + cv
    order = np.argsort(cell_id, kind="stable")
    cell_sorted = cell_id[order]
    z_sorted = z[order]
    boundaries = np.concatenate(([0], np.flatnonzero(np.diff(cell_sorted)) + 1, [len(cell_sorted)]))

    keep_local = np.zeros(idx_in.size, dtype=bool)
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        zs = z_sorted[start:end]
        z_min = float(zs.min())
        tol = max(float(args.camera_depth_tol_abs), float(args.camera_depth_tol_rel) * max(z_min, 1e-6))
        keep_local[order[start:end]] = zs <= z_min + tol

    surface = np.zeros_like(frustum)
    surface[idx_in[keep_local]] = True
    if int(surface.sum()) < max(20, int(0.02 * int(frustum.sum()))):
        return frustum, frustum
    return surface, frustum


# ------------------------------------------------------------------ scene sampling
def sample_surface_world(
    mesh_path: str,
    n_samples: int,
    scene_scale: float,
    world_pos: np.ndarray,
    world_rot: np.ndarray,
    mesh_import_rot: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    mesh = load_combined_mesh(mesh_path)
    if len(mesh.faces) == 0:
        raise ValueError(f"Mesh has no faces: {mesh_path}")

    points_local, face_idx = mesh.sample(n_samples, return_index=True)
    normals_local = mesh.face_normals[face_idx]

    points_import = points_local @ mesh_import_rot.T
    normals_import = normals_local @ mesh_import_rot.T
    points_world = (points_import * float(scene_scale)) @ world_rot.T + world_pos[None, :]
    normals_world = normals_import @ world_rot.T
    normals_world /= np.maximum(np.linalg.norm(normals_world, axis=1, keepdims=True), 1e-12)

    if len(points_world) > n_samples:
        idx = rng.choice(len(points_world), size=n_samples, replace=False)
        return points_world[idx], normals_world[idx]
    return points_world, normals_world


# ------------------------------------------------------------------ workspace bounds
def workspace_mask(points: np.ndarray, bounds: tuple[float, float, float, float, float, float]) -> np.ndarray:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    return (
        (points[:, 0] >= xmin)
        & (points[:, 0] <= xmax)
        & (points[:, 1] >= ymin)
        & (points[:, 1] <= ymax)
        & (points[:, 2] >= zmin)
        & (points[:, 2] <= zmax)
    )


def upward_normal_mask(normals: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    nz = normals[:, 2]
    if bool(args.normal_abs):
        return np.abs(nz) >= args.normal_thresh
    return nz >= args.normal_thresh


def point_bounds(points: np.ndarray) -> tuple[float, float, float, float, float, float] | None:
    if len(points) == 0:
        return None
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    return (
        float(mins[0]),
        float(maxs[0]),
        float(mins[1]),
        float(maxs[1]),
        float(mins[2]),
        float(maxs[2]),
    )


def expanded_bounds(
    bounds: tuple[float, float, float, float, float, float],
    margin: float,
) -> tuple[float, float, float, float, float, float]:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    return (
        xmin - margin,
        xmax + margin,
        ymin - margin,
        ymax + margin,
        zmin - margin,
        zmax + margin,
    )


def bounds_from_points(points: np.ndarray, margin: float = 0.0) -> tuple[float, float, float, float, float, float] | None:
    bounds = point_bounds(points)
    if bounds is None:
        return None
    return expanded_bounds(bounds, margin)


def bounds_center(bounds: tuple[float, float, float, float, float, float]) -> np.ndarray:
    return np.array(
        [
            0.5 * (bounds[0] + bounds[1]),
            0.5 * (bounds[2] + bounds[3]),
            0.5 * (bounds[4] + bounds[5]),
        ],
        dtype=np.float64,
    )


def bounds_size(bounds: tuple[float, float, float, float, float, float]) -> np.ndarray:
    return np.array(
        [bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]],
        dtype=np.float64,
    )


def format_bounds(bounds: tuple[float, float, float, float, float, float] | None) -> str:
    if bounds is None:
        return "None"
    return (
        f"x=[{bounds[0]:.3f}, {bounds[1]:.3f}] "
        f"y=[{bounds[2]:.3f}, {bounds[3]:.3f}] "
        f"z=[{bounds[4]:.3f}, {bounds[5]:.3f}]"
    )


def bounds_args(bounds: tuple[float, float, float, float, float, float] | None) -> str:
    if bounds is None:
        return ""
    return " ".join(f"{v:.4f}" for v in bounds)


def resolved_cluster_z_thresh(args: argparse.Namespace) -> float | None:
    if args.cluster_z_thresh is not None:
        return float(args.cluster_z_thresh)
    return None


def connected_components_radius(
    points: np.ndarray,
    radius: float,
    min_points: int,
    z_thresh: float | None = None,
) -> list[np.ndarray]:
    if len(points) == 0:
        return []
    tree = cKDTree(points)
    visited = np.zeros(len(points), dtype=bool)
    components: list[np.ndarray] = []

    for start in range(len(points)):
        if visited[start]:
            continue
        queue = [start]
        visited[start] = True
        comp = []
        while queue:
            idx = queue.pop()
            comp.append(idx)
            for nbr in tree.query_ball_point(points[idx], radius):
                if z_thresh is not None and abs(float(points[nbr, 2] - points[idx, 2])) > float(z_thresh):
                    continue
                if not visited[nbr]:
                    visited[nbr] = True
                    queue.append(nbr)
        if len(comp) >= min_points:
            components.append(np.asarray(comp, dtype=np.int64))

    components.sort(key=len, reverse=True)
    return components


def auto_workspace_bounds(
    points: np.ndarray,
    normals: np.ndarray,
    requested_bounds: tuple[float, float, float, float, float, float],
    args: argparse.Namespace,
) -> tuple[tuple[float, float, float, float, float, float] | None, dict[str, Any]]:
    upward = upward_normal_mask(normals, args)
    upward_points = points[upward]
    stats: dict[str, Any] = {
        "auto_crop_upward_count": int(len(upward_points)),
        "auto_crop_cluster_count": 0,
    }
    if len(upward_points) == 0:
        return None, stats

    clusters = connected_components_radius(
        upward_points,
        radius=float(args.auto_crop_cluster_radius),
        min_points=int(args.auto_crop_min_points),
        z_thresh=resolved_cluster_z_thresh(args),
    )
    stats["auto_crop_cluster_count"] = int(len(clusters))
    if not clusters:
        return None, stats

    target = (
        np.asarray(args.auto_crop_target, dtype=np.float64)
        if args.auto_crop_target is not None
        else bounds_center(requested_bounds)
    )
    best_score = -float("inf")
    best_bounds = None
    best_info: dict[str, Any] = {}

    for cluster_id, cluster_idx in enumerate(clusters):
        cluster_points = upward_points[cluster_idx]
        raw_bounds = bounds_from_points(cluster_points, margin=0.0)
        if raw_bounds is None:
            continue
        size = bounds_size(raw_bounds)
        if size[0] < args.min_side or size[1] < args.min_side:
            continue

        center = bounds_center(raw_bounds)
        xy_area = max(float(size[0] * size[1]), 1e-9)
        z_span = max(float(size[2]), 0.0)
        count_score = math.log1p(float(len(cluster_points)))
        area_score = math.log1p(xy_area)
        target_dist = float(np.linalg.norm(center - target))
        target_score = -target_dist / max(float(args.auto_crop_target_falloff), 1e-6)
        z_penalty = z_span / max(float(args.auto_crop_z_span_falloff), 1e-6)
        score = count_score + 0.75 * area_score + target_score - z_penalty

        if score > best_score:
            best_score = score
            best_bounds = expanded_bounds(raw_bounds, margin=float(args.auto_crop_margin))
            best_info = {
                "auto_crop_selected_cluster": int(cluster_id),
                "auto_crop_selected_count": int(len(cluster_points)),
                "auto_crop_selected_score": float(score),
                "auto_crop_selected_center": center.tolist(),
                "auto_crop_selected_raw_bounds": list(raw_bounds),
                "auto_crop_target": target.tolist(),
            }

    stats.update(best_info)
    return best_bounds, stats


# ------------------------------------------------------------------ footprints and occupancy masks
def plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z_axis = np.asarray(normal, dtype=np.float64)
    z_axis /= max(np.linalg.norm(z_axis), 1e-12)
    seed = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(seed, z_axis))) > 0.9:
        seed = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u = seed - np.dot(seed, z_axis) * z_axis
    u /= max(np.linalg.norm(u), 1e-12)
    v = np.cross(z_axis, u)
    v /= max(np.linalg.norm(v), 1e-12)
    return u, v


def largest_occupied_uv_component(
    uv: np.ndarray,
    grid_size: float,
    min_cell_points: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    stats = {
        "footprint_grid_enabled": True,
        "footprint_grid_size": float(grid_size),
        "footprint_grid_min_cell_points": int(min_cell_points),
        "footprint_grid_input_count": int(len(uv)),
        "footprint_grid_filtered_count": int(len(uv)),
        "footprint_grid_cell_count": 0,
        "footprint_grid_component_count": 0,
        "footprint_grid_used": False,
    }
    if len(uv) == 0 or grid_size <= 0.0:
        return np.ones(len(uv), dtype=bool), stats

    uv_min = uv.min(axis=0)
    cells = np.floor((uv - uv_min[None, :]) / float(grid_size)).astype(np.int64)
    cell_to_indices: dict[tuple[int, int], list[int]] = {}
    for idx, cell in enumerate(cells):
        key = (int(cell[0]), int(cell[1]))
        cell_to_indices.setdefault(key, []).append(idx)

    occupied = {
        cell: indices
        for cell, indices in cell_to_indices.items()
        if len(indices) >= max(1, int(min_cell_points))
    }
    stats["footprint_grid_cell_count"] = int(len(occupied))
    if not occupied:
        return np.ones(len(uv), dtype=bool), stats

    unvisited = set(occupied.keys())
    components: list[set[tuple[int, int]]] = []
    neighbor_offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]
    while unvisited:
        seed = unvisited.pop()
        comp = {seed}
        queue = [seed]
        while queue:
            cx, cy = queue.pop()
            for dx, dy in neighbor_offsets:
                nbr = (cx + dx, cy + dy)
                if nbr in unvisited:
                    unvisited.remove(nbr)
                    comp.add(nbr)
                    queue.append(nbr)
        components.append(comp)

    stats["footprint_grid_component_count"] = int(len(components))
    best = max(components, key=lambda comp: sum(len(occupied[cell]) for cell in comp))
    keep_indices = sorted({idx for cell in best for idx in occupied[cell]})
    if len(keep_indices) < max(8, int(min_cell_points) * 4):
        return np.ones(len(uv), dtype=bool), stats

    mask = np.zeros(len(uv), dtype=bool)
    mask[keep_indices] = True
    stats["footprint_grid_filtered_count"] = int(mask.sum())
    stats["footprint_grid_used"] = True
    return mask, stats


def largest_true_mask_component(mask: np.ndarray) -> np.ndarray:
    if mask.size == 0 or not mask.any():
        return np.zeros_like(mask, dtype=bool)

    rows, cols = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    best_cells: list[tuple[int, int]] = []
    neighbor_offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    for row in range(rows):
        for col in range(cols):
            if not mask[row, col] or visited[row, col]:
                continue
            cells: list[tuple[int, int]] = []
            queue = [(row, col)]
            visited[row, col] = True
            while queue:
                cur_row, cur_col = queue.pop()
                cells.append((cur_row, cur_col))
                for d_row, d_col in neighbor_offsets:
                    nbr_row = cur_row + d_row
                    nbr_col = cur_col + d_col
                    if (
                        0 <= nbr_row < rows
                        and 0 <= nbr_col < cols
                        and mask[nbr_row, nbr_col]
                        and not visited[nbr_row, nbr_col]
                    ):
                        visited[nbr_row, nbr_col] = True
                        queue.append((nbr_row, nbr_col))
            if len(cells) > len(best_cells):
                best_cells = cells

    component = np.zeros_like(mask, dtype=bool)
    if best_cells:
        rr, cc = zip(*best_cells)
        component[np.asarray(rr), np.asarray(cc)] = True
    return component


def erode_true_mask(mask: np.ndarray, iterations: int) -> np.ndarray:
    eroded = mask.astype(bool, copy=True)
    for _ in range(max(0, int(iterations))):
        if not eroded.any():
            break
        padded = np.pad(eroded, 1, constant_values=False)
        next_mask = np.ones_like(eroded, dtype=bool)
        for d_row in range(3):
            for d_col in range(3):
                next_mask &= padded[d_row:d_row + eroded.shape[0], d_col:d_col + eroded.shape[1]]
        eroded = next_mask
    return eroded


def largest_axis_aligned_true_rectangle(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    if mask.size == 0 or not mask.any():
        return None

    rows, cols = mask.shape
    heights = np.zeros(cols, dtype=np.int64)
    best_area = 0
    best_rect: tuple[int, int, int, int] | None = None

    for row in range(rows):
        heights = np.where(mask[row], heights + 1, 0)
        stack: list[int] = []
        for col in range(cols + 1):
            cur_height = int(heights[col]) if col < cols else 0
            while stack and cur_height < int(heights[stack[-1]]):
                height_col = stack.pop()
                height = int(heights[height_col])
                left = stack[-1] + 1 if stack else 0
                right = col - 1
                area = height * (right - left + 1)
                if area > best_area:
                    best_area = area
                    best_rect = (row - height + 1, row, left, right)
            stack.append(col)
    return best_rect


def estimate_uv_support_radius(uv: np.ndarray, grid_size: float, requested_radius: float | None) -> float:
    if requested_radius is not None and float(requested_radius) > 0.0:
        return float(requested_radius)
    if len(uv) < 2:
        return max(float(grid_size) * 2.5, float(grid_size))

    tree = cKDTree(uv)
    distances, _ = tree.query(uv, k=min(2, len(uv)))
    nearest = distances[:, -1] if distances.ndim == 2 else distances
    nearest = nearest[np.isfinite(nearest)]
    if len(nearest) == 0:
        return max(float(grid_size) * 2.5, float(grid_size))
    spacing = float(np.quantile(nearest, 0.75))
    return max(float(grid_size) * 2.5, spacing * 1.5)


def inscribed_supported_uv_bounds(
    uv: np.ndarray,
    grid_size: float,
    support_radius: float | None,
    erosion_cells: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    if len(uv) == 0 or grid_size <= 0.0:
        return None

    grid_size = float(grid_size)
    resolved_radius = estimate_uv_support_radius(uv, grid_size, support_radius)
    pad = resolved_radius + grid_size
    uv_lo = uv.min(axis=0) - pad
    uv_hi = uv.max(axis=0) + pad
    span = np.maximum(uv_hi - uv_lo, grid_size)
    n_cols = max(1, int(math.ceil(float(span[0]) / grid_size)))
    n_rows = max(1, int(math.ceil(float(span[1]) / grid_size)))

    x_centers = uv_lo[0] + (np.arange(n_cols, dtype=np.float64) + 0.5) * grid_size
    y_centers = uv_lo[1] + (np.arange(n_rows, dtype=np.float64) + 0.5) * grid_size
    xx, yy = np.meshgrid(x_centers, y_centers, indexing="xy")
    cell_centers = np.column_stack((xx.ravel(), yy.ravel()))
    distances, _ = cKDTree(uv).query(cell_centers, k=1)
    supported = distances.reshape(n_rows, n_cols) <= resolved_radius
    supported = largest_true_mask_component(supported)
    if not supported.any():
        return None

    eroded = erode_true_mask(supported, int(erosion_cells))
    rectangle_mask = eroded if eroded.any() else supported
    rect = largest_axis_aligned_true_rectangle(rectangle_mask)
    if rect is None:
        return None
    row_min, row_max, col_min, col_max = rect
    rect_uv_min = uv_lo + np.array([col_min, row_min], dtype=np.float64) * grid_size
    rect_uv_max = uv_lo + np.array([col_max + 1, row_max + 1], dtype=np.float64) * grid_size
    stats = {
        "footprint_inner_rectangle_used": True,
        "footprint_inner_grid_size": grid_size,
        "footprint_inner_support_radius": resolved_radius,
        "footprint_inner_erosion_cells": int(erosion_cells),
        "footprint_inner_supported_cell_count": int(supported.sum()),
        "footprint_inner_eroded_cell_count": int(eroded.sum()),
        "footprint_inner_rectangle_cell_count": int(
            (row_max - row_min + 1) * (col_max - col_min + 1)
        ),
    }
    return rect_uv_min, rect_uv_max, stats


def supported_uv_mask(
    uv: np.ndarray,
    grid_size: float,
    support_radius: float | None,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]] | None:
    if len(uv) == 0 or grid_size <= 0.0:
        return None

    grid_size = float(grid_size)
    resolved_radius = estimate_uv_support_radius(uv, grid_size, support_radius)
    pad = resolved_radius + grid_size
    uv_lo = uv.min(axis=0) - pad
    uv_hi = uv.max(axis=0) + pad
    span = np.maximum(uv_hi - uv_lo, grid_size)
    n_cols = max(1, int(math.ceil(float(span[0]) / grid_size)))
    n_rows = max(1, int(math.ceil(float(span[1]) / grid_size)))

    x_centers = uv_lo[0] + (np.arange(n_cols, dtype=np.float64) + 0.5) * grid_size
    y_centers = uv_lo[1] + (np.arange(n_rows, dtype=np.float64) + 0.5) * grid_size
    xx, yy = np.meshgrid(x_centers, y_centers, indexing="xy")
    cell_centers = np.column_stack((xx.ravel(), yy.ravel()))
    distances, _ = cKDTree(uv).query(cell_centers, k=1)
    supported = distances.reshape(n_rows, n_cols) <= resolved_radius
    supported = largest_true_mask_component(supported)
    if not supported.any():
        return None
    stats = {
        "footprint_support_mask_grid_size": grid_size,
        "footprint_support_mask_radius": resolved_radius,
        "footprint_support_mask_cell_count": int(supported.sum()),
    }
    return supported, uv_lo, grid_size, stats


def decompose_true_mask_into_rectangles(
    mask: np.ndarray,
    max_rectangles: int,
    min_rectangle_cells: int,
) -> list[tuple[int, int, int, int]]:
    remaining = mask.astype(bool, copy=True)
    rectangles: list[tuple[int, int, int, int]] = []
    for _ in range(max(1, int(max_rectangles))):
        rect = largest_axis_aligned_true_rectangle(remaining)
        if rect is None:
            break
        row_min, row_max, col_min, col_max = rect
        area_cells = (row_max - row_min + 1) * (col_max - col_min + 1)
        if area_cells < max(1, int(min_rectangle_cells)):
            break
        rectangles.append(rect)
        remaining[row_min:row_max + 1, col_min:col_max + 1] = False
    return rectangles


def box_from_uv_bounds(
    uv_min: np.ndarray,
    uv_max: np.ndarray,
    origin: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    normal: np.ndarray,
    shrink: float,
    thickness: float,
    clearance: float,
    min_side: float,
) -> dict[str, Any] | None:
    raw_extents = uv_max - uv_min
    shrunk_extents = raw_extents - 2.0 * float(shrink)
    if np.any(shrunk_extents < min_side):
        return None

    uv_center = 0.5 * (uv_min + uv_max)
    plane_center = origin + uv_center[0] * u + uv_center[1] * v
    center = plane_center + normal * (float(clearance) + 0.5 * float(thickness))

    rot = np.column_stack((u, v, normal))
    v_for_rot = v
    if np.linalg.det(rot) < 0.0:
        v_for_rot = -v
        rot = np.column_stack((u, v_for_rot, normal))
    quat = matrix_to_quat_wxyz(rot)
    size = np.array([shrunk_extents[0], shrunk_extents[1], thickness], dtype=np.float64)
    return {
        "center": center,
        "quat_wxyz": quat,
        "size": size,
        "basis_u": u,
        "basis_v": v_for_rot,
        "extents": shrunk_extents,
        "area": float(shrunk_extents[0] * shrunk_extents[1]),
    }


def oriented_box_from_plane(
    inlier_points: np.ndarray,
    normal: np.ndarray,
    offset: float,
    shrink: float,
    footprint_trim_quantile: float,
    footprint_padding: float,
    footprint_grid_filter: bool,
    footprint_grid_size: float,
    footprint_grid_min_cell_points: int,
    thickness: float,
    clearance: float,
    min_side: float,
    footprint_inner_rectangle: bool = False,
    footprint_inner_support_radius: float | None = None,
    footprint_inner_erosion_cells: int = 1,
    footprint_points: np.ndarray | None = None,
    footprint_multi_rectangles: bool = False,
    footprint_max_rectangles: int = 6,
    footprint_min_rectangle_cells: int = 4,
) -> dict[str, Any] | None:
    footprint_input_points = (
        np.asarray(footprint_points, dtype=np.float64)
        if footprint_points is not None and len(footprint_points) > 0
        else np.asarray(inlier_points, dtype=np.float64)
    )
    projected = (
        footprint_input_points
        - (footprint_input_points @ normal + offset)[:, None] * normal[None, :]
    )
    origin = projected.mean(axis=0)

    u0, v0 = plane_basis(normal)
    uv0 = np.column_stack(((projected - origin) @ u0, (projected - origin) @ v0))
    cov = np.cov(uv0, rowvar=False)
    if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
        return None
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]

    u = eigvecs[0, 0] * u0 + eigvecs[1, 0] * v0
    v = eigvecs[0, 1] * u0 + eigvecs[1, 1] * v0
    u /= max(np.linalg.norm(u), 1e-12)
    v = np.cross(normal, u)
    v /= max(np.linalg.norm(v), 1e-12)

    uv = np.column_stack(((projected - origin) @ u, (projected - origin) @ v))
    footprint_stats = {
        "footprint_grid_enabled": bool(footprint_grid_filter),
        "footprint_grid_used": False,
        "footprint_grid_input_count": int(len(uv)),
        "footprint_grid_filtered_count": int(len(uv)),
    }
    if footprint_grid_filter:
        keep_mask, footprint_stats = largest_occupied_uv_component(
            uv,
            grid_size=float(footprint_grid_size),
            min_cell_points=int(footprint_grid_min_cell_points),
        )
        if keep_mask.any() and int(keep_mask.sum()) < len(uv):
            uv = uv[keep_mask]
            projected = projected[keep_mask]
            origin = projected.mean(axis=0)

            # Once disconnected coplanar points have been removed, recompute
            # the in-plane PCA frame from the kept tabletop component itself.
            # Otherwise a small but distant stray component can still rotate
            # the final rectangle even though it no longer contributes to the
            # extents.
            uv0 = np.column_stack(((projected - origin) @ u0, (projected - origin) @ v0))
            cov = np.cov(uv0, rowvar=False)
            if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
                return None
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = np.argsort(eigvals)[::-1]
            eigvecs = eigvecs[:, order]

            u = eigvecs[0, 0] * u0 + eigvecs[1, 0] * v0
            u /= max(np.linalg.norm(u), 1e-12)
            v = np.cross(normal, u)
            v /= max(np.linalg.norm(v), 1e-12)
            uv = np.column_stack(((projected - origin) @ u, (projected - origin) @ v))

    support_boxes: list[dict[str, Any]] = []
    if footprint_multi_rectangles:
        support_mask_data = supported_uv_mask(
            uv,
            grid_size=float(footprint_grid_size),
            support_radius=footprint_inner_support_radius,
        )
        if support_mask_data is None:
            return None
        support_mask, support_uv_lo, support_grid_size, support_stats = support_mask_data
        rectangles = decompose_true_mask_into_rectangles(
            support_mask,
            max_rectangles=int(footprint_max_rectangles),
            min_rectangle_cells=int(footprint_min_rectangle_cells),
        )
        if not rectangles:
            return None
        for row_min, row_max, col_min, col_max in rectangles:
            rect_uv_min = support_uv_lo + np.array([col_min, row_min], dtype=np.float64) * support_grid_size
            rect_uv_max = support_uv_lo + np.array([col_max + 1, row_max + 1], dtype=np.float64) * support_grid_size
            patch = box_from_uv_bounds(
                rect_uv_min,
                rect_uv_max,
                origin=origin,
                u=u,
                v=v,
                normal=normal,
                shrink=shrink,
                thickness=thickness,
                clearance=clearance,
                min_side=min_side,
            )
            if patch is not None:
                support_boxes.append(patch)
        if not support_boxes:
            return None
        support_boxes.sort(key=lambda patch: float(patch["area"]), reverse=True)
        footprint_stats.update(support_stats)
        footprint_stats.update(
            {
                "footprint_multi_rectangles_used": True,
                "footprint_multi_rectangle_count": int(len(support_boxes)),
                "footprint_multi_rectangle_total_area": float(sum(box["area"] for box in support_boxes)),
            }
        )
        primary_box = support_boxes[0]
        return {
            **primary_box,
            "area": float(sum(box["area"] for box in support_boxes)),
            "support_boxes": support_boxes,
            "footprint_stats": footprint_stats,
        }
    elif footprint_inner_rectangle:
        inner_bounds = inscribed_supported_uv_bounds(
            uv,
            grid_size=float(footprint_grid_size),
            support_radius=footprint_inner_support_radius,
            erosion_cells=int(footprint_inner_erosion_cells),
        )
        if inner_bounds is None:
            return None
        uv_min, uv_max, inner_stats = inner_bounds
        footprint_stats.update(inner_stats)
    else:
        trim_q = float(np.clip(footprint_trim_quantile, 0.0, 0.20))
        if trim_q > 0.0 and len(uv) >= 20:
            uv_min = np.quantile(uv, trim_q, axis=0)
            uv_max = np.quantile(uv, 1.0 - trim_q, axis=0)
        else:
            uv_min = uv.min(axis=0)
            uv_max = uv.max(axis=0)
        padding = max(0.0, float(footprint_padding))
        uv_min = uv_min - padding
        uv_max = uv_max + padding
    box = box_from_uv_bounds(
        uv_min,
        uv_max,
        origin=origin,
        u=u,
        v=v,
        normal=normal,
        shrink=shrink,
        thickness=thickness,
        clearance=clearance,
        min_side=min_side,
    )
    if box is None:
        return None
    return {
        **box,
        "support_boxes": [box],
        "footprint_stats": footprint_stats,
    }


# ------------------------------------------------------------------ descriptors
def selected_plane(selected: dict[str, Any]) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """``(normal, offset, u, v)`` of a descriptor's selected surface (plane ``n.x + offset = 0``)."""
    normal =np.asarray(selected.get("plane_normal", [0.0, 0.0, 1.0]), dtype=np.float64)
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    offset = float(selected.get("plane_offset", -float(np.dot(normal, selected["center"]))))

    u = np.asarray(selected.get("basis_u", [1.0, 0.0, 0.0]), dtype=np.float64)
    u = u - np.dot(u, normal) * normal
    u /= max(float(np.linalg.norm(u)), 1e-12)
    v = np.asarray(selected.get("basis_v", np.cross(normal, u)), dtype=np.float64)
    v = v - np.dot(v, normal) * normal
    v /= max(float(np.linalg.norm(v)), 1e-12)
    if np.linalg.det(np.column_stack((u, v, normal))) < 0.0:
        v = -v
    return normal, offset, u, v


def footprint_mask_from_world_points(points_world: np.ndarray, footprint: dict[str, Any]) -> np.ndarray:
    """Test points against a smooth polygon, with grid fallback for old descriptors."""
    points = np.asarray(points_world, dtype=np.float64)
    origin = np.asarray(footprint["plane_origin"], dtype=np.float64)
    u = np.asarray(footprint["basis_u"], dtype=np.float64)
    v = np.asarray(footprint["basis_v"], dtype=np.float64)
    rel = points - origin
    uv = np.stack((rel @ u, rel @ v), axis=-1)
    boundary = np.asarray(footprint.get("boundary_uv") or [], dtype=np.float64)
    if len(boundary) >= 3:
        flat = uv.reshape(-1, 2)
        x = flat[:, 0]
        y = flat[:, 1]

        def ring_mask(ring: np.ndarray) -> np.ndarray:
            inside_ring = np.zeros(len(flat), dtype=bool)
            xj, yj = ring[-1]
            for xi, yi in ring:
                crosses = (yi > y) != (yj > y)
                edge_x = (xj - xi) * (y - yi) / (yj - yi + 1e-15) + xi
                inside_ring ^= crosses & (x < edge_x)
                xj, yj = xi, yi
            return inside_ring

        inside = ring_mask(boundary)
        for hole in footprint.get("holes_uv") or []:
            hole_ring = np.asarray(hole, dtype=np.float64)
            if len(hole_ring) >= 3:
                inside &= ~ring_mask(hole_ring)
        return inside.reshape(uv.shape[:-1])
    uv_lo = np.asarray(footprint["uv_lo"], dtype=np.float64)
    grid_size = float(footprint["grid_size"])
    cells = np.floor((uv - uv_lo) / grid_size).astype(np.int64)
    occupied = {tuple(int(value) for value in cell) for cell in footprint.get("occupied_cells_rc", [])}
    flat = cells.reshape(-1, 2)
    inside = np.fromiter(((int(cell[1]), int(cell[0])) in occupied for cell in flat), dtype=bool, count=len(flat))
    return inside.reshape(cells.shape[:-1])
