"""Step 6 ``surface.receiver``: smooth support-aligned receiver polygon from the trimmed splat.

Inputs:  ``surface/operation_surface_plane_fit.json`` (step 4), ``world/gaussian_trimmed.ply``
         (step 5), ``align/aligned_pose_refined.npy`` and ``..._scene_scale_refined.npy``.
Output:  ``surface/operation_surface_shadow_v2.json`` -- the v2 render-receiver descriptor whose
         ``selected.footprints.{physics,lighting}`` polygons the evaluator loads as the
         support collider and the Nyx shadow catcher.

Ported from the old ``operation_surface_render_receiver_fit.py`` with the flags of
``evaluate_pi_shadow_visual.sh`` as defaults. Only the production path is kept:
``--receiver-mode polygon`` (the rectangle-only modes ``multi-rectangles``,
``coverage-rectangles``, ``inner-rectangle`` and ``support-bounds`` are gone), the
``support-edge`` basis (plus ``surface``, i.e. no rotation), no camera filter, no
opacity/scale prefilter (step 5 already trims), no reference-edge snap and no ``auto``
basis selection. Keys of the removed modes are still written with their fixed values so
the descriptor schema matches the old files.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from r2s.common.io import read_json, write_json
from r2s.common.quat import quat_wxyz_to_matrix
from r2s.common.transforms import splat_world_pose, transform_gaussian_centers
from r2s.layout import SceneLayout
from r2s.steps import Step
from r2s.surface.geometry import (
    box_from_uv_bounds,
    decompose_true_mask_into_rectangles,
    erode_true_mask,
    largest_true_mask_component,
    selected_plane,
)

# Score bias the old ``auto`` basis selection gave the unrotated ``surface`` basis; it is
# still added to ``surface`` candidates so their reported scores match the old files.
SURFACE_SCORE_BIAS = 0.055

# Options of the removed modes. They no longer change anything, but the descriptor still
# records them with the values every production run used, so it keeps the old schema.
FIXED_PARAMETERS: dict[str, Any] = {
    "receiver_mode": "polygon",
    "receiver_basis_max_samples": 20000,
    "receiver_basis_min_area_gain": 0.12,
    "receiver_auto_score_margin": 0.035,
    "receiver_auto_surface_score_bias": SURFACE_SCORE_BIAS,
    "receiver_edge_alignment_score_margin": 0.12,
    "receiver_edge_alignment_angle_threshold_deg": 10.0,
    "receiver_support_bounds_quantile": 0.05,
    "receiver_reference_edge_snap": False,
    "receiver_reference_edge_band": 0.05,
    "receiver_reference_edge_min_points": 40,
    "receiver_reference_edge_min_span_fraction": 0.55,
    "receiver_reference_edge_trim_quantile": 0.01,
    "receiver_coverage_max_empty_fraction": 0.0,
    "receiver_coverage_max_gap_cells": 0,
    "receiver_coverage_boundary_weight": 1.0,
    "receiver_coverage_boundary_refine": True,
    "receiver_coverage_boundary_pass_fraction": 0.5,
    "receiver_coverage_smooth_boundary_cells": 1,
    "receiver_coverage_grid_aligned_edges": True,
    "receiver_coverage_stop_after_coverage": 0.0,
    "receiver_coverage_min_new_point_fraction": 0.0,
    "min_opacity": None,
    "max_gaussian_scale": None,
    "camera_filter_enabled": False,
}


def read_gaussian_ply(path: str) -> dict[str, np.ndarray]:
    """Read Gaussian vertex properties from binary or ASCII PLY."""
    from plyfile import PlyData

    ply = PlyData.read(str(path))
    vertex = ply["vertex"]
    return {name: np.asarray(vertex[name]) for name in vertex.data.dtype.names or ()}


def gaussian_vertex_filter_mask(props: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, int]]:
    """Keep gaussians with finite centers (opacity/scale floaters are trimmed by step 5)."""
    for name in ("x", "y", "z"):
        if name not in props:
            raise ValueError(f"Gaussian PLY is missing vertex property '{name}'")
    xyz = np.column_stack((props["x"], props["y"], props["z"])).astype(np.float64)
    mask = np.all(np.isfinite(xyz), axis=1)
    stats = {"input_count": int(len(xyz)), "finite_count": int(mask.sum())}

    stats["kept_count"] = int(mask.sum())
    return mask, stats


def gaussian_surface_normal_confidence(
    props: dict[str, np.ndarray],
    *,
    splat_quat_wxyz: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Estimate confidence that each splat represents a horizontal surface.

    A 3DGS splat's thinnest principal axis is its surface-normal estimate.  We
    combine its alignment with world +Z and its anisotropy, so nearly spherical
    splats do not receive high normal confidence merely due to an arbitrary
    quaternion.  Missing rotation/scale properties preserve compatibility by
    returning confidence one for every vertex.
    """
    count = len(np.asarray(props.get("x", [])))
    scale_names = [f"scale_{index}" for index in range(3)]
    rotation_names = [f"rot_{index}" for index in range(4)]
    if not all(name in props for name in (*scale_names, *rotation_names)):
        return np.ones(count, dtype=np.float64), "unavailable"

    scales = np.column_stack([np.exp(np.asarray(props[name], dtype=np.float64)) for name in scale_names])
    quaternions = np.column_stack([np.asarray(props[name], dtype=np.float64) for name in rotation_names])
    quaternions /= np.maximum(np.linalg.norm(quaternions, axis=1, keepdims=True), 1e-12)
    local_normals = np.eye(3, dtype=np.float64)[np.argmin(scales, axis=1)]
    w = quaternions[:, :1]
    vector = quaternions[:, 1:]
    rotated = (
        local_normals
        + 2.0 * w * np.cross(vector, local_normals)
        + 2.0 * np.cross(vector, np.cross(vector, local_normals))
    )
    rotated_world = rotated @ quat_wxyz_to_matrix(splat_quat_wxyz).T
    orientation = np.abs(rotated_world[:, 2])
    ordered_scales = np.sort(scales, axis=1)
    anisotropy = 1.0 - ordered_scales[:, 0] / np.maximum(ordered_scales[:, 1], 1e-12)
    return np.clip(orientation * anisotropy, 0.0, 1.0), "gaussian_thinnest_axis"


def estimate_low_frequency_table_material(
    props: dict[str, np.ndarray],
    points_world: np.ndarray,
    selected: dict[str, Any],
    *,
    keep_mask: np.ndarray,
    gaussian_plane_band: float,
) -> dict[str, Any]:
    """Estimate a conservative diffuse table color from Gaussian SH DC terms."""
    dc_names = [f"f_dc_{index}" for index in range(3)]
    if not all(name in props for name in dc_names):
        return {
            "albedo_linear": [0.5, 0.5, 0.5],
            "roughness": 0.9,
            "metallic": 0.0,
            "source": "neutral_fallback_missing_sh_dc",
            "sample_count": 0,
        }

    sh_dc = np.column_stack([np.asarray(props[name], dtype=np.float64) for name in dc_names])
    # Standard 3DGS stores RGB as 0.5 + C0 * SH_DC.
    rgb = np.clip(0.5 + 0.28209479177387814 * sh_dc, 0.0, 1.0)[np.asarray(keep_mask, dtype=bool)]
    normal, offset, u, v = selected_plane(selected)
    points = np.asarray(points_world, dtype=np.float64)
    support = np.abs(points @ normal + float(offset)) <= float(gaussian_plane_band)
    support &= _reference_footprint_mask(
        selected,
        points,
        normal=normal,
        offset=float(offset),
        u=u,
        v=v,
        margin=0.0,
    )
    samples = rgb[support]
    if len(samples) < 8:
        return {
            "albedo_linear": [0.5, 0.5, 0.5],
            "roughness": 0.9,
            "metallic": 0.0,
            "source": "neutral_fallback_insufficient_surface_samples",
            "sample_count": int(len(samples)),
        }

    # Trim extreme dark shadows and highlights before taking the low-frequency color.
    luminance = samples @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)
    lo, hi = np.quantile(luminance, [0.15, 0.85])
    trimmed = samples[(luminance >= lo) & (luminance <= hi)]
    if len(trimmed) == 0:
        trimmed = samples
    albedo = np.clip(np.median(trimmed, axis=0), 0.03, 0.90)
    return {
        "albedo_linear": albedo.tolist(),
        "roughness": 0.9,
        "metallic": 0.0,
        "source": "gaussian_sh_dc_trimmed_median",
        "sample_count": int(len(trimmed)),
    }


def _descriptor_geometry_defaults(selected: dict[str, Any]) -> tuple[float, float]:
    size = selected.get("size") or [0.0, 0.0, 0.01]
    thickness = float(size[2]) if len(size) >= 3 else 0.01
    normal = np.asarray(selected.get("plane_normal", [0.0, 0.0, 1.0]), dtype=np.float64)
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    plane_offset = float(selected.get("plane_offset", -float(np.dot(normal, selected["center"]))))
    center = np.asarray(selected["center"], dtype=np.float64)
    plane_center = center - (float(np.dot(center, normal)) + plane_offset) * normal
    clearance = float(np.dot(center - plane_center, normal) - 0.5 * thickness)
    return thickness, clearance


def _reference_footprint_mask(
    selected: dict[str, Any],
    points_world: np.ndarray,
    *,
    normal: np.ndarray,
    offset: float,
    u: np.ndarray,
    v: np.ndarray,
    margin: float,
) -> np.ndarray:
    size = np.asarray(selected.get("size", [0.0, 0.0, 0.0]), dtype=np.float64)
    if len(size) < 2 or size[0] <= 0.0 or size[1] <= 0.0:
        return np.ones(len(points_world), dtype=bool)
    center = np.asarray(selected["center"], dtype=np.float64)
    plane_center = center - (float(np.dot(center, normal)) + float(offset)) * normal
    projected = points_world - (points_world @ normal + float(offset))[:, None] * normal[None, :]
    rel = projected - plane_center[None, :]
    # The reference box remains expressed in its own descriptor basis even
    # while candidate polygon bases rotate during scoring.
    reference_u = np.asarray(selected.get("basis_u", u), dtype=np.float64)
    reference_u -= float(np.dot(reference_u, normal)) * normal
    if float(np.linalg.norm(reference_u)) < 1e-8:
        reference_u = np.asarray(u, dtype=np.float64)
    reference_u /= max(float(np.linalg.norm(reference_u)), 1e-12)
    reference_v = np.asarray(selected.get("basis_v", v), dtype=np.float64)
    reference_v -= float(np.dot(reference_v, normal)) * normal
    reference_v -= float(np.dot(reference_v, reference_u)) * reference_u
    if float(np.linalg.norm(reference_v)) < 1e-8:
        reference_v = np.cross(normal, reference_u)
    reference_v /= max(float(np.linalg.norm(reference_v)), 1e-12)
    clip_margin = max(float(margin), 0.0)
    return (
        (np.abs(rel @ reference_u) <= 0.5 * float(size[0]) + clip_margin)
        & (np.abs(rel @ reference_v) <= 0.5 * float(size[1]) + clip_margin)
    )


def _occupied_grid(uv: np.ndarray, grid_size: float) -> tuple[np.ndarray, np.ndarray]:
    if len(uv) == 0 or grid_size <= 0.0:
        return np.zeros((0, 0), dtype=bool), np.zeros(2, dtype=np.float64)
    uv_lo = uv.min(axis=0) - 1e-9
    ij = np.floor((uv - uv_lo[None, :]) / float(grid_size)).astype(np.int64)
    n_rows = int(ij[:, 1].max()) + 1
    n_cols = int(ij[:, 0].max()) + 1
    mask = np.zeros((n_rows, n_cols), dtype=bool)
    mask[ij[:, 1], ij[:, 0]] = True
    return mask, uv_lo


def _binary_dilate(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Small dependency-free 8-neighbour binary dilation."""
    result = np.asarray(mask, dtype=bool).copy()
    for _ in range(max(0, int(iterations))):
        padded = np.pad(result, 1, mode="constant")
        result = np.zeros_like(result)
        for row_shift in range(3):
            for col_shift in range(3):
                result |= padded[
                    row_shift : row_shift + result.shape[0],
                    col_shift : col_shift + result.shape[1],
                ]
    return result


def _binary_erode(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Small dependency-free 8-neighbour binary erosion."""
    result = np.asarray(mask, dtype=bool).copy()
    for _ in range(max(0, int(iterations))):
        padded = np.pad(result, 1, mode="constant")
        eroded = np.ones_like(result)
        for row_shift in range(3):
            for col_shift in range(3):
                eroded &= padded[
                    row_shift : row_shift + result.shape[0],
                    col_shift : col_shift + result.shape[1],
                ]
        result = eroded
    return result


def _binary_close(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Close only small occupancy-grid holes and one-cell boundary gaps."""
    if not np.asarray(mask, dtype=bool).any():
        return np.asarray(mask, dtype=bool).copy()
    closed = _binary_erode(_binary_dilate(mask, iterations), iterations)
    # Closing must never delete the original connected support.
    return closed | np.asarray(mask, dtype=bool)


def _binary_fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill enclosed false regions while preserving the measured outer boundary."""
    occupied = np.asarray(mask, dtype=bool)
    if occupied.size == 0 or occupied.all():
        return occupied.copy()
    exterior = np.zeros_like(occupied)
    pending: list[tuple[int, int]] = []
    n_rows, n_cols = occupied.shape
    for col in range(n_cols):
        pending.extend(((0, col), (n_rows - 1, col)))
    for row in range(n_rows):
        pending.extend(((row, 0), (row, n_cols - 1)))
    while pending:
        row, col = pending.pop()
        if row < 0 or row >= n_rows or col < 0 or col >= n_cols:
            continue
        if occupied[row, col] or exterior[row, col]:
            continue
        exterior[row, col] = True
        pending.extend(((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)))
    return occupied | (~occupied & ~exterior)


def _triangulated_grid_footprint(
    mask: np.ndarray,
    *,
    uv_lo: np.ndarray,
    grid_size: float,
    plane_origin: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    normal: np.ndarray,
    thickness: float,
    clearance: float,
    role: str,
) -> dict[str, Any] | None:
    """Convert a concave occupancy mask into one indexed planar mesh descriptor.

    Each occupied grid cell contributes two top-surface triangles. Vertices are
    shared, so the runtime creates one mesh entity with no receiver-panel seams.
    The descriptor retains plane-local UVs for camera projection and diagnostics.
    """
    occupied = np.argwhere(np.asarray(mask, dtype=bool))
    if len(occupied) == 0:
        return None

    vertex_indices: dict[tuple[int, int], int] = {}
    vertices_uv: list[list[float]] = []
    triangles: list[list[int]] = []

    def vertex(row_corner: int, col_corner: int) -> int:
        key = (int(row_corner), int(col_corner))
        index = vertex_indices.get(key)
        if index is not None:
            return index
        uv = np.asarray(uv_lo, dtype=np.float64) + np.array(
            [float(col_corner), float(row_corner)], dtype=np.float64
        ) * float(grid_size)
        index = len(vertices_uv)
        vertex_indices[key] = index
        vertices_uv.append([float(uv[0]), float(uv[1])])
        return index

    for row, col in occupied:
        i00 = vertex(int(row), int(col))
        i10 = vertex(int(row), int(col) + 1)
        i11 = vertex(int(row) + 1, int(col) + 1)
        i01 = vertex(int(row) + 1, int(col))
        triangles.append([i00, i10, i11])
        triangles.append([i00, i11, i01])

    vertices_uv_array = np.asarray(vertices_uv, dtype=np.float64)
    plane_vertices = (
        np.asarray(plane_origin, dtype=np.float64)[None, :]
        + vertices_uv_array[:, :1] * np.asarray(u, dtype=np.float64)[None, :]
        + vertices_uv_array[:, 1:] * np.asarray(v, dtype=np.float64)[None, :]
    )
    # ``vertices_world`` describes the support surface (the top face).  The
    # runtime extrudes this face downward by ``thickness`` when it writes the
    # closed collision/proxy mesh, matching the legacy box convention where
    # ``clearance`` is measured from the fitted plane to the box bottom.
    top_offset = float(clearance) + float(thickness)
    vertices_world = plane_vertices + top_offset * np.asarray(normal, dtype=np.float64)[None, :]
    area = float(len(occupied)) * float(grid_size) ** 2
    return {
        "type": "triangulated_grid",
        "role": str(role),
        "vertices_uv": vertices_uv_array.tolist(),
        "vertices_world": vertices_world.tolist(),
        "triangles": triangles,
        "plane_origin": np.asarray(plane_origin, dtype=np.float64).tolist(),
        "basis_u": np.asarray(u, dtype=np.float64).tolist(),
        "basis_v": np.asarray(v, dtype=np.float64).tolist(),
        "plane_normal": np.asarray(normal, dtype=np.float64).tolist(),
        "grid_size": float(grid_size),
        "uv_lo": np.asarray(uv_lo, dtype=np.float64).tolist(),
        "occupied_cells_rc": [[int(row), int(col)] for row, col in occupied],
        "thickness": float(thickness),
        "clearance": float(clearance),
        "occupied_cell_count": int(len(occupied)),
        "area": area,
    }


def _grid_mask_geometry(mask: np.ndarray, *, uv_lo: np.ndarray, grid_size: float) -> Any | None:
    """Return the exact union of occupied UV grid cells as a Shapely geometry."""
    occupied = np.argwhere(np.asarray(mask, dtype=bool))
    if len(occupied) == 0:
        return None
    import shapely

    row = occupied[:, 0].astype(np.float64)
    col = occupied[:, 1].astype(np.float64)
    x_min = float(uv_lo[0]) + col * float(grid_size)
    y_min = float(uv_lo[1]) + row * float(grid_size)
    cells = shapely.box(x_min, y_min, x_min + float(grid_size), y_min + float(grid_size))
    geometry = shapely.make_valid(shapely.union_all(cells))
    # Occupancy connectivity is 8-neighbour. A tiny close makes cells that
    # touch only at a corner one polygon without changing the metric boundary.
    epsilon = max(float(grid_size) * 1e-6, 1e-12)
    return shapely.make_valid(
        geometry.buffer(epsilon, join_style=2).buffer(-epsilon, join_style=2)
    )


def _largest_polygon(geometry: Any) -> Any | None:
    """Extract the largest non-empty polygon from a valid Shapely geometry."""
    if geometry is None or geometry.is_empty:
        return None
    from shapely.geometry import Polygon

    geometry = __import__("shapely").make_valid(geometry)
    if isinstance(geometry, Polygon):
        return geometry
    polygons: list[Any] = []

    def collect(item: Any) -> None:
        if isinstance(item, Polygon):
            polygons.append(item)
            return
        for child in getattr(item, "geoms", ()):
            collect(child)

    collect(geometry)
    return max(polygons, key=lambda item: float(item.area), default=None)


def _unsupported_polygon_cell_count(
    polygon: Any,
    *,
    support_envelope_mask: np.ndarray,
    uv_lo: np.ndarray,
    grid_size: float,
) -> int:
    """Count polygon-covered grid centers outside the allowed support envelope."""
    if polygon is None or polygon.is_empty:
        return 0
    import shapely

    min_x, min_y, max_x, max_y = (float(value) for value in polygon.bounds)
    col_min = int(math.floor((min_x - float(uv_lo[0])) / float(grid_size)))
    col_max = int(math.floor((max_x - float(uv_lo[0]) - 1e-12) / float(grid_size)))
    row_min = int(math.floor((min_y - float(uv_lo[1])) / float(grid_size)))
    row_max = int(math.floor((max_y - float(uv_lo[1]) - 1e-12) / float(grid_size)))
    if col_max < col_min or row_max < row_min:
        return 0

    cols = np.arange(col_min, col_max + 1, dtype=np.int64)
    rows = np.arange(row_min, row_max + 1, dtype=np.int64)
    grid_cols, grid_rows = np.meshgrid(cols, rows)
    centers_x = float(uv_lo[0]) + (grid_cols.astype(np.float64) + 0.5) * float(grid_size)
    centers_y = float(uv_lo[1]) + (grid_rows.astype(np.float64) + 0.5) * float(grid_size)
    covered = np.asarray(shapely.contains_xy(polygon, centers_x, centers_y), dtype=bool)

    allowed = np.zeros_like(covered)
    in_bounds = (
        (grid_rows >= 0)
        & (grid_rows < support_envelope_mask.shape[0])
        & (grid_cols >= 0)
        & (grid_cols < support_envelope_mask.shape[1])
    )
    allowed[in_bounds] = support_envelope_mask[
        grid_rows[in_bounds],
        grid_cols[in_bounds],
    ]
    return int(np.count_nonzero(covered & ~allowed))


def _support_aligned_polygon_footprint(
    support_uv: np.ndarray,
    *,
    fallback_mask: np.ndarray,
    support_envelope_mask: np.ndarray,
    uv_lo: np.ndarray,
    grid_size: float,
    plane_origin: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    normal: np.ndarray,
    thickness: float,
    clearance: float,
    role: str,
    concavity_ratio: float,
    convex_max_overcoverage: float,
    simplify_tolerance: float,
    boundary_offset: float,
    input_support_point_count: int,
) -> dict[str, Any] | None:
    """Fit and triangulate a smooth contour whose vertices come from support.

    The occupancy grid remains in the descriptor for diagnostics and fallback,
    but is no longer used as the visible mesh boundary. Shapely's concave hull
    selects actual Gaussian support samples; metric simplification removes
    brittle point-to-point noise, and the role-specific offset separates stable
    physics coverage from conservative no-spill lighting coverage.
    """
    fallback = lambda: _triangulated_grid_footprint(
        fallback_mask,
        uv_lo=uv_lo,
        grid_size=grid_size,
        plane_origin=plane_origin,
        u=u,
        v=v,
        normal=normal,
        thickness=thickness,
        clearance=clearance,
        role=role,
    )
    points = np.asarray(support_uv, dtype=np.float64)
    if len(points) < 4:
        return fallback()
    try:
        import shapely
        from shapely.geometry import MultiPoint, Polygon
        from shapely.geometry.polygon import orient

        support_geometry = MultiPoint(points)
        concave_polygon = shapely.concave_hull(
            support_geometry,
            ratio=float(np.clip(concavity_ratio, 0.0, 1.0)),
            allow_holes=False,
        )
        concave_polygon = _largest_polygon(concave_polygon)
        if not isinstance(concave_polygon, Polygon) or concave_polygon.is_empty:
            return fallback()
        concave_area = float(concave_polygon.area)
        convex_polygon = support_geometry.convex_hull
        convex_overcoverage = (
            max(float(convex_polygon.area) - concave_area, 0.0) / max(concave_area, 1e-12)
        )
        support_envelope_geometry = _grid_mask_geometry(
            support_envelope_mask,
            uv_lo=uv_lo,
            grid_size=float(grid_size),
        )
        if support_envelope_geometry is None:
            return fallback()
        concave_unsupported_cells = _unsupported_polygon_cell_count(
            concave_polygon,
            support_envelope_mask=support_envelope_mask,
            uv_lo=uv_lo,
            grid_size=float(grid_size),
        )
        convex_unsupported_cells = (
            _unsupported_polygon_cell_count(
                convex_polygon,
                support_envelope_mask=support_envelope_mask,
                uv_lo=uv_lo,
                grid_size=float(grid_size),
            )
            if isinstance(convex_polygon, Polygon)
            else 0
        )
        contour_source = "gaussian_support_concave_hull"
        contour_mode = "concave"
        polygon = concave_polygon
        support_clip_area_retention = 1.0
        if (
            isinstance(convex_polygon, Polygon)
            and convex_overcoverage <= float(convex_max_overcoverage)
            and convex_unsupported_cells == 0
        ):
            polygon = convex_polygon
            contour_source = "gaussian_support_convex_hull_comparable"
            contour_mode = "convex"
        elif concave_unsupported_cells > 0:
            clipped_concave = _largest_polygon(concave_polygon.intersection(support_envelope_geometry))
            support_clip_area_retention = (
                float(clipped_concave.area) / max(float(concave_polygon.area), 1e-12)
                if clipped_concave is not None
                else 0.0
            )
            if clipped_concave is not None and support_clip_area_retention >= 0.95:
                polygon = clipped_concave
                contour_source = "gaussian_support_concave_hull_clipped"
            else:
                polygon = _largest_polygon(
                    _grid_mask_geometry(
                        fallback_mask,
                        uv_lo=uv_lo,
                        grid_size=float(grid_size),
                    )
                )
                contour_source = "connected_occupancy_fallback"
                contour_mode = "occupancy_fallback"
                if polygon is None:
                    return fallback()

        # Simplification may move vertices outside the validated support. Clip
        # the simplified base contour back to the one-cell envelope before the
        # deliberate role-specific padding/inset is applied.
        if float(simplify_tolerance) > 0.0:
            polygon = polygon.simplify(float(simplify_tolerance), preserve_topology=True)
        polygon = _largest_polygon(polygon.intersection(support_envelope_geometry))
        if polygon is None:
            return fallback()
        if float(simplify_tolerance) > 0.0:
            resimplified = _largest_polygon(
                polygon.simplify(float(simplify_tolerance), preserve_topology=True)
            )
            if (
                resimplified is not None
                and float(resimplified.difference(support_envelope_geometry).area) <= 1e-10
            ):
                polygon = resimplified
        selected_unsupported_cells = _unsupported_polygon_cell_count(
            polygon,
            support_envelope_mask=support_envelope_mask,
            uv_lo=uv_lo,
            grid_size=float(grid_size),
        )
        if float(boundary_offset):
            polygon = polygon.buffer(float(boundary_offset), join_style=2)
        polygon = shapely.make_valid(polygon)
        if polygon.geom_type == "MultiPolygon":
            polygon = max(polygon.geoms, key=lambda item: float(item.area))
        if not isinstance(polygon, Polygon) or polygon.is_empty or float(polygon.area) <= 1e-8:
            return fallback()
        polygon = orient(polygon, sign=1.0)

        triangulation = shapely.constrained_delaunay_triangles(polygon)
        vertex_indices: dict[tuple[float, float], int] = {}
        vertices_uv: list[list[float]] = []
        triangles: list[list[int]] = []

        def vertex_index(coordinate: tuple[float, float]) -> int:
            key = (round(float(coordinate[0]), 10), round(float(coordinate[1]), 10))
            if key not in vertex_indices:
                vertex_indices[key] = len(vertices_uv)
                vertices_uv.append([float(coordinate[0]), float(coordinate[1])])
            return vertex_indices[key]

        for triangle in triangulation.geoms:
            coordinates = list(triangle.exterior.coords)[:-1]
            if len(coordinates) != 3:
                continue
            triangles.append([vertex_index(coordinate) for coordinate in coordinates])
        if not triangles:
            return fallback()

        vertices_uv_array = np.asarray(vertices_uv, dtype=np.float64)
        top_offset = float(clearance) + float(thickness)
        vertices_world = (
            np.asarray(plane_origin, dtype=np.float64)[None, :]
            + vertices_uv_array[:, :1] * np.asarray(u, dtype=np.float64)[None, :]
            + vertices_uv_array[:, 1:] * np.asarray(v, dtype=np.float64)[None, :]
            + top_offset * np.asarray(normal, dtype=np.float64)[None, :]
        )
        boundary_uv = np.asarray(polygon.exterior.coords[:-1], dtype=np.float64)
        holes_uv = [
            np.asarray(interior.coords[:-1], dtype=np.float64).tolist()
            for interior in polygon.interiors
            if len(interior.coords) >= 4
        ]
        occupied = np.argwhere(np.asarray(fallback_mask, dtype=bool))
        return {
            "type": "support_aligned_polygon",
            "role": str(role),
            "vertices_uv": vertices_uv_array.tolist(),
            "vertices_world": vertices_world.tolist(),
            "triangles": triangles,
            "boundary_uv": boundary_uv.tolist(),
            "holes_uv": holes_uv,
            "plane_origin": np.asarray(plane_origin, dtype=np.float64).tolist(),
            "basis_u": np.asarray(u, dtype=np.float64).tolist(),
            "basis_v": np.asarray(v, dtype=np.float64).tolist(),
            "plane_normal": np.asarray(normal, dtype=np.float64).tolist(),
            "grid_size": float(grid_size),
            "uv_lo": np.asarray(uv_lo, dtype=np.float64).tolist(),
            "occupied_cells_rc": [[int(row), int(col)] for row, col in occupied],
            "thickness": float(thickness),
            "clearance": float(clearance),
            "occupied_cell_count": int(len(occupied)),
            "area": float(polygon.area),
            "contour": {
                "source": contour_source,
                "selected_mode": contour_mode,
                "concavity_ratio": float(concavity_ratio),
                "concave_area": float(concave_area),
                "convex_area": float(convex_polygon.area),
                "convex_overcoverage_ratio": float(convex_overcoverage),
                "convex_max_overcoverage": float(convex_max_overcoverage),
                "concave_unsupported_cell_count": int(concave_unsupported_cells),
                "convex_unsupported_cell_count": int(convex_unsupported_cells),
                "selected_unsupported_cell_count": int(selected_unsupported_cells),
                "support_clip_area_retention": float(support_clip_area_retention),
                "support_envelope_cells": 1,
                "input_support_point_count": int(input_support_point_count),
                "retained_support_point_count": int(len(points)),
                "rejected_support_point_count": int(input_support_point_count - len(points)),
                "simplify_tolerance": float(simplify_tolerance),
                "boundary_offset": float(boundary_offset),
                "boundary_vertex_count": int(len(boundary_uv)),
                "hole_count": int(len(holes_uv)),
            },
        }
    except Exception as exc:
        result = fallback()
        if result is not None:
            result["contour"] = {
                "source": "triangulated_grid_exception_fallback",
                "selected_mode": "grid_fallback",
                "error": f"{type(exc).__name__}: {exc}",
                "input_support_point_count": int(input_support_point_count),
                "retained_support_point_count": int(len(points)),
                "rejected_support_point_count": int(input_support_point_count - len(points)),
                "selected_unsupported_cell_count": 0,
                "boundary_offset": float(boundary_offset),
            }
        return result


def _convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """Return a counter-clockwise 2D convex hull using Andrew's monotone chain."""
    unique = sorted({(float(point[0]), float(point[1])) for point in np.asarray(points, dtype=np.float64)})
    if len(unique) <= 2:
        return np.asarray(unique, dtype=np.float64)

    def cross(o, a, b) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def _polygon_area_2d(vertices: np.ndarray) -> float:
    polygon = np.asarray(vertices, dtype=np.float64)
    if len(polygon) < 3:
        return 0.0
    return 0.5 * abs(float(np.dot(polygon[:, 0], np.roll(polygon[:, 1], -1)) - np.dot(polygon[:, 1], np.roll(polygon[:, 0], -1))))


def _refine_horizontal_plane_offset(
    points: np.ndarray,
    *,
    initial_offset: float,
    band: float,
) -> tuple[float, dict[str, Any]]:
    """Select the densest nearby height layer and robustly refine its Z offset."""
    xyz = np.asarray(points, dtype=np.float64)
    if len(xyz) < 8:
        return float(initial_offset), {"plane_height_refined": False, "plane_height_sample_count": int(len(xyz))}
    heights = xyz[:, 2]
    finite = heights[np.isfinite(heights)]
    if len(finite) < 8:
        return float(initial_offset), {"plane_height_refined": False, "plane_height_sample_count": int(len(finite))}

    bin_width = max(min(float(band) / 6.0, 0.005), 0.001)
    lo = float(finite.min()) - 0.5 * bin_width
    hi = float(finite.max()) + 0.5 * bin_width
    bins = max(1, int(math.ceil((hi - lo) / bin_width)))
    counts, edges = np.histogram(finite, bins=bins, range=(lo, hi))
    peak = int(np.argmax(counts))
    center = 0.5 * float(edges[peak] + edges[peak + 1])
    layer_half_width = max(2.0 * bin_width, min(float(band) * 0.40, 0.012))
    layer = finite[np.abs(finite - center) <= layer_half_width]
    if len(layer) < 8:
        layer = finite

    estimate = float(np.median(layer))
    # A few Huber-style reweighting steps reduce the influence of remaining floaters.
    for _ in range(4):
        residual = layer - estimate
        scale = max(1.4826 * float(np.median(np.abs(residual))), 1e-4)
        cutoff = 1.5 * scale
        weights = np.minimum(1.0, cutoff / np.maximum(np.abs(residual), 1e-12))
        estimate = float(np.sum(weights * layer) / max(float(weights.sum()), 1e-12))
    residual = np.abs(layer - estimate)
    return -estimate, {
        "plane_height_refined": True,
        "plane_height_method": "histogram_consensus_huber",
        "plane_height_sample_count": int(len(layer)),
        "plane_height_initial": float(-initial_offset),
        "plane_height_refined_value": estimate,
        "plane_height_residual_p50": float(np.quantile(residual, 0.50)),
        "plane_height_residual_p90": float(np.quantile(residual, 0.90)),
        "plane_height_histogram_bin_width": float(bin_width),
    }


def _fit_support_basis_angle(
    uv: np.ndarray,
    *,
    angle_step_deg: float,
    quantile: float,
    max_samples: int,
) -> tuple[float, float]:
    points = np.asarray(uv, dtype=np.float64)
    if len(points) < 3:
        return 0.0, 0.0
    if max_samples > 0 and len(points) > max_samples:
        indices = np.linspace(0, len(points) - 1, int(max_samples), dtype=np.int64)
        points = points[indices]

    step = max(float(angle_step_deg), 0.1)
    angles = np.deg2rad(np.arange(0.0, 90.0, step, dtype=np.float64))
    if len(angles) == 0:
        angles = np.asarray([0.0], dtype=np.float64)
    q = min(max(float(quantile), 0.0), 0.20)
    use_quantiles = q > 0.0 and len(points) >= 20

    best_angle = 0.0
    best_area = math.inf
    best_preference = math.inf
    for angle in angles:
        c = math.cos(float(angle))
        s = math.sin(float(angle))
        a = points[:, 0] * c + points[:, 1] * s
        b = -points[:, 0] * s + points[:, 1] * c
        if use_quantiles:
            a_lo, a_hi = np.quantile(a, [q, 1.0 - q])
            b_lo, b_hi = np.quantile(b, [q, 1.0 - q])
        else:
            a_lo, a_hi = np.min(a), np.max(a)
            b_lo, b_hi = np.min(b), np.max(b)
        area = max(float(a_hi - a_lo), 0.0) * max(float(b_hi - b_lo), 0.0)
        preference = min(abs(float(angle)), abs(float(angle) - 0.5 * math.pi))
        if area < best_area * 0.999 or (abs(area - best_area) <= max(best_area, 1.0) * 1e-6 and preference < best_preference):
            best_area = area
            best_angle = float(angle)
            best_preference = preference
    return best_angle, best_area


def _boundary_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return mask.copy()
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    neighbors = (
        padded[1:-1, :-2]
        & padded[1:-1, 2:]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
    )
    return mask & ~neighbors


def _fit_support_edge_angle(
    uv: np.ndarray,
    *,
    grid_size: float,
    angle_step_deg: float,
) -> tuple[float, float, int, float]:
    points = np.asarray(uv, dtype=np.float64)
    if len(points) < 8:
        return 0.0, 0.0, 0, 0.0
    raw_mask, uv_lo = _occupied_grid(points, float(grid_size))
    component = largest_true_mask_component(raw_mask)
    boundary = _boundary_mask(component)
    rows, cols = np.nonzero(boundary)
    if len(rows) < 8:
        fallback_angle = _fit_support_basis_angle(
            points,
            angle_step_deg=angle_step_deg,
            quantile=0.0,
            max_samples=20000,
        )[0]
        return fallback_angle, 0.0, int(len(rows)), fallback_angle

    boundary_uv = uv_lo[None, :] + (np.column_stack((cols, rows)).astype(np.float64) + 0.5) * float(grid_size)
    step = max(float(angle_step_deg), 0.25)
    angles = np.deg2rad(np.arange(0.0, 180.0, step, dtype=np.float64))
    strip = max(float(grid_size) * 2.0, 1e-6)
    best_line_angle = 0.0
    best_score = -math.inf
    best_count = 0

    def line_score(along: np.ndarray, across: np.ndarray) -> tuple[float, int]:
        if len(along) == 0:
            return 0.0, 0
        bins = np.floor((across - across.min()) / strip).astype(np.int64)
        min_count = max(4, int(0.004 * len(along)))
        best: tuple[float, int] = (0.0, 0)
        for bin_id in np.unique(bins):
            values = along[bins == bin_id]
            if len(values) < min_count:
                continue
            lo, hi = np.quantile(values, [0.03, 0.97]) if len(values) >= 20 else (values.min(), values.max())
            span = max(float(hi - lo), 0.0)
            score = span * math.log1p(float(len(values)))
            if score > best[0]:
                best = (float(score), int(len(values)))
        return best

    for angle in angles:
        c = math.cos(float(angle))
        s = math.sin(float(angle))
        a = boundary_uv[:, 0] * c + boundary_uv[:, 1] * s
        b = -boundary_uv[:, 0] * s + boundary_uv[:, 1] * c
        score, count = line_score(a, b)
        basis_angle = float(angle) % (0.5 * math.pi)
        preference = min(abs(basis_angle), abs(basis_angle - 0.5 * math.pi))
        score -= 0.01 * preference
        if score > best_score:
            best_score = score
            best_line_angle = float(angle)
            best_count = int(count)

    best_angle = best_line_angle % (0.5 * math.pi)
    return best_angle, float(best_score), int(best_count), float(best_line_angle)


def _trimmed_bbox_area_at_angle(uv: np.ndarray, angle: float, quantile: float) -> float:
    points = np.asarray(uv, dtype=np.float64)
    if len(points) == 0:
        return 0.0
    c = math.cos(float(angle))
    s = math.sin(float(angle))
    a = points[:, 0] * c + points[:, 1] * s
    b = -points[:, 0] * s + points[:, 1] * c
    q = min(max(float(quantile), 0.0), 0.20)
    if q > 0.0 and len(points) >= 20:
        a_lo, a_hi = np.quantile(a, [q, 1.0 - q])
        b_lo, b_hi = np.quantile(b, [q, 1.0 - q])
    else:
        a_lo, a_hi = np.min(a), np.max(a)
        b_lo, b_hi = np.min(b), np.max(b)
    return max(float(a_hi - a_lo), 0.0) * max(float(b_hi - b_lo), 0.0)


def _receiver_plane_basis(
    projected_support_points: np.ndarray,
    *,
    plane_origin: np.ndarray,
    normal: np.ndarray,
    surface_u: np.ndarray,
    surface_v: np.ndarray,
    mode: str,
    grid_size: float,
    angle_step_deg: float,
    quantile: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """In-plane receiver basis: ``surface`` keeps the plane-fit axes, ``support-edge`` turns
    them onto the dominant straight boundary edge of the Gaussian support."""
    rel = projected_support_points - plane_origin[None, :]
    seed_uv = np.column_stack((rel @ surface_u, rel @ surface_v))
    meta: dict[str, Any] = {
        "receiver_basis_mode": mode,
        "receiver_basis_selected_mode": "surface",
        "receiver_basis_angle_deg": 0.0,
        "receiver_basis_point_count": int(len(projected_support_points)),
    }
    if mode == "surface":
        return surface_u, surface_v, meta
    if mode != "support-edge":
        raise ValueError(f"Unknown receiver basis mode: {mode}")

    basis_uv = seed_uv
    surface_area = _trimmed_bbox_area_at_angle(basis_uv, 0.0, float(quantile))
    angle, edge_score, edge_point_count, edge_line_angle = _fit_support_edge_angle(
        basis_uv,
        grid_size=float(grid_size),
        angle_step_deg=float(angle_step_deg),
    )
    trimmed_area = _trimmed_bbox_area_at_angle(basis_uv, angle, float(quantile))
    area_gain = 0.0
    if surface_area > 1e-12:
        area_gain = max(0.0, (float(surface_area) - float(trimmed_area)) / float(surface_area))

    c = math.cos(angle)
    s = math.sin(angle)
    u = c * surface_u + s * surface_v
    v = -s * surface_u + c * surface_v
    u = u - np.dot(u, normal) * normal
    u /= max(float(np.linalg.norm(u)), 1e-12)
    v = v - np.dot(v, normal) * normal - np.dot(v, u) * u
    v /= max(float(np.linalg.norm(v)), 1e-12)
    if np.linalg.det(np.column_stack((u, v, normal))) < 0.0:
        v = -v
    meta.update(
        {
            "receiver_basis_selected_mode": mode,
            "receiver_basis_fit_mode": mode,
            "receiver_basis_angle_deg": float(np.rad2deg(angle)),
            "receiver_basis_candidate_angle_deg": float(np.rad2deg(angle)),
            "receiver_basis_point_count": int(len(basis_uv)),
            "receiver_basis_edge_score": float(edge_score),
            "receiver_basis_edge_point_count": int(edge_point_count),
            "receiver_basis_edge_line_angle_deg": float(np.rad2deg(edge_line_angle)),
            "receiver_basis_surface_trimmed_area": float(surface_area),
            "receiver_basis_trimmed_area": float(trimmed_area),
            "receiver_basis_support_trimmed_area": float(trimmed_area),
            "receiver_basis_area_gain": float(area_gain),
            "receiver_basis_min_area_gain": float(FIXED_PARAMETERS["receiver_basis_min_area_gain"]),
            "receiver_basis_quantile": float(quantile),
        }
    )
    return u, v, meta


def _box_to_json(box: dict[str, Any]) -> dict[str, Any]:
    return {
        "center": np.asarray(box["center"], dtype=np.float64).tolist(),
        "quat_wxyz": np.asarray(box["quat_wxyz"], dtype=np.float64).tolist(),
        "size": np.asarray(box["size"], dtype=np.float64).tolist(),
        "basis_u": np.asarray(box["basis_u"], dtype=np.float64).tolist(),
        "basis_v": np.asarray(box["basis_v"], dtype=np.float64).tolist(),
        "extents": np.asarray(box["extents"], dtype=np.float64).tolist(),
        "area": float(box["area"]),
    }


def _mask_membership(uv: np.ndarray, mask: np.ndarray, uv_lo: np.ndarray, grid_size: float) -> np.ndarray:
    if len(uv) == 0 or mask.size == 0:
        return np.zeros(len(uv), dtype=bool)
    ij = np.floor((uv - uv_lo[None, :]) / float(grid_size)).astype(np.int64)
    rows = np.clip(ij[:, 1], 0, mask.shape[0] - 1)
    cols = np.clip(ij[:, 0], 0, mask.shape[1] - 1)
    return mask[rows, cols]


def _build_receiver_for_basis(
    selected: dict[str, Any],
    projected_support_points: np.ndarray,
    *,
    normal: np.ndarray,
    offset: float,
    plane_origin: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    basis_stats: dict[str, Any],
    grid_size: float,
    erosion_cells: int,
    min_area: float,
    max_rectangles: int,
    min_rectangle_cells: int,
    tighten_to_support: bool,
    support_edge_inset: float,
    contour_concavity_ratio: float,
    contour_convex_max_overcoverage: float,
    contour_simplify_tolerance: float,
    physics_boundary_padding: float,
    lighting_boundary_inset: float,
    surface_lift: float,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    rel = projected_support_points - plane_origin[None, :]
    uv = np.column_stack((rel @ u, rel @ v))

    raw_mask, uv_lo = _occupied_grid(uv, float(grid_size))
    uv_ij = np.floor((uv - uv_lo[None, :]) / float(grid_size)).astype(np.int64) if len(uv) else np.zeros((0, 2), dtype=np.int64)
    component = largest_true_mask_component(raw_mask)
    physics_mask = _binary_close(component, 1)
    # Fill true interior scan holes while they are still distinguishable from
    # boundary-connected empty space.  Closing first can seal a one-cell mouth
    # and then turn an arbitrarily large exterior "air" pocket into a hole that
    # `_binary_fill_holes` incorrectly authorizes as receiver support.
    support_core_mask = _binary_fill_holes(component)
    support_envelope_mask = _binary_dilate(support_core_mask, 1)
    eroded = erode_true_mask(physics_mask, int(erosion_cells))
    final_mask = eroded if eroded.any() else physics_mask
    if not final_mask.any():
        stats = {
            "raw_cell_count": int(raw_mask.sum()),
            "component_cell_count": int(component.sum()),
            "eroded_cell_count": int(eroded.sum()),
            "final_cell_count": 0,
            "receiver_area": 0.0,
            **basis_stats,
        }
        return None, stats

    # Retain boxes solely as a backward-compatible runtime fallback. The
    # preferred geometry is the single triangulated footprint emitted below.
    rectangles = decompose_true_mask_into_rectangles(
        final_mask,
        max_rectangles=int(max_rectangles),
        min_rectangle_cells=int(min_rectangle_cells),
    )

    thickness, clearance = _descriptor_geometry_defaults(selected)
    clearance += float(surface_lift)
    footprints: dict[str, Any] = {}
    contour_membership = _mask_membership(uv, component, uv_lo, float(grid_size))
    contour_uv = uv[contour_membership]
    if len(contour_uv) == 0:
        contour_uv = uv
    physics_footprint = _support_aligned_polygon_footprint(
        contour_uv,
        fallback_mask=support_core_mask,
        support_envelope_mask=support_envelope_mask,
        uv_lo=uv_lo,
        grid_size=float(grid_size),
        plane_origin=plane_origin,
        u=u,
        v=v,
        normal=normal,
        thickness=thickness,
        clearance=clearance,
        role="physics",
        concavity_ratio=float(contour_concavity_ratio),
        convex_max_overcoverage=float(contour_convex_max_overcoverage),
        simplify_tolerance=float(contour_simplify_tolerance),
        boundary_offset=float(physics_boundary_padding),
        input_support_point_count=int(len(uv)),
    )
    lighting_footprint = _support_aligned_polygon_footprint(
        contour_uv,
        fallback_mask=support_core_mask,
        support_envelope_mask=support_envelope_mask,
        uv_lo=uv_lo,
        grid_size=float(grid_size),
        plane_origin=plane_origin,
        u=u,
        v=v,
        normal=normal,
        thickness=thickness,
        clearance=clearance,
        role="lighting",
        concavity_ratio=float(contour_concavity_ratio),
        convex_max_overcoverage=float(contour_convex_max_overcoverage),
        simplify_tolerance=float(contour_simplify_tolerance),
        boundary_offset=-float(lighting_boundary_inset),
        input_support_point_count=int(len(uv)),
    )
    if physics_footprint is not None:
        footprints["physics"] = physics_footprint
    if lighting_footprint is not None:
        footprints["lighting"] = lighting_footprint
    hull_uv = _convex_hull_2d(contour_uv)
    hull_area = _polygon_area_2d(hull_uv)
    polygon_area = float(lighting_footprint["area"]) if lighting_footprint is not None else 0.0
    hull_overcoverage = max(hull_area - polygon_area, 0.0) / max(polygon_area, 1e-12)
    footprints["convex_hull_diagnostic"] = {
        "vertices_uv": hull_uv.tolist(),
        "area": float(hull_area),
        "overcoverage_ratio": float(hull_overcoverage),
        "acceptable_as_fallback": bool(
            hull_overcoverage <= float(contour_convex_max_overcoverage)
        ),
    }
    boxes: list[dict[str, Any]] = []
    rectangle_cell_count = 0
    tightened_rectangle_count = 0
    for row_min, row_max, col_min, col_max in rectangles:
        rectangle_cell_count += int((row_max - row_min + 1) * (col_max - col_min + 1))
        rect_uv_min = uv_lo + np.array([col_min, row_min], dtype=np.float64) * float(grid_size)
        rect_uv_max = uv_lo + np.array([col_max + 1, row_max + 1], dtype=np.float64) * float(grid_size)
        if tighten_to_support and len(uv_ij):
            in_rect = (
                (uv_ij[:, 1] >= row_min)
                & (uv_ij[:, 1] <= row_max)
                & (uv_ij[:, 0] >= col_min)
                & (uv_ij[:, 0] <= col_max)
            )
            rect_points = uv[in_rect]
            if len(rect_points):
                inset = max(float(support_edge_inset), 0.0)
                support_uv_min = rect_points.min(axis=0) + inset
                support_uv_max = rect_points.max(axis=0) - inset
                if np.all(support_uv_max > support_uv_min):
                    rect_uv_min = support_uv_min
                    rect_uv_max = support_uv_max
                    tightened_rectangle_count += 1
        box = box_from_uv_bounds(
            rect_uv_min,
            rect_uv_max,
            origin=plane_origin,
            u=u,
            v=v,
            normal=normal,
            shrink=0.0,
            thickness=thickness,
            clearance=clearance,
            min_side=float(grid_size),
        )
        if box is not None:
            boxes.append(box)

    eligible_points = _mask_membership(uv, final_mask, uv_lo, float(grid_size))
    covered_points = eligible_points.copy()
    support_point_count = int(eligible_points.sum())
    covered_support_point_count = int(covered_points.sum())
    point_coverage = float(covered_support_point_count) / max(float(support_point_count), 1.0)
    coverage_stats = {
        "receiver_support_point_count": support_point_count,
        "receiver_covered_support_point_count": covered_support_point_count,
        "receiver_point_coverage": point_coverage,
    }

    if "lighting" in footprints:
        total_area = float(footprints["lighting"]["area"])
    else:
        total_area = float(sum(float(box["area"]) for box in boxes))
    final_cell_count = int(final_mask.sum())
    coverage = min(float(rectangle_cell_count) / max(float(final_cell_count), 1.0), 1.0)
    receiver_coverage = coverage
    polygon_precision = 0.0
    reference_mask_recall = 0.0
    reference_mask_iou = 0.0
    if final_cell_count:
        occupied_rc = np.argwhere(final_mask)
        occupied_uv = np.asarray(uv_lo, dtype=np.float64)[None, :] + np.column_stack(
            (occupied_rc[:, 1] + 0.5, occupied_rc[:, 0] + 0.5)
        ) * float(grid_size)
        occupied_world = (
            plane_origin[None, :]
            + occupied_uv[:, :1] * u[None, :]
            + occupied_uv[:, 1:] * v[None, :]
        )
        inside_reference = _reference_footprint_mask(
            selected,
            occupied_world,
            normal=normal,
            offset=float(offset),
            u=u,
            v=v,
            margin=0.0,
        )
        intersection_area = float(inside_reference.sum()) * float(grid_size) ** 2
        reference_size = np.asarray(selected.get("size", [0.0, 0.0]), dtype=np.float64)
        reference_area = (
            float(reference_size[0] * reference_size[1]) if len(reference_size) >= 2 else 0.0
        )
        polygon_area = float(final_cell_count) * float(grid_size) ** 2
        polygon_precision = intersection_area / max(polygon_area, 1e-12)
        reference_mask_recall = intersection_area / max(reference_area, 1e-12)
        reference_mask_iou = intersection_area / max(
            polygon_area + reference_area - intersection_area,
            1e-12,
        )
    stats: dict[str, Any] = {
        "raw_cell_count": int(raw_mask.sum()),
        "component_cell_count": int(component.sum()),
        "support_core_cell_count": int(support_core_mask.sum()),
        "filled_interior_hole_cell_count": int((support_core_mask & ~component).sum()),
        "closure_added_cell_count": int((physics_mask & ~component).sum()),
        "support_envelope_cell_count": int(support_envelope_mask.sum()),
        "eroded_cell_count": int(eroded.sum()),
        "final_cell_count": final_cell_count,
        "rectangle_cell_count": int(rectangle_cell_count),
        "tightened_rectangle_count": int(tightened_rectangle_count),
        "rectangle_count": int(len(boxes)),
        "receiver_area": total_area,
        "receiver_geometry": str(footprints.get("lighting", {}).get("type", "triangulated_grid")),
        "physics_boundary_vertex_count": int(
            len(footprints.get("physics", {}).get("boundary_uv", []))
        ),
        "lighting_boundary_vertex_count": int(
            len(footprints.get("lighting", {}).get("boundary_uv", []))
        ),
        "physics_polygon_cell_count": int(physics_mask.sum()),
        "lighting_polygon_cell_count": int(final_mask.sum()),
        "convex_hull_area": float(footprints.get("convex_hull_diagnostic", {}).get("area", 0.0)),
        "convex_hull_overcoverage_ratio": float(
            footprints.get("convex_hull_diagnostic", {}).get("overcoverage_ratio", 0.0)
        ),
        "receiver_coverage": float(receiver_coverage),
        "polygon_precision": float(polygon_precision),
        "reference_mask_recall": float(reference_mask_recall),
        "reference_mask_overlap_iou": float(reference_mask_iou),
        "receiver_cell_coverage": float(coverage),
        # Settings of the removed coverage-rectangles mode, as polygon runs always wrote them.
        "receiver_coverage_smooth_boundary_cells": 0,
        "receiver_coverage_grid_aligned_edges": False,
        "receiver_tighten_to_support": bool(tighten_to_support),
        "receiver_support_edge_inset": float(support_edge_inset),
        "receiver_support_bounds_quantile": float(FIXED_PARAMETERS["receiver_support_bounds_quantile"]),
        "receiver_surface_lift": float(surface_lift),
        "receiver_physics_boundary_padding": float(physics_boundary_padding),
        "receiver_lighting_boundary_inset": float(lighting_boundary_inset),
        "grid_size": float(grid_size),
        "erosion_cells": int(erosion_cells),
        **coverage_stats,
        **basis_stats,
    }
    contour_stats = footprints.get("lighting", {}).get("contour", {})
    stats.update(
        {
            "contour_input_point_count": int(contour_stats.get("input_support_point_count", len(uv))),
            "contour_retained_point_count": int(contour_stats.get("retained_support_point_count", len(uv))),
            "contour_rejected_point_count": int(contour_stats.get("rejected_support_point_count", 0)),
            "contour_concave_area": float(contour_stats.get("concave_area", 0.0)),
            "contour_convex_area": float(contour_stats.get("convex_area", 0.0)),
            "contour_convex_overcoverage_ratio": float(
                contour_stats.get("convex_overcoverage_ratio", 0.0)
            ),
            "contour_unsupported_cell_count": int(
                contour_stats.get("selected_unsupported_cell_count", 0)
            ),
            "contour_hole_count": int(contour_stats.get("hole_count", 0)),
            "contour_selected_mode": str(contour_stats.get("selected_mode", "grid_fallback")),
        }
    )
    if (not boxes and not footprints) or total_area < float(min_area):
        return None, stats

    boxes.sort(key=lambda box: float(box["area"]), reverse=True)
    primary = boxes[0]
    receiver = {
        **_box_to_json(primary),
        "schema_version": 2 if footprints else 1,
        "area": total_area,
        "plane_normal": normal.tolist(),
        "plane_offset": float(offset),
        "support_boxes": [_box_to_json(box) for box in boxes],
        "footprints": footprints,
        "footprint_stats": stats,
    }
    return receiver, stats


def _receiver_candidate_score(stats: dict[str, Any], *, surface_bias: float) -> float:
    coverage = float(stats.get("receiver_point_coverage", stats.get("receiver_coverage", 0.0)))
    area_gain = min(max(float(stats.get("receiver_basis_area_gain", 0.0)), 0.0), 0.35)
    rectangle_count = max(int(stats.get("rectangle_count", 0)), 1)
    empty_cells = max(int(stats.get("receiver_empty_cell_count", 0)), 0)
    final_cells = max(int(stats.get("final_cell_count", 0)), 1)
    empty_fraction = min(float(empty_cells) / float(final_cells), 1.0)
    score = coverage + 0.20 * area_gain - 0.010 * float(rectangle_count - 1) - 0.030 * empty_fraction
    if stats.get("receiver_basis_selected_mode") == "surface":
        score += float(surface_bias)
    if stats.get("receiver_basis_selected_mode") == "support-edge":
        edge_angle = abs(float(stats.get("receiver_basis_angle_deg", 0.0))) % 90.0
        edge_angle = min(edge_angle, 90.0 - edge_angle)
        score += min(0.08, 0.002 * edge_angle)
    return float(score)


def _receiver_candidate_sort_key(result: tuple[str, dict[str, Any] | None, dict[str, Any], float]) -> tuple[float, float]:
    _, _, stats, score = result
    area = float(stats.get("receiver_area", 0.0))
    return float(score), area


def fit_render_receiver_from_world_points(
    selected: dict[str, Any],
    points_world: np.ndarray,
    *,
    gaussian_plane_band: float = 0.025,
    refine_horizontal_plane_height: bool = True,
    grid_size: float = 0.025,
    erosion_cells: int = 1,
    min_area: float = 0.02,
    max_rectangles: int = 8,
    min_rectangle_cells: int = 2,
    receiver_basis_mode: str = "support-edge",
    receiver_basis_angle_step_deg: float = 1.0,
    receiver_basis_quantile: float = 0.02,
    receiver_tighten_to_support: bool = True,
    receiver_support_edge_inset: float = 0.0,
    receiver_grid_scales: tuple[float, ...] = (1.0, 1.4),
    receiver_clip_to_reference: bool = True,
    receiver_reference_margin: float = 0.08,
    receiver_basis_unclipped_support: bool = True,
    receiver_contour_concavity_ratio: float = 0.12,
    receiver_contour_convex_max_overcoverage: float = 0.05,
    receiver_contour_simplify_tolerance: float = 0.012,
    receiver_physics_boundary_padding: float = 0.003,
    receiver_lighting_boundary_inset: float = 0.006,
    receiver_surface_lift: float = 0.0,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Fit the polygon receiver to world-space Gaussian centers near the selected plane.

    One candidate per grid scale; the best ``(score, area)`` wins. Returns
    ``(receiver or None, stats)``.
    """
    fitted_normal, fitted_offset, surface_u, _ = selected_plane(selected)
    fitted_origin = -float(fitted_offset) * fitted_normal
    normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    offset = -float(fitted_origin[2])
    surface_u = surface_u - float(np.dot(surface_u, normal)) * normal
    if float(np.linalg.norm(surface_u)) < 1e-8:
        surface_u = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    surface_u /= max(float(np.linalg.norm(surface_u)), 1e-12)
    surface_v = np.cross(normal, surface_u)
    surface_v /= max(float(np.linalg.norm(surface_v)), 1e-12)
    points = np.asarray(points_world, dtype=np.float64)
    signed = points @ normal + offset
    near_plane = np.abs(signed) <= float(gaussian_plane_band)
    # Search a wider workspace-local height interval before selecting the
    # densest layer.  This lets refinement recover from a noticeably misplaced
    # legacy box while the final support still uses the conservative plane band.
    height_search_band = max(4.0 * float(gaussian_plane_band), 0.08)
    preliminary_support = np.abs(signed) <= height_search_band
    reference_mask = np.ones(len(points), dtype=bool)
    if receiver_clip_to_reference:
        reference_mask = _reference_footprint_mask(
            selected,
            points,
            normal=normal,
            offset=float(offset),
            u=surface_u,
            v=surface_v,
            margin=float(receiver_reference_margin),
        )
        preliminary_support &= reference_mask

    height_stats: dict[str, Any] = {
        "plane_height_refined": False,
        "plane_height_initial": float(-offset),
        "plane_height_search_band": float(height_search_band),
    }
    if refine_horizontal_plane_height:
        offset, height_stats = _refine_horizontal_plane_offset(
            points[preliminary_support],
            initial_offset=float(offset),
            band=float(gaussian_plane_band),
        )
        signed = points @ normal + float(offset)
        near_plane = np.abs(signed) <= float(gaussian_plane_band)
        reference_mask = np.ones(len(points), dtype=bool)
        if receiver_clip_to_reference:
            reference_mask = _reference_footprint_mask(
                selected,
                points,
                normal=normal,
                offset=float(offset),
                u=surface_u,
                v=surface_v,
                margin=float(receiver_reference_margin),
            )

    support = near_plane.copy()
    support &= reference_mask
    basis_support = near_plane.copy()
    if not receiver_basis_unclipped_support:
        basis_support = support.copy()

    stats: dict[str, Any] = {
        "world_point_count": int(len(points)),
        "near_plane_count": int(near_plane.sum()),
        "basis_support_count": int(basis_support.sum()),
        "reference_footprint_count": int(reference_mask.sum()),
        "support_count": int(support.sum()),
        "receiver_clip_to_reference": bool(receiver_clip_to_reference),
        "receiver_reference_margin": float(receiver_reference_margin),
        "receiver_basis_unclipped_support": bool(receiver_basis_unclipped_support),
        "plane_fitted_normal": fitted_normal.tolist(),
        "plane_stabilized_normal": normal.tolist(),
        "plane_fitted_offset": float(fitted_offset),
        "plane_stabilized_offset": float(offset),
        **height_stats,
    }
    if int(support.sum()) == 0:
        return None, stats

    plane_origin = -offset * normal
    support_points = points[support]
    projected = support_points - (support_points @ normal + offset)[:, None] * normal[None, :]
    basis_points = points[basis_support]
    projected_basis = basis_points - (basis_points @ normal + offset)[:, None] * normal[None, :]

    basis_mode = receiver_basis_mode
    grid_scales = tuple(float(scale) for scale in receiver_grid_scales if float(scale) > 0.0)
    if not grid_scales:
        grid_scales = (1.0,)
    candidate_results: list[tuple[str, dict[str, Any] | None, dict[str, Any], float]] = []
    for grid_scale in grid_scales:
        candidate_grid_size = float(grid_size) * float(grid_scale)
        u, v, basis_stats = _receiver_plane_basis(
            projected_basis if basis_mode == "support-edge" else projected,
            plane_origin=plane_origin,
            normal=normal,
            surface_u=surface_u,
            surface_v=surface_v,
            mode=basis_mode,
            grid_size=candidate_grid_size,
            angle_step_deg=float(receiver_basis_angle_step_deg),
            quantile=float(receiver_basis_quantile),
        )
        basis_stats["receiver_grid_scale"] = float(grid_scale)
        receiver, candidate_stats = _build_receiver_for_basis(
            selected,
            projected,
            normal=normal,
            offset=float(offset),
            plane_origin=plane_origin,
            u=u,
            v=v,
            basis_stats=basis_stats,
            grid_size=candidate_grid_size,
            erosion_cells=int(erosion_cells),
            min_area=float(min_area),
            max_rectangles=int(max_rectangles),
            min_rectangle_cells=int(min_rectangle_cells),
            tighten_to_support=bool(receiver_tighten_to_support),
            support_edge_inset=float(receiver_support_edge_inset),
            contour_concavity_ratio=float(receiver_contour_concavity_ratio),
            contour_convex_max_overcoverage=float(receiver_contour_convex_max_overcoverage),
            contour_simplify_tolerance=float(receiver_contour_simplify_tolerance),
            physics_boundary_padding=float(receiver_physics_boundary_padding),
            lighting_boundary_inset=float(receiver_lighting_boundary_inset),
            surface_lift=float(receiver_surface_lift),
        )
        candidate_score = (
            _receiver_candidate_score(candidate_stats, surface_bias=SURFACE_SCORE_BIAS)
            if receiver is not None
            else -math.inf
        )
        candidate_stats["receiver_candidate_score"] = float(candidate_score)
        candidate_results.append((basis_mode, receiver, candidate_stats, float(candidate_score)))

    valid_candidates = [result for result in candidate_results if result[1] is not None]
    if not valid_candidates:
        best_failed = max(candidate_results, key=lambda result: result[3], default=None)
        if best_failed is not None:
            failed_stats = {**stats, **best_failed[2]}
            failed_stats["receiver_auto_candidates"] = [
                {
                    "basis_mode": mode,
                    "score": score if math.isfinite(score) else None,
                    "receiver_area": float(candidate_stats.get("receiver_area", 0.0)),
                    "rectangle_count": int(candidate_stats.get("rectangle_count", 0)),
                }
                for mode, _, candidate_stats, score in candidate_results
            ]
            return None, failed_stats
        return None, stats

    selected_mode, receiver, selected_stats, selected_score = max(valid_candidates, key=_receiver_candidate_sort_key)

    final_stats = {**stats, **selected_stats}
    final_stats["receiver_basis_mode"] = receiver_basis_mode
    final_stats["receiver_basis_chosen_candidate"] = selected_mode
    final_stats["receiver_candidate_score"] = float(selected_score)
    for key in (
        "receiver_auto_score_margin",
        "receiver_auto_surface_score_bias",
        "receiver_edge_alignment_score_margin",
        "receiver_edge_alignment_angle_threshold_deg",
    ):
        final_stats[key] = float(FIXED_PARAMETERS[key])
    final_stats["receiver_auto_candidates"] = [
        {
            "basis_mode": mode,
            "score": score if math.isfinite(score) else None,
            "receiver_area": float(candidate_stats.get("receiver_area", 0.0)),
            "rectangle_count": int(candidate_stats.get("rectangle_count", 0)),
            "basis_angle_deg": float(candidate_stats.get("receiver_basis_angle_deg", 0.0)),
            "basis_area_gain": float(candidate_stats.get("receiver_basis_area_gain", 0.0)),
            "basis_edge_score": float(candidate_stats.get("receiver_basis_edge_score", 0.0)),
            "coverage": float(candidate_stats.get("receiver_coverage", 0.0)),
            "grid_size": float(candidate_stats.get("grid_size", grid_size)),
            "grid_scale": float(candidate_stats.get("receiver_grid_scale", 1.0)),
            "reference_edge_valid_count": int(candidate_stats.get("receiver_reference_edge_valid_count", 0)),
        }
        for mode, _, candidate_stats, score in candidate_results
    ]
    if receiver is not None:
        receiver["footprint_stats"] = final_stats
    return receiver, final_stats


def build_render_receiver_descriptor(
    *,
    operation_surface_json: str | Path,
    gaussian_ply: str | Path,
    result: str | Path,
    scene_scale: str | Path,
    selected: dict[str, Any],
    receiver: dict[str, Any] | None,
    stats: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    fixed = FIXED_PARAMETERS
    return {
        "version": 2,
        "type": "operation_surface_render_receiver",
        "source": {
            "operation_surface_json": os.path.abspath(operation_surface_json),
            "gaussian_ply": os.path.abspath(gaussian_ply),
            "result": os.path.abspath(result),
            "scene_scale": os.path.abspath(scene_scale),
        },
        "parameters": {
            "gaussian_plane_band": float(args.gaussian_plane_band),
            "refine_horizontal_plane_height": bool(args.refine_horizontal_plane_height),
            "receiver_mode": str(fixed["receiver_mode"]),
            "receiver_basis_mode": str(args.receiver_basis_mode),
            "receiver_basis_angle_step_deg": float(args.receiver_basis_angle_step_deg),
            "receiver_basis_quantile": float(args.receiver_basis_quantile),
            "receiver_basis_max_samples": int(fixed["receiver_basis_max_samples"]),
            "receiver_basis_min_area_gain": float(fixed["receiver_basis_min_area_gain"]),
            "receiver_auto_score_margin": float(fixed["receiver_auto_score_margin"]),
            "receiver_auto_surface_score_bias": float(fixed["receiver_auto_surface_score_bias"]),
            "receiver_edge_alignment_score_margin": float(fixed["receiver_edge_alignment_score_margin"]),
            "receiver_edge_alignment_angle_threshold_deg": float(fixed["receiver_edge_alignment_angle_threshold_deg"]),
            "receiver_tighten_to_support": bool(args.receiver_tighten_to_support),
            "receiver_support_edge_inset": float(args.receiver_support_edge_inset),
            "receiver_support_bounds_quantile": float(fixed["receiver_support_bounds_quantile"]),
            "receiver_grid_scales": [float(scale) for scale in args.receiver_grid_scales],
            "receiver_clip_to_reference": bool(args.receiver_clip_to_reference),
            "receiver_reference_margin": float(args.receiver_reference_margin),
            "receiver_basis_unclipped_support": bool(args.receiver_basis_unclipped_support),
            "receiver_reference_edge_snap": bool(fixed["receiver_reference_edge_snap"]),
            "receiver_reference_edge_band": float(fixed["receiver_reference_edge_band"]),
            "receiver_reference_edge_min_points": int(fixed["receiver_reference_edge_min_points"]),
            "receiver_reference_edge_min_span_fraction": float(fixed["receiver_reference_edge_min_span_fraction"]),
            "receiver_reference_edge_trim_quantile": float(fixed["receiver_reference_edge_trim_quantile"]),
            "receiver_coverage_max_empty_fraction": float(fixed["receiver_coverage_max_empty_fraction"]),
            "receiver_coverage_max_gap_cells": int(fixed["receiver_coverage_max_gap_cells"]),
            "receiver_coverage_boundary_weight": float(fixed["receiver_coverage_boundary_weight"]),
            "receiver_coverage_boundary_refine": bool(fixed["receiver_coverage_boundary_refine"]),
            "receiver_coverage_boundary_pass_fraction": float(fixed["receiver_coverage_boundary_pass_fraction"]),
            "receiver_coverage_smooth_boundary_cells": int(fixed["receiver_coverage_smooth_boundary_cells"]),
            "receiver_coverage_grid_aligned_edges": bool(fixed["receiver_coverage_grid_aligned_edges"]),
            "receiver_coverage_stop_after_coverage": float(fixed["receiver_coverage_stop_after_coverage"]),
            "receiver_coverage_min_new_point_fraction": float(fixed["receiver_coverage_min_new_point_fraction"]),
            "grid_size": float(args.grid_size),
            "erosion_cells": int(args.erosion_cells),
            "receiver_contour_concavity_ratio": float(args.receiver_contour_concavity_ratio),
            "receiver_contour_convex_max_overcoverage": float(args.receiver_contour_convex_max_overcoverage),
            "receiver_contour_simplify_tolerance": float(args.receiver_contour_simplify_tolerance),
            "receiver_physics_boundary_padding": float(args.receiver_physics_boundary_padding),
            "receiver_lighting_boundary_inset": float(args.receiver_lighting_boundary_inset),
            "receiver_surface_lift": float(args.receiver_surface_lift),
            "min_area": float(args.min_area),
            "max_rectangles": int(args.max_rectangles),
            "min_rectangle_cells": int(args.min_rectangle_cells),
            "min_opacity": fixed["min_opacity"],
            "max_gaussian_scale": fixed["max_gaussian_scale"],
            "min_gaussian_normal_confidence": float(args.min_gaussian_normal_confidence),
            "camera_filter_enabled": bool(fixed["camera_filter_enabled"]),
        },
        "reference_surface": {
            "center": selected.get("center"),
            "size": selected.get("size"),
            "quat_wxyz": selected.get("quat_wxyz"),
            "plane_normal": selected.get("plane_normal"),
            "plane_offset": selected.get("plane_offset"),
        },
        "stats": stats,
        "selected": receiver,
        "candidates": [receiver] if receiver is not None else [],
    }


# ---------------------------------------------------------------------------
# Plain function + step
# ---------------------------------------------------------------------------

def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Receiver options. Defaults are the ``evaluate_pi_shadow_visual.sh`` production values."""
    parser.add_argument("--gaussian-plane-band", type=float, default=0.025)
    parser.add_argument(
        "--refine-horizontal-plane-height",
        dest="refine_horizontal_plane_height",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-refine-horizontal-plane-height",
        dest="refine_horizontal_plane_height",
        action="store_false",
    )
    parser.add_argument(
        "--min-gaussian-normal-confidence",
        type=float,
        default=0.3,
        help="Minimum horizontal thinnest-axis/anisotropy confidence for support splats; zero disables it.",
    )
    parser.add_argument(
        "--receiver-basis-mode",
        choices=["support-edge", "surface"],
        default="support-edge",
        help="'support-edge': align to the support's straight boundary edge; 'surface': keep the plane-fit axes.",
    )
    parser.add_argument("--receiver-basis-angle-step-deg", type=float, default=1.0)
    parser.add_argument("--receiver-basis-quantile", type=float, default=0.02)
    parser.add_argument("--receiver-tighten-to-support", dest="receiver_tighten_to_support", action="store_true", default=True)
    parser.add_argument("--no-receiver-tighten-to-support", dest="receiver_tighten_to_support", action="store_false")
    parser.add_argument("--receiver-support-edge-inset", type=float, default=0.0)
    parser.add_argument("--receiver-grid-scales", type=float, nargs="+", default=(1.0, 1.4))
    parser.add_argument("--receiver-clip-to-reference", dest="receiver_clip_to_reference", action="store_true", default=True)
    parser.add_argument("--no-receiver-clip-to-reference", dest="receiver_clip_to_reference", action="store_false")
    parser.add_argument("--receiver-reference-margin", type=float, default=0.08)
    parser.add_argument("--receiver-basis-unclipped-support", dest="receiver_basis_unclipped_support", action="store_true", default=True)
    parser.add_argument("--receiver-basis-clipped-support", dest="receiver_basis_unclipped_support", action="store_false")
    parser.add_argument("--grid-size", type=float, default=0.025)
    parser.add_argument("--erosion-cells", type=int, default=1)
    parser.add_argument(
        "--receiver-contour-concavity-ratio",
        type=float,
        default=0.12,
        help="Shapely concave-hull ratio: larger values produce smoother, more convex support-aligned edges.",
    )
    parser.add_argument(
        "--receiver-contour-simplify-tolerance",
        type=float,
        default=0.012,
        help="Metric RDP-style contour simplification tolerance in meters.",
    )
    parser.add_argument(
        "--receiver-contour-convex-max-overcoverage",
        type=float,
        default=0.05,
        help="Use the smoother convex support hull when its added area is no more than this fraction.",
    )
    parser.add_argument("--receiver-physics-boundary-padding", type=float, default=0.003)
    parser.add_argument("--receiver-lighting-boundary-inset", type=float, default=0.006)
    parser.add_argument(
        "--receiver-surface-lift",
        type=float,
        default=0.015,
        help="Shared upward lift in meters applied to physics, lighting, and fallback receiver geometry.",
    )
    parser.add_argument("--min-area", type=float, default=0.02)
    parser.add_argument("--max-rectangles", type=int, default=8)
    parser.add_argument("--min-rectangle-cells", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0, help="Viewer subsampling seed.")
    parser.add_argument("--viewer", action="store_true", help="Serve a viser view of the fit and block until Ctrl-C.")
    parser.add_argument("--viser-port", type=int, default=8081)
    parser.add_argument("--viser-point-size", type=float, default=0.006)
    parser.add_argument("--viser-max-points", type=int, default=120000)
    parser.add_argument("--viser-max-support-points", type=int, default=60000)
    parser.add_argument("--viser-plane-opacity", type=float, default=0.24)
    parser.add_argument("--viser-receiver-opacity", type=float, default=0.58)
    parser.add_argument("--viser-plane-lift", type=float, default=0.004)


def default_options(**overrides: Any) -> argparse.Namespace:
    """The step's default (production) options, with ``overrides`` applied."""
    parser = argparse.ArgumentParser(add_help=False)
    add_arguments(parser)
    args = parser.parse_args([])
    for key, value in overrides.items():
        if not hasattr(args, key):
            raise TypeError(f"unknown receiver option {key!r}")
        setattr(args, key, value)
    return args


def validate_options(args: argparse.Namespace) -> None:
    if args.grid_size <= 0.0:
        raise ValueError("--grid-size must be positive")
    if args.gaussian_plane_band <= 0.0:
        raise ValueError("--gaussian-plane-band must be positive")
    if args.receiver_support_edge_inset < 0.0:
        raise ValueError("--receiver-support-edge-inset must be non-negative")
    if args.receiver_reference_margin < 0.0:
        raise ValueError("--receiver-reference-margin must be non-negative")
    if not args.receiver_grid_scales or any(scale <= 0.0 for scale in args.receiver_grid_scales):
        raise ValueError("--receiver-grid-scales values must be positive")
    if not 0.0 <= args.min_gaussian_normal_confidence <= 1.0:
        raise ValueError("--min-gaussian-normal-confidence must be in [0, 1]")
    if not 0.0 <= args.receiver_contour_concavity_ratio <= 1.0:
        raise ValueError("--receiver-contour-concavity-ratio must be in [0, 1]")
    if not 0.0 <= args.receiver_contour_convex_max_overcoverage <= 1.0:
        raise ValueError("--receiver-contour-convex-max-overcoverage must be in [0, 1]")
    if args.receiver_contour_simplify_tolerance < 0.0:
        raise ValueError("--receiver-contour-simplify-tolerance must be non-negative")
    if args.receiver_physics_boundary_padding < 0.0 or args.receiver_lighting_boundary_inset < 0.0:
        raise ValueError("Receiver contour padding/inset values must be non-negative")
    if args.receiver_surface_lift < 0.0:
        raise ValueError("--receiver-surface-lift must be non-negative")


def fit_receiver(
    plane_fit_json: str | Path,
    gaussian_ply: str | Path,
    pose_path: str | Path,
    scale_path: str | Path,
    output_json: str | Path,
    *,
    options: argparse.Namespace | None = None,
) -> dict[str, Any]:
    """Fit the v2 render receiver and write its descriptor to ``output_json``; returns the descriptor."""
    args = options if options is not None else default_options()
    validate_options(args)

    operation_descriptor = read_json(plane_fit_json)
    selected = operation_descriptor.get("selected")
    if selected is None:
        raise ValueError(f"Operation-surface descriptor has no selected candidate: {plane_fit_json}")

    props = read_gaussian_ply(gaussian_ply)
    splat_pos, splat_quat, scene_scale = splat_world_pose(pose_path, scale_path)
    gaussian_keep_mask, gaussian_stats = gaussian_vertex_filter_mask(props)
    normal_confidence, normal_confidence_source = gaussian_surface_normal_confidence(
        props,
        splat_quat_wxyz=splat_quat,
    )
    normal_mask = normal_confidence >= float(args.min_gaussian_normal_confidence)
    gaussian_keep_mask &= normal_mask
    gaussian_stats.update(
        {
            "normal_confidence_source": normal_confidence_source,
            "normal_confidence_count": int(normal_mask.sum()),
            "normal_confidence_threshold": float(args.min_gaussian_normal_confidence),
            "kept_count": int(gaussian_keep_mask.sum()),
        }
    )
    xyz_all = np.column_stack((props["x"], props["y"], props["z"])).astype(np.float64)
    xyz = xyz_all[gaussian_keep_mask]
    points_world = transform_gaussian_centers(
        xyz,
        splat_pos=splat_pos,
        splat_quat_wxyz=splat_quat,
        scene_scale=scene_scale,
    )

    receiver, stats = fit_render_receiver_from_world_points(
        selected,
        points_world,
        gaussian_plane_band=args.gaussian_plane_band,
        refine_horizontal_plane_height=args.refine_horizontal_plane_height,
        grid_size=args.grid_size,
        erosion_cells=args.erosion_cells,
        min_area=args.min_area,
        max_rectangles=args.max_rectangles,
        min_rectangle_cells=args.min_rectangle_cells,
        receiver_basis_mode=args.receiver_basis_mode,
        receiver_basis_angle_step_deg=args.receiver_basis_angle_step_deg,
        receiver_basis_quantile=args.receiver_basis_quantile,
        receiver_tighten_to_support=args.receiver_tighten_to_support,
        receiver_support_edge_inset=args.receiver_support_edge_inset,
        receiver_grid_scales=tuple(args.receiver_grid_scales),
        receiver_clip_to_reference=args.receiver_clip_to_reference,
        receiver_reference_margin=args.receiver_reference_margin,
        receiver_basis_unclipped_support=args.receiver_basis_unclipped_support,
        receiver_contour_concavity_ratio=args.receiver_contour_concavity_ratio,
        receiver_contour_convex_max_overcoverage=args.receiver_contour_convex_max_overcoverage,
        receiver_contour_simplify_tolerance=args.receiver_contour_simplify_tolerance,
        receiver_physics_boundary_padding=args.receiver_physics_boundary_padding,
        receiver_lighting_boundary_inset=args.receiver_lighting_boundary_inset,
        receiver_surface_lift=args.receiver_surface_lift,
    )
    table_material = estimate_low_frequency_table_material(
        props,
        points_world,
        receiver or selected,
        keep_mask=gaussian_keep_mask,
        gaussian_plane_band=args.gaussian_plane_band,
    )
    if receiver is not None:
        receiver["material"] = table_material
        fitted_normal = np.asarray(
            stats.get("plane_fitted_normal", receiver["plane_normal"]), dtype=np.float64
        )
        fitted_offset = float(stats.get("plane_fitted_offset", receiver["plane_offset"]))
        stabilized_normal = np.asarray(
            stats.get("plane_stabilized_normal", [0.0, 0.0, 1.0]), dtype=np.float64
        )
        stabilized_offset = float(stats.get("plane_stabilized_offset", receiver["plane_offset"]))
        receiver["fitted_plane"] = {
            "origin": (-stabilized_offset * stabilized_normal).tolist(),
            "fitted_origin": (-fitted_offset * fitted_normal).tolist(),
            "fitted_normal": fitted_normal.tolist(),
            "fitted_offset": fitted_offset,
            "stabilized_origin": (-stabilized_offset * stabilized_normal).tolist(),
            "stabilized_normal": stabilized_normal.tolist(),
            "stabilized_offset": stabilized_offset,
            "stabilized_horizontal": True,
        }
        receiver["surface_gaussian_prior"] = {
            "space": "world",
            "selection": "lighting_footprint_and_stabilized_plane_band",
            "plane_band": float(args.gaussian_plane_band),
            "normal_confidence_threshold": float(args.min_gaussian_normal_confidence),
            "fit_support_count": int(stats.get("receiver_support_point_count", 0)),
            "runtime_subset": "persistent",
        }
    stats["gaussian"] = gaussian_stats

    stats = {**stats, "gaussian": gaussian_stats, "table_material": table_material}
    descriptor = build_render_receiver_descriptor(
        operation_surface_json=plane_fit_json,
        gaussian_ply=gaussian_ply,
        result=pose_path,
        scene_scale=scale_path,
        selected=selected,
        receiver=receiver,
        stats=stats,
        args=args,
    )
    write_json(output_json, descriptor)
    _print_summary(output_json, receiver, stats, args)
    if args.viewer:
        from r2s.surface.viz import launch_render_receiver_viz_viewer

        launch_render_receiver_viz_viewer(
            points_world=points_world,
            selected=selected,
            receiver=receiver,
            args=args,
        )
    return descriptor


def _print_summary(
    output_path: str | Path,
    receiver: dict[str, Any] | None,
    stats: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    if receiver is None:
        print(f"[write] no safe render receiver emitted; diagnostics saved to {output_path}")
        return
    chosen = stats.get("receiver_basis_chosen_candidate", stats.get("receiver_basis_selected_mode", "unknown"))
    candidate_bits = []
    for candidate in stats.get("receiver_auto_candidates", []):
        score = candidate.get("score")
        score_text = "none" if score is None else f"{float(score):.3f}"
        candidate_bits.append(
            f"{candidate.get('basis_mode')} score={score_text} "
            f"area={float(candidate.get('receiver_area', 0.0)):.4f} "
            f"angle={float(candidate.get('basis_angle_deg', 0.0)):.1f} "
            f"grid={float(candidate.get('grid_size', args.grid_size)):.4f}"
        )
    print(
        f"[write] render receiver saved to {output_path} "
        f"area={receiver['area']:.4f} boxes={len(receiver['support_boxes'])} "
        f"basis={chosen}"
    )
    print(
        "[fit] contour "
        f"mode={stats.get('contour_selected_mode', 'n/a')} "
        f"points={int(stats.get('contour_retained_point_count', 0))}/"
        f"{int(stats.get('contour_input_point_count', 0))} "
        f"rejected={int(stats.get('contour_rejected_point_count', 0))} "
        f"cells={int(stats.get('component_cell_count', 0))}/"
        f"{int(stats.get('raw_cell_count', 0))} "
        f"core={int(stats.get('support_core_cell_count', 0))} "
        f"filled_holes={int(stats.get('filled_interior_hole_cell_count', 0))} "
        f"closure_added={int(stats.get('closure_added_cell_count', 0))} "
        f"convex_overcoverage={float(stats.get('contour_convex_overcoverage_ratio', 0.0)):.3f} "
        f"unsupported={int(stats.get('contour_unsupported_cell_count', 0))} "
        f"cutouts={int(stats.get('contour_hole_count', 0))} "
        f"physics_padding={float(stats.get('receiver_physics_boundary_padding', 0.0)):.3f}m "
        f"lighting_inset={float(stats.get('receiver_lighting_boundary_inset', 0.0)):.3f}m "
        f"surface_lift={float(stats.get('receiver_surface_lift', 0.0)):.3f}m"
    )
    if candidate_bits:
        print("[fit] candidate comparison: " + " | ".join(candidate_bits))


def run(layout: SceneLayout, args: argparse.Namespace) -> dict[str, Any]:
    descriptor = fit_receiver(
        layout.plane_fit_json,
        layout.gaussian_trimmed_ply,
        layout.refined_pose,
        layout.refined_scale,
        layout.receiver_json,
        options=args,
    )
    receiver = descriptor["selected"]
    stats = descriptor["stats"]
    return {
        "receiver_found": receiver is not None,
        "receiver_area": float(receiver["area"]) if receiver is not None else 0.0,
        "contour_selected_mode": stats.get("contour_selected_mode"),
        "basis_angle_deg": stats.get("receiver_basis_angle_deg"),
        "grid_size": stats.get("grid_size"),
    }


STEP = Step(
    name="surface.receiver",
    help="Fit the smooth polygon render/physics receiver from the trimmed splat",
    add_arguments=add_arguments,
    run=run,
    inputs=lambda layout, args: [
        layout.plane_fit_json,
        layout.gaussian_trimmed_ply,
        layout.refined_pose,
        layout.refined_scale,
    ],
    outputs=lambda layout, args: [layout.receiver_json],
)
