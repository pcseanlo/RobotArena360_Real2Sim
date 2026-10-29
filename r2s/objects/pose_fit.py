"""Step 8b ``objects.posefit``: move each object on the table to match the real first frame.

Port of the old ``object_pose_refine.py`` (``fit`` and ``check-render``). The
evaluator places each reconstructed object at the pose optimized during scene
generation; measured on droid2/302/820 those poses sit 1-10 cm away from where
the object is in the real recording. This step moves each object on its support
plane (XY translation + yaw about the table normal) so that the object's
silhouette, projected through the calibrated camera, lines up with the real
first-frame segmentation mask:

1. align silhouette centroids (Gauss-Newton on the XY offset),
2. sweep yaw by mask IoU where the silhouette is informative,
3. polish XY on IoU,

then reject fits that do not beat the original IoU, whose area is far from the
mask's, or whose footprints swallow a neighbour. Before fitting, objects whose
thin axis is tilted 3-30 deg are levelled and objects floating > 1 cm above the
dump's ``physics_top`` (else ``z_plane``) are dropped onto it.

Input: ``layout.object_dump`` (``objects/stock/object_poses.json`` from
``objects.dump``: the poses the evaluator loaded, same mesh frame as the sim),
the scene's calibration and first-frame masks. Output: ``layout.refined_poses``
(the ``--object-pose-override`` file) and ``overlays/``. Defaults are the
production settings (every scene ran the script defaults).

Roll/pitch are never changed: objects rest on the table, so only yaw is free.
Yaw is only accepted when it improves IoU by more than ``yaw_min_gain``;
rotationally symmetric objects keep their reconstructed yaw (``yaw_iou_range``).
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from r2s.common.io import prepare_output
from r2s.layout import SceneLayout
from r2s.objects.placement_types import ObjectPlacementSpec, quat_mul_wxyz, yaw_quat_wxyz
from r2s.steps import Step

DEFAULT_HEIGHT = 360
DEFAULT_WIDTH = 640


# ---------------------------------------------------------------------------
# Camera + rasterization
# ---------------------------------------------------------------------------
@dataclass
class PinholeCamera:
    K: np.ndarray  # (3,3)
    T_w2c: np.ndarray  # (4,4) world -> camera (OpenCV: +z forward)
    height: int
    width: int

    @classmethod
    def from_files(cls, intrinsics_path, extrinsics_path, height: int = 360, width: int = 640) -> "PinholeCamera":
        K = np.load(intrinsics_path).astype(np.float64).reshape(3, 3)
        T = np.load(extrinsics_path).astype(np.float64).reshape(4, 4)
        return cls(K=K, T_w2c=T, height=int(height), width=int(width))

    @classmethod
    def from_inputs(cls, inputs, height: int = 360, width: int = 640) -> "PinholeCamera":
        """From a scene's step-0 inputs (``SceneLayout.inputs``)."""
        return cls.from_files(inputs.intrinsics, inputs.extrinsics, height=height, width=width)

    def project(self, points_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return pixel coordinates (N,2) as (u,v) and camera depth (N,)."""
        pts = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
        cam = pts @ self.T_w2c[:3, :3].T + self.T_w2c[:3, 3][None, :]
        depth = cam[:, 2]
        safe = np.where(np.abs(depth) < 1e-9, 1e-9, depth)
        u = self.K[0, 0] * cam[:, 0] / safe + self.K[0, 2]
        v = self.K[1, 1] * cam[:, 1] / safe + self.K[1, 2]
        return np.stack([u, v], axis=1), depth

    def ray_plane_xy(self, uv: np.ndarray, z_plane: float) -> np.ndarray:
        """World XY where the pixel ray hits the horizontal plane z=z_plane."""
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        R = self.T_w2c[:3, :3]
        t = self.T_w2c[:3, 3]
        cam_center = -R.T @ t
        dirs_cam = np.stack(
            [(uv[:, 0] - self.K[0, 2]) / self.K[0, 0], (uv[:, 1] - self.K[1, 2]) / self.K[1, 1], np.ones(len(uv))],
            axis=1,
        )
        dirs_world = dirs_cam @ R  # R.T @ d for each row
        s = (z_plane - cam_center[2]) / dirs_world[:, 2]
        hit = cam_center[None, :] + s[:, None] * dirs_world
        return hit[:, :2]


def rasterize_silhouette(
    uv: np.ndarray, depth: np.ndarray, faces: np.ndarray | None, height: int, width: int
) -> np.ndarray:
    """Binary silhouette of a triangle mesh already projected to pixels."""
    import cv2

    mask = np.zeros((height, width), dtype=np.uint8)
    if faces is None or len(faces) == 0:
        # Vertex cloud fallback: convex hull of the visible points.
        pts = uv[depth > 1e-6]
        if len(pts) >= 3:
            hull = cv2.convexHull(np.round(pts).astype(np.int32))
            cv2.fillPoly(mask, [hull], 1)
        return mask.astype(bool)
    tri = uv[faces]  # (M,3,2)
    keep = (depth[faces] > 1e-6).all(axis=1)
    tri = tri[keep]
    if len(tri) == 0:
        return mask.astype(bool)
    # Drop triangles entirely outside a generous margin so fillPoly stays fast.
    margin = 64
    inside = (
        (tri[..., 0].max(axis=1) >= -margin)
        & (tri[..., 0].min(axis=1) <= width + margin)
        & (tri[..., 1].max(axis=1) >= -margin)
        & (tri[..., 1].min(axis=1) <= height + margin)
    )
    tri = tri[inside]
    if len(tri) == 0:
        return mask.astype(bool)
    polys = np.round(tri).astype(np.int32)
    # One call per triangle: a single fillPoly over many contours applies the
    # even-odd rule, so overlapping front/back faces would cancel out.
    # ~2 us per triangle -> ~0.1 s for a 50k-face reconstruction.
    for poly in polys:
        cv2.fillConvexPoly(mask, poly, 1)
    return mask.astype(bool)


def mask_centroid(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return np.array([xs.mean(), ys.mean()], dtype=np.float64)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(a, b).sum() / union)


def mask_principal_angle(mask: np.ndarray) -> float | None:
    """Orientation (rad, image frame) of the mask's major axis, in [0, pi)."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 5:
        return None
    x = xs - xs.mean()
    y = ys - ys.mean()
    cov = np.array([[np.mean(x * x), np.mean(x * y)], [np.mean(x * y), np.mean(y * y)]])
    evals, evecs = np.linalg.eigh(cov)
    major = evecs[:, int(np.argmax(evals))]
    return float(np.arctan2(major[1], major[0]) % np.pi)


# ---------------------------------------------------------------------------
# Object model
# ---------------------------------------------------------------------------
def _axis_angle_quat_wxyz(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    n = float(np.linalg.norm(axis))
    if n < 1e-12 or abs(angle) < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = axis / n
    return np.array([math.cos(angle / 2), *(math.sin(angle / 2) * axis)])


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / max(float(np.linalg.norm(q)), 1e-12)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def minor_axis_tilt(centered_vertices: np.ndarray) -> tuple[np.ndarray, float]:
    """Unit minor PCA axis (pointing up) and its angle to world +z in degrees."""
    c = centered_vertices - centered_vertices.mean(axis=0)
    _, U = np.linalg.eigh(c.T @ c)
    axis = U[:, 0]
    if axis[2] < 0:
        axis = -axis
    return axis, math.degrees(math.acos(min(1.0, max(-1.0, float(axis[2])))))


@dataclass
class ObjectModel:
    spec: ObjectPlacementSpec
    camera: PinholeCamera
    level_deg: float = 0.0  # tilt removed from the thin axis (0 = untouched)
    z_delta: float = 0.0  # vertical shift applied to rest on the support plane
    minz_before: float | None = None
    load_pos: np.ndarray | None = None  # evaluator's load-time pose (before levelling/dropping)
    load_quat: np.ndarray | None = None

    @classmethod
    def from_dump_entry(
        cls,
        entry: dict,
        camera: PinholeCamera,
        *,
        z_plane: float | None = None,
        clearance: float = 0.0005,
        level_min_deg: float = 3.0,
        level_max_deg: float = 30.0,
        drop_tolerance: float = 0.002,
        min_float: float = 0.01,
    ) -> "ObjectModel":
        """Build the model in the evaluator's mesh frame, then rest it on the table.

        Two load-time defects are corrected before fitting, because a pose fit
        cannot absorb them: an object whose thin axis (bowl height, bottle
        lying flat) is tilted by ``level_min_deg``..``level_max_deg`` is rotated
        upright about its root, and an object floating more than ``min_float``
        above ``z_plane`` is dropped so its lowest vertex sits ``clearance``
        above the plane.  Otherwise the object's original lowest-vertex height
        is preserved (also after levelling): the evaluator's collision guard
        chose it with the collision hull, which for thin objects (brush,
        scissors) is fatter than the visual mesh, and resting the visual mesh
        lower makes physics pop the object out.  Pass ``level_max_deg=0`` /
        ``z_plane=None`` to disable.
        """
        spec = ObjectPlacementSpec.from_asset(
            safe_key=entry["safe_key"],
            count=int(entry["count"]),
            object_name=entry.get("object_name", entry["safe_key"]),
            asset_file=entry["asset_file"],
            scale=float(entry["scale"]),
            base_pos=np.asarray(entry["pos"], dtype=np.float64),
            base_quat_wxyz=np.asarray(entry["quat_wxyz"], dtype=np.float64),
            max_vertices=0,  # keep faces for rasterization
        )
        minz_before = spec.base_min_z
        model = cls(
            spec=spec,
            camera=camera,
            minz_before=minz_before,
            load_pos=spec.base_pos.copy(),
            load_quat=spec.base_quat_wxyz.copy(),
        )
        axis, tilt = minor_axis_tilt(spec.centered_vertices)
        if level_max_deg > 0 and level_min_deg <= tilt <= level_max_deg:
            q = _axis_angle_quat_wxyz(np.cross(axis, np.array([0.0, 0.0, 1.0])), math.radians(tilt))
            R = _quat_to_matrix(q)
            spec.centered_vertices = spec.centered_vertices @ R.T
            spec.base_quat_wxyz = quat_mul_wxyz(q, spec.base_quat_wxyz)
            model.level_deg = float(tilt)
        target = minz_before  # keep the guard's resting height by default
        if z_plane is not None:
            plane_rest = float(z_plane) + float(clearance)
            if minz_before - plane_rest > min_float:
                target = plane_rest  # clearly floating: drop onto the plane
        delta = target - spec.base_min_z
        if abs(delta) > drop_tolerance:
            spec.base_pos = spec.base_pos.copy()
            spec.base_pos[2] += delta
            model.z_delta = float(delta)
        return model

    def pos_for(self, xy_delta) -> np.ndarray:
        pos = self.spec.base_pos.copy()
        pos[0] += float(xy_delta[0])
        pos[1] += float(xy_delta[1])
        return pos

    def quat_for(self, yaw_rad: float) -> np.ndarray:
        return quat_mul_wxyz(yaw_quat_wxyz(yaw_rad), self.spec.base_quat_wxyz)

    def silhouette(self, xy_delta, yaw_rad: float) -> np.ndarray:
        verts = self.spec.world_vertices(self.pos_for(xy_delta), yaw_rad)
        uv, depth = self.camera.project(verts)
        return rasterize_silhouette(uv, depth, self.spec.faces, self.camera.height, self.camera.width)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------
@dataclass
class FitResult:
    safe_key: str
    count: int
    object_name: str
    status: str  # refined | unchanged | skipped_no_mask | skipped_no_silhouette
    xy_delta: np.ndarray = field(default_factory=lambda: np.zeros(2))
    yaw_rad: float = 0.0
    iou_before: float = 0.0
    iou_after: float = 0.0
    iou_after_xy_only: float = 0.0
    centroid_real: np.ndarray | None = None
    centroid_before: np.ndarray | None = None
    centroid_after: np.ndarray | None = None
    yaw_iou_range: float | None = None
    yaw_mode: str = "off"
    yaw_curve: list[tuple[float, float]] = field(default_factory=list)
    iterations: int = 0
    area_ratio: float | None = None  # sim silhouette area / real mask area at the refined pose
    suggested_scale: float | None = None  # mesh scale multiplier implied by the area ratio
    notes: list[str] = field(default_factory=list)

    def to_json_dict(self, model: ObjectModel | None) -> dict:
        out = {
            "area_ratio": None if self.area_ratio is None else float(self.area_ratio),
            "suggested_scale": None if self.suggested_scale is None else float(self.suggested_scale),
            "safe_key": self.safe_key,
            "count": int(self.count),
            "object_name": self.object_name,
            "status": self.status,
            "apply": self.status == "refined",
            "xy_delta_m": [float(v) for v in self.xy_delta],
            "shift_cm": float(100.0 * np.linalg.norm(self.xy_delta)),
            "yaw_delta_deg": float(math.degrees(self.yaw_rad)),
            "yaw_mode": self.yaw_mode,
            "iou_before": float(self.iou_before),
            "iou_after_xy_only": float(self.iou_after_xy_only),
            "iou_after": float(self.iou_after),
            "yaw_iou_range": None if self.yaw_iou_range is None else float(self.yaw_iou_range),
            "yaw_curve_deg_iou": [[float(a), float(b)] for a, b in self.yaw_curve],
            "centroid_real_px": None if self.centroid_real is None else [float(v) for v in self.centroid_real],
            "centroid_before_px": None if self.centroid_before is None else [float(v) for v in self.centroid_before],
            "centroid_after_px": None if self.centroid_after is None else [float(v) for v in self.centroid_after],
            "iterations": int(self.iterations),
            "notes": list(self.notes),
        }
        if model is not None:
            out.update(
                {
                    "asset_file": model.spec.asset_file,
                    "scale": float(model.spec.scale),
                    "base_pos": [float(v) for v in model.spec.base_pos],
                    "base_quat_wxyz": [float(v) for v in model.spec.base_quat_wxyz],
                    "pos": [float(v) for v in model.pos_for(self.xy_delta)],
                    "quat_wxyz": [float(v) for v in model.quat_for(self.yaw_rad)],
                    "load_pos": None if model.load_pos is None else [float(v) for v in model.load_pos],
                    "load_quat_wxyz": None if model.load_quat is None else [float(v) for v in model.load_quat],
                    "level_deg": float(model.level_deg),
                    "z_delta_m": float(model.z_delta),
                    "minz_before_m": None if model.minz_before is None else float(model.minz_before),
                }
            )
            # A levelled or dropped object is a real change even when XY/yaw stay put.
            if self.status == "unchanged" and (model.level_deg or model.z_delta):
                out["apply"] = True
                out["status"] = "rested_only"
        return out


def align_centroid(
    model: ObjectModel,
    real_centroid: np.ndarray,
    xy0,
    yaw_rad: float,
    *,
    max_iters: int = 8,
    tol_px: float = 0.25,
    fd_step_m: float = 0.01,
    max_step_m: float = 0.15,
) -> tuple[np.ndarray, int]:
    """Gauss-Newton on the XY offset so the silhouette centroid hits ``real_centroid``."""
    xy = np.asarray(xy0, dtype=np.float64).copy()
    iters = 0
    for _ in range(max_iters):
        c = mask_centroid(model.silhouette(xy, yaw_rad))
        if c is None:
            break
        r = real_centroid - c
        iters += 1
        if np.linalg.norm(r) < tol_px:
            break
        J = np.zeros((2, 2))
        for axis in range(2):
            d = np.zeros(2)
            d[axis] = fd_step_m
            c_plus = mask_centroid(model.silhouette(xy + d, yaw_rad))
            c_minus = mask_centroid(model.silhouette(xy - d, yaw_rad))
            if c_plus is None or c_minus is None:
                return xy, iters
            J[:, axis] = (c_plus - c_minus) / (2.0 * fd_step_m)
        step, *_ = np.linalg.lstsq(J, r, rcond=None)
        norm = float(np.linalg.norm(step))
        if norm > max_step_m:
            step *= max_step_m / norm
        xy = xy + step
    return xy, iters


def polish_xy_iou(model: ObjectModel, real_mask: np.ndarray, xy, yaw_rad: float, *, radius_m: float = 0.015, step_m: float = 0.005):
    best_xy = np.asarray(xy, dtype=np.float64).copy()
    best = mask_iou(model.silhouette(best_xy, yaw_rad), real_mask)
    offsets = np.arange(-radius_m, radius_m + 1e-9, step_m)
    for dx in offsets:
        for dy in offsets:
            cand = np.asarray(xy, dtype=np.float64) + np.array([dx, dy])
            score = mask_iou(model.silhouette(cand, yaw_rad), real_mask)
            if score > best + 1e-6:
                best, best_xy = score, cand
    return best_xy, best


def fit_object(
    model: ObjectModel,
    real_mask: np.ndarray | None,
    *,
    yaw_mode: str = "sweep",
    yaw_step_deg: float = 15.0,
    yaw_min_gain: float = 0.02,
    polish: bool = True,
    min_area_ratio: float = 0.5,
    max_area_ratio: float = 2.0,
    min_iou: float = 0.3,
) -> FitResult:
    spec = model.spec
    result = FitResult(safe_key=spec.safe_key, count=spec.count, object_name=spec.object_name, status="unchanged", yaw_mode=yaw_mode)
    if real_mask is None or not real_mask.any():
        result.status = "skipped_no_mask"
        return result
    sil0 = model.silhouette(np.zeros(2), 0.0)
    if not sil0.any():
        result.status = "skipped_no_silhouette"
        result.notes.append("sim silhouette is empty (object outside the calibrated view?)")
        return result
    real_c = mask_centroid(real_mask)
    result.centroid_real = real_c
    result.centroid_before = mask_centroid(sil0)
    result.iou_before = mask_iou(sil0, real_mask)

    # 1. translation only
    xy, iters = align_centroid(model, real_c, np.zeros(2), 0.0)
    result.iterations = iters
    iou_xy = mask_iou(model.silhouette(xy, 0.0), real_mask)
    result.iou_after_xy_only = iou_xy
    best_xy, best_yaw, best_iou = xy, 0.0, iou_xy

    # 2. yaw (about the table normal, through the object origin)
    if yaw_mode != "off":
        if yaw_mode == "c4":
            candidates = [0.0, 90.0, 180.0, 270.0]
        else:
            candidates = list(np.arange(0.0, 360.0, float(yaw_step_deg)))
        curve = []
        for deg in candidates:
            yaw = math.radians(deg)
            xy_c, _ = align_centroid(model, real_c, xy, yaw, max_iters=4)
            curve.append((deg, mask_iou(model.silhouette(xy_c, yaw), real_mask), xy_c))
        ious = np.array([c[1] for c in curve])
        result.yaw_iou_range = float(ious.max() - ious.min())
        result.yaw_curve = [(c[0], c[1]) for c in curve]
        k = int(np.argmax(ious))
        coarse_deg, coarse_iou, coarse_xy = curve[k]
        # fine refinement around the coarse winner
        fine_best = (coarse_deg, coarse_iou, coarse_xy)
        if yaw_mode != "c4":
            half = float(yaw_step_deg)
            for deg in np.arange(coarse_deg - half, coarse_deg + half + 1e-9, max(half / 5.0, 1.0)):
                yaw = math.radians(deg)
                xy_c, _ = align_centroid(model, real_c, coarse_xy, yaw, max_iters=4)
                score = mask_iou(model.silhouette(xy_c, yaw), real_mask)
                if score > fine_best[1] + 1e-6:
                    fine_best = (deg, score, xy_c)
        deg, score, xy_c = fine_best
        gain = score - iou_xy
        if gain > yaw_min_gain and abs(((deg + 180.0) % 360.0) - 180.0) > 1e-6:
            best_xy, best_yaw, best_iou = xy_c, math.radians(((deg + 180.0) % 360.0) - 180.0), score
            result.notes.append(f"yaw accepted: +{gain:.3f} IoU over translation-only")
        else:
            result.notes.append(
                f"yaw rejected: best {deg:.0f} deg gives +{gain:.3f} IoU (< {yaw_min_gain}); "
                f"sweep range {result.yaw_iou_range:.3f}"
            )

    # 3. small IoU polish on XY (centroid alignment is a proxy for overlap)
    if polish:
        best_xy, best_iou = polish_xy_iou(model, real_mask, best_xy, best_yaw)

    result.xy_delta = np.asarray(best_xy, dtype=np.float64)
    result.yaw_rad = float(best_yaw)
    result.iou_after = float(best_iou)
    sil_after = model.silhouette(best_xy, best_yaw)
    result.centroid_after = mask_centroid(sil_after)
    result.area_ratio = float(sil_after.sum() / max(int(real_mask.sum()), 1))
    def revert(status: str, note: str) -> None:
        result.status = status
        result.notes.append(note)
        result.xy_delta = np.zeros(2)
        result.yaw_rad = 0.0
        result.iou_after = result.iou_before
        result.centroid_after = result.centroid_before

    if best_iou + 1e-6 < result.iou_before:
        revert("unchanged", "refined pose did not beat the original IoU; keeping the original")
    elif not (min_area_ratio <= result.area_ratio <= max_area_ratio):
        # A silhouette this far from the mask's size is a reconstruction scale
        # error; matching centroids then only slides the object along the view
        # ray (droid28 thermos: 4x area, dragged 53 cm toward the camera).
        result.suggested_scale = float(1.0 / math.sqrt(result.area_ratio))
        revert(
            "size_mismatch_not_applied",
            f"sim silhouette is {result.area_ratio:.2f}x the real mask area (allowed "
            f"{min_area_ratio}-{max_area_ratio}); keeping the original pose, mesh scale "
            f"would need x{result.suggested_scale:.2f}",
        )
    elif best_iou < min_iou:
        revert(
            "low_iou_not_applied",
            f"refined IoU {best_iou:.2f} < {min_iou}; keeping the original pose (mask occluded, "
            "wrong object, or reconstruction shape mismatch)",
        )
    else:
        result.status = "refined"
    return result


def _hull_contains_fraction(container_xy: np.ndarray, points_xy: np.ndarray) -> float:
    """Fraction of ``points_xy`` inside the convex hull of ``container_xy``."""
    from scipy.spatial import ConvexHull, Delaunay

    if len(container_xy) < 3 or len(points_xy) == 0:
        return 0.0
    try:
        hull = ConvexHull(container_xy)
    except Exception:
        return 0.0
    tri = Delaunay(container_xy[hull.vertices])
    return float((tri.find_simplex(points_xy) >= 0).mean())


def resolve_footprint_conflicts(
    models: dict[str, ObjectModel],
    results: dict[str, FitResult],
    *,
    max_overlap: float = 0.25,
    flat_height: float = 0.025,
) -> list[str]:
    """Do not apply refined poses whose footprint swallows another object's.

    A reconstruction that is too large (or whose real mask is occluded) can
    only match its mask by sitting on top of a neighbour; in the simulator
    that becomes an interpenetration that launches the neighbour.  When two
    refined objects overlap by more than ``max_overlap`` of the smaller
    footprint, the one that moved further keeps its original pose.

    Objects thinner than ``flat_height`` (placemats, plates, cutlery) are
    exempt: a plate on a mat is a legitimate stack that physics settles.
    """
    notes: list[str] = []
    keys = [k for k, r in results.items() if r.status == "refined"]
    foot = {k: models[k].spec.world_vertices(models[k].pos_for(results[k].xy_delta), results[k].yaw_rad)[:, :2] for k in keys}
    height = {k: float(np.ptp(models[k].spec.centered_vertices[:, 2])) for k in keys}
    changed = True
    while changed:
        changed = False
        for i, a in enumerate(keys):
            for b in keys[i + 1 :]:
                if results[a].status != "refined" or results[b].status != "refined":
                    continue
                if height[a] < flat_height or height[b] < flat_height:
                    continue
                overlap = max(_hull_contains_fraction(foot[a], foot[b]), _hull_contains_fraction(foot[b], foot[a]))
                if overlap <= max_overlap:
                    continue
                loser = a if np.linalg.norm(results[a].xy_delta) >= np.linalg.norm(results[b].xy_delta) else b
                other = b if loser == a else a
                res = results[loser]
                res.status = "conflict_not_applied"
                res.notes.append(
                    f"footprint overlaps {results[other].object_name} [{results[other].count}] by {overlap:.2f}; "
                    "keeping the original pose (reconstruction too large or mask occluded)"
                )
                notes.append(f"{res.object_name} [{res.count}] not applied: overlaps {results[other].object_name} ({overlap:.2f})")
                changed = True
    return notes


# ---------------------------------------------------------------------------
# Real-data loading
# ---------------------------------------------------------------------------
def load_real_masks(bbox_masks_dir) -> dict[int, tuple[str, np.ndarray]]:
    """First-frame segmentation masks by object count, from ``bounding_boxes_and_masks/``.

    The first file (sorted) of ``frame_*_bounding_boxes_and_masks.json`` is frame 0.
    """
    files = sorted(glob.glob(os.path.join(str(bbox_masks_dir), "frame_*_bounding_boxes_and_masks.json")))
    if not files:
        raise FileNotFoundError(f"no frame_*_bounding_boxes_and_masks.json under {bbox_masks_dir}")
    with open(files[0], encoding="utf-8") as handle:
        entries = json.load(handle)
    masks: dict[int, tuple[str, np.ndarray]] = {}
    for entry in entries:
        count = entry.get("count")
        if isinstance(count, list):
            count = count[0]
        masks[int(count)] = (str(entry.get("object_name", count)), np.asarray(entry["mask"], dtype=bool))
    return masks


FIRST_IMAGE_NAMES = ("0.jpg", "0.png", "00000.jpg")


def load_first_real_image(images_dir) -> np.ndarray | None:
    """Real frame 0 (RGB) from ``images0/`` or None (overlays then use a black canvas)."""
    import cv2

    for name in FIRST_IMAGE_NAMES:
        path = os.path.join(str(images_dir), name)
        if os.path.isfile(path):
            return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    return None


def load_object_pose_dump(path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# Overlays
# ---------------------------------------------------------------------------
def draw_contour(img: np.ndarray, mask: np.ndarray, color, thickness: int = 1) -> None:
    import cv2

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, color, thickness)


def write_overlays(out_dir: str, image: np.ndarray | None, camera: PinholeCamera, models: dict[str, ObjectModel], results: dict[str, FitResult], real_masks: dict[int, tuple[str, np.ndarray]]) -> list[str]:
    import cv2

    out_dir = str(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    if image is None:
        image = np.zeros((camera.height, camera.width, 3), dtype=np.uint8)
    combined = image.copy()
    written = []
    for key, res in results.items():
        model = models[key]
        real = real_masks.get(res.count, (None, None))[1]
        before = model.silhouette(np.zeros(2), 0.0)
        after = model.silhouette(res.xy_delta, res.yaw_rad)
        panel = image.copy()
        for canvas in (panel, combined):
            if real is not None:
                draw_contour(canvas, real, (255, 255, 255), 2)
            draw_contour(canvas, before, (255, 60, 60), 1)
            draw_contour(canvas, after, (60, 255, 60), 1)
        label = f"{res.object_name} [{res.count}] {res.status} IoU {res.iou_before:.2f}->{res.iou_after:.2f} shift {100*np.linalg.norm(res.xy_delta):.1f}cm yaw {math.degrees(res.yaw_rad):+.0f}"
        cv2.putText(panel, label, (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1, cv2.LINE_AA)
        path = os.path.join(out_dir, f"overlay_{res.count:02d}_{re.sub(r'[^a-z0-9]+', '_', res.object_name.lower())}.png")
        cv2.imwrite(str(prepare_output(path)), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        written.append(path)
    cv2.putText(combined, "white=real mask  red=sim before  green=sim after", (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1, cv2.LINE_AA)
    path = os.path.join(out_dir, "overlay_all.png")
    cv2.imwrite(str(prepare_output(path)), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    written.append(path)
    return written


def overlay_real_masks_on_render(render_path: str, real_masks: dict[int, tuple[str, np.ndarray]], out_path: str, *, real_hw=(360, 640)) -> str:
    """Draw the real first-frame masks on a sim render (e.g. preview camera.png)."""
    import cv2

    render = cv2.imread(str(render_path))
    if render is None:
        raise FileNotFoundError(f"render not readable: {render_path}")
    h = min(render.shape[0], real_hw[0])
    w = min(render.shape[1], real_hw[1])
    canvas = render[:h, :w].copy()
    for count, (name, mask) in sorted(real_masks.items()):
        draw_contour(canvas, mask[:h, :w], (255, 255, 255), 2)
        c = mask_centroid(mask[:h, :w])
        if c is not None:
            cv2.putText(canvas, f"{count}:{name}", (int(c[0]), int(c[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)
    out_path = str(prepare_output(out_path))
    cv2.imwrite(out_path, canvas)
    return out_path




# ---------------------------------------------------------------------------
# Scene-level fit (old ``cmd_fit``) and check-render
# ---------------------------------------------------------------------------
def select_objects(dump: dict, selectors: list[str] | None) -> list[dict]:
    entries = dump["objects"]
    if not selectors:
        return entries
    chosen = []
    for entry in entries:
        name = str(entry.get("object_name", "")).lower()
        for sel in selectors:
            s = sel.strip().lower()
            if s == str(entry["count"]) or s == entry["safe_key"].lower() or s in name:
                chosen.append(entry)
                break
    return chosen


@dataclass
class PoseFitSettings:
    """``object_pose_refine.py fit`` options; the defaults are the production values."""

    objects: list[str] | None = None
    yaw: str = "sweep"
    yaw_objects: list[str] | None = None
    yaw_step_deg: float = 15.0
    yaw_min_gain: float = 0.02
    no_polish: bool = False
    min_area_ratio: float = 0.5
    max_area_ratio: float = 2.0
    min_iou: float = 0.3
    no_level: bool = False
    level_max_deg: float = 30.0
    no_drop: bool = False
    clearance: float = 0.0005
    allow_overlap: bool = False
    max_overlap: float = 0.25
    height: int = DEFAULT_HEIGHT
    width: int = DEFAULT_WIDTH

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "PoseFitSettings":
        return cls(**{k: getattr(args, k) for k in cls.__dataclass_fields__})


def fit_scene_poses(
    dump_path,
    *,
    scene: str,
    intrinsics,
    extrinsics,
    bbox_masks_dir,
    out_path,
    settings: PoseFitSettings | None = None,
    overlay_dir=None,
    images_dir=None,
    scene_dir: str | None = None,
    log=print,
) -> dict[str, Any]:
    """Fit every selected object of ``dump_path`` and write the override JSON to ``out_path``.

    Same procedure, log lines and JSON as the old ``object_pose_refine.py fit``.
    ``scene_dir`` is only recorded in the JSON (provenance). Overlays are written
    when ``overlay_dir`` is given. Returns the written payload.
    """
    st = settings or PoseFitSettings()
    camera = PinholeCamera.from_files(intrinsics, extrinsics, height=st.height, width=st.width)
    dump = load_object_pose_dump(dump_path)
    real_masks = load_real_masks(bbox_masks_dir)
    entries = select_objects(dump, st.objects)
    if not entries:
        raise ValueError(f"[fit] no objects selected from {dump_path} (selectors {st.objects})")
    yaw_objects = {s.strip().lower() for s in (st.yaw_objects or [])}

    models: dict[str, ObjectModel] = {}
    results: dict[str, FitResult] = {}
    # Objects settle on the physics receiver surface at runtime, which can be
    # below the load-time z_plane (droid2: 1.8 cm, droid41: 5.9 cm); rest the
    # silhouettes there, or the fitted XY is biased by the later drop.
    rest_plane = dump.get("physics_top")
    if rest_plane is None:
        rest_plane = dump.get("z_plane")
    z_plane = None if st.no_drop else rest_plane
    if z_plane is not None and dump.get("z_plane") is not None and abs(float(z_plane) - float(dump["z_plane"])) > 1e-4:
        log(f"[fit] resting objects on physics_top={float(z_plane):.4f} (load-time z_plane={float(dump['z_plane']):.4f})")
    for entry in entries:
        key = entry["safe_key"]
        model = ObjectModel.from_dump_entry(
            entry,
            camera,
            z_plane=z_plane,
            clearance=st.clearance,
            level_max_deg=0.0 if st.no_level else st.level_max_deg,
        )
        models[key] = model
        if model.level_deg or model.z_delta:
            log(
                f"[fit] {entry.get('object_name', key):22s} [{entry['count']}] rested: "
                f"levelled {model.level_deg:.1f} deg, z {model.z_delta:+.3f} m "
                f"(lowest vertex was {model.minz_before:.3f} m, plane {z_plane})"
            )
        real = real_masks.get(int(entry["count"]))
        real_mask = None if real is None else real[1]
        if real is not None and real[0].strip().lower() != str(entry.get("object_name", "")).strip().lower():
            log(f"[fit] note: count {entry['count']} is '{real[0]}' in the mask file, '{entry.get('object_name')}' in the dump")
        mode = st.yaw
        if yaw_objects and not any(
            s == str(entry["count"]) or s == key.lower() or s in str(entry.get("object_name", "")).lower() for s in yaw_objects
        ):
            mode = "off"
        res = fit_object(
            model,
            real_mask,
            yaw_mode=mode,
            yaw_step_deg=st.yaw_step_deg,
            yaw_min_gain=st.yaw_min_gain,
            polish=not st.no_polish,
            min_area_ratio=st.min_area_ratio,
            max_area_ratio=st.max_area_ratio,
            min_iou=st.min_iou,
        )
        results[key] = res
        log(
            f"[fit] {res.object_name:22s} [{res.count}] {res.status:22s} IoU {res.iou_before:.3f} -> {res.iou_after:.3f} "
            f"(xy-only {res.iou_after_xy_only:.3f}) shift ({100*res.xy_delta[0]:+.1f},{100*res.xy_delta[1]:+.1f}) cm "
            f"yaw {math.degrees(res.yaw_rad):+.1f} deg"
            + (f" yaw-range {res.yaw_iou_range:.3f}" if res.yaw_iou_range is not None else "")
            + (f" area sim/real {res.area_ratio:.2f}" if res.area_ratio is not None else "")
        )
        for note in res.notes:
            log(f"       {note}")

    if not st.allow_overlap:
        for note in resolve_footprint_conflicts(models, results, max_overlap=st.max_overlap):
            log(f"[fit] conflict: {note}")

    payload = {
        "version": 1,
        "scene": dump.get("scene", scene),
        "scene_dir": scene_dir,
        "source_dump": os.path.abspath(str(dump_path)),
        "settings": {
            "yaw": st.yaw,
            "yaw_step_deg": st.yaw_step_deg,
            "yaw_min_gain": st.yaw_min_gain,
            "polish": not st.no_polish,
            "min_area_ratio": st.min_area_ratio,
            "max_area_ratio": st.max_area_ratio,
            "min_iou": st.min_iou,
            "level": not st.no_level,
            "level_max_deg": st.level_max_deg,
            "drop_to_z_plane": z_plane,
            "clearance": st.clearance,
            "allow_overlap": st.allow_overlap,
            "max_overlap": st.max_overlap,
            "objects": st.objects,
            "yaw_objects": st.yaw_objects,
        },
        "objects": [results[k].to_json_dict(models[k]) for k in results],
    }
    out = prepare_output(out_path)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    log(f"[fit] wrote {out}")
    if overlay_dir is not None:
        image = load_first_real_image(images_dir) if images_dir is not None else None
        for path in write_overlays(overlay_dir, image, camera, models, results, real_masks):
            log(f"[fit] overlay {path}")
    return payload


def check_render(layout: SceneLayout, render_path, out_path) -> str:
    """Draw the real first-frame masks on a sim render (old ``object_pose_refine.py check-render``)."""
    real_masks = load_real_masks(layout.inputs.bbox_masks_dir)
    return overlay_real_masks_on_render(render_path, real_masks, out_path)


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------
def overlay_dir_for(layout: SceneLayout) -> Path:
    return layout.posefit_dir / "overlays"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    d = PoseFitSettings()
    parser.add_argument("--objects", nargs="*", default=d.objects, help="restrict to these counts / names (substring)")
    parser.add_argument("--yaw", choices=["off", "sweep", "c4"], default=d.yaw)
    parser.add_argument("--yaw-objects", nargs="*", default=d.yaw_objects, help="only sweep yaw for these objects (others: off)")
    parser.add_argument("--yaw-step-deg", type=float, default=d.yaw_step_deg)
    parser.add_argument("--yaw-min-gain", type=float, default=d.yaw_min_gain, help="IoU gain needed to accept a yaw change")
    parser.add_argument("--no-polish", action="store_true")
    parser.add_argument("--min-area-ratio", type=float, default=d.min_area_ratio, help="reject fits whose silhouette area / mask area is below this")
    parser.add_argument("--max-area-ratio", type=float, default=d.max_area_ratio, help="reject fits whose silhouette area / mask area is above this")
    parser.add_argument("--min-iou", type=float, default=d.min_iou, help="reject fits whose final IoU is below this")
    parser.add_argument("--no-level", action="store_true", help="do not rotate tilted thin-axis objects upright")
    parser.add_argument("--level-max-deg", type=float, default=d.level_max_deg, help="only level tilts up to this angle (3 deg minimum)")
    parser.add_argument("--no-drop", action="store_true", help="do not move objects vertically onto the dump's physics_top / z_plane")
    parser.add_argument("--clearance", type=float, default=d.clearance, help="rest height above the plane (evaluator support clearance)")
    parser.add_argument("--allow-overlap", action="store_true", help="apply refined poses even when footprints interpenetrate")
    parser.add_argument("--max-overlap", type=float, default=d.max_overlap, help="footprint containment fraction that counts as a conflict")
    parser.add_argument("--no-overlays", action="store_true")
    parser.add_argument("--height", type=int, default=d.height)
    parser.add_argument("--width", type=int, default=d.width)


def run(layout: SceneLayout, args: argparse.Namespace) -> dict[str, Any]:
    inputs = layout.inputs
    payload = fit_scene_poses(
        layout.object_dump,
        scene=layout.scene,
        intrinsics=inputs.intrinsics,
        extrinsics=inputs.extrinsics,
        bbox_masks_dir=inputs.bbox_masks_dir,
        out_path=layout.refined_poses,
        settings=PoseFitSettings.from_args(args),
        overlay_dir=None if args.no_overlays else overlay_dir_for(layout),
        images_dir=inputs.images_dir,
        scene_dir=str(inputs.meta_dir),
    )
    statuses: dict[str, int] = {}
    for obj in payload["objects"]:
        statuses[obj["status"]] = statuses.get(obj["status"], 0) + 1
    return {"statuses": statuses, "n_objects": len(payload["objects"])}


def _inputs(layout: SceneLayout, args: argparse.Namespace) -> list[Path]:
    inputs = layout.inputs
    return [layout.object_dump, inputs.intrinsics, inputs.extrinsics, inputs.bbox_masks_dir]


def _outputs(layout: SceneLayout, args: argparse.Namespace) -> list[Path]:
    out = [layout.refined_poses]
    if not args.no_overlays:
        out.append(overlay_dir_for(layout) / "overlay_all.png")
    return out


STEP = Step(
    name="objects.posefit",
    help="fit each object's XY + yaw on the table to the real first-frame masks (refined_poses.json)",
    add_arguments=add_arguments,
    run=run,
    inputs=_inputs,
    outputs=_outputs,
    needs="cpu",
)
