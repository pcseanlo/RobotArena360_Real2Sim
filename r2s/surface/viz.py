"""Optional visualizations of the support-surface fits.

* :func:`write_surface_viz_ply` -- colored point/plane PLY of the plane fit (``--viz-ply``)
* :func:`launch_surface_viz_viewer` / :func:`launch_render_receiver_viz_viewer` -- viser
  viewers (``--viewer``). They block until Ctrl-C; viser is imported only here, lazily.

Moved from the old ``operation_surface_precompute.py`` and
``operation_surface_render_receiver_fit.py``.
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from r2s.common.camera import load_camera_pose_from_extrinsics
from r2s.common.io import prepare_output
from r2s.common.quat import quat_normalize
from r2s.surface.geometry import (
    PlaneCandidate,
    footprint_mask_from_world_points,
    selected_plane,
    upward_normal_mask,
    workspace_mask,
)


def downsample_mesh_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
    max_faces: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if max_faces <= 0 or len(faces) <= max_faces:
        return vertices, faces
    face_idx = rng.choice(len(faces), size=int(max_faces), replace=False)
    selected_faces = faces[face_idx]
    used_vertices, inverse = np.unique(selected_faces.reshape(-1), return_inverse=True)
    remapped_faces = inverse.reshape((-1, 3)).astype(np.int64)
    return vertices[used_vertices], remapped_faces


def _support_box_surface_corners(
    support_box: dict[str, Any],
    normal: np.ndarray,
    clearance: float,
    lift: float,
) -> np.ndarray:
    u = np.asarray(support_box["basis_u"], dtype=np.float64)
    v = np.asarray(support_box["basis_v"], dtype=np.float64)
    center = np.asarray(support_box["center"], dtype=np.float64)
    thickness = float(support_box["size"][2]) if len(support_box["size"]) >= 3 else 0.0
    surface_center = center - normal * (float(clearance) + 0.5 * thickness) + normal * float(lift)
    half_u = 0.5 * float(support_box["extents"][0])
    half_v = 0.5 * float(support_box["extents"][1])
    return np.stack(
        (
            surface_center - half_u * u - half_v * v,
            surface_center + half_u * u - half_v * v,
            surface_center + half_u * u + half_v * v,
            surface_center - half_u * u + half_v * v,
        ),
        axis=0,
    )


def _format_ply_vertex(point: np.ndarray, color: tuple[int, int, int]) -> str:
    return (
        f"{float(point[0]):.8f} {float(point[1]):.8f} {float(point[2]):.8f} "
        f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
    )


def build_surface_viz_geometry(
    points: np.ndarray,
    normals: np.ndarray,
    bounds: tuple[float, float, float, float, float, float],
    candidates: list[PlaneCandidate],
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    max_points = int(args.viz_max_points)
    if max_points > 0 and len(points) > max_points:
        idx = rng.choice(len(points), size=max_points, replace=False)
        viz_points = points[idx]
        viz_normals = normals[idx]
    else:
        viz_points = points
        viz_normals = normals

    reachable = workspace_mask(viz_points, bounds)
    upward = upward_normal_mask(viz_normals, args)
    support = reachable & upward

    point_colors = np.full((len(viz_points), 3), (85, 165, 235), dtype=np.uint8)
    point_colors[~reachable] = (70, 70, 70)
    point_colors[support] = (65, 185, 95)

    selected_index = int(args.select_index) if args.select_index is not None and candidates else 0
    if candidates and (selected_index < 0 or selected_index >= len(candidates)):
        raise ValueError(f"--select-index {selected_index} is out of range for {len(candidates)} candidates.")
    selected_candidates = candidates[selected_index : selected_index + 1] if candidates else []
    if selected_candidates:
        selected = selected_candidates[0]
        normal = np.asarray(selected.plane_normal, dtype=np.float64)
        normal /= max(float(np.linalg.norm(normal)), 1e-12)
        plane_dist = np.abs(viz_points @ normal + float(selected.plane_offset))
        plane_thresh = args.viz_selected_point_thresh
        if plane_thresh is None or float(plane_thresh) <= 0.0:
            plane_thresh = max(float(args.inlier_thresh), float(args.expand_inlier_thresh or 0.0))
        selected_point_mask = np.zeros(len(viz_points), dtype=bool)
        for support_box in selected.support_boxes:
            basis_u = np.asarray(support_box["basis_u"], dtype=np.float64)
            basis_v = np.asarray(support_box["basis_v"], dtype=np.float64)
            center = np.asarray(support_box["center"], dtype=np.float64)
            thickness = float(support_box["size"][2]) if len(support_box["size"]) >= 3 else 0.0
            surface_center = center - normal * (float(args.clearance) + 0.5 * thickness)
            rel = viz_points - surface_center[None, :]
            uv = np.column_stack((rel @ basis_u, rel @ basis_v))
            selected_point_mask |= (
                reachable
                & (plane_dist <= float(plane_thresh))
                & (
                    np.abs(uv[:, 0])
                    <= 0.5 * float(support_box["extents"][0]) + float(args.viz_selected_point_padding)
                )
                & (
                    np.abs(uv[:, 1])
                    <= 0.5 * float(support_box["extents"][1]) + float(args.viz_selected_point_padding)
                )
            )
        point_colors[selected_point_mask] = (255, 215, 45)
    if args.viz_selected_only:
        draw_candidates = selected_candidates
    else:
        max_candidates = max(0, int(args.viz_max_candidates))
        ranked = [candidate for i, candidate in enumerate(candidates) if i != selected_index]
        draw_candidates = (selected_candidates + ranked)[:max_candidates]

    palette = [
        (35, 135, 255),   # selected
        (255, 156, 55),
        (245, 215, 65),
        (215, 85, 190),
        (70, 205, 210),
        (175, 130, 255),
        (255, 105, 105),
        (120, 215, 120),
    ]
    plane_vertices: list[tuple[np.ndarray, tuple[int, int, int]]] = []
    plane_faces: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = []

    drawn_patches = 0
    for draw_index, candidate in enumerate(draw_candidates):
        color = palette[min(draw_index, len(palette) - 1)]
        lift = float(args.viz_plane_lift) * float(draw_index + 1)
        normal = np.asarray(candidate.plane_normal, dtype=np.float64)
        normal /= max(float(np.linalg.norm(normal)), 1e-12)
        support_boxes = candidate.support_boxes or [
            {
                "center": candidate.center,
                "size": candidate.size,
                "basis_u": candidate.basis_u,
                "basis_v": candidate.basis_v,
                "extents": candidate.extents,
            }
        ]
        for support_box in support_boxes:
            corners = _support_box_surface_corners(support_box, normal, args.clearance, lift)
            base = len(viz_points) + len(plane_vertices)
            for corner in corners:
                plane_vertices.append((corner, color))
            plane_faces.append(((base, base + 1, base + 2), color))
            plane_faces.append(((base, base + 2, base + 3), color))
            drawn_patches += 1

    if plane_vertices:
        plane_points = np.stack([point for point, _ in plane_vertices], axis=0)
        plane_colors = np.asarray([color for _, color in plane_vertices], dtype=np.uint8)
    else:
        plane_points = np.zeros((0, 3), dtype=np.float64)
        plane_colors = np.zeros((0, 3), dtype=np.uint8)
    if plane_faces:
        faces = np.asarray([face for face, _ in plane_faces], dtype=np.int64)
    else:
        faces = np.zeros((0, 3), dtype=np.int64)

    return viz_points, point_colors, plane_points, plane_colors, faces, drawn_patches


def write_surface_viz_ply(
    path: str,
    points: np.ndarray,
    normals: np.ndarray,
    bounds: tuple[float, float, float, float, float, float],
    candidates: list[PlaneCandidate],
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> int:
    viz_points, point_colors, plane_points, plane_colors, faces, drawn = build_surface_viz_geometry(
        points=points,
        normals=normals,
        bounds=bounds,
        candidates=candidates,
        args=args,
        rng=rng,
    )

    out_path = prepare_output(path)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment operation_surface_precompute visualization\n")
        f.write("comment point colors: dark gray=outside workspace, blue=inside crop non-support, green=upward support samples, yellow=selected plane inliers\n")
        f.write("comment plane colors: blue=selected candidate, warm/bright colors=next ranked candidates\n")
        f.write(f"element vertex {len(viz_points) + len(plane_points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for point, color in zip(viz_points, point_colors):
            f.write(_format_ply_vertex(point, (int(color[0]), int(color[1]), int(color[2]))))
        for point, color in zip(plane_points, plane_colors):
            f.write(_format_ply_vertex(point, (int(color[0]), int(color[1]), int(color[2]))))
        for face in faces:
            color = plane_colors[int(face[0]) - len(viz_points)] if len(plane_colors) else (255, 255, 255)
            f.write(
                f"3 {face[0]} {face[1]} {face[2]} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )
    return drawn


def _viser_add_frame(
    server: Any,
    name: str,
    position: np.ndarray,
    quat_wxyz: np.ndarray,
    axes_length: float,
    axes_radius: float,
) -> None:
    add_frame = getattr(server.scene, "add_frame", None)
    if add_frame is None:
        print("[viser] installed viser does not support coordinate frame overlays")
        return
    handle = add_frame(
        name=name,
        axes_length=float(axes_length),
        axes_radius=float(axes_radius),
    )
    handle.position = np.asarray(position, dtype=np.float64)
    handle.wxyz = quat_normalize(np.asarray(quat_wxyz, dtype=np.float64))


def launch_surface_viz_viewer(
    points: np.ndarray,
    normals: np.ndarray,
    bounds: tuple[float, float, float, float, float, float],
    candidates: list[PlaneCandidate],
    args: argparse.Namespace,
    rng: np.random.Generator,
    mesh_viz: tuple[np.ndarray, np.ndarray] | None = None,
) -> None:
    try:
        import viser
    except ImportError as exc:
        raise RuntimeError(
            "`--viewer` needs the `viser` Python package (pip install viser). "
            "Use `--viz-ply` for dependency-free inspection."
        ) from exc

    viz_points, point_colors, plane_points, plane_colors, faces, drawn = build_surface_viz_geometry(
        points=points,
        normals=normals,
        bounds=bounds,
        candidates=candidates,
        args=args,
        rng=rng,
    )

    server = viser.ViserServer(port=int(args.viser_port))
    if hasattr(server.scene, "set_up_direction"):
        server.scene.set_up_direction("+z")
    frame_len = float(args.viser_frame_length)
    frame_rad = max(frame_len * 0.035, 0.001)
    _viser_add_frame(
        server,
        "/frames/world",
        np.zeros(3, dtype=np.float64),
        np.array((1.0, 0.0, 0.0, 0.0), dtype=np.float64),
        axes_length=frame_len,
        axes_radius=frame_rad,
    )
    _viser_add_frame(
        server,
        "/frames/collider_root",
        np.asarray(args.collider_root_pos, dtype=np.float64),
        np.asarray(args.collider_root_quat_wxyz, dtype=np.float64),
        axes_length=frame_len,
        axes_radius=frame_rad,
    )
    if args.extrinsics:
        cam_pos, cam_quat = load_camera_pose_from_extrinsics(args.extrinsics)
        _viser_add_frame(
            server,
            "/frames/camera",
            cam_pos,
            cam_quat,
            axes_length=frame_len,
            axes_radius=frame_rad,
        )
    server.scene.add_point_cloud(
        "/operation_surface/samples",
        points=viz_points.astype(np.float32),
        colors=point_colors,
        point_size=float(args.viser_point_size),
    )
    if mesh_viz is not None:
        mesh_vertices, mesh_faces = downsample_mesh_faces(
            vertices=mesh_viz[0],
            faces=mesh_viz[1],
            max_faces=int(args.viser_mesh_max_faces),
            rng=rng,
        )
        try:
            server.scene.add_mesh_simple(
                "/scene/full_mesh",
                vertices=mesh_vertices.astype(np.float32),
                faces=mesh_faces.astype(np.int64),
                color=(160, 160, 160),
                opacity=float(args.viser_mesh_opacity),
                side="double",
            )
        except TypeError:
            server.scene.add_mesh_simple(
                "/scene/full_mesh",
                vertices=mesh_vertices.astype(np.float32),
                faces=mesh_faces.astype(np.int64),
                color=(160, 160, 160),
            )

    for candidate_idx in range(drawn):
        start = candidate_idx * 4
        end = start + 4
        if end > len(plane_points):
            break
        local_vertices = plane_points[start:end].astype(np.float32)
        local_faces = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
        color = tuple(int(v) for v in plane_colors[start])
        name = "/operation_surface/selected" if candidate_idx == 0 else f"/operation_surface/candidate_{candidate_idx:02d}"
        try:
            server.scene.add_mesh_simple(
                name,
                vertices=local_vertices,
                faces=local_faces,
                color=color,
                opacity=float(args.viser_plane_opacity),
                side="double",
            )
        except TypeError:
            server.scene.add_mesh_simple(
                name,
                vertices=local_vertices,
                faces=local_faces,
                color=color,
            )

    print(
        f"[viser] open http://localhost:{args.viser_port} "
        f"(surfaces={drawn}; full_mesh={'yes' if mesh_viz is not None else 'no'}; "
        "cyan=inside crop non-support, green=upward support samples)"
    )
    print("[viser] surface viewer running; Ctrl-C to exit")
    import time

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


# ------------------------------------------------------------------ receiver viewer
def _rectangle_vertices_from_box(box: dict[str, Any], *, lift: float = 0.0) -> np.ndarray:
    center = np.asarray(box["center"], dtype=np.float64)
    u = np.asarray(box.get("basis_u", [1.0, 0.0, 0.0]), dtype=np.float64)
    v = np.asarray(box.get("basis_v", [0.0, 1.0, 0.0]), dtype=np.float64)
    normal = np.asarray(box.get("plane_normal", np.cross(u, v)), dtype=np.float64)
    u /= max(float(np.linalg.norm(u)), 1e-12)
    v = v - np.dot(v, u) * u
    v /= max(float(np.linalg.norm(v)), 1e-12)
    normal = normal - np.dot(normal, u) * u - np.dot(normal, v) * v
    normal /= max(float(np.linalg.norm(normal)), 1e-12)

    size = np.asarray(box.get("size", [0.0, 0.0, 0.0]), dtype=np.float64)
    extents = np.asarray(box.get("extents", size[:2]), dtype=np.float64)
    thickness = float(size[2]) if len(size) >= 3 else 0.0
    face_center = center + normal * (0.5 * thickness + float(lift))
    half_u = 0.5 * float(extents[0])
    half_v = 0.5 * float(extents[1])
    return np.asarray(
        [
            face_center - half_u * u - half_v * v,
            face_center + half_u * u - half_v * v,
            face_center + half_u * u + half_v * v,
            face_center - half_u * u + half_v * v,
        ],
        dtype=np.float64,
    )


def _selected_reference_box(selected: dict[str, Any]) -> dict[str, Any]:
    return {
        "center": selected["center"],
        "size": selected["size"],
        "extents": selected.get("extents", selected["size"][:2]),
        "basis_u": selected.get("basis_u", [1.0, 0.0, 0.0]),
        "basis_v": selected.get("basis_v", [0.0, 1.0, 0.0]),
        "plane_normal": selected.get("plane_normal", [0.0, 0.0, 1.0]),
    }


def _receiver_box_mask(points_world: np.ndarray, receiver: dict[str, Any] | None) -> np.ndarray:
    points = np.asarray(points_world, dtype=np.float64)
    mask = np.zeros(len(points), dtype=bool)
    if receiver is None or len(points) == 0:
        return mask
    for box in receiver.get("support_boxes") or [receiver]:
        center = np.asarray(box["center"], dtype=np.float64)
        u = np.asarray(box.get("basis_u", [1.0, 0.0, 0.0]), dtype=np.float64)
        v = np.asarray(box.get("basis_v", [0.0, 1.0, 0.0]), dtype=np.float64)
        extents = np.asarray(box.get("extents", box.get("size", [0.0, 0.0])[:2]), dtype=np.float64)
        u /= max(float(np.linalg.norm(u)), 1e-12)
        v = v - np.dot(v, u) * u
        v /= max(float(np.linalg.norm(v)), 1e-12)
        rel = points - center[None, :]
        mask |= (
            (np.abs(rel @ u) <= 0.5 * float(extents[0]))
            & (np.abs(rel @ v) <= 0.5 * float(extents[1]))
        )
    return mask


def _receiver_geometry_mask(points_world: np.ndarray, receiver: dict[str, Any] | None) -> np.ndarray:
    """Prefer the conservative lighting polygon for visualization selection."""
    if receiver is None:
        return np.zeros(len(points_world), dtype=bool)
    footprint = (receiver.get("footprints") or {}).get("lighting")
    if not footprint:
        return _receiver_box_mask(points_world, receiver)
    return footprint_mask_from_world_points(points_world, footprint)


def _footprint_mesh_arrays(
    footprint: dict[str, Any],
    *,
    lift: float,
) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(footprint.get("vertices_world") or [], dtype=np.float64)
    triangles = np.asarray(footprint.get("triangles") or [], dtype=np.int64)
    if len(vertices) and float(lift):
        normal = np.asarray(footprint.get("plane_normal", [0.0, 0.0, 1.0]), dtype=np.float64)
        normal /= max(float(np.linalg.norm(normal)), 1e-12)
        vertices = vertices + float(lift) * normal[None, :]
    return vertices, triangles


def _render_receiver_support_mask(
    points_world: np.ndarray,
    selected: dict[str, Any],
    *,
    gaussian_plane_band: float,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    from r2s.surface.receiver_fit import _reference_footprint_mask

    points = np.asarray(points_world, dtype=np.float64)
    normal, offset, surface_u, surface_v = selected_plane(selected)
    near_plane = np.abs(points @ normal + offset) <= float(gaussian_plane_band)
    support = near_plane.copy()
    if bool(getattr(args, "receiver_clip_to_reference", True)):
        support &= _reference_footprint_mask(
            selected,
            points,
            normal=normal,
            offset=float(offset),
            u=surface_u,
            v=surface_v,
            margin=float(getattr(args, "receiver_reference_margin", 0.0)),
        )
    return near_plane, support


def _sample_masked_points(
    points_world: np.ndarray,
    mask: np.ndarray,
    *,
    max_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    indices = np.flatnonzero(np.asarray(mask, dtype=bool))
    if len(indices) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if max_points > 0 and len(indices) > max_points:
        indices = rng.choice(indices, size=int(max_points), replace=False)
    return np.asarray(points_world, dtype=np.float64)[indices]


def _sample_viser_points(
    points_world: np.ndarray,
    selected: dict[str, Any],
    *,
    receiver: dict[str, Any] | None,
    gaussian_plane_band: float,
    args: argparse.Namespace,
    max_points: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    points = np.asarray(points_world, dtype=np.float64)
    if len(points) == 0:
        empty_colors = np.zeros((0, 3), dtype=np.uint8)
        return points.reshape(0, 3), empty_colors, points.reshape(0, 3), points.reshape(0, 3), {}

    near_plane, support = _render_receiver_support_mask(
        points,
        selected,
        gaussian_plane_band=float(gaussian_plane_band),
        args=args,
    )
    selected_support = support & _receiver_geometry_mask(points, receiver)
    count = len(points)
    if max_points > 0 and count > max_points:
        indices = rng.choice(count, size=int(max_points), replace=False)
    else:
        indices = np.arange(count)
    colors = np.full((len(indices), 3), (150, 150, 150), dtype=np.uint8)
    colors[near_plane[indices]] = np.asarray((70, 145, 255), dtype=np.uint8)
    colors[support[indices]] = np.asarray((45, 205, 115), dtype=np.uint8)
    colors[selected_support[indices]] = np.asarray((255, 155, 35), dtype=np.uint8)

    max_support_points = int(getattr(args, "viser_max_support_points", max_points))
    support_points = _sample_masked_points(
        points,
        support,
        max_points=max_support_points,
        rng=rng,
    )
    selected_support_points = _sample_masked_points(
        points,
        selected_support,
        max_points=max_support_points,
        rng=rng,
    )
    stats = {
        "near_plane_count": int(near_plane.sum()),
        "receiver_support_count": int(support.sum()),
        "selected_receiver_support_count": int(selected_support.sum()),
    }
    return points[indices], colors, support_points, selected_support_points, stats


def launch_render_receiver_viz_viewer(
    *,
    points_world: np.ndarray,
    selected: dict[str, Any],
    receiver: dict[str, Any] | None,
    args: argparse.Namespace,
) -> None:
    try:
        import viser
    except ImportError as exc:
        raise RuntimeError(
            "`--viewer` needs the `viser` Python package (pip install viser)."
        ) from exc

    rng = np.random.default_rng(int(args.seed))
    viz_points, point_colors, support_points, selected_support_points, support_stats = _sample_viser_points(
        points_world,
        selected,
        receiver=receiver,
        gaussian_plane_band=float(args.gaussian_plane_band),
        args=args,
        max_points=int(args.viser_max_points),
        rng=rng,
    )

    server = viser.ViserServer(port=int(args.viser_port))
    if hasattr(server.scene, "set_up_direction"):
        server.scene.set_up_direction("+z")

    if len(viz_points):
        server.scene.add_point_cloud(
            "/render_receiver/gaussians",
            points=viz_points.astype(np.float32),
            colors=point_colors,
            point_size=float(args.viser_point_size),
        )

    if len(support_points):
        server.scene.add_point_cloud(
            "/render_receiver/receiver_support_points",
            points=support_points.astype(np.float32),
            colors=np.full((len(support_points), 3), (0, 235, 110), dtype=np.uint8),
            point_size=float(args.viser_point_size) * 1.5,
        )

    if len(selected_support_points):
        server.scene.add_point_cloud(
            "/render_receiver/selected_receiver_support_points",
            points=selected_support_points.astype(np.float32),
            colors=np.full((len(selected_support_points), 3), (255, 155, 35), dtype=np.uint8),
            point_size=float(args.viser_point_size) * 2.0,
        )

    faces = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
    reference_vertices = _rectangle_vertices_from_box(
        _selected_reference_box(selected),
        lift=float(args.viser_plane_lift),
    )
    try:
        server.scene.add_mesh_simple(
            "/render_receiver/reference_operation_surface",
            vertices=reference_vertices.astype(np.float32),
            faces=faces,
            color=(255, 90, 80),
            opacity=float(args.viser_plane_opacity),
            side="double",
        )
    except TypeError:
        server.scene.add_mesh_simple(
            "/render_receiver/reference_operation_surface",
            vertices=reference_vertices.astype(np.float32),
            faces=faces,
            color=(255, 90, 80),
        )

    footprint_count = 0
    box_count = 0
    if receiver is not None:
        footprints = receiver.get("footprints") or {}
        footprint_styles = {
            "physics": ((40, 205, 235), 0.34, 2.0),
            "lighting": ((255, 205, 45), float(args.viser_receiver_opacity), 3.0),
        }
        for role in ("physics", "lighting"):
            footprint = footprints.get(role)
            if not footprint:
                continue
            vertices, footprint_faces = _footprint_mesh_arrays(
                footprint,
                lift=float(args.viser_plane_lift) * footprint_styles[role][2],
            )
            if not len(vertices) or not len(footprint_faces):
                continue
            color, opacity, _ = footprint_styles[role]
            try:
                server.scene.add_mesh_simple(
                    f"/render_receiver/footprints/{role}",
                    vertices=vertices.astype(np.float32),
                    faces=footprint_faces,
                    color=color,
                    opacity=opacity,
                    side="double",
                )
            except TypeError:
                server.scene.add_mesh_simple(
                    f"/render_receiver/footprints/{role}",
                    vertices=vertices.astype(np.float32),
                    faces=footprint_faces,
                    color=color,
                )
            footprint_count += 1

        # Old descriptors have no polygon; retain their box visualization as a
        # fallback. New descriptors keep boxes in JSON for diagnostics only.
        if footprint_count == 0:
            for index, box in enumerate(receiver.get("support_boxes") or [receiver]):
                vertices = _rectangle_vertices_from_box(box, lift=float(args.viser_plane_lift) * 2.0)
                color = (255, 210, 70) if index == 0 else (55, 220, 135)
                name = "/render_receiver/selected" if index == 0 else f"/render_receiver/box_{index:02d}"
                try:
                    server.scene.add_mesh_simple(
                        name,
                        vertices=vertices.astype(np.float32),
                        faces=faces,
                        color=color,
                        opacity=float(args.viser_receiver_opacity),
                        side="double",
                    )
                except TypeError:
                    server.scene.add_mesh_simple(
                        name,
                        vertices=vertices.astype(np.float32),
                        faces=faces,
                        color=color,
                    )
                box_count += 1

    print(
        f"[viser] open http://localhost:{args.viser_port} "
        f"(points={len(viz_points)}; receiver_footprints={footprint_count}; "
        f"fallback_boxes={box_count}; "
        f"support={support_stats.get('receiver_support_count', 0)}; "
        f"selected_support={support_stats.get('selected_receiver_support_count', 0)}; "
        "gray=all gaussians, blue=near plane, green=support, orange=inside lighting polygon, "
        "cyan=physics footprint, yellow=lighting footprint, red=legacy reference)"
    )
    print("[viser] render receiver viewer running; Ctrl-C to exit")
    import time

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
