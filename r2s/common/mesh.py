"""Mesh helpers shared by the surface, collision and object steps."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_combined_mesh(mesh_path: str | Path):
    """Load a GLB/OBJ/PLY as one ``trimesh.Trimesh`` with scene-graph transforms applied."""
    import trimesh

    loaded = trimesh.load(str(mesh_path), force="scene")
    if isinstance(loaded, trimesh.Trimesh):
        return loaded.copy()

    meshes = []
    if hasattr(loaded, "graph") and hasattr(loaded.graph, "nodes_geometry"):
        for node_name in loaded.graph.nodes_geometry:
            try:
                transform, geom_name = loaded.graph.get(frame_to=node_name)
            except TypeError:
                transform, geom_name = loaded.graph.get(node_name)
            geom = loaded.geometry.get(geom_name)
            if isinstance(geom, trimesh.Trimesh):
                mesh = geom.copy()
                mesh.apply_transform(transform)
                meshes.append(mesh)

    if not meshes:
        dumped = loaded.dump()
        if isinstance(dumped, trimesh.Trimesh):
            meshes = [dumped]
        else:
            meshes = [m for m in dumped if isinstance(m, trimesh.Trimesh)]

    if not meshes:
        raise ValueError(f"No mesh geometry found in {mesh_path}")
    return trimesh.util.concatenate(meshes)


def mesh_world_geometry(
    mesh_path: str | Path,
    scene_scale: float,
    world_pos: np.ndarray,
    world_rot: np.ndarray,
    mesh_import_rot: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """World-space vertices and faces of a posed, scaled mesh."""
    mesh = load_combined_mesh(mesh_path)
    if len(mesh.faces) == 0:
        raise ValueError(f"Mesh has no faces: {mesh_path}")

    vertices_import = np.asarray(mesh.vertices, dtype=np.float64) @ mesh_import_rot.T
    vertices_world = (vertices_import * float(scene_scale)) @ world_rot.T + world_pos[None, :]
    faces = np.asarray(mesh.faces, dtype=np.int64)
    return vertices_world, faces
