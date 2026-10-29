"""Lay out one scene's outputs the way the old evaluator expects them.

    <to>/optimization_results/<s>/aligned_pose_refined.npy, ..._scene_scale_refined.npy,
        operation_surface_plane_fit.json, operation_surface_shadow_v2.json, cropped_collision/
    <to>/echo2/<s>/gaussian.ply, gaussian_trimmed.ply, collider.glb
    <to>/layout/<s>/candidate.json                    (object layout, step 10)

Point the evaluator at it with ``--result <to>/optimization_results/<s>/aligned_pose_refined.npy
--scene_scale <to>/optimization_results/<s>/aligned_pose_scene_scale_refined.npy`` and an
Eden asset path that resolves ``echo2/<s>/...`` to ``<to>/echo2/<s>/...``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from r2s.layout import SceneLayout


def export_scene(layout: SceneLayout, to: Path, *, copy: bool = False) -> dict[str, Path]:
    """Symlink (or copy) a scene's outputs into the old evaluator layout under ``to``."""
    s = layout.scene
    opt = Path(to) / "optimization_results" / s
    echo2 = Path(to) / "echo2" / s
    pairs = {
        opt / "aligned_pose.npy": layout.aligned_pose,
        opt / "aligned_pose_scene_scale.npy": layout.aligned_scale,
        opt / "aligned_pose_refined.npy": layout.refined_pose,
        opt / "aligned_pose_scene_scale_refined.npy": layout.refined_scale,
        opt / "operation_surface_plane_fit.json": layout.plane_fit_json,
        opt / "operation_surface_shadow_v2.json": layout.receiver_json,
        opt / "cropped_collision" / layout.cropped_collider_glb.name: layout.cropped_collider_glb,
        opt / "cropped_collision" / (layout.cropped_collider_glb.name + ".json"):
            layout.cropped_collider_glb.with_name(layout.cropped_collider_glb.name + ".json"),
        echo2 / "gaussian.ply": layout.gaussian_ply,
        echo2 / "gaussian_trimmed.ply": layout.gaussian_trimmed_ply,
        echo2 / "collider.glb": layout.collider_glb,
        Path(to) / "layout" / s / "candidate.json": layout.layout_candidate,
    }
    written: dict[str, Path] = {}
    for dst, src in pairs.items():
        if not src.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if copy:
            shutil.copy2(src.resolve(), dst)
        else:
            dst.symlink_to(src.resolve())
        written[str(dst.relative_to(to))] = src
    return written
