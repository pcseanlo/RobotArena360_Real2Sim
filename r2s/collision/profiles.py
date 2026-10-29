"""Topology-aware collision settings for reconstructed scene objects.

Port of the old ``object_collision_profiles.py`` (unchanged algorithm). The sim
assembly calls :func:`select_object_collision_profile` per object with the
evaluator's production arguments (``requested_profile="auto"``,
``detailed_decompose_error_threshold=0.05``, ``detailed_coacd_threshold=0.03``,
``legacy_face_num=1000``) and may then apply
``r2s.objects.overrides.apply_collision_override``.

Reconstructed GLBs are frequently open, fragmented, and much denser than a
collision mesh needs to be.  Feeding those meshes directly to convex
decomposition can create many tiny, uneven contact faces.  This module keeps
the choice deterministic and independent of scene names: clean meshes use the
detailed profile, while risky scans use the established decimated profile.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Literal

import numpy as np

from r2s.common.io import prepare_output
from r2s.common.mesh import load_combined_mesh


CollisionProfileName = Literal["auto", "detailed", "legacy"]


@dataclass(frozen=True)
class MeshCollisionQuality:
    face_count: int
    vertex_count: int
    component_count: int
    watertight: bool
    degenerate_face_fraction: float
    min_extent_ratio: float
    risk_score: int
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ObjectCollisionSelection:
    requested_profile: str
    selected_profile: str
    decimate: bool
    decimate_face_num: int
    decimate_aggressiveness: int
    decompose_object_error_threshold: float
    coacd_threshold: float
    quality: MeshCollisionQuality

    def to_json_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["quality"]["reasons"] = list(self.quality.reasons)
        return result


def inspect_mesh_collision_quality(mesh_path: str) -> MeshCollisionQuality:
    """Return inexpensive, deterministic collision-risk metrics for a mesh."""
    mesh = load_combined_mesh(mesh_path)
    vertices = np.asarray(mesh.vertices, dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 3)
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError(f"Collision mesh has no triangles: {mesh_path}")

    triangles = vertices[faces]
    twice_area = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    scale = max(float(np.linalg.norm(np.ptp(vertices, axis=0))), 1e-12)
    degenerate_fraction = float(np.mean(twice_area <= (scale * scale * 1e-12)))

    try:
        component_count = len(mesh.split(only_watertight=False))
    except Exception:
        component_count = 1
    extents = np.asarray(mesh.extents, dtype=np.float64)
    max_extent = max(float(np.max(extents)), 1e-12)
    min_extent_ratio = float(np.min(extents) / max_extent)

    reasons: list[str] = []
    score = 0
    if not bool(mesh.is_watertight):
        reasons.append("non-watertight")
        score += 1
    if component_count > 4:
        reasons.append(f"fragmented:{component_count}")
        score += 2
    elif component_count > 1:
        reasons.append(f"multi-component:{component_count}")
        score += 1
    if len(faces) > 100000:
        reasons.append(f"extremely-dense:{len(faces)}")
        score += 2
    elif len(faces) > 5000:
        reasons.append(f"dense:{len(faces)}")
        score += 1
    if degenerate_fraction > 0.005:
        reasons.append(f"degenerate:{degenerate_fraction:.3f}")
        score += 1
    if min_extent_ratio < 0.035:
        reasons.append(f"very-thin:{min_extent_ratio:.3f}")
        score += 1

    return MeshCollisionQuality(
        face_count=int(len(faces)),
        vertex_count=int(len(vertices)),
        component_count=int(component_count),
        watertight=bool(mesh.is_watertight),
        degenerate_face_fraction=degenerate_fraction,
        min_extent_ratio=min_extent_ratio,
        risk_score=score,
        reasons=tuple(reasons),
    )


def select_object_collision_profile(
    mesh_path: str,
    *,
    requested_profile: CollisionProfileName = "auto",
    inspect_quality: bool = True,
    decimate_override: bool | None = None,
    detailed_decompose_error_threshold: float = 0.05,
    detailed_coacd_threshold: float = 0.03,
    legacy_face_num: int = 1000,
) -> ObjectCollisionSelection:
    """Choose stable settings without using scene or semantic object names."""
    if requested_profile not in ("auto", "detailed", "legacy"):
        raise ValueError(f"Unknown collision profile: {requested_profile}")
    if requested_profile == "auto" and not inspect_quality:
        raise ValueError("The auto collision profile requires mesh quality inspection")
    quality = (
        inspect_mesh_collision_quality(mesh_path)
        if inspect_quality
        else MeshCollisionQuality(
            face_count=-1,
            vertex_count=-1,
            component_count=-1,
            watertight=False,
            degenerate_face_fraction=0.0,
            min_extent_ratio=0.0,
            risk_score=-1,
            reasons=("not-inspected:explicit-profile",),
        )
    )
    selected = requested_profile
    if selected == "auto":
        # Two independent warning signals avoid downgrading a clean, merely
        # high-resolution visual mesh.  Fragmentation alone is intentionally a
        # strong signal because disconnected scan debris produces bad hulls.
        selected = "legacy" if quality.risk_score >= 2 else "detailed"

    if selected == "legacy":
        decimate = True
        face_num = int(legacy_face_num)
        aggressiveness = 3
        decompose_threshold = 0.15
        coacd_threshold = 0.05
    else:
        decimate = False
        face_num = max(int(legacy_face_num), 1000)
        aggressiveness = 2
        decompose_threshold = float(detailed_decompose_error_threshold)
        coacd_threshold = float(detailed_coacd_threshold)

    if decimate_override is not None:
        decimate = bool(decimate_override)

    return ObjectCollisionSelection(
        requested_profile=requested_profile,
        selected_profile=selected,
        decimate=decimate,
        decimate_face_num=face_num,
        decimate_aggressiveness=aggressiveness,
        decompose_object_error_threshold=decompose_threshold,
        coacd_threshold=coacd_threshold,
        quality=quality,
    )


def write_collision_profile_manifest(
    output_path: str,
    selections: dict[str, ObjectCollisionSelection],
) -> None:
    """Persist the actual per-object settings used by a run."""
    output_path = str(prepare_output(output_path))
    payload = {
        "version": 1,
        "objects": {key: value.to_json_dict() for key, value in selections.items()},
    }
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
