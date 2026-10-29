"""Eden scene options for reconstructed scenes (GPU code only; imports Eden).

``ReconSceneOptions`` is a ``SceneOptions`` with a ``background`` entity (the scene
collider) that adds a default ground plane when none is given -- what the old scale-fit
script's scene did. Splats use Eden's own ``SplatOptions`` / ``PCDOptions`` directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from eden.options.entities import EntityOptions, PlaneOptions, SceneOptions


class ReconSceneOptions(SceneOptions):
    if TYPE_CHECKING:
        plane: PlaneOptions

    background: EntityOptions | None = None

    def __init__(self, is_support_enabled: bool = False, **data):
        if "plane" not in data:
            data["plane"] = PlaneOptions(is_support_enabled=is_support_enabled)
        super().__init__(**data)


__all__ = ["ReconSceneOptions"]
