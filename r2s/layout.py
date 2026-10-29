"""Every per-scene file the pipeline reads or writes, in one place.

Inputs (step 0) follow the scene_generation layout. Step 0 writes them under
``paths.scene_data_out`` (by default the run's own ``<work>/scene_data``); steps
1-10 read them from there once the scene has been imported into the run
(``<scene_data_out>/<scene_type>/<s>/`` exists), else from ``paths.scene_data``
(the existing scene_generation data). Setting ``R2S_SCENE_DATA`` pins the read
root to ``scene_data``. Outputs of steps 1-10 live under ``<work>/scenes/<scene>/``.
File names inside the output folders keep the old repo's names
(``aligned_pose_refined.npy``, ``operation_surface_plane_fit.json``, ...) so
``r2s export`` can hand them to the evaluator unchanged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from r2s.config import Paths, SceneConfig, SceneRegistry

CROP_GLB_NAME = "reachable_collider_r1p100_zm0p150_1p450_s24.glb"
# Set explicitly, this pins SceneLayout.inputs to ``paths.scene_data`` (configs/paths.yaml).
SCENE_DATA_ENV = "R2S_SCENE_DATA"


@dataclass(frozen=True)
class InputLayout:
    """Step-0 outputs for one scene (scene_generation layout)."""

    root: Path  # scene_data or scene_data_out
    scene: str
    scene_type: str

    @property
    def meta_dir(self) -> Path:  # DATA/DROID/<s>
        return self.root / self.scene_type / self.scene

    @property
    def intrinsics(self) -> Path:
        return self.meta_dir / "intrinsics.npy"

    @property
    def extrinsics(self) -> Path:
        return self.meta_dir / "extrinsics.npy"

    @property
    def lang(self) -> Path:
        return self.meta_dir / "lang.txt"

    @property
    def images_dir(self) -> Path:
        return self.meta_dir / "images0"

    @property
    def depth_dir(self) -> Path:
        return self.meta_dir / "depth"

    @property
    def joints(self) -> Path:
        return self.meta_dir / "joints.npy"

    @property
    def joints_old(self) -> Path:
        return self.meta_dir / "joints_old.npy"

    @property
    def grippers(self) -> Path:
        return self.meta_dir / "grippers.npy"

    @property
    def tcp(self) -> Path:
        return self.meta_dir / "tcp.npy"

    @property
    def video(self) -> Path:
        return self.meta_dir / "video1.mp4"

    @property
    def original_meta_dir(self) -> Path:
        """``DROID_original/<s>``: the untouched episode copy some scenes keep (recorded reset pose)."""
        return self.root / f"{self.scene_type}_original" / self.scene

    def recorded_joints(self) -> Path:
        """Joint trajectory for the recorded reset pose, in the evaluator's search order.

        Mirrors the old evaluator's recorded-trajectory loader:
        ``DROID_original/<s>/joints_old.npy``; else ``DROID_original/<s>/joints.npy``
        when its row count equals the frame count (OpenCV ``CAP_PROP_FRAME_COUNT``;
        imageio reports ``inf`` for these videos) of ``DROID_original/<s>/video1.mp4``
        (droid4/7/8/41), otherwise the processed ``DROID/<s>/joints.npy`` (trimmed
        exports: droid17/26/28/...); else ``DROID/<s>/joints.npy``. Nothing usable
        is an error.
        """
        import numpy as np

        def rows(path: Path) -> int:
            arr = np.load(path, mmap_mode="r", allow_pickle=False)
            if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] != 7:
                raise ValueError(f"recorded DROID reset must have shape (T, 7) with T >= 1; got {arr.shape} from {path}")
            return int(arr.shape[0])

        orig = self.original_meta_dir
        if (orig / "joints_old.npy").is_file():
            rows(orig / "joints_old.npy")
            return orig / "joints_old.npy"
        if (orig / "joints.npy").is_file():
            n_rows = rows(orig / "joints.npy")
            video = orig / "video1.mp4"
            n_frames = 0
            if video.is_file():
                import cv2

                capture = cv2.VideoCapture(str(video))
                try:
                    if capture.isOpened():
                        n_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
                finally:
                    capture.release()
            if n_frames >= 1 and n_rows == n_frames:
                return orig / "joints.npy"
            if not self.joints.is_file():
                raise ValueError(
                    f"unsafe recorded DROID reset fallback: {orig / 'joints.npy'} has {n_rows} rows but "
                    f"{video} has {n_frames} frames, and {self.joints} does not exist"
                )
            rows(self.joints)
            return self.joints
        if self.joints.is_file():
            rows(self.joints)
            return self.joints
        raise FileNotFoundError(
            f"recorded DROID reset joints not found: {orig / 'joints_old.npy'}, {orig / 'joints.npy'} or {self.joints}"
        )

    @property
    def masks_dir(self) -> Path:
        return self.meta_dir / "masks"

    @property
    def transformations(self) -> Path:
        return self.masks_dir / "transformations.json"

    @property
    def mask_result(self) -> Path:
        return self.masks_dir / "result.json"

    @property
    def physical_properties(self) -> Path:
        # Older scenes keep it next to the masks folder; the evaluator checks both.
        new = self.masks_dir / "physical_properties.json"
        return new if new.exists() else self.meta_dir / "physical_properties.json"

    @property
    def bbox_masks_dir(self) -> Path:
        return self.meta_dir / "bounding_boxes_and_masks"

    @property
    def pcds_dir(self) -> Path:
        return self.meta_dir / "pcds"

    @property
    def input_mask_dir(self) -> Path:
        """LaMa inputs: binary object masks + the anchor frame (``<root>/input_mask/<s>``)."""
        return self.root / "input_mask" / self.scene

    @property
    def inpainted_depth(self) -> Path:
        return self.meta_dir / "inpainted_depth.npy"

    @property
    def mesh_sheet(self) -> Path:
        return self.meta_dir / "mesh_sheet.glb"

    @property
    def background_png(self) -> Path:
        return self.root / "scene_background" / self.scene / "background.png"

    @property
    def assets_dir(self) -> Path:
        return self.root / "assets" / self.scene

    def segmented_glbs(self) -> list[Path]:
        return sorted(self.assets_dir.glob("image_*_segmented.glb"))


@dataclass(frozen=True)
class SceneLayout:
    scene: str
    paths: Paths
    config: SceneConfig

    @classmethod
    def create(cls, scene: str, paths: Paths | None = None, registry: SceneRegistry | None = None) -> "SceneLayout":
        paths = paths or Paths.load()
        registry = registry or SceneRegistry()
        return cls(scene, paths, registry.get(scene))

    # ---------------------------------------------------------------- inputs
    def inputs_root_source(self) -> tuple[Path, str]:
        """``(root, why)``: where steps 1-10 read the scene's step-0 files from.

        The run's own ``scene_data_out`` once the scene was imported there
        (``inputs.episode``), else ``scene_data``; ``$R2S_SCENE_DATA`` pins ``scene_data``.
        """
        if os.environ.get(SCENE_DATA_ENV):
            return self.paths.scene_data, f"scene_data (${SCENE_DATA_ENV})"
        if (self.paths.scene_data_out / self.config.scene_type / self.scene).is_dir():
            return self.paths.scene_data_out, "scene_data_out (imported into this run)"
        return self.paths.scene_data, "scene_data"

    @property
    def inputs(self) -> InputLayout:
        """Step-0 files read by steps 1-10 (root chosen by :meth:`inputs_root_source`)."""
        return InputLayout(self.inputs_root_source()[0], self.scene, self.config.scene_type)

    @property
    def inputs_out(self) -> InputLayout:
        """Where step 0 writes when regenerating inputs."""
        return InputLayout(self.paths.scene_data_out, self.scene, self.config.scene_type)

    @property
    def inputs_src(self) -> InputLayout:
        """``paths.scene_data`` itself: the default episode source and the saved Gemini answers to replay."""
        return InputLayout(self.paths.scene_data, self.scene, self.config.scene_type)

    # --------------------------------------------------------------- outputs
    @property
    def root(self) -> Path:
        return self.paths.work / "scenes" / self.scene

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    # step 1: Gaussian world and collider
    @property
    def world_dir(self) -> Path:
        return self.root / "world"

    @property
    def gaussian_ply(self) -> Path:
        return self.world_dir / "gaussian.ply"

    @property
    def collider_glb(self) -> Path:
        return self.world_dir / "collider.glb"

    # steps 2-3: alignment
    @property
    def align_dir(self) -> Path:
        return self.root / "align"

    @property
    def aligned_pose(self) -> Path:
        return self.align_dir / "aligned_pose.npy"

    @property
    def aligned_scale(self) -> Path:
        return self.align_dir / "aligned_pose_scene_scale.npy"

    @property
    def relative_transform(self) -> Path:
        return self.align_dir / "aligned_pose_relative_transform.npy"

    @property
    def refined_pose(self) -> Path:
        return self.align_dir / "aligned_pose_refined.npy"

    @property
    def refined_scale(self) -> Path:
        return self.align_dir / "aligned_pose_scene_scale_refined.npy"

    @property
    def align_log_dir(self) -> Path:
        return self.align_dir / "log"

    # steps 4-6: support surface
    @property
    def surface_dir(self) -> Path:
        return self.root / "surface"

    @property
    def plane_fit_json(self) -> Path:
        return self.surface_dir / "operation_surface_plane_fit.json"

    @property
    def plane_fit_ply(self) -> Path:
        return self.surface_dir / "operation_surface_plane_fit.ply"

    @property
    def gaussian_trimmed_ply(self) -> Path:
        return self.world_dir / "gaussian_trimmed.ply"

    @property
    def receiver_json(self) -> Path:
        return self.surface_dir / "operation_surface_shadow_v2.json"

    # step 7: collision crop
    @property
    def collision_dir(self) -> Path:
        return self.root / "cropped_collision"

    @property
    def cropped_collider_glb(self) -> Path:
        return self.collision_dir / CROP_GLB_NAME

    # steps 8-10: objects
    @property
    def objects_dir(self) -> Path:
        return self.root / "objects"

    @property
    def object_dump(self) -> Path:
        """Objects exactly as the evaluator loads them (before any override)."""
        return self.objects_dir / "stock" / "object_poses.json"

    @property
    def posefit_dir(self) -> Path:
        return self.objects_dir / "posefit"

    @property
    def refined_poses(self) -> Path:
        return self.posefit_dir / "refined_poses.json"

    def refine_dir(self, tag: str) -> Path:
        return self.objects_dir / "refine" / tag

    @property
    def layout_dir(self) -> Path:
        return self.objects_dir / "layout"

    @property
    def layout_candidate(self) -> Path:
        return self.layout_dir / "candidate.json"

    def ensure(self, *dirs: Path) -> None:
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
