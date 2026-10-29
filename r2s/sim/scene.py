"""Build the old evaluator's preview-run scene on Eden HEAD (non-RL gsplat ``SplatEnvBase``).

Spec: ``docs/porting/sim_assembly.md``. One call, :func:`build_scene`, assembles

* ``background``: the cropped collider (``layout.cropped_collider_glb``, built through
  ``r2s.collision.crop`` when missing), baked in world space and loaded with
  ``scale=1, pos=0, quat=GLB_WORLDSPACE_LOAD_QUAT_WXYZ, up=(0,0,1)``, convexified
  (coacd 0.01, decimate to 5000 faces, forced decomposition, aggressiveness 2,
  preprocess 150), ``vis_mode="collision"``, SDF cell 0.05;
* ``plane``: ground plane 2 m below the collider origin;
* the objects of an object dump (``layout.object_dump`` format; the dump is the
  source of truth for asset file, scale and load pose), collision profile
  ``detailed``, contact ``default``, material ``auto``; mass / friction from the
  scene metadata (``r2s.objects.dump``'s port of the evaluator's, mass floor 0.03 kg) set after build;
* ``operation_surface``: the physics prism of ``layout.receiver_json``
  (``physics_polygon``, thickness 0.03; ``r2s.surface.runtime``);
* ``robot`` + ``gripper``: the DROID rig (``r2s.sim.robot``), last;
* splat ``layout.gaussian_trimmed_ply`` at ``PCDOptions(scale=s, pos=t, quat=q_rel*ADJ*BASE)``;
* cameras ``camera`` (DROID calibration) and ``wrist_camera``, background entities
  ``["background", "plane"]``; the gsplat path's raster lights;
* env options of timing profile ``current``: ``sim_dt 0.004``, 4 substeps,
  decimation 5 (50 Hz control), noslip 5, ls/solver iterations 100, 2000 pairs.

Then ``env.build()``, snapshot the reset state, ``env.reset()``, set masses and
frictions, and hold the robot's PD targets at the reset pose (the old
``env.step(None)`` stepping "holding the reset action").

Differences forced by HEAD (see the spec): splat ``pos = t`` (not ``t / s``); the arm
file is pinned to the old hand-less Panda; default root poses are written in the
mesh frame (``relative=True``) so the old post-build ``set_quat(..., relative=False)``
is gone; pad boxes are applied on this path too.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from r2s.common.camera import load_camera_model
from r2s.common.io import read_json, write_json
from r2s.common.quat import axis_quat_np, euler_zyx_to_quat, quat_mul_np
from r2s.common.transforms import SCALE_OPT_ADJUSTMENT, SCALE_OPT_BASE_ROT, load_pose, load_scene_scale
from r2s.layout import SceneLayout
from r2s.sim import robot as droid

# timing profile "current" for franka + a 50 Hz policy (resolve_sim_timing)
SIM_DT = 0.004
SIM_SUBSTEPS = 4
DECIMATION = 5
NOSLIP_ITERATIONS = 5
EPISODE_LENGTH_S = 133.04  # max(100, 2000 steps * 0.064 s + 0.04 s delay + 5)
SDF_CELL_SIZE = 0.05
OPERATION_SURFACE_THICKNESS = 0.03
OBJECT_MASS_FLOOR = 0.03
CAMERA_BACKGROUND = ["background", "plane"]
# gsplat path (scale 1.0): bright_vis of the evaluator
AMBIENT_LIGHT = (0.5, 0.5, 0.5)
LIGHTS = (
    {"type": "directional", "dir": (-1, -1, -1), "color": (1.0, 1.0, 1.0), "intensity": 4.0},
    {"type": "directional", "dir": (1, 1, -1), "color": (1.0, 1.0, 1.0), "intensity": 2.0},
    {"type": "directional", "dir": (0, 0, 1), "color": (1.0, 1.0, 1.0), "intensity": 1.5},
)
# evaluator --object-collision-profile detailed (+ its threshold flags at their defaults)
OBJECT_COLLISION_PROFILE = "detailed"
OBJECT_DECOMPOSE_ERROR = 0.05
OBJECT_COACD = 0.03
OBJECT_LEGACY_FACE_NUM = 1000


# --------------------------------------------------------------------------- dump
@dataclass(frozen=True)
class DumpObject:
    safe_key: str
    count: int
    object_name: str
    asset_file: str
    scale: float
    pos: tuple[float, float, float]
    quat_wxyz: tuple[float, float, float, float]
    aabb_min: tuple[float, float, float] | None = None
    aabb_max: tuple[float, float, float] | None = None

    @property
    def key(self) -> str:
        """``transformations.json`` key (``image_0000_segmented``)."""
        return self.safe_key[len("obj_"):] if self.safe_key.startswith("obj_") else self.safe_key


def load_object_dump(path: str | Path) -> tuple[dict[str, Any], list[DumpObject]]:
    """``(header, objects)`` of an ``object_poses.json`` (``r2s.objects.overrides.write_object_pose_dump``)."""
    payload = read_json(path)
    if not isinstance(payload, dict) or "objects" not in payload:
        raise ValueError(f"{path}: not an object pose dump (no 'objects')")
    objects = []
    for o in payload["objects"]:
        asset = str(o["asset_file"])
        if not Path(asset).is_file():
            raise FileNotFoundError(f"{path}: {o['safe_key']}: asset_file not found: {asset}")
        objects.append(
            DumpObject(
                safe_key=str(o["safe_key"]),
                count=int(o["count"]),
                object_name=str(o.get("object_name", o["safe_key"])),
                asset_file=asset,
                scale=float(o["scale"]),
                pos=tuple(float(v) for v in o["pos"]),
                quat_wxyz=tuple(float(v) for v in o["quat_wxyz"]),
                aabb_min=tuple(float(v) for v in o["aabb_min"]) if o.get("aabb_min") is not None else None,
                aabb_max=tuple(float(v) for v in o["aabb_max"]) if o.get("aabb_max") is not None else None,
            )
        )
    header = {k: v for k, v in payload.items() if k != "objects"}
    return header, objects


# --------------------------------------------------------------------------- poses
@dataclass(frozen=True)
class Calibration:
    K: np.ndarray
    width: int
    height: int
    fov_y_deg: float
    cam_pos: np.ndarray
    cam_lookat: np.ndarray
    cam_up: np.ndarray


def load_calibration(layout: SceneLayout) -> Calibration:
    """Old evaluator ``load_calibration`` (2179-2197)."""
    K, W, H, _t_w2c, pos, lookat, up = load_camera_model(layout.inputs.intrinsics, layout.inputs.extrinsics)
    fov_y = 2.0 * math.atan(H / (2.0 * float(K[1, 1]))) * 180.0 / math.pi
    return Calibration(K, W, H, fov_y, pos, lookat, up)


@dataclass(frozen=True)
class WorldPose:
    """Splat / collider placement from the refined alignment (``r2s.common.transforms``)."""

    t: np.ndarray  # collider_pos = splat pos at HEAD
    scale: float
    splat_quat: np.ndarray  # q_rel * ADJ * BASE
    rx: float
    ry: float
    rz: float


def evaluator_splat_quat(rx: float, ry: float, rz: float) -> np.ndarray:
    """``optimised_splat_quat`` bit for bit: ``q_rel * transform_quat_by_quat(BASE, ADJ)``.

    ``transform_quat_by_quat(v, u)`` is Genesis' normalised ``u * v`` (``genesis_quat_mul``);
    the evaluator's ``apply_relative_transform_to_quat`` then returns the raw product.
    """
    from r2s.collision.crop import genesis_quat_mul

    orig = genesis_quat_mul(np.asarray(SCALE_OPT_ADJUSTMENT, dtype=np.float64), np.asarray(SCALE_OPT_BASE_ROT, dtype=np.float64))
    q_rel = axis_quat_np(rz, 2) if (rx == 0.0 and ry == 0.0) else euler_zyx_to_quat(rx, ry, rz)
    return quat_mul_np(q_rel, orig)


def world_pose(layout: SceneLayout) -> WorldPose:
    t, rx, ry, rz = load_pose(layout.refined_pose)
    scale = load_scene_scale(layout.refined_scale)
    return WorldPose(t=t, scale=scale, splat_quat=evaluator_splat_quat(rx, ry, rz), rx=rx, ry=ry, rz=rz)


# --------------------------------------------------------------------------- scene
@dataclass
class SimScene:
    env: Any
    layout: SceneLayout
    run_dir: Path
    calibration: Calibration
    pose: WorldPose
    objects: list[DumpObject]
    object_names: dict[str, str]
    physics: dict[str, dict[str, Any]]
    q0: np.ndarray
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def safe_keys(self) -> list[str]:
        return [o.safe_key for o in self.objects if o.safe_key in self.env.entities]

    def step(self) -> None:
        """One 50 Hz control step holding the reset action (the old ``env.step(None)``)."""
        self.env.step(None)

    def reset(self) -> None:
        """``env.reset()`` and re-hold the robot at its reset pose."""
        self.env.reset()
        hold_reset_pose(self.env, self.q0)

    def render(self, camera: str = "camera") -> np.ndarray:
        return rgb_uint8(self.env.cameras[camera].render_rgb())

    def destroy(self) -> None:
        try:
            self.env.destroy()
        except Exception as exc:  # pragma: no cover - best effort
            print(f"[sim] env.destroy failed: {exc}")


def rgb_uint8(rgb) -> np.ndarray:
    if hasattr(rgb, "detach"):
        rgb = rgb.detach().cpu().numpy()
    rgb = np.asarray(rgb)
    if rgb.ndim == 4:
        rgb = rgb[0]
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 1) if rgb.max() <= 1.0 + 1e-6 else np.clip(rgb / 255.0, 0, 1)
        rgb = (rgb * 255).astype(np.uint8)
    return np.ascontiguousarray(rgb[..., :3])


def hold_reset_pose(env, q0: np.ndarray) -> None:
    """PD targets = reset pose for the arm, open for the gripper drivers."""
    import torch

    arm = env.entities["robot"]
    _, arm_idx = arm.find_named_dofs_idx_local(droid.FRANKA_ARM_DOFS, preserve_order=True)
    arm.control_dofs_pos(torch.as_tensor(np.asarray(q0, dtype=np.float32)[None], device=env.device), dofs_idx_local=arm_idx)
    if "gripper" in env.entities:
        gripper = env.entities["gripper"]
        _, g_idx = gripper.find_named_dofs_idx_local(["left_driver_joint", "right_driver_joint"], preserve_order=True)
        gripper.control_dofs_pos(torch.zeros((1, len(g_idx)), device=env.device), dofs_idx_local=g_idx)


_EDEN_INITED = False


def init_eden(log_dir: Path, *, use_cpu: bool = False) -> None:
    """``en.init(gs.gpu, debug=True, performance_mode=True)`` as the evaluator, logs under the run folder."""
    global _EDEN_INITED
    if _EDEN_INITED:
        return
    import genesis as gs
    import eden as en

    en.init(backend=gs.cpu if use_cpu else gs.gpu, debug=True, performance_mode=True, log_root_path=str(log_dir))
    _EDEN_INITED = True


def collider_entity_options(layout: SceneLayout):
    """Cropped background collider options (crop built through ``r2s.collision.crop`` if missing)."""
    from eden.options.entities import EntityOptions
    from eden.options.materials import RigidMaterialOptions
    from r2s.collision.crop import crop_collider, load_crop_result

    if layout.cropped_collider_glb.is_file():
        crop = load_crop_result(layout.cropped_collider_glb)
    else:
        print(f"[sim] {layout.cropped_collider_glb} missing; cropping the collider now")
        crop = crop_collider(layout.collider_glb, layout.refined_pose, layout.refined_scale, layout.collision_dir)
    opts = EntityOptions(
        file=str(crop.mesh_path),
        scale=crop.load_scale,
        up=tuple(int(v) for v in crop.load_up),
        front=tuple(int(v) for v in crop.load_front),
        default_root_pos=tuple(crop.load_pos),
        default_root_quat=tuple(crop.load_quat_wxyz),
        is_fixed_base=True,
        convexify=True,
        coacd_threshold=0.01,
        decimate_face_num=5000,
        decompose_object_error_threshold=0.0,
        decimate_aggressiveness=2,
        coacd_preprocess_resolution=150,
        vis_mode="collision",
        visualization=True,
        material=RigidMaterialOptions(sdf_cell_size=SDF_CELL_SIZE),
    )
    return opts, crop


def eden_surface(material):
    """Eden surface for a resolved material (old ``resolve_object_visual_material``, 3075-3080); None = the GLB's own."""
    if material.status != "metadata" or material.surface_kind is None:
        return None
    from eden.options.surfaces import BSDFSurfaceOptions, EmissionSurfaceOptions, GlassSurfaceOptions

    if material.surface_kind == "glass":
        return GlassSurfaceOptions(roughness=material.roughness)
    if material.surface_kind == "emission":
        return EmissionSurfaceOptions()
    return BSDFSurfaceOptions(metallic=material.metallic, roughness=material.roughness)


def object_entity_options(
    objects: list[DumpObject],
    layout: SceneLayout,
    *,
    collision_overrides: dict[str, dict[str, Any]] | None = None,
):
    """Per-object ``EntityOptions`` + the physics / material record of each (mass, friction, hull settings)."""
    from eden.options.entities import EntityOptions
    from r2s.collision.profiles import select_object_collision_profile
    from r2s.objects.dump import (
        load_scene_object_metadata,
        metadata_sources,
        resolve_object_mass,
        resolve_object_visual_material,
    )

    metadata = load_scene_object_metadata(metadata_sources(layout.inputs))
    if not metadata:
        print(f"[object metadata] WARNING: no object metadata in {layout.inputs.meta_dir}; using evaluator defaults")
    entities: dict[str, Any] = {}
    physics: dict[str, dict[str, Any]] = {}
    for obj in objects:
        phys = metadata.get(obj.count, {})
        mass, raw_mass = resolve_object_mass(phys, mass_floor=OBJECT_MASS_FLOOR)
        friction = float(phys.get("friction", 0.5))
        material = resolve_object_visual_material(phys, mode="auto", splat_backend="gsplat")
        selection = select_object_collision_profile(
            obj.asset_file,
            requested_profile=OBJECT_COLLISION_PROFILE,
            inspect_quality=False,
            decimate_override=None,
            detailed_decompose_error_threshold=OBJECT_DECOMPOSE_ERROR,
            detailed_coacd_threshold=OBJECT_COACD,
            legacy_face_num=OBJECT_LEGACY_FACE_NUM,
        )
        override = (collision_overrides or {}).get(obj.key)
        if override:
            from r2s.objects.overrides import apply_collision_override

            selection = apply_collision_override(selection, override)
            print(f"[object collision override] {obj.key}: {override}")
        kwargs = {}
        surface = eden_surface(material)
        if surface is not None:
            kwargs["surface"] = surface
        entities[obj.safe_key] = EntityOptions(
            file=obj.asset_file,
            scale=obj.scale,
            default_root_pos=obj.pos,
            default_root_quat=obj.quat_wxyz,
            is_fixed_base=False,
            convexify=True,
            decimate=selection.decimate,
            decimate_face_num=selection.decimate_face_num,
            decimate_aggressiveness=selection.decimate_aggressiveness,
            decompose_object_error_threshold=selection.decompose_object_error_threshold,
            coacd_threshold=selection.coacd_threshold,
            **kwargs,
        )
        physics[obj.safe_key] = {
            "count": obj.count,
            "object_name": str(phys.get("object_name", obj.object_name)),
            "mass": mass,
            "raw_mass": raw_mass,
            "mass_clamped": bool(mass > raw_mass),
            "friction": friction,
            "metadata_sources": list(phys.get("_metadata_sources", [])),
            "collision": selection.to_json_dict(),
            "material": material.to_json_dict(),
        }
        print(
            f"[objects] {obj.object_name:32s} pos={tuple(round(v, 4) for v in obj.pos)} scale={obj.scale:.5f} "
            f"mass={mass} friction={friction} hull={selection.selected_profile}"
        )
    return entities, physics


def build_scene(
    layout: SceneLayout,
    *,
    dump_path: str | Path,
    run_dir: str | Path,
    collision_overrides: dict[str, dict[str, Any]] | None = None,
    use_cpu: bool = False,
) -> SimScene:
    """Assemble, build and reset the preview-run scene. Files the run writes go under ``run_dir``."""
    run_dir = Path(run_dir)
    header, objects = load_object_dump(dump_path)
    calib = load_calibration(layout)
    pose = world_pose(layout)
    q0, q0_path = droid.recorded_reset_joints(layout)
    print(f"[robot] droid_reset_profile=recorded source={q0_path} arm_q={q0.tolist()}")
    print(f"[pose] collider/splat pos={pose.t.tolist()} scale={pose.scale:.6f} splat_quat={pose.splat_quat.tolist()}")

    init_eden(run_dir / "eden_logs", use_cpu=use_cpu)
    import genesis as gs
    from eden.envs.splat import SplatEnvBase
    from eden.options.camera import CameraOptions, CamerasOptions
    from eden.options.entities import PlaneOptions, SceneOptions
    from eden.options.envs import EnvOptions
    from eden.options.splat import PCDOptions, SplatOptions
    from eden.utils.configs import EdenSplatConfig
    from r2s.surface.runtime import load_operation_surface_boxes

    background, crop = collider_entity_options(layout)
    object_entities, physics = object_entity_options(objects, layout, collision_overrides=collision_overrides)
    surface_entities, surface_names = load_operation_surface_boxes(
        str(layout.receiver_json),
        mode="physics_polygon",
        visualization=False,
        collision=True,
        collision_thickness=OPERATION_SURFACE_THICKNESS,
        contact_sol_params=None,
    )
    if not surface_names:
        raise RuntimeError(f"{layout.receiver_json}: no physics operation surface")
    robotiq_xml = droid.write_robotiq_pad_box_mjcf(droid.eden_asset(droid.ROBOTIQ_MJCF), run_dir / "robot" / "2f85.xml")
    robot = droid.build_droid(q0, robotiq_xml)

    reserved = {"background", "plane", "robot", "gripper"}
    clash = reserved.intersection({**object_entities, **surface_entities})
    if clash:
        raise ValueError(f"scene entities use reserved names: {sorted(clash)}")
    scene_kwargs: dict[str, Any] = {
        "background": background,
        "plane": PlaneOptions(default_root_pos=(float(pose.t[0]), float(pose.t[1]), float(pose.t[2]) - 2.0)),
        **object_entities,
        **surface_entities,
        "robot": robot,  # last: the composite expands into robot -> gripper, which must stay consecutive
    }

    class _EnvOptions(EnvOptions):  # the evaluator's EvaluationEnvOptions: lets vis_options reach gs.Scene
        model_config = {**EnvOptions.model_config, "extra": "allow"}

    base = EnvOptions()
    vis = gs.options.VisOptions(
        split_envs=base.env_separate_rigid,
        show_link_frame=base.show_link_frame,
        show_world_frame=base.show_world_frame,
        show_cameras=base.show_camera_frustums,
        segmentation_level=base.segmentation_level,
        background_color=base.background_color,
        shadow=base.rasterizer_shadows,
        ambient_light=AMBIENT_LIGHT,
        lights=[dict(light) for light in LIGHTS],
    )
    env_options = _EnvOptions(
        num_envs=1,
        episode_length_s=EPISODE_LENGTH_S,
        sim_dt=SIM_DT,
        sim_substeps=SIM_SUBSTEPS,
        decimation=DECIMATION,
        noslip_iterations=NOSLIP_ITERATIONS,
        ls_iterations=100,
        iterations=100,
        max_collision_pairs=2000,
        vis_options=vis,
    )
    cameras = CamerasOptions(
        camera=CameraOptions(
            cam_pos=tuple(float(v) for v in calib.cam_pos),
            cam_lookat=tuple(float(v) for v in calib.cam_lookat),
            cam_up=tuple(float(v) for v in calib.cam_up),
            cam_fov=float(calib.fov_y_deg),
            cam_resolution=(int(calib.width), int(calib.height)),
            background_entities=list(CAMERA_BACKGROUND),
        ),
        wrist_camera=droid.wrist_camera_options(CAMERA_BACKGROUND),
    )
    splat = SplatOptions(
        background=PCDOptions(
            path=str(layout.gaussian_trimmed_ply),
            scale=float(pose.scale),
            pos=tuple(float(v) for v in pose.t),
            quat=tuple(float(v) for v in pose.splat_quat),
        )
    )
    cfg = EdenSplatConfig(
        env_options=env_options,
        scene_options=SceneOptions(**scene_kwargs),
        splat_options=splat,
        cameras_options=cameras,
    )
    env = SplatEnvBase.from_config(cfg, show_viewer=False, eval_mode=True)
    order = list(env.entities)
    if order.index("gripper") != order.index("robot") + 1:
        raise RuntimeError(f"robot and gripper are not consecutive before build: {order}")
    env.build()
    env.reset_state = env.scene.get_state()
    env.reset()
    for key, rec in physics.items():
        entity = env.entities.get(key)
        if entity is None:
            continue
        entity.set_mass(rec["mass"])
        entity.set_friction(rec["friction"])
    hold_reset_pose(env, q0)

    info = {
        "dump": str(dump_path),
        "dump_header": header,
        "crop": {"mesh_path": crop.mesh_path, "load_quat_wxyz": list(crop.load_quat_wxyz), "reason": crop.reason},
        "receiver_json": str(layout.receiver_json),
        "surface_names": surface_names,
        "splat": {"ply": str(layout.gaussian_trimmed_ply), "scale": pose.scale, "pos": pose.t.tolist(), "quat_wxyz": pose.splat_quat.tolist()},
        "robot": {"reset_joints": q0.tolist(), "reset_source": str(q0_path), "robotiq_xml": str(robotiq_xml),
                  "franka_xml": str(droid.eden_asset(droid.OLD_FRANKA_MJCF))},
        "timing": {"sim_dt": SIM_DT, "sim_substeps": SIM_SUBSTEPS, "decimation": DECIMATION, "noslip_iterations": NOSLIP_ITERATIONS},
        "physics": physics,
        "entity_order": order,
    }
    write_json(run_dir / "scene_info.json", info)
    return SimScene(
        env=env, layout=layout, run_dir=run_dir, calibration=calib, pose=pose, objects=objects,
        object_names={o.safe_key: physics[o.safe_key]["object_name"] for o in objects}, physics=physics,
        q0=q0, info=info,
    )


__all__ = [
    "Calibration",
    "DumpObject",
    "SimScene",
    "WorldPose",
    "build_scene",
    "hold_reset_pose",
    "init_eden",
    "load_calibration",
    "load_object_dump",
    "rgb_uint8",
    "world_pose",
]
