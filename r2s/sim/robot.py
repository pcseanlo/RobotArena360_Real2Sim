"""The DROID rig of the old evaluator's preview run, on Eden HEAD.

Franka arm + Robotiq 2F85 as ``Droid`` (``robot`` + ``gripper`` entities welded at the
arm's ``attachment`` link), configured as the old evaluator's ``build_robot("franka",
"pi")`` with ``--arm-gain-profile`` auto -> ``pi-sysid`` and ``--gripper-collision-profile
high-fidelity-eef``:

* arm: the hand-less Panda file ``arms/franka_research3/fr3.xml``. At HEAD
  ``FrankaResearch3`` loads a real FR3 (``arms/franka_fr3/fr3.xml``, other limits and
  joint friction), so the old file is pinned explicitly; the ``FrankaResearch3``
  options class (joint names, ``attachment`` link) is unchanged otherwise.
  ``default_dofs_pos`` = the recorded DROID joints at frame 0, pi-sysid kp/kd,
  ``gravity_compensation=1``.
* gripper: ``Robotiq2f85(batch_fixed_verts, is_fixed_base=False, decimate=False,
  gravity_compensation=1, collision_link_patterns=[base, *_driver, *_spring_link,
  *_follower, *pad], default_geoms_sol_params={".*pad": DROID_OBJECT_SOL_PARAMS})`` on a
  copy of ``2f85.xml`` whose two 5-sphere fingertip clusters are replaced by one box
  each (the old Nyx-path MJCF rewrite, written into the run folder).

Only the pad geometry of the old MJCF rewrite is kept; its Nyx material tweaks
(rgba / specular scaling) were visual-only and do not apply to the rasterizer.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from r2s.common.io import prepare_output

FRANKA_ARM_DOFS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]

# the old evaluator's FRANKA_ARM_GAIN_PROFILES["pi-sysid"] (Eden/results/droid_sysid_kpkd_2)
PI_SYSID_KP = (624.04, 2403.96, 1064.94, 2146.25, 366.29, 1568.66, 1015.37)
PI_SYSID_KD = (240.57, 799.99, 356.44, 799.47, 117.23, 446.11, 363.74)

# evaluator GRIPPER_COLLISION_PROFILES["high-fidelity-eef"]
GRIPPER_COLLISION_PATTERNS = ["base", ".*_driver", ".*_spring_link", ".*_follower", ".*pad"]
DROID_OBJECT_SOL_PARAMS = (0.002, 1.0, 0.95, 0.999, 0.001, 0.5, 2.0)

# Old asset paths inside Eden's asset folder (resolved against the imported eden package).
OLD_FRANKA_MJCF = "arms/franka_research3/fr3.xml"  # mjcf model "panda nohand"
ROBOTIQ_MJCF = "grippers/robotiq_2f85/2f85.xml"

# evaluator --wrist-camera-profile current (robot_cfg wrist_* for franka)
WRIST_ATTACH_ENTITY = "gripper"
WRIST_ATTACH_LINK = "base"
WRIST_OFFSET = (0.031, 0.06, -0.025)
WRIST_QUAT = (0.0, 0.0, 1.0, 0.0)
WRIST_INTRINSIC = ((184.5, 0.0, 160.0), (0.0, 184.5, 90.0), (0.0, 0.0, 1.0))

# cached_adjusted_robot_mjcf_for_nyx(..., continuous_robotiq_pads=True)
PAD_BOX = {"class": "pad_sphere", "type": "box", "pos": "0.044258 0 0.1305", "size": "0.00155 0.01055 0.01555"}


def eden_assets_dir() -> Path:
    """``<eden package>/assets`` of the Eden actually imported (never written to)."""
    import eden

    return Path(eden.__file__).resolve().parent / "assets"


def eden_asset(relpath: str) -> Path:
    path = eden_assets_dir() / relpath
    if not path.is_file():
        raise FileNotFoundError(
            f"Eden asset {relpath} not found at {path}; refusing to let Eden download it into its asset folder"
        )
    return path


def write_robotiq_pad_box_mjcf(src_xml: str | Path, out_xml: str | Path) -> Path:
    """Copy of the Robotiq MJCF with each fingertip's 5 spheres replaced by one box.

    Same rewrite as the old ``cached_adjusted_robot_mjcf_for_nyx(profile="robotiq",
    continuous_robotiq_pads=True)``: ``compiler`` mesh/texture dirs are made absolute
    (the copy lives elsewhere), every geom of ``left_pad`` / ``right_pad`` is removed and
    a ``<pad>_box`` geom (class ``pad_sphere`` -> its friction and contact params)
    covering the same envelope is added.
    """
    src_xml = Path(src_xml)
    tree = ET.parse(src_xml)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is not None:
        xml_dir = src_xml.resolve().parent
        for attr in ("meshdir", "texturedir"):
            value = compiler.get(attr)
            if value and not Path(value).is_absolute():
                compiler.set(attr, str((xml_dir / value).resolve()))
    count = 0
    for body in root.iter("body"):
        if body.get("name") not in {"left_pad", "right_pad"}:
            continue
        for geom in list(body.findall("geom")):
            body.remove(geom)
        ET.SubElement(body, "geom", {"class": PAD_BOX["class"], "name": f"{body.get('name')}_box", "type": PAD_BOX["type"],
                                     "pos": PAD_BOX["pos"], "size": PAD_BOX["size"]})
        count += 1
    if count != 2:
        raise ValueError(f"expected left_pad and right_pad bodies in {src_xml}; updated {count}")
    out = prepare_output(out_xml)
    if hasattr(ET, "indent"):
        ET.indent(tree, space="  ")
    tree.write(out, encoding="utf-8", xml_declaration=False)
    return out


def recorded_reset_joints(layout) -> tuple[np.ndarray, Path]:
    """Frame-0 arm joints for ``--droid-reset-profile recorded`` (old ``load_droid_recorded_trajectory``).

    The file comes from ``layout.inputs.recorded_joints()`` (the evaluator's search order:
    ``DROID_original/<s>/joints_old.npy``, ``DROID_original/<s>/joints.npy`` when its rows
    match ``video1.mp4``, else ``DROID/<s>/joints.npy``).
    """
    path = Path(layout.inputs.recorded_joints())
    if not path.is_file():
        raise FileNotFoundError(f"recorded DROID reset joints not found: {path}")
    q0 = np.asarray(_load_joint_array(path)[0], dtype=np.float64)
    if not np.all(np.isfinite(q0)):
        raise ValueError(f"recorded DROID reset contains non-finite values: {path}")
    return q0, path


def _load_joint_array(path: Path) -> np.ndarray:
    joints = np.load(path, allow_pickle=False)
    if joints.ndim != 2 or joints.shape[0] < 1 or joints.shape[1] != len(FRANKA_ARM_DOFS):
        raise ValueError(f"recorded DROID reset must have shape (T, 7) with T >= 1; got {joints.shape} from {path}")
    return joints


def build_droid(q0: np.ndarray, robotiq_xml: str | Path, *, franka_xml: str | Path | None = None):
    """``Droid`` composite options for the preview run (arm + gripper, base at the origin, no mount yaw)."""
    from eden.options.materials import RigidMaterialOptions
    from eden.options.robots.composite.droid import Droid
    from eden.options.robots.franka_emika_panda import FrankaResearch3
    from eden.options.robots.robotiq_2f85 import Robotiq2f85

    q0 = np.asarray(q0, dtype=np.float64).reshape(-1)
    if q0.shape != (len(FRANKA_ARM_DOFS),) or not np.all(np.isfinite(q0)):
        raise ValueError(f"reset joints must be 7 finite values, got {q0}")
    arm = FrankaResearch3(
        file=str(franka_xml or eden_asset(OLD_FRANKA_MJCF)),
        default_dofs_pos=dict(zip(FRANKA_ARM_DOFS, q0.tolist())),
        default_dofs_kp=dict(zip(FRANKA_ARM_DOFS, (float(v) for v in PI_SYSID_KP))),
        default_dofs_kd=dict(zip(FRANKA_ARM_DOFS, (float(v) for v in PI_SYSID_KD))),
        material=RigidMaterialOptions(gravity_compensation=1.0),
    )
    gripper = Robotiq2f85(
        file=str(robotiq_xml),
        batch_fixed_verts=True,
        is_fixed_base=False,
        decimate=False,
        material=RigidMaterialOptions(gravity_compensation=1.0),
        collision_link_patterns=list(GRIPPER_COLLISION_PATTERNS),
        default_geoms_sol_params={".*pad": DROID_OBJECT_SOL_PARAMS},
    )
    return Droid(arm=arm, gripper=gripper)


def wrist_camera_options(background_entities: list[str]):
    from eden.options.camera import CameraOptions

    return CameraOptions(
        cam_intrinsic=np.array(WRIST_INTRINSIC, dtype=np.float64),
        attach_entity_name=WRIST_ATTACH_ENTITY,
        attach_link_name=WRIST_ATTACH_LINK,
        attach_offset=WRIST_OFFSET,
        attach_quat=WRIST_QUAT,
        background_entities=list(background_entities),
    )


__all__ = [
    "DROID_OBJECT_SOL_PARAMS",
    "FRANKA_ARM_DOFS",
    "GRIPPER_COLLISION_PATTERNS",
    "PI_SYSID_KD",
    "PI_SYSID_KP",
    "build_droid",
    "eden_asset",
    "recorded_reset_joints",
    "wrist_camera_options",
    "write_robotiq_pad_box_mjcf",
]
