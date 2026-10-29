"""CPU checks of the sim assembly pieces (no Genesis build): robot rig, metadata, poses, dumps."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from legacy_paths import old_file

from r2s.common.quat import axis_quat_np, quat_mul_np
from r2s.common.transforms import SCALE_OPT_ADJUSTMENT, SCALE_OPT_BASE_ROT
from r2s.config import Paths
from r2s.layout import SceneLayout
from r2s.sim import robot as droid
from r2s.objects.dump import load_scene_object_metadata, metadata_sources, resolve_object_mass

EDEN_ROBOTIQ = Paths.load().eden / "eden/assets" / droid.ROBOTIQ_MJCF


def _layout(scene: str) -> SceneLayout:
    layout = SceneLayout.create(scene)
    if not layout.inputs.meta_dir.is_dir():
        pytest.skip(f"scene data for {scene} not available")
    return layout


def test_pad_box_rewrite(tmp_path):
    if not EDEN_ROBOTIQ.is_file():
        pytest.skip("Eden Robotiq asset not available")
    out = droid.write_robotiq_pad_box_mjcf(EDEN_ROBOTIQ, tmp_path / "robot" / "2f85.xml")
    root = ET.parse(out).getroot()
    meshdir = root.find("compiler").get("meshdir")
    assert Path(meshdir).is_absolute() and Path(meshdir).is_dir()
    for name in ("left_pad", "right_pad"):
        body = next(b for b in root.iter("body") if b.get("name") == name)
        geoms = body.findall("geom")
        assert len(geoms) == 1
        g = geoms[0].attrib
        assert g == {"class": "pad_sphere", "name": f"{name}_box", "type": "box",
                     "pos": "0.044258 0 0.1305", "size": "0.00155 0.01055 0.01555"}
    # every other body is untouched
    src = ET.parse(EDEN_ROBOTIQ).getroot()
    n_src = sum(len(b.findall("geom")) for b in src.iter("body") if b.get("name") not in {"left_pad", "right_pad"})
    n_out = sum(len(b.findall("geom")) for b in root.iter("body") if b.get("name") not in {"left_pad", "right_pad"})
    assert n_src == n_out


def test_gains_and_patterns():
    assert len(droid.PI_SYSID_KP) == len(droid.PI_SYSID_KD) == 7
    assert droid.PI_SYSID_KP[1] == 2403.96 and droid.PI_SYSID_KD[6] == 363.74
    assert droid.GRIPPER_COLLISION_PATTERNS == ["base", ".*_driver", ".*_spring_link", ".*_follower", ".*pad"]


@pytest.mark.parametrize(
    "scene,expected_file,first",
    [
        # the old preview logs: [robot] droid_reset_profile=recorded source=... arm_q=[...]
        ("droid2", "DROID_original/droid2/joints_old.npy", [-0.15425199270248413, -0.38827115297317505, -0.03350615128874779]),
        ("droid820", "DROID/droid820/joints.npy", [0.15707914531230927, -0.2404114156961441, 0.21825861930847168]),
    ],
)
def test_recorded_reset_joints(scene, expected_file, first):
    q0, path = droid.recorded_reset_joints(_layout(scene))
    assert str(path).endswith(expected_file)
    assert np.allclose(q0[:3], first, atol=0, rtol=0)


@pytest.mark.parametrize("scene,tag", [("droid820", "stock"), ("droid2", "stock")])
def test_mass_friction_match_old_manifest(scene, tag):
    layout = _layout(scene)
    manifest = old_file("asset_refine", scene, tag, "rec", "object_material_manifest.json")
    import json

    old = json.loads(manifest.read_text())["objects"]
    meta = load_scene_object_metadata(metadata_sources(layout.inputs))
    for key, rec in old.items():
        phys = meta.get(int(rec["count"]), {})
        mass, raw = resolve_object_mass(phys, mass_floor=0.03)
        assert (mass, raw) == (rec["mass"], rec["raw_mass"]), key
        assert float(phys.get("friction", 0.5)) == rec["friction"], key
        assert phys.get("object_name", key) == rec["object_name"]
        assert list(phys.get("_metadata_sources", [])) == rec["metadata_sources"]


def test_splat_quat_is_evaluators():
    from r2s.collision.crop import genesis_quat_mul
    from r2s.sim.scene import evaluator_splat_quat

    rz = 1.2102925777435303
    q = evaluator_splat_quat(0.0, 0.0, rz)
    orig = genesis_quat_mul(SCALE_OPT_ADJUSTMENT, SCALE_OPT_BASE_ROT)  # transform_quat_by_quat(BASE, ADJ)
    assert np.array_equal(q, quat_mul_np(axis_quat_np(rz, 2), orig))
    assert abs(np.linalg.norm(q) - 1.0) < 1e-12


def test_object_dump_loads_old_stock():
    from r2s.sim.scene import load_object_dump

    dump = old_file("asset_refine", "droid820", "stock", "object_poses.json")
    header, objects = load_object_dump(dump)
    assert header["scene"] == "droid820" and len(objects) == 5
    assert objects[0].safe_key == "obj_image_0000_segmented" and objects[0].key == "image_0000_segmented"
    assert all(Path(o.asset_file).is_file() for o in objects)
