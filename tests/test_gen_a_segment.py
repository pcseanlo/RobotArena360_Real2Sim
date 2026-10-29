"""inputs.segment: replay reconstruction and SG mask persistence (no Gemini, no SAM3)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from r2s.config import Paths
from r2s.gen.sg_glue import gembox, gemini_utils, segment_replay

REPO = Path(__file__).resolve().parents[1]
OLD_DROID = Paths.load().scene_data / "DROID"  # the existing scene_generation data

RAW = {
    "image_size_wh": [640, 360],
    "canvas_side": 640,
    "canvas_pad_xy": [0, 140],
    "items": [
        {"label": "black bowl", "box_2d_raw": [518, 642, 616, 797]},
        {"label": "hand cream", "box_2d_raw": [637, 276, 772, 396]},
        {"label": "red block", "box_2d_raw": [564, 538, 610, 584]},
        {"label": "flat", "box_2d_raw": [100, 100, 100, 200]},
    ],
    "denormalized_xyxy": [
        {"label": "black bowl", "xyxy": [410, 191, 510, 254]},
        {"label": "hand cream", "xyxy": [176, 267, 253, 354]},
        {"label": "red block", "xyxy": [344, 220, 373, 250]},
    ],
    "inpaint_only_denormalized_xyxy": [
        {"label": "robot arm", "xyxy": [0, 0, 100, 100]},
        {"label": "human person", "xyxy": [180, 35, 237, 97]},
    ],
}
RESULT = [
    {"object_name": "black bowl", "count": [0], "mass": 0.2, "friction": 0.3, "surface": "Plastic"},
    {"object_name": "red block", "count": [1], "mass": 0.005, "friction": 0.4, "surface": "Plastic"},
    {"object_name": "table", "count": [2], "mass": 0.5, "friction": 0.5, "surface": "Rough"},
]


def test_inpaint_only_labels():
    assert gemini_utils.is_inpaint_only_label("human person")
    assert gemini_utils.is_inpaint_only_label("robot-gripper")
    assert gemini_utils.is_inpaint_only_label("hand cream")  # SG treats it as inpaint-only too
    assert not gemini_utils.is_inpaint_only_label("rubik's cube")


def test_prompts_from_boxes_raw():
    prompts, notes = segment_replay.prompts_from_boxes_raw(RAW, RESULT)
    assert prompts["robot"] == [("robot arm", (0, 0, 100, 100))]
    assert prompts["human"] == [("human person", (180, 35, 237, 97))]
    assert [lbl for lbl, _ in prompts["objects"]] == ["black bowl", "hand cream", "red block"]
    # hand cream is inpaint-only, so result.json has no physics for it
    assert prompts["object_items"] == [
        {"mass": 0.2, "friction": 0.3, "surface_type": "Plastic"},
        {},
        {"mass": 0.005, "friction": 0.4, "surface_type": "Plastic"},
    ]
    assert any("'table'" in n for n in notes)
    assert gembox.gemini_boxes_raw(prompts) == RAW


def test_boxes_raw_must_be_consistent():
    bad = json.loads(json.dumps(RAW))
    bad["denormalized_xyxy"][0]["xyxy"][0] += 1
    with pytest.raises(ValueError):
        segment_replay.prompts_from_boxes_raw(bad, RESULT)


def test_old_file_without_inpaint_key_and_dropped_object():
    raw = {k: v for k, v in RAW.items() if k != "inpaint_only_denormalized_xyxy"}
    prompts, notes = segment_replay.prompts_from_boxes_raw(raw, RESULT[1:2])  # bowl was dropped by old SAM3
    assert prompts["robot"] == [] and prompts["human"] == []
    assert prompts["object_items"][0] == {} and prompts["object_items"][2]["mass"] == 0.005
    assert any("black bowl" in n for n in notes)


def test_split_inpaint_only_is_strict():
    with pytest.raises(ValueError):
        segment_replay.split_inpaint_only([("gripper", (0, 0, 1, 1))])


def test_answers_roundtrip():
    answers = {
        "image_size_wh": [640, 360],
        "robot": {"canvas_side": 640, "canvas_pad_xy": [0, 140], "items": [{"box_2d": [0, 0, 300, 300], "label": "arm"}]},
        "human": {"canvas_side": 640, "canvas_pad_xy": [0, 140], "items": []},
        "objects": {"canvas_side": 640, "canvas_pad_xy": [0, 140], "items": [
            {"box_2d": [518, 642, 616, 797], "label": "black bowl", "mass": 0.2, "friction": 0.3, "surface_type": "Plastic"},
            {"label": "no box"},
        ]},
    }
    p = gembox.prompts_from_answers(answers)
    assert p["robot"][0][0] == "robot arm"  # prefixed like SG
    assert p["objects"] == [("black bowl", (410, 191, 510, 254))]
    assert p["object_items"][0]["mass"] == 0.2 and len(p["raw_items"]) == 1


@pytest.mark.parametrize("scene", ["droid820", "droid2"])
def test_old_scene_files_replay(scene):
    raw = OLD_DROID / scene / "masks" / "gemini_boxes_raw.json"
    if not raw.exists():
        pytest.skip("old scene data not available")
    prompts, notes, kind = segment_replay.load_replay(str(raw))
    assert kind == "gemini_boxes_raw" and len(prompts["objects"]) == 5
    old = json.loads(raw.read_text())
    assert {k: v for k, v in gembox.gemini_boxes_raw(prompts).items() if k in old} == old


def test_plot_segmentation_masks_layout(tmp_path):
    img = Image.fromarray((np.arange(60 * 80 * 3) % 251).astype(np.uint8).reshape(60, 80, 3))
    obj = np.zeros((60, 80), np.uint8)
    obj[10:40, 20:50] = 255
    rob = np.zeros((60, 80), np.uint8)
    rob[0:5, 0:5] = 255
    masks = [
        gemini_utils.SegmentationMask(0, 0, 5, 5, rob, "robot arm"),
        gemini_utils.SegmentationMask(10, 20, 40, 50, obj, "cup", mass=0.1, friction=0.4, surface="Plastic"),
    ]
    mask_dir, im_dir = tmp_path / "masks", tmp_path / "input_mask"
    mask_dir.mkdir()
    _, dicts = gemini_utils.plot_segmentation_masks(img, masks, str(mask_dir), str(im_dir), None)
    assert [d["object_name"] for d in dicts] == ["cup"]
    assert json.loads((mask_dir / "result.json").read_text()) == [
        {"object_name": "cup", "count": [0], "mass": 0.1, "friction": 0.4, "surface": "Plastic"}]
    assert sorted(p.name for p in im_dir.iterdir()) == ["image_0000.png", "image_0000_mask.png",
                                                         "image_0001.png", "image_0001_mask.png"]
    rgb = np.asarray(Image.open(mask_dir / "image_0000_mask.png"))
    assert (rgb[obj == 0] == 0).all() and (rgb[obj > 0] == np.asarray(img)[obj > 0]).all()
    assert Image.open(mask_dir / "cropped" / "image_0000_segmented.png").size == (80, 60)  # box + 40 px, clamped


def test_lama_worker_is_python36():
    py = Paths.load().conda_root / "envs" / "lama" / "bin" / "python"
    if not py.exists():
        pytest.skip("lama env not available")
    src = REPO / "r2s" / "gen" / "workers" / "lama_inpaint.py"
    subprocess.run([str(py), "-c", f"compile(open({str(src)!r}).read(), {str(src)!r}, 'exec')"], check=True,
                   env={"PYTHONDONTWRITEBYTECODE": "1", "PATH": "/usr/bin:/bin"})
