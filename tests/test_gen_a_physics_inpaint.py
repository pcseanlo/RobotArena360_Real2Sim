"""inputs.physics replay validation and inputs.inpaint mask ordering."""

from __future__ import annotations

import pytest

from r2s.gen import inpaint, physics

RESULT = [
    {"object_name": "black bowl", "count": [0], "mass": 0.2, "friction": 0.3, "surface": "Plastic"},
    {"object_name": "cup ", "count": [1], "mass": 0.1, "friction": 0.4, "surface": "Plastic"},
]


def _phys(name, count):
    return {"object_name": name, "count": count, "width": 0.1, "height": 0.1, "length": 0.1,
            "mass": 0.2, "friction": 0.5, "surface": "Plastic"}


def test_physics_replay_matches_result():
    physics.check_physics([_phys("black bowl", [0]), _phys("cup", [1])], RESULT)


@pytest.mark.parametrize("bad", [
    [_phys("black bowl", [0])],
    [_phys("black bowl", [0]), _phys("mug", [1])],
    [_phys("black bowl", [0]), {"object_name": "cup", "count": [1]}],
])
def test_physics_replay_rejects_mismatch(bad):
    with pytest.raises(ValueError):
        physics.check_physics(bad, RESULT)


def test_mask_order_is_lamas_glob(tmp_path):
    for name in ("image_0010_mask.png", "image_0002_mask.png", "image_0002.png", "image_0000_mask.png"):
        (tmp_path / name).write_bytes(b"")
    assert [p.name for p in inpaint.mask_files(tmp_path)] == [
        "image_0000_mask.png", "image_0002_mask.png", "image_0010_mask.png"]
