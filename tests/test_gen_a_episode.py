"""inputs.episode: what gets linked/copied, and the write-root guard."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest

from r2s.config import Paths
from r2s.gen import episode


def _fake_episode(root: Path, intrinsics_shape=(3, 3)) -> Path:
    src = root / "src"
    (src / "images0").mkdir(parents=True)
    (src / "depth").mkdir()
    for i in range(3):
        (src / "images0" / f"{i}.jpg").write_bytes(b"jpg%d" % i)
        np.save(src / "depth" / f"{i}.npy", np.full((2, 2), i, np.float32))
    (src / "images0" / "super.png").write_bytes(b"superres output")
    np.save(src / "intrinsics.npy", np.ones(intrinsics_shape, np.float32))
    for name in ("extrinsics.npy", "joints.npy", "grippers.npy", "tcp.npy"):
        np.save(src / name, np.zeros(4))
    (src / "lang.txt").write_text("put the cup in the bowl\n")
    (src / "metadata_X.json").write_text("{}")
    for later in ("inpainted_depth.npy", "mesh_sheet.glb", "whole_mask.png", "tracker_data.pth"):
        (src / later).write_bytes(b"later step")
    (src / "masks").mkdir()
    (src / "masks" / "result.json").write_text("[]")
    return src


def test_link_import_takes_only_raw_files(tmp_path):
    src = _fake_episode(tmp_path)
    dst = tmp_path / "out" / "DROID" / "s"
    stats = episode.import_episode(src, dst)
    assert stats["frames"] == 3 and stats["depth_frames"] == 3
    assert sorted(p.name for p in (dst / "images0").iterdir()) == ["0.jpg", "1.jpg", "2.jpg"]
    assert (dst / "images0" / "1.jpg").is_symlink() and not (dst / "images0").is_symlink()
    assert (dst / "lang.txt").read_text().startswith("put")
    assert (dst / "metadata_X.json").exists()
    for later in ("inpainted_depth.npy", "mesh_sheet.glb", "whole_mask.png", "tracker_data.pth", "masks"):
        assert not (dst / later).exists()
    assert stats["missing_optional"] == ["joints_old.npy"]


def test_copy_and_reimport_drops_stale_frames(tmp_path):
    src = _fake_episode(tmp_path)
    dst = tmp_path / "out"
    episode.import_episode(src, dst, copy=True)
    assert not (dst / "joints.npy").is_symlink()
    (src / "images0" / "2.jpg").unlink()
    (dst / "images0" / "super.png").write_bytes(b"from a superres run")
    stats = episode.import_episode(src, dst)
    assert not (dst / "images0" / "2.jpg").exists()
    assert stats["images0_other_files"] == ["super.png"]


def test_intrinsics_normalised_like_sg(tmp_path):
    src = _fake_episode(tmp_path, intrinsics_shape=(5, 3, 3))
    dst = tmp_path / "out"
    stats = episode.import_episode(src, dst)
    assert stats["intrinsics_normalised_from"] == [5, 3, 3]
    assert not (dst / "intrinsics.npy").is_symlink()
    assert np.load(dst / "intrinsics.npy").shape == (3, 3)


def test_missing_required_file_fails(tmp_path):
    src = _fake_episode(tmp_path)
    (src / "tcp.npy").unlink()
    with pytest.raises(FileNotFoundError):
        episode.import_episode(src, tmp_path / "out")


def test_refuses_to_write_into_scene_generation(tmp_path, monkeypatch):
    from r2s.layout import SceneLayout

    sg = tmp_path / "sg"
    (sg / "data").mkdir(parents=True)
    monkeypatch.setenv("R2S_SCENE_GENERATION", str(sg))
    monkeypatch.setenv("R2S_SCENE_DATA_OUT", str(sg / "data"))
    layout = SceneLayout.create("droid2", Paths.load())
    with pytest.raises(SystemExit):
        episode.output_inputs(layout)
    monkeypatch.setenv("R2S_SCENE_DATA_OUT", str(tmp_path / "fresh"))
    assert episode.output_inputs(SceneLayout.create("droid2", Paths.load())).root == tmp_path / "fresh"


def test_step_default_source_is_read_root(tmp_path, monkeypatch):
    from r2s.layout import SceneLayout

    monkeypatch.setenv("R2S_SCENE_DATA", str(tmp_path / "data"))
    layout = SceneLayout.create("droid2", Paths.load())
    args = argparse.Namespace(src=None, copy=False, extra=[])
    assert episode.source_dir(layout, args) == tmp_path / "data" / "DROID" / "droid2"
