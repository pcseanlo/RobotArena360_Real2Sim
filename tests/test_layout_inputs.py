"""Per-scene input root: steps 1-10 read the run's own step-0 outputs once the scene was imported."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from r2s.config import Paths, SceneConfig
from r2s.gen import episode, physics, segment
from r2s.layout import SceneLayout


@pytest.fixture
def layout(tmp_path, monkeypatch) -> SceneLayout:
    monkeypatch.delenv("R2S_SCENE_DATA", raising=False)
    fields = {f: tmp_path / f for f in Paths.__dataclass_fields__}
    fields["scene_data"] = tmp_path / "sg_data"
    fields["scene_data_out"] = tmp_path / "work" / "scene_data"
    return SceneLayout("droidX", Paths(**fields), SceneConfig("droidX", {"scene_type": "DROID"}))


def test_reads_scene_data_until_imported(layout, tmp_path, monkeypatch):
    assert layout.inputs.root == tmp_path / "sg_data"
    assert layout.inputs_root_source() == (tmp_path / "sg_data", "scene_data")
    (tmp_path / "work" / "scene_data" / "DROID" / "droidX").mkdir(parents=True)  # inputs.episode ran in this run
    assert layout.inputs.root == tmp_path / "work" / "scene_data"
    assert layout.inputs.background_png == tmp_path / "work" / "scene_data" / "scene_background" / "droidX" / "background.png"
    assert "imported" in layout.inputs_root_source()[1]
    monkeypatch.setenv("R2S_SCENE_DATA", str(tmp_path / "sg_data"))  # an explicit override wins
    assert layout.inputs_root_source() == (tmp_path / "sg_data", "scene_data ($R2S_SCENE_DATA)")
    # step 0 always writes to scene_data_out; the replay / episode sources stay scene_data
    assert layout.inputs_out.root == tmp_path / "work" / "scene_data"
    assert layout.inputs_src.root == tmp_path / "sg_data"


def test_defaults_that_point_at_the_existing_data(layout, tmp_path):
    (tmp_path / "work" / "scene_data" / "DROID" / "droidX").mkdir(parents=True)
    src = tmp_path / "sg_data" / "DROID" / "droidX"
    assert episode.source_dir(layout, argparse.Namespace(src=None)) == src
    gem, res = segment.replay_paths(layout, argparse.Namespace(replay_gemini=segment.AUTO, replay_result=None))
    assert gem == src / "masks" / "gemini_boxes_raw.json" and res == src / "masks" / "result.json"
    assert src in physics.replay_path(layout, argparse.Namespace(replay=physics.AUTO)).parents


def test_manifest_records_the_inputs_root(tmp_path):
    from r2s.manifest import Manifest, step_record

    m = tmp_path / "manifest.json"
    with step_record(m, "surface.plane", args={}, inputs=[], outputs=[], inputs_root=Path("/some/root")):
        pass
    assert Manifest(m).get("surface.plane")["inputs_root"] == "/some/root"
