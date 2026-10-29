"""``r2s.align.scale_core`` vs the old scale-fit script (bit-identical on synthetic data).

The old module is imported by path (its top level only needs numpy/scipy), with
bytecode writing disabled so nothing lands in the read-only old repo.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from legacy_paths import old_file

from r2s.align import scale_core as core
from r2s.common import transforms

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def old():
    path = old_file("world_lab_scale_optimization.py")
    prev = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location("_old_scale_fit_script", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = prev
    return module


def _camera():
    K = np.array([[262.0, 0.0, 320.0], [0.0, 262.0, 185.0], [0.0, 0.0, 1.0]])
    W, H = 640, 370
    cam_pos = np.array([0.35, 0.43, 0.31])
    cam_lookat = np.array([1.18, 0.03, -0.05])
    forward = (cam_lookat - cam_pos) / np.linalg.norm(cam_lookat - cam_pos)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    cam_up = np.cross(right, forward)
    return K, W, H, cam_pos, cam_lookat, cam_up


def _table_scene(rng, n=6000, scale=1.0, pivot=None):
    """A tabletop plane plus a box, in front of the camera."""
    xy = rng.uniform([0.5, -0.4], [1.2, 0.4], size=(n, 2))
    plane = np.column_stack([xy, np.zeros(n)])
    box = rng.uniform([0.75, -0.1, 0.0], [0.9, 0.05, 0.12], size=(n // 4, 3))
    pts = np.concatenate([plane, box])
    if pivot is not None:
        pts = pivot + scale * (pts - pivot)
    return pts


def test_constants_match(old):
    assert np.array_equal(core._BASE_ROT, old._BASE_ROT)
    assert np.array_equal(core._ADJUSTMENT, old._ADJUSTMENT)


def test_splat_frame_constants_defined_once():
    """scale_core uses r2s.common.transforms' constants; no other file spells them out."""
    assert core._BASE_ROT is transforms.SCALE_OPT_BASE_ROT
    assert core._ADJUSTMENT is transforms.SCALE_OPT_ADJUSTMENT
    literals = ("0.01745240643728351", "0.7071067811865476, 0.7071067811865476")
    hits = sorted(
        str(p.relative_to(REPO))
        for root in ("r2s", "tools", "tests", "scripts")
        for p in (REPO / root).rglob("*.py")
        if p.name != Path(__file__).name and any(lit in p.read_text() for lit in literals)
    )
    assert hits == ["r2s/common/transforms.py"]


def test_masks_bit_identical(old):
    rng = np.random.default_rng(0)
    K, W, H, cp, cl, cu = _camera()
    pts = _table_scene(rng)
    for kw in (dict(), dict(truncate_w=0.1, truncate_h=0.3), dict(truncate_left=0.4, truncate_top=0.4, truncate_right=0.2, truncate_bottom=0.2)):
        a = core._visible_mask(pts, cp, cl, cu, K, W, H, **kw)
        b = old._visible_mask(pts, cp, cl, cu, K, W, H, **kw)
        assert np.array_equal(a, b)
        for cell in (0, 1, 3):
            sa, fa = core._visible_surface_mask(pts, cp, cl, cu, K, W, H, cell_px=cell, **kw)
            sb, fb = old._visible_surface_mask(pts, cp, cl, cu, K, W, H, cell_px=cell, **kw)
            assert np.array_equal(sa, sb) and np.array_equal(fa, fb)


def test_outlier_filter_and_sampling_identical(old):
    rng = np.random.default_rng(1)
    pts = _table_scene(rng, n=3000)
    pts[:20, 2] += rng.uniform(0.2, 0.5, size=20)  # floaters
    a, ka = core._statistical_outlier_filter(pts, k=60, std_ratio=0.5)
    b, kb = old._statistical_outlier_filter(pts, k=60, std_ratio=0.5)
    assert np.array_equal(a, b) and np.array_equal(ka, kb)
    assert not ka[:20].any()

    verts = rng.normal(size=(50, 3))
    faces = rng.integers(0, 50, size=(80, 3))
    sa = core._sample_triangles_area_weighted(verts, faces, 500, np.random.default_rng(7))
    sb = old._sample_triangles_area_weighted(verts, faces, 500, np.random.default_rng(7))
    assert np.array_equal(sa, sb)


class _Geom:
    def __init__(self, v, f):
        self._v, self.init_faces = v, f

    def get_verts(self):
        return self._v


class _GsEntity:
    def __init__(self, geoms):
        self.geoms = geoms

    def get_verts(self):
        return np.concatenate([g._v for g in self.geoms])


class _Entity:
    def __init__(self, geoms):
        self._entity = _GsEntity(geoms)


def test_entity_surface_samples_identical(old):
    rng = np.random.default_rng(2)
    geoms = [_Geom(rng.normal(size=(30, 3)) + i, rng.integers(0, 30, size=(40, 3))) for i in range(4)]
    ent = _Entity(geoms)
    a = core._get_world_surface_samples(ent, 2000, rng_seed=0)
    b = old._get_world_surface_samples(ent, 2000, rng_seed=0)
    assert np.array_equal(a, b)


def test_calibration_and_quats_identical(old, tmp_path):
    K, W, H, cp, cl, cu = _camera()
    T = np.eye(4)
    T[:3, :3] = np.array([[0.0, -1.0, 0.0], [0.2, 0.0, -0.98], [0.98, 0.0, 0.2]])
    T[:3, :3] = np.linalg.qr(T[:3, :3])[0]
    T[:3, 3] = [0.1, 0.2, 0.9]
    np.save(tmp_path / "K.npy", K)
    np.save(tmp_path / "E.npy", T)
    a = core.load_calibration(str(tmp_path / "K.npy"), str(tmp_path / "E.npy"))
    b = old.load_calibration(str(tmp_path / "K.npy"), str(tmp_path / "E.npy"))
    for x, y in zip(a, b):
        assert np.array_equal(np.asarray(x), np.asarray(y))
    assert core.lookat_to_quat(cp, cl) == old.lookat_to_quat(cp, cl)
    for args in ((0.3,), (0.3, 0.01, -0.02)):
        q = np.array([0.7, 0.1, 0.0, 0.7])
        assert core.apply_relative_transform_to_quat(q, *args) == old.apply_relative_transform_to_quat(q, *args)


def test_find_scale_chamfer_identical_and_recovers_scale(old):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(3)
    K, W, H, cp, cl, cu = _camera()
    true_scale, s_init = 0.75, 1.0
    metric = _table_scene(rng, n=3000)
    # collider sampled at s_init: points that would land on the metric surface at true_scale
    coll_true = _table_scene(np.random.default_rng(4), n=12000)
    coll_init = cp + (s_init / true_scale) * (coll_true - cp)
    kw = dict(K=K, cam_pos=cp, cam_lookat=cl, cam_up=cu, W=W, H=H, n_steps=150, lr=5e-3,
              s_min_ratio=0.3, s_max_ratio=2.0, trim_quantile=0.8, truncate_w=0.1, truncate_h=0.3)
    torch.manual_seed(0)
    sa, la = core.find_scale_chamfer(coll_init, metric, cp, s_init, **kw)
    torch.manual_seed(0)
    sb, lb = old.find_scale_chamfer(coll_init, metric, cp, s_init, **kw)
    assert sa == sb and la == lb
    kw["n_steps"] = 400
    s_fit, _ = core.find_scale_chamfer(coll_init, metric, cp, s_init, **kw)
    assert math.isclose(s_fit, true_scale, rel_tol=0.02)


def test_find_metric_mesh(tmp_path):
    class Inputs:
        mesh_sheet = tmp_path / "DROID" / "s" / "mesh_sheet.glb"
        assets_dir = tmp_path / "assets" / "s"

    Inputs.assets_dir.mkdir(parents=True)
    for i in (0, 2, 10):
        (Inputs.assets_dir / f"image_{i:04d}_segmented.glb").write_bytes(b"x")
    (Inputs.assets_dir / "image_0011_segmented_scaled.glb").write_bytes(b"x")
    path, source = core.find_metric_mesh(Inputs)
    assert Path(path).name == "image_0010_segmented.glb" and source == "segmented_fallback"
    Inputs.mesh_sheet.parent.mkdir(parents=True)
    Inputs.mesh_sheet.write_bytes(b"x")
    assert core.find_metric_mesh(Inputs) == (str(Inputs.mesh_sheet), "mesh_sheet")
