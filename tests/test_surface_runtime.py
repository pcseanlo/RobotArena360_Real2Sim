"""Tests for ``r2s.surface.runtime`` (ported from the old ``test_operation_surface_runtime.py``).

Eden is replaced by a stub in ``sys.modules`` for each test, so these run without
Genesis; the last test checks the CPU steps never import Eden or viser.
"""

import json
import subprocess
import sys
import tempfile
import types
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

import numpy as np

from r2s.surface import runtime


@dataclass
class _BoxOptions:
    size: tuple[float, float, float]
    default_root_pos: tuple[float, float, float]
    default_root_quat: tuple[float, float, float, float]
    is_fixed_base: bool
    visualization: bool
    collision: bool = True
    surface: object | None = None
    default_geoms_sol_params: dict | None = None


@dataclass
class _EntityOptions:
    file: str
    up: tuple[int, int, int]
    front: tuple[int, int, int]
    scale: float
    default_root_pos: tuple[float, float, float]
    default_root_quat: tuple[float, float, float, float]
    is_fixed_base: bool
    visualization: bool
    collision: bool
    decimate: bool
    convexify: bool
    surface: object | None = None
    decompose_object_error_threshold: float = 0.15
    coacd_threshold: float = 0.05
    coacd_preprocess_resolution: int = 100
    default_geoms_sol_params: dict | None = None


def _eden_stub() -> dict[str, types.ModuleType]:
    eden = types.ModuleType("eden")
    options = types.ModuleType("eden.options")
    entities = types.ModuleType("eden.options.entities")
    entities.BoxOptions = _BoxOptions
    entities.EntityOptions = _EntityOptions
    options.entities = entities
    eden.options = options
    return {"eden": eden, "eden.options": options, "eden.options.entities": entities}


def _descriptor_path(tmpdir: str) -> str:
    path = Path(tmpdir) / "operation_surface.json"
    path.write_text(
        json.dumps(
            {
                "selected": {
                    "center": [1.0, 2.0, 3.0],
                    "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                    "size": [4.0, 5.0, 0.1],
                    "support_boxes": [
                        {
                            "center": [1.0, 2.0, 3.0],
                            "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                            "size": [1.0, 2.0, 0.1],
                        },
                        {
                            "center": [6.0, 7.0, 8.0],
                            "quat_wxyz": [0.0, 1.0, 0.0, 0.0],
                            "size": [2.0, 3.0, 0.1],
                        },
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    return str(path)


def _polygon_descriptor_path(tmpdir: str) -> str:
    path = Path(tmpdir) / "operation_surface_polygon.json"
    path.write_text(
        json.dumps(
            {
                "version": 2,
                "selected": {
                    "center": [0.5, 0.5, 0.0],
                    "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                    "size": [1.0, 1.0, 0.01],
                    "support_boxes": [],
                    "footprints": {
                        "lighting": {
                            "vertices_world": [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
                            "triangles": [[0, 1, 2], [0, 2, 3]],
                            "plane_normal": [0, 0, 1],
                            "thickness": 0.01,
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return str(path)


class OperationSurfaceRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        patcher = mock.patch.dict(sys.modules, _eden_stub())
        patcher.start()
        self.addCleanup(patcher.stop)
        # Keep the footprint OBJ cache of these tests in a folder that is removed afterwards.
        cache_root = tempfile.TemporaryDirectory()
        self.addCleanup(cache_root.cleanup)
        tmp_patcher = mock.patch.object(tempfile, "tempdir", cache_root.name)
        tmp_patcher.start()
        self.addCleanup(tmp_patcher.stop)
        self.runtime = runtime

    def test_solid_mode_loads_one_selected_box_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            entities, names = self.runtime.load_operation_surface_boxes(_descriptor_path(tmpdir))

        self.assertEqual(names, ["operation_surface"])
        self.assertEqual(entities["operation_surface"].size, (4.0, 5.0, 0.1))
        self.assertEqual(entities["operation_surface"].default_root_pos, (1.0, 2.0, 3.0))
        self.assertFalse(entities["operation_surface"].visualization)

    def test_support_boxes_mode_loads_decomposed_boxes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            entities, names = self.runtime.load_operation_surface_boxes(
                _descriptor_path(tmpdir),
                mode="support_boxes",
            )

        self.assertEqual(names, ["operation_surface", "operation_surface_01"])
        self.assertEqual(entities["operation_surface"].size, (1.0, 2.0, 0.1))
        self.assertEqual(entities["operation_surface_01"].default_root_pos, (6.0, 7.0, 8.0))

    def test_custom_name_prefix_supports_visual_only_receiver(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            entities, names = self.runtime.load_operation_surface_boxes(
                _descriptor_path(tmpdir),
                mode="support_boxes",
                name_prefix="operation_surface_render_receiver",
                visualization=True,
                collision=False,
            )

        self.assertEqual(names, ["operation_surface_render_receiver", "operation_surface_render_receiver_01"])
        self.assertFalse(entities["operation_surface_render_receiver"].collision)
        self.assertTrue(entities["operation_surface_render_receiver"].visualization)

    def test_unknown_mode_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "Unknown operation-surface mode"):
                self.runtime.load_operation_surface_boxes(_descriptor_path(tmpdir), mode="crust")

    def test_lighting_polygon_loads_one_mesh_entity(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            entities, names = self.runtime.load_operation_surface_boxes(
                _polygon_descriptor_path(tmpdir),
                mode="lighting_polygon",
                visualization=True,
                collision=False,
            )

        self.assertEqual(names, ["operation_surface"])
        entity = entities["operation_surface"]
        self.assertTrue(entity.file.endswith(".obj"))
        self.assertTrue(Path(entity.file).exists())
        self.assertFalse(entity.collision)

    def test_polygon_top_faces_are_wound_toward_plane_normal(self) -> None:
        footprint = {
            "vertices_world": [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
            # Clockwise when viewed from +Z, matching the real droid8 descriptor.
            "triangles": [[0, 1, 2], [0, 2, 3]],
            "plane_normal": [0, 0, 1],
            "thickness": 0.01,
        }

        mesh_path = Path(self.runtime._write_footprint_obj(footprint, z_offset=0.0))
        vertices = []
        faces = []
        for line in mesh_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("v "):
                vertices.append([float(value) for value in line.split()[1:]])
            elif line.startswith("f "):
                faces.append([int(value.split("/")[0]) - 1 for value in line.split()[1:]])

        vertices = np.asarray(vertices)
        for face in faces[:2]:
            a, b, c = vertices[face]
            self.assertGreater(float(np.cross(b - a, c - a)[2]), 0.0)

        texture_coordinates = [
            line for line in mesh_path.read_text(encoding="utf-8").splitlines()
            if line.startswith("vt ")
        ]
        self.assertEqual(len(texture_coordinates), 4)
        self.assertTrue(all("/" in token for token in mesh_path.read_text().splitlines()[-1].split()[1:]))

    def test_collision_thickness_extends_only_downward(self) -> None:
        footprint = {
            "vertices_world": [[0, 0, 0.125], [1, 0, 0.125], [1, 1, 0.125], [0, 1, 0.125]],
            "triangles": [[0, 1, 2], [0, 2, 3]],
            "plane_normal": [0, 0, 1],
            "thickness": 0.01,
        }

        mesh_path = Path(
            self.runtime._write_footprint_obj(
                footprint,
                z_offset=0.0,
                minimum_thickness=0.03,
            )
        )
        vertices = [
            [float(value) for value in line.split()[1:]]
            for line in mesh_path.read_text(encoding="utf-8").splitlines()
            if line.startswith("v ")
        ]

        vertices = np.asarray(vertices)
        self.assertTrue(np.array_equal(vertices[:4], np.asarray(footprint["vertices_world"])))
        self.assertTrue(np.allclose(vertices[4:, :2], vertices[:4, :2]))
        self.assertTrue(np.allclose(vertices[4:, 2], 0.095))
        self.assertEqual(footprint["thickness"], 0.01)

    def test_physics_polygon_receives_stiff_contact_and_fine_decomposition(self) -> None:
        contact = {".*": (0.002, 1.0, 0.95, 0.999, 0.001, 0.5, 2.0)}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(_polygon_descriptor_path(tmpdir))
            descriptor = json.loads(path.read_text(encoding="utf-8"))
            descriptor["selected"]["footprints"]["physics"] = descriptor["selected"][
                "footprints"
            ]["lighting"]
            path.write_text(json.dumps(descriptor), encoding="utf-8")
            entities, _ = self.runtime.load_operation_surface_boxes(
                str(path),
                mode="physics_polygon",
                collision=True,
                collision_thickness=0.03,
                contact_sol_params=contact,
            )

        entity = entities["operation_surface"]
        self.assertEqual(entity.default_geoms_sol_params, contact)
        self.assertEqual(entity.decompose_object_error_threshold, 0.0)
        self.assertEqual(entity.coacd_threshold, 0.03)
        self.assertEqual(entity.coacd_preprocess_resolution, 150)


class LazyImportTests(unittest.TestCase):
    def test_cpu_steps_do_not_import_eden_or_viser(self) -> None:
        code = (
            "import sys\n"
            "import r2s.surface.plane_fit, r2s.surface.trim, r2s.surface.receiver_fit, r2s.surface.runtime\n"
            "print(','.join(m for m in ('eden', 'genesis', 'torch', 'viser') if m in sys.modules))\n"
        )
        repo = Path(__file__).resolve().parents[1]
        out = subprocess.run(
            [sys.executable, "-c", code], cwd=repo, capture_output=True, text=True, check=True,
            env={"PYTHONDONTWRITEBYTECODE": "1", "PATH": "/usr/bin:/bin"},
        )
        self.assertEqual(out.stdout.strip(), "")


if __name__ == "__main__":
    unittest.main()
