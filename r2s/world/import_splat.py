"""Step 1a ``world.import``: bring the scene's Gaussian splat into the run.

The splat is whatever image-to-world generator produced for the scene's
inpainted background (today: World Labs Marble worlds downloaded as PLY, 2.2M
gaussians, SH degree 1), stored as ``paths.splat_source_dir/<scene>.ply``. This step links (default)
or copies that file to ``<run>/scenes/<s>/world/gaussian.ply`` and checks it is a
3DGS PLY with the properties every later step reads.

Use ``--src`` to import from anywhere else, e.g. an existing Eden asset folder
(``.../eden/assets/echo2/<s>/gaussian.ply``).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from r2s.common.io import prepare_output
from r2s.layout import SceneLayout
from r2s.steps import Step

REQUIRED_PROPERTIES = (
    "x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3", "f_dc_0", "f_dc_1", "f_dc_2",
)


def source_path(layout: SceneLayout, args: argparse.Namespace) -> Path:
    return Path(args.src) if args.src else layout.paths.splat_source_dir / f"{layout.scene}.ply"


def check_gaussian_ply(path: Path) -> dict:
    from plyfile import PlyData

    ply = PlyData.read(str(path), mmap="c")
    vertex = ply["vertex"]
    names = set(vertex.data.dtype.names or ())
    missing = [p for p in REQUIRED_PROPERTIES if p not in names]
    if missing:
        raise ValueError(f"{path} is not a 3DGS PLY: missing {missing}")
    rest = sorted(n for n in names if n.startswith("f_rest_"))
    # f_rest has 3 * ((d + 1)^2 - 1) entries for SH degree d.
    sh_degree = {0: 0, 9: 1, 24: 2, 45: 3}.get(len(rest), -1)
    return {"num_gaussians": int(len(vertex.data)), "sh_degree": sh_degree}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--src", default=None, help="PLY to import (default: <splat_source_dir>/<scene>.ply)")
    parser.add_argument("--copy", action="store_true", help="copy instead of symlinking")


def run(layout: SceneLayout, args: argparse.Namespace) -> dict:
    src = source_path(layout, args).resolve()
    stats = check_gaussian_ply(src)
    dst = prepare_output(layout.gaussian_ply)
    if dst.exists():
        dst.unlink()
    if args.copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src)
    return {"source": str(src), **stats}


STEP = Step(
    name="world.import",
    help="Link or copy the scene's generated splat PLY into world/gaussian.ply",
    add_arguments=add_arguments,
    run=run,
    inputs=lambda layout, args: [source_path(layout, args)],
    outputs=lambda layout, args: [layout.gaussian_ply],
)
