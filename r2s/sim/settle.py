"""Preview / settle runs in the sim (GPU): the old asset-refine preview run, on Eden HEAD.

One run = build the preview scene (``r2s.sim.scene``) from an object dump, reset,
optionally push a candidate's poses (``absolute``/``delta``; an object that jumps more
than ``max_displacement`` in the override's 5 settle steps is reverted to its load
pose), run the settle probe (``r2s.objects.settle``), then render the previews:

    <out_dir>/object_poses.json   poses this run loaded (before the pose override)
    <out_dir>/settle.json         commanded vs settled pose per object (steps > 0)
    <out_dir>/settle.mp4          main camera, one frame before step 1 + one per step
    <out_dir>/preview/camera.png, preview/wrist_camera.png   after the settle
    <out_dir>/check.png           real first-frame masks drawn on camera.png
    <out_dir>/sim/                scene_info.json, the pad-box Robotiq MJCF, Eden logs

``run_settle`` is the hook ``r2s.objects.refine.pipeline`` calls for its
``stock-settle`` (``candidate=None, mode="free", steps=150``) and ``candidate-settle``
(``mode="upright", steps=150, hold_steps=20``) stages; ``objects.posefit`` previews
use :func:`run_preview` (pose override, no settle probe).

Load poses come from an object dump, the source of truth for asset, scale and pose:
``layout.object_dump`` for stock runs, or ``dump_path`` for a run whose candidate
changes the load (scale / asset overrides re-run the load-time collision guard, which
is ``objects.dump``'s job: :func:`resolve_dump` calls its library to write
``<out_dir>/object_poses.json``, or pass ``dump_path`` to load a given dump as is).

    python -m r2s.sim.settle --scene droid820 --out DIR [--candidate C.json] [--mode upright] [--steps 150]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from r2s.common.io import read_json, write_json
from r2s.layout import SceneLayout

POSE_MODES = ("absolute", "delta")
DEFAULT_POSE_MODE = "absolute"  # old asset-refine launcher's POSE_MODE default (since 2026-09-12; delta before)
DEFAULT_MAX_DISPLACEMENT = 0.05  # old asset-refine launcher's POSE_MAX_DISP
OVERRIDE_SETTLE_STEPS = 5
SAMPLE_EVERY = 5
VIDEO_FPS = 15
PREVIEW_CAMERAS = ("camera", "wrist_camera")


class _LoadSpec:
    """The two fields ``resolve_override_for_run`` reads from an ``ObjectPlacementSpec``."""

    def __init__(self, pos, quat):
        self.base_pos = np.asarray(pos, dtype=np.float64)
        self.base_quat_wxyz = np.asarray(quat, dtype=np.float64)


def candidate_load_overrides(candidate_path: str | Path | None) -> dict[str, Any]:
    """Load-time overrides a candidate carries: scale multipliers, replacement assets, hull settings."""
    from r2s.objects.overrides import (
        load_object_asset_overrides,
        load_object_collision_overrides,
        load_object_scale_multipliers,
    )

    if candidate_path is None:
        return {"scale": {}, "asset": {}, "collision": {}}
    return {
        "scale": load_object_scale_multipliers(candidate_path),
        "asset": load_object_asset_overrides(candidate_path),
        "collision": load_object_collision_overrides(candidate_path),
    }


def resolve_dump(
    layout: SceneLayout,
    candidate_path: str | Path | None,
    loads: dict[str, Any],
    dump_path: str | Path | None,
    out_dir: Path,
) -> Path:
    """Which object dump this run loads (see the module docstring)."""
    if dump_path is not None:
        return Path(dump_path)
    if not loads["scale"] and not loads["asset"]:
        return layout.object_dump
    # objects.dump owns the load-time placement (reprojection + collision guard) at the candidate's scale.
    from r2s.objects.dump import DumpSettings, dump_scene_objects, load_scene_objects_from_layout

    settings = DumpSettings(
        scale_override=str(candidate_path) if loads["scale"] else None,
        asset_override=str(candidate_path) if loads["asset"] else None,
        collision_override=str(candidate_path) if loads["collision"] else None,
    )
    out = out_dir / "object_poses.json"
    dump_scene_objects(load_scene_objects_from_layout(layout, settings), out)
    print(f"[sim] re-placed the objects at the candidate's scale with objects.dump -> {out}")
    return out


def _copy_dump(src: Path, dst: Path) -> None:
    if dst.exists() and dst.resolve() == src.resolve():
        return
    write_json(dst, read_json(src))


def _write_check(layout: SceneLayout, render_path: Path, out_path: Path) -> str:
    """The old pose-fit ``check-render``: real first-frame masks over the render."""
    from r2s.objects.pose_fit import check_render

    return check_render(layout, render_path, out_path)


def run_settle(
    layout: SceneLayout,
    candidate_path: Path | None,
    out_dir: Path,
    mode: str,
    steps: int,
    hold_steps: int,
    *,
    pose_mode: str = DEFAULT_POSE_MODE,
    max_displacement: float = DEFAULT_MAX_DISPLACEMENT,
    dump_path: Path | None = None,
    apply_scale: bool = True,
    apply_pose: bool = True,
    video: bool = True,
    check: bool = True,
    use_cpu: bool = False,
) -> dict[str, Any]:
    """Settle the scene and write ``out_dir/{object_poses.json, settle.json, settle.mp4, preview/, check.png}``.

    ``steps`` counts every probe step including the ``hold_steps`` of ``gentle`` /
    ``upright`` (``free`` ignores the hold). ``steps <= 0`` skips the probe (preview only).
    """
    from r2s.objects.overrides import apply_object_pose_override, load_object_pose_override, resolve_override_for_run
    from r2s.objects.settle import SETTLE_MODES, probe_settle, write_settle_dump, write_settle_video
    from r2s.sim.scene import build_scene, load_object_dump

    if mode not in SETTLE_MODES:
        raise ValueError(f"unknown settle mode {mode!r}; expected one of {SETTLE_MODES}")
    if pose_mode not in POSE_MODES:
        raise ValueError(f"unknown pose override mode {pose_mode!r}")
    out_dir = Path(out_dir)
    candidate_path = None if candidate_path is None else Path(candidate_path)
    if candidate_path is not None and not candidate_path.is_file():
        raise FileNotFoundError(f"candidate not found: {candidate_path}")
    started = time.time()

    loads = candidate_load_overrides(candidate_path)
    if not apply_scale:
        loads["scale"] = {}
    dump = resolve_dump(layout, candidate_path, loads, dump_path, out_dir)
    if not dump.is_file():
        raise FileNotFoundError(f"object dump not found: {dump} (run objects.dump first)")
    _copy_dump(dump, out_dir / "object_poses.json")
    _header, objects = load_object_dump(dump)

    scene = build_scene(layout, dump_path=dump, run_dir=out_dir / "sim", collision_overrides=loads["collision"] or None, use_cpu=use_cpu)
    env = scene.env
    try:
        # write_camera_previews: reset once, then the override + settle hook, then render.
        scene.reset()
        override_lines: list[str] = []
        if candidate_path is not None and apply_pose:
            override = load_object_pose_override(candidate_path)
            specs = {o.safe_key: _LoadSpec(o.pos, o.quat_wxyz) for o in objects}
            override, notes = resolve_override_for_run(override, specs, mode=pose_mode)
            override_lines += [f"note: {n}" for n in notes]
            override_lines += apply_object_pose_override(
                env, override, settle_steps=OVERRIDE_SETTLE_STEPS, max_displacement=max_displacement
            )
            for line in override_lines:
                print(f"  [object pose override] {line}")

        report = None
        frames: list = []
        if steps > 0:
            report = probe_settle(
                env.entities,
                scene.step,
                [o.safe_key for o in objects],
                steps=steps,
                sample_every=SAMPLE_EVERY,
                object_names=scene.object_names,
                mode=mode,
                hold_steps=hold_steps,
                frame_callback=(lambda: scene.render("camera")) if video else None,
                frames_out=frames if video else None,
            )
            for line in report.log_lines():
                print(f"  [object settle] {line}")
            write_settle_dump(out_dir / "settle.json", report, scene_name=layout.scene, source=str(candidate_path or ""))
            if video:
                write_settle_video(str(out_dir / "settle.mp4"), frames, fps=VIDEO_FPS)

        import imageio.v3 as iio

        from r2s.common.io import prepare_output

        previews = {}
        for cam in PREVIEW_CAMERAS:
            path = prepare_output(out_dir / "preview" / f"{cam}.png")
            iio.imwrite(path, scene.render(cam))
            previews[cam] = str(path)
        check_path = _write_check(layout, out_dir / "preview" / "camera.png", out_dir / "check.png") if check else None
    finally:
        scene.destroy()

    stats: dict[str, Any] = {
        "out_dir": str(out_dir),
        "dump": str(dump),
        "candidate": None if candidate_path is None else str(candidate_path),
        "mode": mode,
        "steps": int(steps),
        "hold_steps": int(hold_steps),
        "pose_mode": pose_mode if candidate_path is not None and apply_pose else None,
        "max_displacement": max_displacement,
        "scale_overrides": loads["scale"],
        "collision_overrides": loads["collision"],
        "previews": previews,
        "check": check_path,
        "override_log": override_lines,
        "seconds": round(time.time() - started, 1),
    }
    if report is not None:
        stats["objects"] = {
            o.safe_key: {"xy_drift_m": o.xy_drift, "tilt_deg": o.tilt, "z_drop_m": o.z_drop, "unstable": bool(o.unstable)}
            for o in report.objects
        }
        stats["n_unstable"] = int(sum(o.unstable for o in report.objects))
    write_json(out_dir / "sim" / "run_stats.json", stats)
    return stats


def run_preview(
    layout: SceneLayout,
    override_path: Path | None,
    out_dir: Path,
    *,
    pose_mode: str = "delta",
    max_displacement: float = 0.01,
    dump_path: Path | None = None,
    use_cpu: bool = False,
) -> dict[str, Any]:
    """``objects.posefit`` preview (the old pose-fit launcher's ``preview``): pose override, no settle probe.

    The old launcher passed only ``--object-pose-override``, so the evaluator defaults
    applied: ``delta`` mode, 1 cm revert threshold, and no scale override.
    """
    return run_settle(
        layout, override_path, out_dir, "free", 0, 0,
        pose_mode=pose_mode, max_displacement=max_displacement, dump_path=dump_path,
        apply_scale=False, video=False, use_cpu=use_cpu,
    )


def main(argv: list[str] | None = None) -> int:
    from r2s.config import Paths, SceneRegistry
    from r2s.objects.settle import SETTLE_MODES

    ap = argparse.ArgumentParser(prog="python -m r2s.sim.settle", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--candidate", type=Path, default=None, help="candidate.json / refined_poses.json (poses + scale multipliers)")
    ap.add_argument("--dump", type=Path, default=None, help="object dump to load (default: layout.object_dump)")
    ap.add_argument("--mode", choices=SETTLE_MODES, default="free")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--hold-steps", type=int, default=20)
    ap.add_argument("--pose-mode", choices=POSE_MODES, default=DEFAULT_POSE_MODE)
    ap.add_argument("--max-displacement", type=float, default=DEFAULT_MAX_DISPLACEMENT)
    ap.add_argument("--no-scale", action="store_true", help="apply the candidate's poses but not its scale multipliers")
    ap.add_argument("--no-pose", action="store_true", help="apply the candidate's scale multipliers but not its poses")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    layout = SceneLayout.create(args.scene, Paths.load(args.config), SceneRegistry())
    stats = run_settle(
        layout, args.candidate, args.out, args.mode, args.steps, args.hold_steps,
        pose_mode=args.pose_mode, max_displacement=args.max_displacement, dump_path=args.dump,
        apply_scale=not args.no_scale, apply_pose=not args.no_pose, video=not args.no_video,
    )
    print(json.dumps({k: v for k, v in stats.items() if k in ("out_dir", "n_unstable", "seconds")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
