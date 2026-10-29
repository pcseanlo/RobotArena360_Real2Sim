"""``r2s check --scene S [--inputs-only] [--src DIR]``: validate a scene's episode folder.

Checks the folder step 0a reads (``layout.inputs.meta_dir``, or ``--src``) against
docs/custom_capture.md §1, one OK / WARN / FAIL line per item:

* ``images0/{0,1}.jpg`` and ``depth/{0,1}.npy`` exist; the two frames have one size;
* depth: 2-D float32 metres, the RGB's ``(H, W)``, plausible range (0.05-20 m), few holes;
* ``intrinsics.npy``: 3x3 with ``fx, fy > 0``; the principal point is centred
  (``|2cx - W|``, ``|2cy - H|``: WARN above 2 %, FAIL above 10 % of the side), because
  every step derives the image size as ``2 c``;
* ``extrinsics.npy``: 4x4 world-to-camera with an orthonormal rotation (det 1), and the
  optical axis reaches the robot workspace (the ray crosses z = 0.3 .. z_min of the
  workspace box inside its x/y bounds);
* ``joints.npy (T, 7)``, ``grippers.npy (T,)``, ``tcp.npy (T, 4, 4)`` or ``(T, 3)`` with
  one ``T`` (``joints_old.npy`` stands in for ``joints.npy`` as in step 9); missing
  files are a WARN (no grasp term in step 9, no recorded reset pose in the sim);
* ``lang.txt`` (WARN when missing).

Then (unless ``--inputs-only``) every step's declared inputs are checked for
existence with its default options, next to its manifest record. Exit 1 on any FAIL.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import NamedTuple

import numpy as np

DEPTH_RANGE_M = (0.05, 20.0)
PRINCIPAL_POINT_WARN, PRINCIPAL_POINT_FAIL = 0.02, 0.10
HOLE_WARN, HOLE_FAIL = 0.01, 0.10
AXIS_TOP_Z = 0.3  # the optical axis must reach the workspace between this height and the box floor


class Finding(NamedTuple):
    level: str  # "OK", "WARN" or "FAIL"
    item: str
    message: str


def print_findings(findings: list[Finding], title: str) -> int:
    """Print one line per finding and a summary; returns the number of FAILs."""
    print(title)
    for f in findings:
        print(f"{f.level:<5} {f.item:<19} {f.message}")
    counts = {lvl: sum(f.level == lvl for f in findings) for lvl in ("OK", "WARN", "FAIL")}
    print(f"summary: {counts['OK']} OK, {counts['WARN']} WARN, {counts['FAIL']} FAIL")
    return counts["FAIL"]


# --------------------------------------------------------------------------- episode folder
def _depth_findings(path: Path, hw: tuple[int, int] | None) -> list[Finding]:
    item = f"depth/{path.name}"
    depth = np.load(path, allow_pickle=False)
    if depth.ndim != 2:
        return [Finding("FAIL", item, f"shape {depth.shape}; needs (H, W)")]
    if not np.issubdtype(depth.dtype, np.floating):
        return [Finding("FAIL", item, f"dtype {depth.dtype}; needs float32 metres (integer depth is usually mm)")]
    fails, warns = [], []
    if depth.dtype != np.float32:
        warns.append(f"dtype {depth.dtype}, the pipeline expects float32")
    if hw is not None and depth.shape != hw:
        fails.append(f"shape {depth.shape} != the RGB's (H, W) {hw}")
    holes = ~np.isfinite(depth) | (depth <= 0)
    frac = float(holes.mean())
    if frac == 1.0:
        return [Finding("FAIL", item, "; ".join(["no valid depth (all non-finite or <= 0)", *fails]))]
    valid = depth[~holes].astype(np.float64)
    lo, hi = DEPTH_RANGE_M
    med = float(np.median(valid))
    outside = float(((valid < lo) | (valid > hi)).mean())
    if not lo <= med <= hi:
        fails.append(f"median outside {lo}-{hi} m: not metres?")
    elif outside > 0.01:
        warns.append(f"{100 * outside:.1f}% of pixels outside {lo}-{hi} m")
    if frac > HOLE_FAIL:
        fails.append(f"{100 * frac:.1f}% holes (non-finite or <= 0); needs dense depth")
    elif frac > HOLE_WARN:
        warns.append(f"{100 * frac:.1f}% holes (non-finite or <= 0)")
    stats = (f"{depth.shape[1]}x{depth.shape[0]} {depth.dtype}, {valid.min():.3f} / {med:.3f} / {valid.max():.3f} m "
             f"(min/median/max), {100 * frac:.2f}% holes")
    level = "FAIL" if fails else "WARN" if warns else "OK"
    return [Finding(level, item, "; ".join([stats, *fails, *warns]))]


def _intrinsics_findings(path: Path, wh: tuple[int, int] | None) -> list[Finding]:
    K = np.load(path, allow_pickle=True)
    out = []
    if K.ndim == 3 and K.shape[1:] == (3, 3):
        out.append(Finding("WARN", "intrinsics", f"shape {K.shape}; inputs.episode keeps intrinsics[0]"))
        K = K[0]
    if K.shape != (3, 3):
        return [Finding("FAIL", "intrinsics", f"shape {K.shape}; needs (3, 3)")]
    K = K.astype(np.float64)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    if not (np.isfinite(K).all() and fx > 0 and fy > 0):
        return out + [Finding("FAIL", "intrinsics", f"fx={fx:g} fy={fy:g}; both must be > 0")]
    if abs(K[0, 1]) > 1e-6 or not np.allclose(K[2], (0.0, 0.0, 1.0)):
        out.append(Finding("WARN", "intrinsics", f"not a plain pinhole K (skew {K[0, 1]:g}, last row {K[2].tolist()})"))
    out.append(Finding("OK", "intrinsics", f"fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}"))
    if wh is not None:
        for axis, c, side in (("x", cx, wh[0]), ("y", cy, wh[1])):
            off = abs(2.0 * c - side)
            rel = off / side
            msg = (f"|2c{axis} - {'W' if axis == 'x' else 'H'}| = {off:.1f} px ({100 * rel:.1f}% of {side}); "
                   f"steps use the image size 2c = {round(2 * cx)}x{round(2 * cy)}")
            if rel > PRINCIPAL_POINT_FAIL:
                out.append(Finding("FAIL", "principal pt", msg + "; crop/pad the frames so c is centred"))
            elif rel > PRINCIPAL_POINT_WARN:
                out.append(Finding("WARN", "principal pt", msg))
            else:
                out.append(Finding("OK", "principal pt", msg))
    return out


def optical_axis_hit(extrinsics: np.ndarray, workspace) -> tuple[bool, str]:
    """Does the camera's optical axis cross the workspace box between z = 0.3 and its floor?"""
    R, t = extrinsics[:3, :3], extrinsics[:3, 3]
    cam = -R.T @ t
    axis = R[2]  # camera +z (OpenCV forward) in world coordinates
    x0, x1, y0, y1, z0, _ = workspace
    if axis[2] >= -1e-6:
        return False, f"camera at {np.round(cam, 3).tolist()} looks level or up (axis z {axis[2]:+.2f})"
    z_top = min(AXIS_TOP_Z, cam[2])
    if z_top <= z0:
        return False, f"camera at {np.round(cam, 3).tolist()} is below the workspace floor z={z0}"
    ts = np.linspace((z_top - cam[2]) / axis[2], (z0 - cam[2]) / axis[2], 256)
    pts = cam[None] + ts[:, None] * axis[None]
    inside = (pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)
    a, b = np.round(pts[0], 2).tolist(), np.round(pts[-1], 2).tolist()
    where = f"axis from camera {np.round(cam, 2).tolist()} crosses z={z_top:.2f}..{z0} at {a} .. {b}"
    if not inside.any():
        return False, f"{where}, outside the workspace x {x0}..{x1}, y {y0}..{y1}"
    first = np.round(pts[np.argmax(inside)], 2).tolist()
    return True, f"{where}; inside the workspace from {first}"


def _extrinsics_findings(path: Path, workspace) -> list[Finding]:
    E = np.load(path, allow_pickle=False).astype(np.float64)
    if E.shape != (4, 4):
        return [Finding("FAIL", "extrinsics", f"shape {E.shape}; needs (4, 4) world-to-camera")]
    out = []
    if not np.allclose(E[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6):
        out.append(Finding("WARN", "extrinsics", f"last row {E[3].tolist()}, expected [0, 0, 0, 1]"))
    R = E[:3, :3]
    ortho = float(np.abs(R @ R.T - np.eye(3)).max())
    det = float(np.linalg.det(R))
    if ortho > 1e-3 or abs(det - 1.0) > 1e-3:
        return out + [Finding("FAIL", "extrinsics", f"rotation not orthonormal (|R R^T - I| = {ortho:.2e}, det {det:.4f})")]
    out.append(Finding("OK", "extrinsics", f"rotation orthonormal (|R R^T - I| = {ortho:.1e}, det {det:.6f})"))
    ok, msg = optical_axis_hit(E, workspace)
    if ok:
        out.append(Finding("OK", "optical axis", msg))
    else:
        hint = ""
        if optical_axis_hit(np.linalg.inv(E), workspace)[0]:
            hint = "; its inverse passes: extrinsics.npy looks camera-to-world (needs world-to-camera)"
        out.append(Finding("FAIL", "optical axis", msg + hint))
    return out


def _robot_findings(meta: Path, n_frames: int) -> list[Finding]:
    missing = [n for n in ("joints.npy", "grippers.npy", "tcp.npy") if not (meta / n).is_file()]
    out = []
    if missing:
        out.append(Finding("WARN", "robot state", f"missing {', '.join(missing)}: no grasp term in objects.refine "
                                                  "and no recorded reset pose for the sim"))
    lengths: dict[str, int] = {}
    joints = "joints_old.npy" if (meta / "joints_old.npy").is_file() else "joints.npy"
    specs = {joints: "(T, 7)", "grippers.npy": "(T,)", "tcp.npy": "(T, 4, 4) or (T, 3)"}
    for name, want in specs.items():
        if not (meta / name).is_file():
            continue
        a = np.load(meta / name, mmap_mode="r", allow_pickle=False)
        ok = {
            joints: a.ndim == 2 and a.shape[1] == 7,
            "grippers.npy": a.ndim == 1 or (a.ndim == 2 and a.shape[1] == 1),
            "tcp.npy": (a.ndim == 3 and a.shape[1:] == (4, 4)) or (a.ndim == 2 and a.shape[1] >= 3),
        }[name]
        if not ok:
            out.append(Finding("FAIL", "robot state", f"{name} has shape {a.shape}; needs {want}"))
            continue
        lengths[name] = int(a.shape[0])
    if lengths:
        desc = ", ".join(f"{k} {v}" for k, v in lengths.items())
        if len(set(lengths.values())) == 1:
            out.append(Finding("OK", "robot state", f"T = {next(iter(lengths.values()))} ({desc}; {n_frames} frames)"))
        else:
            out.append(Finding("WARN", "robot state", f"lengths differ ({desc}): joints not frame-aligned, the grasp "
                                                      "anchor falls back to tcp"))
    return out


def check_episode(meta: Path, workspace) -> list[Finding]:
    """Findings for one episode folder (see the module docstring)."""
    meta = Path(meta)
    if not meta.is_dir():
        return [Finding("FAIL", "episode", f"{meta} does not exist (run inputs.episode, or pass --src)")]
    out = [Finding("OK", "episode", str(meta))]
    sizes = {}
    for n in (0, 1):
        rgb = meta / "images0" / f"{n}.jpg"
        if not rgb.is_file():
            out.append(Finding("FAIL", f"images0/{n}.jpg", "missing"))
            continue
        from PIL import Image

        with Image.open(rgb) as im:
            sizes[n] = im.size
            out.append(Finding("OK" if im.mode == "RGB" else "WARN", f"images0/{n}.jpg", f"{im.size[0]}x{im.size[1]} {im.mode}"))
    if len(set(sizes.values())) > 1:
        out.append(Finding("FAIL", "images0", f"frames 0 and 1 differ in size: {sizes}"))
    wh = sizes.get(1, sizes.get(0))
    hw = (wh[1], wh[0]) if wh else None
    n_frames = len([p for p in (meta / "images0").glob("*.jpg")]) if (meta / "images0").is_dir() else 0

    checks = [(f"depth/{n}.npy", meta / "depth" / f"{n}.npy", lambda p: _depth_findings(p, hw)) for n in (0, 1)]
    checks += [("intrinsics", meta / "intrinsics.npy", lambda p: _intrinsics_findings(p, wh)),
               ("extrinsics", meta / "extrinsics.npy", lambda p: _extrinsics_findings(p, workspace))]
    for item, path, fn in checks:
        if not path.is_file():
            out.append(Finding("FAIL", item, f"missing {path.name}"))
            continue
        try:
            out += fn(path)
        except Exception as exc:  # an unreadable file is a finding, not a crash
            out.append(Finding("FAIL", item, f"cannot read {path.name}: {type(exc).__name__}: {exc}"))
    try:
        out += _robot_findings(meta, n_frames)
    except Exception as exc:
        out.append(Finding("FAIL", "robot state", f"{type(exc).__name__}: {exc}"))
    lang = meta / "lang.txt"
    if lang.is_file() and lang.read_text().strip():
        out.append(Finding("OK", "lang.txt", repr(lang.read_text().strip().splitlines()[0][:80])))
    else:
        out.append(Finding("WARN", "lang.txt", "missing or empty (eval prompt / physics context)"))
    return out


# --------------------------------------------------------------------------- steps
def short_path(path: Path, layout) -> str:
    """``path`` with the docs/pipeline.md shorthands: ``S`` (scene outputs), ``OUT`` / ``IN`` (step-0 folders)."""
    prefixes = [(layout.root, "S"), (layout.inputs_out.meta_dir, "OUT"), (layout.inputs_out.root, "<scene_data_out>"),
                (layout.inputs.meta_dir, "IN"), (layout.inputs.root, "<scene_data>")]
    for prefix, short in prefixes:
        if Path(path) == prefix or prefix in Path(path).parents:
            return f"{short}/{Path(path).relative_to(prefix)}".rstrip("/.")
    return str(path)


def step_lines(layout) -> list[str]:
    """One line per step: are its inputs there (default options), has it run (manifest)."""
    from r2s.cli import missing_inputs, step_parser
    from r2s.manifest import Manifest
    from r2s.steps import OPTIONAL_STEPS, STEP_MODULES, load_step

    records = Manifest(layout.manifest).load().get("steps", {})
    lines = [f"  S = {layout.root}", f"  OUT = {layout.inputs_out.meta_dir}", f"  IN = {layout.inputs.meta_dir}"]
    for name, (number, _) in STEP_MODULES.items():
        try:
            step = load_step(name)
            args = step_parser(step, layout).parse_args([])
            missing = missing_inputs(step, layout, args)
            if missing:
                more = f" (+{len(missing) - 1} more)" if len(missing) > 1 else ""
                state = f"missing {short_path(missing[0], layout)}{more}"
            else:
                state = "inputs ok"
        except BaseException as exc:  # a step that cannot even list its inputs here
            state = f"inputs ? ({type(exc).__name__}: {exc})"
        rec = records.get(name)
        ran = "not run" if rec is None else f"{rec.get('status')} {rec.get('started')}"
        opt = " (optional)" if name in OPTIONAL_STEPS else ""
        lines.append(f"{number:>5}  {name:<20} {ran:<28} {state}{opt}")
    return lines


def main(argv: list[str]) -> int:
    from r2s.config import Paths, SceneRegistry
    from r2s.layout import SceneLayout
    from r2s.surface.geometry import DEFAULT_WORKSPACES

    parser = argparse.ArgumentParser(prog="python -m r2s check", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--src", type=Path, default=None,
                        help="episode folder to validate (default: the scene's inputs, layout.inputs.meta_dir)")
    parser.add_argument("--inputs-only", action="store_true", help="only validate the episode folder")
    parser.add_argument("--config", default=None)
    parser.add_argument("--scenes-file", default=None)
    args = parser.parse_args(argv)

    layout = SceneLayout.create(args.scene, Paths.load(args.config), SceneRegistry(args.scenes_file))
    workspace = DEFAULT_WORKSPACES.get(layout.config.robot, DEFAULT_WORKSPACES["franka"])
    meta = args.src or layout.inputs.meta_dir
    root, why = layout.inputs_root_source()
    title = f"[r2s check] {layout.scene}: episode {meta}" + ("" if args.src else f"  (inputs root: {why})")
    fails = print_findings(check_episode(meta, workspace), title)
    if not args.inputs_only:
        print(f"\nsteps  (inputs root {root}; manifest {layout.manifest})")
        for line in step_lines(layout):
            print(line)
    return 1 if fails else 0
