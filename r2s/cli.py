"""``python -m r2s`` entry point.

    python -m r2s list [--names [--all]]                # steps, run order, where they run
    python -m r2s run <step> --scene droid2 [options]   # one step for one scene
    python -m r2s pipeline --scene droid2 [--from STEP] [--to STEP] [--skip-done] [--include-optional]
    python -m r2s status --scene droid2                 # what exists, what ran with which args
    python -m r2s check --scene droid2 [--inputs-only]  # validate the episode folder, list step inputs
    python -m r2s doctor                                # env, paths, conda envs, weights, GPU
    python -m r2s export --scene droid2 --to DIR        # old-evaluator layout (optimization_results/, echo2/)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from r2s.config import REPO_ROOT, ConfigError, Paths, SceneRegistry
from r2s.layout import SceneLayout
from r2s.manifest import Manifest, step_record
from r2s.steps import OPTIONAL_STEPS, STEP_MODULES, Step, load_step


def _cmd_list(argv: list[str]) -> int:
    if "--names" in argv:  # machine-readable run order
        names = [n for n in STEP_MODULES if "--all" in argv or n not in OPTIONAL_STEPS]
        print("\n".join(names))
        return 0
    for name, (number, module) in STEP_MODULES.items():
        try:
            needs = load_step(name).needs
        except Exception as exc:  # a step whose module is not importable here still lists
            needs = f"(not importable: {type(exc).__name__})"
        tag = "optional" if name in OPTIONAL_STEPS else ""
        print(f"{number:>5}  {name:<20} {needs:<26} {tag:<9} {module}")
    return 0


# --------------------------------------------------------------------------- one step
def step_parser(step: Step, layout: SceneLayout) -> argparse.ArgumentParser:
    """The step's argparse parser, with the scene's scenes.yaml options as defaults."""
    parser = argparse.ArgumentParser(prog=f"python -m r2s run {step.name}", description=step.help)
    step.add_arguments(parser)
    # Per-scene settings from scenes.yaml become defaults; explicit flags still win.
    overrides = layout.config.step_options(step.name)
    unknown = [k for k in overrides if not any(a.dest == k for a in parser._actions)]
    if unknown:
        raise SystemExit(f"scenes.yaml sets unknown options for {step.name}: {unknown}")
    parser.set_defaults(**overrides)
    return parser


def missing_inputs(step: Step, layout: SceneLayout, args: argparse.Namespace) -> list[Path]:
    """Declared inputs that do not exist (inputs the step treats as optional excluded)."""
    optional = {Path(p) for p in step.optional_inputs(layout, args)}
    return [Path(p) for p in step.inputs(layout, args) if not Path(p).exists() and Path(p) not in optional]


def _cmd_run(argv: list[str]) -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("step")
    pre.add_argument("--scene", required=True)
    pre.add_argument("--config", default=None, help="paths.yaml (default: configs/paths.yaml or $R2S_CONFIG)")
    pre.add_argument("--scenes-file", default=None, help="scenes.yaml (default: configs/scenes.yaml)")
    known, rest = pre.parse_known_args(argv)

    step = load_step(known.step)
    layout = SceneLayout.create(known.scene, Paths.load(known.config), SceneRegistry(known.scenes_file))
    args = step_parser(step, layout).parse_args(rest)

    inputs = step.inputs(layout, args)
    root, why = layout.inputs_root_source()
    print(f"[r2s] {step.name} {layout.scene}: inputs root {root} ({why})")
    missing = missing_inputs(step, layout, args)
    if missing:
        print(f"[r2s] {step.name} for {layout.scene}: missing inputs:", file=sys.stderr)
        for p in missing:
            print(f"  {p}", file=sys.stderr)
        return 2

    outputs = step.outputs(layout, args)
    with step_record(layout.manifest, step.name, args=vars(args), inputs=inputs, outputs=outputs,
                     inputs_root=root) as extra:
        result = step.run(layout, args) or {}
        extra.update(result)
    print(f"[r2s] {step.name} {layout.scene}: ok")
    for p in outputs:
        print(f"  -> {p}")
    return 0


# --------------------------------------------------------------------------- pipeline
def step_range(first: str | None, last: str | None, *, include_optional: bool = False) -> list[tuple[str, bool]]:
    """``(step, runs)`` for every step from ``first`` to ``last`` (inclusive), in run order.

    Optional steps get ``runs=False`` unless ``include_optional`` or they are named
    as ``first`` / ``last``.
    """
    order = list(STEP_MODULES)
    for name in (first, last):
        if name is not None and name not in STEP_MODULES:
            raise SystemExit(f"unknown step {name!r}; run `python -m r2s list`")
    i = order.index(first) if first else 0
    j = order.index(last) if last else len(order) - 1
    if i > j:
        raise SystemExit(f"--from {first} comes after --to {last} in the run order (see `python -m r2s list`)")
    return [(n, include_optional or n not in OPTIONAL_STEPS or n in (first, last)) for n in order[i : j + 1]]


def step_done(layout: SceneLayout, step: str) -> bool:
    """Manifest status ``ok`` and every recorded output still exists."""
    rec = Manifest(layout.manifest).get(step) or {}
    outputs = rec.get("outputs") or []
    return rec.get("status") == "ok" and bool(outputs) and all(Path(o["path"]).exists() for o in outputs)


def gpu_available(python: str) -> tuple[bool, str]:
    """``torch.cuda.is_available()`` in a child interpreter (this process never imports torch)."""
    code = "import torch; ok = torch.cuda.is_available(); print(torch.cuda.get_device_name(0) if ok else 'none')"
    try:
        proc = subprocess.run([python, "-c", code], capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"cannot run {python}: {exc}"
    if proc.returncode != 0:
        return False, (proc.stderr.strip().splitlines() or [f"exit {proc.returncode}"])[-1]
    name = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "none"
    return name != "none", name


def run_step_process(python: str, step: str, scene: str, extra: list[str]) -> int:
    """``python -m r2s run <step> --scene <scene>`` in its own process (Genesis/torch state never leaks)."""
    cmd = [python, "-m", "r2s", "run", step, "--scene", scene, *extra]
    return subprocess.run(cmd, cwd=REPO_ROOT).returncode


def _cmd_pipeline(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m r2s pipeline",
        description="Run a range of steps for one scene in run order, each with its production defaults "
                    "(+ scenes.yaml) in its own process; stops at the first failure.",
    )
    parser.add_argument("--scene", required=True)
    parser.add_argument("--from", dest="first", default=None, metavar="STEP", help="first step (default: the first)")
    parser.add_argument("--to", dest="last", default=None, metavar="STEP", help="last step (default: the last)")
    parser.add_argument("--skip-done", action="store_true",
                        help="skip a step whose manifest status is ok and whose recorded outputs all exist")
    parser.add_argument("--include-optional", action="store_true",
                        help=f"also run optional steps inside the range ({', '.join(sorted(OPTIONAL_STEPS))})")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and run nothing")
    parser.add_argument("--config", default=None, help="paths.yaml (passed to every step)")
    parser.add_argument("--scenes-file", default=None, help="scenes.yaml (passed to every step)")
    args = parser.parse_args(argv)

    layout = SceneLayout.create(args.scene, Paths.load(args.config), SceneRegistry(args.scenes_file))
    tag = f"[r2s] pipeline {layout.scene}"
    plan: list[tuple[str, str | None]] = []  # (step, why it is skipped)
    for name, runs in step_range(args.first, args.last, include_optional=args.include_optional):
        if not runs:
            plan.append((name, "optional (name it in --from/--to or pass --include-optional)"))
        elif args.skip_done and step_done(layout, name):
            plan.append((name, "already done (--skip-done)"))
        else:
            plan.append((name, None))
    todo = [name for name, skip in plan if skip is None]
    print(f"{tag}: {plan[0][0]} -> {plan[-1][0]}, {len(todo)} to run  (work {layout.paths.work})")
    for name, skip in plan:
        print(f"  {STEP_MODULES[name][0]:>5}  {name:<20} {skip or 'run'}")
    sys.stdout.flush()  # the plan before any error below
    if args.dry_run or not todo:
        return 0

    steps = {name: load_step(name) for name in todo}
    for name, step in steps.items():
        if step.conda_env and not (layout.paths.conda_root / "envs" / step.conda_env / "bin" / "python").exists():
            print(f"{tag}: {name} needs conda env {step.conda_env!r}, not found under {layout.paths.conda_root}/envs "
                  "(see docs/setup.md)", file=sys.stderr)
            return 2
    python = os.environ.get("R2S_PYTHON") or sys.executable
    gpu_steps = [name for name, step in steps.items() if step.gpu]
    if gpu_steps:
        ok, detail = gpu_available(python)
        if not ok:
            print(f"{tag}: {gpu_steps[0]} needs a GPU node (torch.cuda.is_available() is False here: {detail}); "
                  f"GPU steps in this range: {', '.join(gpu_steps)}.\n"
                  f"  Run it in a job:  scripts/slurm/submit.sh scripts/slurm/step.sbatch pipeline {' '.join(argv)}",
                  file=sys.stderr)
            return 2
        print(f"{tag}: GPU {detail}")

    extra: list[str] = []
    if args.config:
        extra += ["--config", str(Path(args.config).resolve())]
    if args.scenes_file:
        extra += ["--scenes-file", str(Path(args.scenes_file).resolve())]
    started = time.time()
    for i, name in enumerate(todo, 1):
        print(f"{tag}: [{i}/{len(todo)}] {name}", flush=True)
        t0 = time.time()
        code = run_step_process(python, name, layout.scene, extra)
        if code != 0:
            resume = " ".join([f"r2s pipeline --scene {layout.scene} --from {name}",
                               *([f"--to {args.last}"] if args.last else []),
                               *(["--include-optional"] if args.include_optional else []), *extra])
            print(f"{tag}: FAILED at {name} (exit {code}) after {time.time() - t0:.0f} s; "
                  f"{len(todo) - i} step(s) not run. Resume with:\n  {resume}", file=sys.stderr)
            return code
        print(f"{tag}: {name} ok ({time.time() - t0:.0f} s)", flush=True)
    print(f"{tag}: done, {len(todo)} step(s) in {time.time() - started:.0f} s")
    return 0


# --------------------------------------------------------------------------- reports
def _cmd_status(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m r2s status")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    layout = SceneLayout.create(args.scene, Paths.load(args.config))
    records = Manifest(layout.manifest).load().get("steps", {})
    root, why = layout.inputs_root_source()
    print(f"scene {layout.scene}  ({layout.root})")
    print(f"inputs root {root}  ({why})")
    for name, (number, _) in STEP_MODULES.items():
        rec = records.get(name)
        state = "-" if rec is None else f"{rec.get('status')} {rec.get('started')} ({rec.get('seconds')} s)"
        print(f"{number:>5}  {name:<20} {state}")
    return 0


def _cmd_check(argv: list[str]) -> int:
    from r2s.check import main as check_main

    return check_main(argv)


def _cmd_doctor(argv: list[str]) -> int:
    from r2s.doctor import main as doctor_main

    return doctor_main(argv)


def _cmd_export(argv: list[str]) -> int:
    from r2s.export import export_scene

    parser = argparse.ArgumentParser(prog="python -m r2s export", description=export_scene.__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--to", required=True, type=Path)
    parser.add_argument("--config", default=None)
    parser.add_argument("--copy", action="store_true", help="copy files instead of symlinking")
    args = parser.parse_args(argv)
    layout = SceneLayout.create(args.scene, Paths.load(args.config))
    written = export_scene(layout, args.to, copy=args.copy)
    print(json.dumps({k: str(v) for k, v in written.items()}, indent=2))
    return 0


COMMANDS = {
    "list": _cmd_list,
    "run": _cmd_run,
    "pipeline": _cmd_pipeline,
    "status": _cmd_status,
    "check": _cmd_check,
    "doctor": _cmd_doctor,
    "export": _cmd_export,
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"} or argv[0] not in COMMANDS:
        print(__doc__)
        return 0 if not argv or argv[0] in {"-h", "--help"} else 2
    try:
        return COMMANDS[argv[0]](argv[1:])
    except ConfigError as exc:
        print(f"r2s: {exc}", file=sys.stderr)
        return 2
