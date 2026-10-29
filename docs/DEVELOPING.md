# Developing r2s

Conventions for anyone (human or agent) adding or porting a step.

## Hard rules

- **Read-only sources.** Never write into the old repo
  (`scene_gaussian_alignment`, `$R2S_OLD_REPO`), `scene_generation`, the
  Eden checkout, Eden's `echo2` assets, or the scene data folder. Run Python with
  `PYTHONDONTWRITEBYTECODE=1` when importing anything from those folders.
- **Every output goes through `r2s.common.io.prepare_output(path)`** (or
  `write_json` / `save_npy`). Work folders get seeded with symlinks to read-only
  inputs; `prepare_output` unlinks a symlink before writing so nothing is ever
  written through one.
- **No hardcoded paths.** Every path comes from `SceneLayout` (`r2s/layout.py`)
  or `Paths` (`configs/paths.yaml`). Machine folders belong in the gitignored
  `configs/local.env` only: no absolute path in code, tests, scripts or docs, no
  Eden-asset fallbacks, no "look next to the result file" guessing.
- **Fail loudly.** A missing input is an error, not a silent fallback (the old
  repo's scale-0.5, unrefined-pose and v1-receiver fallbacks are gone).

## Numerics must match the old repo

Port algorithms by moving code, not rewriting it. Keep function bodies, constants
and operation order as they were, so the new step reproduces the old output on the
same inputs (bit-identical for the deterministic CPU steps). Allowed changes:

- imports (shared helpers now live in `r2s/common/`),
- paths and CLI plumbing,
- deleting dead or legacy modes no production run used,
- making optional viewers (viser) lazy imports that only load with `--viewer`.

Shared helpers: `r2s.common.quat` (wxyz quaternions), `r2s.common.transforms`
(pose/scale loading and the canonical splat/collider world poses),
`r2s.common.camera` (DROID calibration), `r2s.common.mesh` (GLB loading).
If an old module had its own private copy of one of these and it is numerically
identical, use the shared one; if it differs in any way, keep the private copy.
The splat-frame constants (`SCALE_OPT_BASE_ROT`, `SCALE_OPT_ADJUSTMENT`) are defined
only in `r2s/common/transforms.py`; modules that kept the old names import them
under those names (`scale_core._BASE_ROT`), and a test fails if the literals appear
anywhere else.

## Step module contract

Each step is a module listed in `r2s/steps.py:STEP_MODULES` exposing

```python
STEP = Step(
    name="surface.plane",
    help="one-line description",
    add_arguments=add_arguments,      # argparse options; defaults = the production values
    run=run,                          # run(layout, args) -> dict of stats (stored in the manifest)
    inputs=lambda layout, args: [...],
    outputs=lambda layout, args: [...],
    needs="cpu",                      # "cpu", "gpu", "conda:<env>" or "gpu+conda:<env>"
)
```

- **`needs`** says where the step runs: `cpu` (repo env), `gpu` (repo env, needs
  CUDA), `conda:<env>` (CPU in a model conda env) or `gpu+conda:<env>`. Anything
  else is rejected when the step is loaded. Audit the worker: a model or helper that
  defaults to `device="cuda"` makes it a GPU step. `r2s pipeline` uses it to fail
  fast without a GPU or a conda env; `r2s doctor` to list the envs to check.
- **Register the step** in `r2s/steps.py:STEP_MODULES` at its place in the run order
  (the number is the one in docs/pipeline.md); `OPTIONAL_STEPS` are left out of a
  `r2s pipeline` range unless named.
- **Read inputs through `layout.inputs`** (the scene's input root: the run's own
  `scene_data_out` once imported, else `scene_data`), write step-0 files through
  `layout.inputs_out`; `layout.inputs_src` is `scene_data` itself (default episode
  source, saved answers to replay). The CLI records the root in the manifest
  (`inputs_root`).
- **Defaults are the production values.** The old launchers passed many
  non-default flags; bake those in as the new defaults so
  `python -m r2s run <step> --scene X` reproduces production with no flags.
- Per-scene deviations go in `configs/scenes.yaml` under the step name.
- The CLI checks inputs exist, records args/inputs/outputs in the scene's
  `manifest.json`, then calls `run`. Don't re-implement that in the step.
- Keep the algorithm callable as a plain function (e.g.
  `fit_plane(collider_glb, pose_path, scale_path, ...) -> dict`) so tests and
  other steps can use it without the CLI.

## Verification

`python tools/legacy/old_repo.py seed --scene S --through <previous step> --run verify`
fills a work run with the old repo's outputs up to the step under test (small
files copied, big PLY/GLB symlinked). Then run the new step and compare against
`tools/legacy/old_repo.py reference`. The comparison scripts live in
`tools/legacy/` (see its README); they need the old checkouts, import each other
as siblings (`from old_repo import OLD`) and write results under the work folder,
never into the repo. `bash tools/legacy/cpu_chain.sh S RUN` reruns steps 4 -> 9 (fit)
on the old alignment and compares every output.

## Tests

`$R2S_PYTHON -m pytest -q tests` (CPU, ~30 s). A test that reads the old repo gets
its path from `tests/legacy_paths.py` (`old_file(...)` skips when the file is absent),
so the suite runs without it; tests that need `scene_generation`, Eden assets or a
conda env skip likewise. Without `configs/local.env` (a fresh clone) `conftest.py`
points every unset folder at `/nonexistent/<VAR>`, so those tests skip too. Slow checks are opt-in:
`R2S_TEST_GENESIS=1` (crop vs Genesis itself), `R2S_TEST_CONDA=1` (worker imports in
the `match` / `moge` envs).

## Environments

- Repo env: the Eden HEAD environment (`paths.python`), built by
  `scripts/setup_env.sh`. GPU steps run on the host (the `eden.sif` container's nvcc
  cannot build gsplat; see docs/setup.md). `r2s doctor` checks it.
- Step-0 models run in their own conda envs (`gemini`, `invsr`, `lama`,
  `sam3d-objects`, `match`, `moge`) through
  `r2s.gen.conda.run_in_env`. Workers (`r2s/gen/workers/`)
  never import `r2s`; the ones that use moved scene_generation code append `r2s/gen`
  to `sys.path` and import the package (`from sg_glue import gembox`), and modules
  inside `sg_glue/` import each other relatively (`from .gemini_utils import ...`).
- `r2s pipeline` runs each step in its own process, so a step may initialise Genesis
  or torch freely; never import them at module level of a step (`r2s list` loads
  every step module).
