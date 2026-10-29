# CLAUDE.md

Orientation for coding agents working in this repo. Read this first, then the doc
for the task at hand.

**What this is:** `r2s` turns one real DROID episode into a sim-ready Gaussian scene
for [Eden](https://github.com/Kashu7100/Eden) (`robotarena360` branch, verified at
`b7c85cc`; README.md "Dependencies"). The output is an aligned splat background, a
collider, a support table and an object layout. The pipeline has 22 steps
(`r2s list`), from `inputs.episode` to `objects.freeze`, and each step is one CLI
command. This repo is a cleaned-up port of the scene-reconstruction half of
`scene_gaussian_alignment` plus the asset and inpainting pipeline of
`scene_generation`. Policy evaluation and rendering fidelity (Nyx, shadow
compositing) are out of scope.

| Need | Read |
|---|---|
| Method, install, quickstart, Eden mapping, troubleshooting | [docs/user_guide.md](docs/user_guide.md) |
| One step's inputs, outputs, options, old equivalent; file formats | [docs/pipeline.md](docs/pipeline.md) |
| The episode folder, DROID preprocessing, own recordings | [docs/custom_capture.md](docs/custom_capture.md) |
| Repo tree and a run's output tree | [docs/structure.md](docs/structure.md) |
| Eden, `configs/local.env`, environment, conda envs, Slurm | [docs/setup.md](docs/setup.md) |
| Rules for adding or porting a step | [docs/DEVELOPING.md](docs/DEVELOPING.md) |
| Behaviour changes, old-code bugs, verification results | [docs/changes.md](docs/changes.md) |
| Old file → new module | [docs/porting/old_to_new.md](docs/porting/old_to_new.md) |

## Every session

If `CLAUDE.local.md` exists, read it first. It is gitignored and holds this
machine's notes: folders, cluster and partition, held GPU jobs, earlier runs.

```bash
cd <repo> && source scripts/env.sh
```

`env.sh` does the following:

- loads this machine's folders and Slurm settings from `configs/local.env`
  (gitignored; template `configs/local.env.example`). A variable already set in the
  environment wins. If the file is missing, ask the user for the folders; never
  guess them.
- puts the repo env first on `PATH` (`$R2S_PYTHON`: Eden's lock + gsplat, in
  `$R2S_WORK_ROOT/envs/eden`);
- sets `TORCH_EXTENSIONS_DIR`, `TORCH_CUDA_ARCH_LIST`, `NUMBA_CACHE_DIR` and
  `PYTHONDONTWRITEBYTECODE=1`, and puts `$CUDA_HOME/bin` on `PATH`;
- defines `r2s` as `python -m r2s`.

Always use `$R2S_PYTHON` (or `r2s`), never the system Python, Eden's `.venv` or the
`eden.sif` container. The container's nvcc cannot build gsplat.

## Commands

| Command | Use |
|---|---|
| `r2s list` | steps in run order, where each runs (`cpu`, `gpu`, `conda:<env>`, `gpu+conda:<env>`), optional steps |
| `r2s run <step> --scene <s> [opts]` | one step with production defaults + `configs/scenes.yaml`; `--help` lists its options |
| `r2s pipeline --scene <s> --from A --to B [--skip-done] [--dry-run]` | a range, one process per step; stops at the first failure and prints the resume command |
| `r2s status --scene <s>` | what ran, with which args, ok / failed |
| `r2s check --scene <s> [--src DIR] [--inputs-only]` | validate an episode folder, then list which step inputs exist |
| `r2s doctor` | env, gsplat build, GPU, Eden commit, paths, conda envs, weights; OK / WARN / FAIL (the only expected WARN is a missing `GEMINI_API_KEY`) |
| `r2s export --scene <s> --to DIR` | symlink a scene into the old evaluator's layout |

**Scratch runs.** Set `R2S_WORK=$R2S_WORK_ROOT/runs/<name>` for any experiment, so
its outputs never mix with another run's. Earlier runs under `$R2S_WORK_ROOT/runs/`
(`e2e`, `verify*`, `hardening_*`, `mesh_sheet_fill`, `collider_repo_env`) are evidence cited in
docs/changes.md. Do not overwrite or delete them.

## Where things run

- **On a login / CPU node:** `cpu` and `conda:<env>` steps run in place.
- **GPU steps** (`r2s list` shows `gpu`) need a GPU node. `r2s pipeline` refuses a
  range with a GPU step on a node without a GPU, before running anything.
  - Batch: `scripts/slurm/submit.sh scripts/slurm/step.sbatch run <step> --scene <s>`,
    or `... pipeline --scene <s> --from A --to B`, or `... --cmd "<shell command>"`.
    `submit.sh` is `sbatch` plus the partition, QOS, account and GPU type from
    `configs/local.env` (`R2S_SLURM_*`). Plain `sbatch` uses the cluster defaults,
    which may be the wrong partition.
  - A scratch run passes through the environment:
    `R2S_WORK=$R2S_WORK scripts/slurm/submit.sh ...`.
  - Logs: `$R2S_WORK_ROOT/logs/r2s_<jobname>_<jobid>.log`.
- **A GPU you already hold** (an idle interactive or hold job, see `squeue -u $USER`):

  ```bash
  srun --jobid=<JOBID> --overlap -n1 bash -c "cd $R2S_REPO && source scripts/env.sh && r2s run align.pose --scene droid2"
  ```

- **Timings:**
  - `align.pose`: 1-3 min.
  - `align.scale`: 10-15 min on the first run per scene (CoACD), then ~70 s.
  - `objects.refine`: 3-4 min per settle stage.
  - `world.collider`: ~4 min.
  - Everything else: seconds.

## Hard rules

1. **Read-only sources.** Never write into any folder `configs/local.env` points at
   except `R2S_WORK_ROOT`, not even `__pycache__`. These are:
   - the Eden checkout (`R2S_EDEN`), including the old `assets/echo2`
     (`R2S_OLD_ECHO2`);
   - the old repo `scene_gaussian_alignment` (`R2S_OLD_REPO`);
   - `scene_generation` and its `data/` (`R2S_SCENE_GENERATION`, `R2S_DATA`);
   - the splats, `sam-3d-objects`, the layout refiner, the task notes and the conda
     install.

   `PYTHONDONTWRITEBYTECODE=1` is set for this reason. Keep it set when importing
   from those folders.
2. **Every output goes through `r2s.common.io.prepare_output`** (or `write_json` /
   `save_npy`). Run folders are seeded with symlinks to read-only data, and
   `prepare_output` unlinks a symlink before writing. Never `open(path, "w")` an
   output directly.
3. **No secrets in files.** API keys come only from `GEMINI_API_KEY` / `WLT_API_KEY`
   in the environment. The old repos contain plain-text keys (`generate_marble.sh`,
   SG `configs/config_DROID.yaml`, `sam3d.py`). Never print, copy or quote them.
4. **Do not call Gemini** (a live `inputs.segment` / `inputs.physics` run) unless
   the user asks. Use `--replay-gemini` / `--replay`, which reuse the saved answers.
5. **No hardcoded paths and no silent fallbacks.** Paths come from
   `configs/paths.yaml` via `Paths` / `SceneLayout`. A missing input raises an error.
   Machine paths and cluster names (partition, QOS) go only into the gitignored
   `configs/local.env` and `CLAUDE.local.md`. No committed file, whether code, test,
   script or doc, may contain one. Before committing, check with
   `git grep -nE '/(data|home|scratch)/'`.
6. **Numerics must match the old repo.** Port by moving code, not rewriting it. The
   CPU steps are bit-identical to the old code, and the tests guard that. A change
   that moves numbers is a behaviour change: put it behind a flag with the old
   default, or record it in docs/changes.md.
7. **Git:** commit or push only when the user asks. Never commit
   `configs/local.env` or `CLAUDE.local.md`.
8. **Never edit a script while a job is running it.** Bash reads scripts
   incrementally; an earlier mid-run edit of `run_scene.sh` broke a job.

## Where to change what

| To change | Edit |
|---|---|
| Where a root folder is on this machine | `configs/local.env` (gitignored) |
| A new root folder | a key in `configs/paths.yaml` reading a new `R2S_*` variable, a field in `Paths` (`r2s/config.py`), and a line in `configs/local.env.example` |
| Slurm partition, QOS, account, GPU type | `R2S_SLURM_*` in `configs/local.env` |
| The verified Eden commit | `EDEN_VERIFIED` in `r2s/doctor.py`, plus README.md "Dependencies" and docs/setup.md §1 |
| A per-scene option (collider mode, scale truncation, frozen round, ...) | `configs/scenes.yaml`, under the step name; comment where the value came from |
| A production default | the step's `add_arguments`; then update docs/pipeline.md |
| Any per-scene input or output path | `r2s/layout.py` (`InputLayout`, `SceneLayout`); steps never build paths themselves |
| Splat / collider world placement, frame constants (`SCALE_OPT_BASE_ROT`, `SCALE_OPT_ADJUSTMENT`) | `r2s/common/transforms.py` only; a test fails if the literals appear elsewhere |
| Run order, optional steps, step registry | `r2s/steps.py` (`STEP_MODULES`, `OPTIONAL_STEPS`) |
| How a step is run, recorded, checked | `r2s/cli.py`, `r2s/manifest.py`, `r2s/check.py`, `r2s/doctor.py` |
| A step-0 model call | the step in `r2s/gen/<step>.py` plus its worker in `r2s/gen/workers/`; workers run in a conda env via `r2s.gen.conda.run_in_env` and never import `r2s` |
| The Eden scene used by settle runs | `r2s/sim/` (`scene.py:build_scene`, `robot.py`, `settle.py`) |

## The step contract (short version)

Each step module exposes this (full rules in docs/DEVELOPING.md):

```python
STEP = Step(name, help, add_arguments, run, inputs, outputs, needs, optional_inputs)
```

- `run(layout, args) -> dict` returns stats, which go into the manifest. The CLI
  checks that the inputs exist and records everything; the step must not do that
  itself.
- `needs` ∈ {`cpu`, `gpu`, `conda:<env>`, `gpu+conda:<env>`}. Audit the worker: a
  helper that defaults to `device="cuda"` makes it a GPU step.
- Read step-0 files through `layout.inputs`, which resolves the per-scene root: the
  run's `scene_data_out` once imported, else `scene_data`. Write step-0 files
  through `layout.inputs_out`. `layout.inputs_src` is `scene_data` itself, for
  replay sources and the default episode.
- Never import torch, Genesis or gsplat at module level, because `r2s list` imports
  every step module. Import them inside `run`.
- Keep the algorithm a plain function, so tests and other steps can call it without
  the CLI.

## Tests and verification

```bash
$R2S_PYTHON -m pytest -q tests      # CPU only, ~30 s; currently 243 passed, 6 skipped
```

- Tests that read the old repo or `scene_generation` skip when those are absent.
  Without `configs/local.env` (a fresh clone), `conftest.py` points every unset
  folder at `/nonexistent/<VAR>`, and 209 pass / 40 skip.
- Opt-in slow tests: `R2S_TEST_GENESIS=1` (crop vs Genesis) and `R2S_TEST_CONDA=1`
  (worker imports in the conda envs).
- To compare a step against the old code:
  1. `python tools/legacy/old_repo.py seed --scene S --through <previous step> --run <name>`
     seeds a run with the old outputs.
  2. Run the new step.
  3. Compare with the matching `tools/legacy/*_compare.py` / `*_verify.py`, or
     `bash tools/legacy/cpu_chain.sh S RUN` for steps 4-9.

  See tools/legacy/README.md. These tools need the old checkouts; nothing in `r2s/`
  imports them.
- After a GPU change, rerun the affected step on droid2 and droid820, the two fully
  verified scenes, and compare with the tables in docs/changes.md.

## Pitfalls

- **Eden version:** r2s was verified on `robotarena360` @ `b7c85cc`. A different
  commit can change the lock (torch / Genesis / gsplat), the splat placement or the
  arm asset. `r2s doctor` warns. Re-verify as described in docs/setup.md §1.
- **gsplat build:** gsplat compiles its CUDA extension on first use.
  - It needs `ninja` and an nvcc matching torch's CUDA (13.0). Set `CUDA_HOME` in
    `configs/local.env`.
  - The build is cached in `$TORCH_EXTENSIONS_DIR` for every architecture in
    `TORCH_CUDA_ARCH_LIST`.
  - A new GPU type means adding its arch and rebuilding once on that node.
- **Eden HEAD splat placement is `world = R(s·x) + pos`.** Pass
  `PCDOptions(scale=s, pos=t, quat=q)`. The old code passed `pos=t/s` for the
  removed 2DGS renderer.
- **Arm asset:** `r2s/sim/robot.py` pins Eden's old hand-less Panda file
  (`arms/franka_research3/fr3.xml`) and rewrites the gripper pads. At HEAD,
  `FrankaResearch3` is a real FR3. Do not switch without re-verifying the settle runs.
- **Image size is `2·principal point`**, not the JPEG size. DROID's `2·cy` = 370 for
  360-px frames, by design.
- **Settle verdicts depend on the Genesis version** (1.4.4 here). Two known flips are
  listed in docs/changes.md.
- **`objects.dump --support-query-rotation`:** `legacy` is the default and
  reproduces an old bug; `identity` is correct. Changing the default moves every
  object layout.
- **`--viewer` options block** until Ctrl-C (viser). Never pass them in batch jobs.
- **Optional step:** `r2s pipeline` skips `inputs.superres` unless it is named.
- **Genesis' numba cache** must stay in `NUMBA_CACHE_DIR`; site-packages is read
  only on compute nodes.

## Keeping docs in sync

- Changing a step's options, inputs or outputs → update its section in
  docs/pipeline.md.
- Adding a step → also update the step table in README.md, docs/user_guide.md (the
  Mermaid diagram) and docs/structure.md.
- Changing behaviour → add an entry to docs/changes.md.
