# Setup

## 1. Eden

r2s needs a checkout of [Eden](https://github.com/Kashu7100/Eden) on branch
`robotarena360`. Clone it anywhere. r2s only reads it.

```bash
git clone -b robotarena360 https://github.com/Kashu7100/Eden.git
```

- **Verified commit:** `b7c85cc` (2026-10-14).
  - Every result in docs/changes.md was produced with this commit plus one
    unpushed commit, `f7ab779a`.
  - `f7ab779a` changes only the Nyx renderer (`eden/envs/nyx_splat.py`,
    `NyxSplatOptions`), which r2s does not use.
  - `r2s doctor` prints the checkout's branch and commit, and warns when it does not
    contain `b7c85cc`.
- **A newer commit** usually works. Eden changes that can move r2s results:
  - Eden's `uv.lock` (torch, genesis-world, gsplat). Settle verdicts depend on the
    Genesis version: docs/changes.md lists two flips between the old Genesis and
    1.4.4.
  - Splat placement: `PCDOptions(scale, pos, quat)` places a splat at
    `world = R(s·x) + pos`. `r2s/common/transforms.py` relies on this.
  - The arm asset: `r2s/sim/robot.py` pins `arms/franka_research3/fr3.xml`, the
    hand-less Panda file the old `FrankaResearch3` loaded, and rewrites the gripper
    pads.

  After moving Eden, rerun `align.scale` and `objects.refine` on droid2 and droid820.
  Compare the results with docs/changes.md.

## 2. This machine's folders: `configs/local.env`

Every folder outside the repo is set in one file, which is gitignored:

```bash
cp configs/local.env.example configs/local.env
$EDITOR configs/local.env
```

`scripts/env.sh` loads it. A variable already set in the environment keeps its
value, so `R2S_EDEN=/other/Eden source scripts/env.sh` points one shell at another
checkout. `configs/paths.yaml` turns the variables into the pipeline's root folders
and holds no machine paths. An unset variable stops every command with a message
naming it.

| Variable | What |
|---|---|
| `R2S_WORK_ROOT` | large writable folder: runs (`runs/<name>/`), the Python env (`envs/eden`), build caches, Slurm logs |
| `R2S_EDEN` | the Eden checkout from section 1 |
| `CUDA_HOME` | CUDA 13.0 toolkit, for gsplat's one-time build |
| `R2S_DATA` | existing scene data in the scene_generation layout (`DROID/<s>/`, `scene_background/<s>/`, `assets/<s>/`) |
| `R2S_SPLAT_SOURCE_DIR` | one generated 3DGS splat per scene, `<scene>.ply` |
| `R2S_SCENE_GENERATION`, `R2S_SAM3D_OBJECTS`, `R2S_CONDA_ROOT` | step-0 model code, weights and conda envs (section 4) |
| `R2S_LAYOUT_REFINER`, `R2S_TASKS_DIR` | reviewed layouts for `objects.freeze --refiner-round`; object size priors for `objects.refine` |
| `R2S_SLURM_PARTITION`, `_QOS`, `_ACCOUNT`, `_GRES` | Slurm settings for `scripts/slurm/submit.sh` (section 6) |
| `R2S_OLD_REPO`, `R2S_OLD_ECHO2`, `R2S_OLD_PATH_ALIASES` | old checkouts, for `tools/legacy` and the tests that compare against them only |

## 3. Repo environment (steps 1-10)

r2s runs in an environment built from Eden's lock file, with Eden's `vis` extra
(gsplat, plyfile, viser).

- On top of the lock, `setup_env.sh` adds cma, ninja, pytest and open3d 0.19.0.
  Open3D is for `world.collider`'s Poisson mesher, at the version the old ply2glb
  env had.
- It pins every package the lock installed, so nothing locked changes.
- Eden and this repo are put on the path with `.pth` files, so nothing is installed
  into, or written inside, the Eden checkout.

```bash
bash scripts/setup_env.sh          # ~12 min; env goes to $R2S_WORK_ROOT/envs/eden (7 GB)
```

gsplat compiles its CUDA extension the first time it is used.

- Do it once on a GPU node; it takes about 6 minutes.
- The build is cached in `$TORCH_EXTENSIONS_DIR` for every GPU type in
  `TORCH_CUDA_ARCH_LIST`. The default `8.0;8.6;8.9;9.0;12.0` covers A100 (8.0),
  A40 / A6000 (8.6), L40S and RTX 6000 Ada (8.9), H100 (9.0) and RTX Blackwell
  (12.0).
- For another GPU type, add its compute capability and build once on that node.

```bash
source scripts/env.sh
python -c "from gsplat.cuda._backend import _C"
```

**Why not Eden's own `.venv`:** at `robotarena360` Eden renders splats through
gsplat, which that venv does not have, so `import eden.envs.splat` fails there.

**CUDA:** torch in the new env is built for CUDA 13.0, so `CUDA_HOME` must point at a
13.0 toolkit. A container whose nvcc is older (Eden's `eden.sif` has 12.4) cannot
build the extension, so run the GPU steps on the host.

`scripts/env.sh` also defines `r2s` as a shell function (`python -m r2s`).
`r2s doctor` checks all of this without loading any model:

- the imports, the CUDA build, the gsplat build and the GPU;
- the Eden commit;
- the folders, the conda envs and the model weights below.

## 4. Step-0 model environments

Step 0 calls models that each need their own conda env, under
`$R2S_CONDA_ROOT/envs`. Steps 1-10 need none.

| Env | Model |
|---|---|
| `gemini` | Gemini + SAM3 |
| `invsr` | InvSR |
| `lama` | LaMa |
| `sam3d-objects` | SAM 3D Objects |
| `match` | MINIMA + LightGlue |
| `moge` | Depth-Anything |

- r2s runs its worker scripts with those interpreters (`r2s/gen/conda.py`).
- Model code and weights stay where they are: `R2S_SCENE_GENERATION`,
  `R2S_SAM3D_OBJECTS` and the Hugging Face cache.

Gemini is only needed for a live `inputs.segment` / `inputs.physics` run. Export
`GEMINI_API_KEY`; nothing reads a key from a file. Both steps can replay saved
Gemini answers instead (see docs/pipeline.md).

Model weights the doctor looks for:

- `<scene_generation>/lama/big-lama/models/best.ckpt`
- `<scene_generation>/MINIMA/weights/minima_lightglue.pth`, plus
  `third_party/LightGlue`
- `<sam3d_objects>/checkpoints/hf/pipeline.yaml`
- the Hugging Face cache entries of `facebook/sam3`,
  `depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf` and
  `Ruicheng/moge-vitl`

The Hugging Face cache is `$HF_HUB_CACHE`, `$HF_HOME/hub`, `$TRANSFORMERS_CACHE` or
`~/.cache/huggingface/hub`.

## 5. Runs and inputs

Every run writes under `work` (`$R2S_WORK_ROOT/runs/default` unless `R2S_WORK` is
set). Set `R2S_WORK` to start a separate run:

```bash
R2S_WORK=$R2S_WORK_ROOT/runs/try1 r2s run align.pose --scene droid2
```

Step-0 files are read per scene:

- from the run's own `scene_data_out`, once `inputs.episode` has imported the scene
  there;
- otherwise from `scene_data` (`$R2S_DATA`).

There is nothing to repoint after regenerating step 0. Set `R2S_SCENE_DATA` only to
force every read to one folder.

## 6. Running on a Slurm cluster

`scripts/slurm/submit.sh` is `sbatch` plus the settings from `configs/local.env`:

- `R2S_SLURM_PARTITION`, `R2S_SLURM_QOS`, `R2S_SLURM_ACCOUNT` and `R2S_SLURM_GRES`
  (default `gpu:1`; empty means the cluster default);
- the log file `$R2S_WORK_ROOT/logs/r2s_<job name>_<job id>.log`.

Its arguments go to `sbatch` unchanged, and the job starts in the repo:

```bash
scripts/slurm/submit.sh scripts/slurm/step.sbatch run align.pose --scene droid2
R2S_WORK=$R2S_WORK_ROOT/runs/try1 scripts/slurm/submit.sh scripts/slurm/step.sbatch pipeline --scene droid2 --from align.pose --to collision.crop
scripts/slurm/submit.sh --time=8:00:00 scripts/slurm/step.sbatch run objects.refine --scene droid2
```

`r2s pipeline` refuses to start a range with a GPU step on a node without a visible
GPU ("needs a GPU node"). A CPU login shell therefore fails in seconds instead of
halfway through.

To reuse a GPU allocation you already hold (for example an idle interactive or
"hold" job), run inside it instead of queueing:

```bash
srun --jobid=<JOBID> --overlap -n1 bash -c "cd $R2S_REPO && source scripts/env.sh && r2s run align.pose --scene droid2"
```
