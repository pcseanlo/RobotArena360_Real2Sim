# real2sim_recon

Turn one real DROID episode into a sim-ready Gaussian scene for
[Eden](https://github.com/Kashu7100/Eden) (branch `robotarena360`): splat
background, collider, support table and object layout, all aligned to the real
camera.

This repo is the cleaned-up scene-reconstruction half of
`scene_gaussian_alignment` plus the asset / inpainting pipeline of
`scene_generation`. Policy evaluation and rendering fidelity (Nyx, shadows) are
out of scope.

## Pipeline

| # | Step | Runs on | What it does | Main output |
|---|---|---|---|---|
| 0a | `inputs.episode` | CPU | Link the raw DROID episode (frames, depth, calibration, joints) | `DROID/<s>/` |
| 0c-1 | `inputs.segment` | GPU, conda `gemini` | Gemini boxes + SAM3 masks (replayable without Gemini) | `masks/`, `result.json` |
| 0c-2 | `inputs.physics` | CPU, conda `gemini` | Gemini guess of size, mass, friction (API calls only) | `physical_properties.json` |
| 0c-3 | `inputs.superres` | GPU, conda `invsr` | Optional InvSR (off in production) | `images0/super.png` |
| 0c-4 | `inputs.inpaint` | CPU, conda `lama` | LaMa removes the objects | `scene_background/<s>/background.png` |
| 0c-5 | `inputs.meshes` | GPU, conda `sam3d-objects` | SAM 3D object meshes | `assets/<s>/image_k_segmented.glb` |
| 0c-6 | `inputs.place` | GPU, conda `match` | MINIMA matching: object scale/pose | `masks/transformations.json` |
| 0c-7 | `inputs.table_depth` | GPU, conda `moge` | Metric depth of the inpainted background | `inpainted_depth.npy` |
| 0c-8 | `inputs.mesh_sheet` | GPU, conda `match` | Metric mesh of the scene | `mesh_sheet.glb` |
| 1a | `world.import` | CPU | Bring in the generated splat PLY (Marble today) | `world/gaussian.ply` |
| 1b | `world.collider` | CPU | Poisson collider from the splat (Open3D) | `world/collider.glb` |
| 2 | `align.pose` | GPU | Fit splat pose to the inpainted view (gsplat, CMA-ES + Adam) | `align/aligned_pose.npy` |
| 3 | `align.scale` | GPU | Fit scene scale against `mesh_sheet.glb` | `align/aligned_pose*_refined.npy` |
| 3b | `align.check` | CPU | Check the alignment against the real sensor depth | `align/depth_check.json` |
| 4 | `surface.plane` | CPU | Table plane from the collider | `surface/operation_surface_plane_fit.json` |
| 5 | `surface.trim` | CPU | Drop floaters near the table | `world/gaussian_trimmed.ply` |
| 6 | `surface.receiver` | CPU | Table polygon for physics and shadows | `surface/operation_surface_shadow_v2.json` |
| 7 | `collision.crop` | CPU | Crop the collider to the robot's reach | `cropped_collision/*.glb` |
| 8a | `objects.dump` | CPU | Objects exactly as the sim loads them | `objects/stock/object_poses.json` |
| 8b | `objects.posefit` | CPU | Fit object XY/yaw to the real masks | `objects/posefit/refined_poses.json` |
| 9 | `objects.refine` | GPU | Method B: scale/pose fit + settle check | `objects/refine/<tag>/candidate.json` |
| 10 | `objects.freeze` | CPU | Pin the chosen layout (with hash manifest) | `objects/layout/candidate.json` |

`inputs.superres` is optional and off in production; `r2s pipeline` skips it
unless it is named.

## Dependencies

**Eden is required.** r2s builds its scenes for Eden and runs inside Eden's Python
environment.

| | |
|---|---|
| Repository | [Kashu7100/Eden](https://github.com/Kashu7100/Eden), branch `robotarena360` |
| Verified commit | `b7c85cc` (2026-10-14). `r2s doctor` warns when the checkout does not contain it. |
| How it is installed | `scripts/setup_env.sh` builds the env from Eden's `uv.lock` with the `vis` extra, then puts the Eden checkout and this repo on `sys.path` (`.pth` files). Nothing is installed into the checkout or written inside it. |
| Versions pinned by that lock | Python 3.12, torch 2.11 (CUDA 13.0), genesis-world 1.4.4, gsplat 1.5.3; `setup_env.sh` adds cma, ninja, pytest and open3d 0.19.0 without changing any locked package |
| Steps that import Eden | `align.pose` (Eden's splat PLY loader), `align.scale` and `objects.refine` (they build Eden splat scenes) |
| What r2s relies on | `SplatEnvBase`, and `PCDOptions` placing a splat at `world = R(s·x) + pos`; the DROID robot options (Franka + Robotiq 2F-85) and their MJCF assets; Eden's scene, entity, camera, material and surface options |

The other steps don't import Eden, but they follow its conventions: the collider
crop reproduces Genesis' load transform, and `r2s export` writes the evaluator's
layout. See [docs/setup.md](docs/setup.md) for a different Eden commit and for the
arm-asset pin.

The other external pieces are set per machine in `configs/local.env`:

| Dependency | Used by | Setting |
|---|---|---|
| [scene_generation](https://github.com/offjangir/scene_generation) (`feature/sam3d_generation`): LaMa + big-lama weights, InvSR, MINIMA weights, LightGlue | step 0 | `R2S_SCENE_GENERATION` |
| [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects) with `checkpoints/hf` | `inputs.meshes` | `R2S_SAM3D_OBJECTS` |
| Hugging Face models `facebook/sam3`, `depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf`, `Ruicheng/moge-vitl` | `inputs.segment`, `inputs.table_depth`, `inputs.meshes` | HF cache |
| conda envs `gemini`, `invsr`, `lama`, `sam3d-objects`, `match`, `moge` | step 0 | `R2S_CONDA_ROOT` |
| one generated 3DGS splat per scene (World Labs Marble today) | `world.import` | `R2S_SPLAT_SOURCE_DIR` |
| scene data in the scene_generation layout | step-0 inputs | `R2S_DATA` |
| reviewed object layouts from the layout refiner | `objects.freeze --refiner-round` | `R2S_LAYOUT_REFINER` |
| task notes with object size priors | `objects.refine` | `R2S_TASKS_DIR` |
| Gemini API key | live `inputs.segment` / `inputs.physics` only | `GEMINI_API_KEY` (environment only) |

## Documentation

| Doc | Covers |
|---|---|
| [docs/user_guide.md](docs/user_guide.md) | method, install, quickstart, concepts, using the result in Eden, troubleshooting |
| [docs/pipeline.md](docs/pipeline.md) | step reference: each step's required inputs, outputs, options, old-repo equivalent; file formats |
| [docs/custom_capture.md](docs/custom_capture.md) | the episode folder, DROID preprocessing, running your own recording |
| [docs/structure.md](docs/structure.md) | folder layout of the repo and of a run |
| [docs/setup.md](docs/setup.md) | environment, conda envs, paths, Slurm |
| [docs/changes.md](docs/changes.md) | behaviour changes, old-code bugs, verification results |
| [docs/DEVELOPING.md](docs/DEVELOPING.md) | rules for adding or porting a step |
| [CLAUDE.md](CLAUDE.md) | orientation for coding agents: commands, hard rules, where to change what |

## Quickstart

```bash
cp configs/local.env.example configs/local.env   # once: this machine's folders (gitignored), then edit it
source scripts/env.sh                         # loads local.env; repo env + gsplat build settings
r2s doctor                                    # env, paths, conda envs, weights, GPU: OK / WARN / FAIL
r2s list                                      # steps, run order, where they run
r2s check --scene droid2                      # validate the episode folder, see which step inputs exist
r2s run surface.plane --scene droid2          # one step, production defaults
r2s pipeline --scene droid2 --from align.pose --to objects.dump [--skip-done]   # a range, one process per step
r2s status --scene droid2                     # what ran, with which args
scripts/slurm/submit.sh scripts/slurm/step.sbatch pipeline --scene droid2 --from align.pose --to objects.refine   # GPU steps
r2s export --scene droid2 --to DIR                                # old evaluator layout
```

This machine's folders and Slurm settings live in `configs/local.env`, which is
gitignored. Its template is [configs/local.env.example](configs/local.env.example),
and `scripts/env.sh` loads it. [configs/paths.yaml](configs/paths.yaml) maps those
variables to the pipeline's root folders and holds no machine paths. Environment
variables win over both, and `R2S_WORK=/some/run` gives a fresh run. Steps 1-10 read a scene's
step-0 files from the run's own `scene_data_out` once `inputs.episode` imported it
there, else from the existing `scene_data` (`R2S_SCENE_DATA` pins that). Per-scene
settings live in [configs/scenes.yaml](configs/scenes.yaml). Every run records its
exact arguments, input root, input fingerprints and outputs in
`<work>/scenes/<s>/manifest.json`.

## Setup

[docs/setup.md](docs/setup.md): get Eden, fill in `configs/local.env`, build the
environment (`scripts/setup_env.sh`), compile gsplat once, conda envs for step 0,
Slurm. `r2s doctor` says what is missing.
Own recordings: [docs/custom_capture.md](docs/custom_capture.md).

## Differences from the old repos

[docs/changes.md](docs/changes.md) lists every behaviour change, the bugs found
during the port, and the verification against the old outputs. The comparison
scripts are in [tools/legacy/](tools/legacy/README.md); they need the old checkouts
and are not needed to use the pipeline.
