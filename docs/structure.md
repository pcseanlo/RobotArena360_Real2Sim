# Folder structure

## The repo

```
real2sim_recon/
├── README.md                 overview, step table, quickstart
├── CLAUDE.md                 orientation for coding agents: commands, rules, where to change what
├── configs/
│   ├── local.env.example     template for local.env: this machine's folders + Slurm settings (local.env is gitignored)
│   ├── paths.yaml            every root folder, from the local.env variables; nothing else hardcodes a path
│   └── scenes.yaml           scene registry + per-scene step options (collider preset, scale truncation, frozen round, ...)
├── r2s/                      the package; `python -m r2s` is the CLI
│   ├── cli.py                list / run / pipeline / status / check / doctor / export
│   ├── steps.py              step registry: run order, optional steps, `needs`, step name -> module (imported lazily)
│   ├── config.py             loads paths.yaml and scenes.yaml
│   ├── layout.py             every per-scene input and output path (InputLayout, SceneLayout; per-scene input root)
│   ├── manifest.py           per-scene manifest.json: args, inputs root, input fingerprints, outputs of each run
│   ├── check.py              `r2s check`: episode-folder validation + which step inputs exist
│   ├── doctor.py             `r2s doctor`: env, gsplat build, GPU, Eden commit, paths, conda envs, weights
│   ├── export.py             symlink a scene into the old evaluator's layout
│   ├── common/               shared helpers, numerically identical to the old code
│   │   ├── quat.py           wxyz quaternions
│   │   ├── transforms.py     the one place alignment results become splat / collider world poses
│   │   ├── camera.py         DROID calibration -> camera pose, image size
│   │   ├── mesh.py           GLB loading
│   │   └── io.py             prepare_output (never writes through a symlink), json / npy writers
│   ├── gen/                  step 0: real-scene inputs from a DROID episode
│   │   ├── episode.py ... mesh_sheet.py   one module per sub-step (runs in the repo env)
│   │   ├── conda.py          runs a worker inside a model conda env
│   │   ├── workers/          scripts that run inside those envs (never import r2s; `from sg_glue import ...`)
│   │   └── sg_glue/          code moved from scene_generation (Gemini/SAM3 glue, MINIMA matching, depth, sheet)
│   │                         + segment_replay.py (saved Gemini answers -> SAM3 prompts); relative imports inside
│   ├── world/                step 1: the Gaussian world
│   │   ├── import_splat.py   world.import
│   │   ├── collider.py       world.collider (runs poisson_mesher.py, the collider mesher, in the repo env)
│   │   ├── poisson_mesher.py Poisson mesher (the user's original ply2glb.py code, pinned by a test)
│   │   └── providers/marble.py   optional World Labs Marble API client (one provider among possible others)
│   ├── align/                steps 2-3b
│   │   ├── render.py         differentiable gsplat renderer with Eden HEAD's conventions
│   │   ├── pose_fit.py       align.pose
│   │   ├── scale_core.py     scale-fit numerics (verbatim port)
│   │   ├── scale_fit.py      align.scale
│   │   └── depth_check.py    align.check: the aligned collider vs the real sensor depth
│   ├── surface/              steps 4-6 (+ runtime.py: table boxes / polygon prism for the sim)
│   ├── collision/            step 7 (crop.py) + the load-time support query and collision profiles
│   ├── objects/              steps 8-10
│   │   ├── dump.py           objects.dump (the evaluator's object loading, CPU)
│   │   ├── pose_fit.py       objects.posefit
│   │   ├── refine/           objects.refine (Method B: fit, metrics, harness, settle orchestration)
│   │   ├── freeze.py         objects.freeze
│   │   └── overrides.py, settle.py, placement_types.py   shared by the sim and the evaluator
│   └── sim/                  Eden HEAD scene assembly for settle runs and previews
│       ├── scene.py          build_scene(): collider, table prism, objects, robot, cameras, splat
│       ├── robot.py          DROID Franka (old hand-less arm file) + Robotiq 2F-85 with pad boxes
│       ├── options.py        small scene/splat option classes
│       └── settle.py         run_settle() / run_preview()
├── scripts/
│   ├── setup_env.sh          build the repo env from Eden's lock file
│   ├── env.sh                source before any run (python, CUDA for gsplat, caches)
│   ├── run_scene.sh          compatibility wrapper: `r2s pipeline --scene S --from A --to B [--skip-done]`
│   └── slurm/                submit.sh (sbatch + local.env Slurm settings), step.sbatch (any step / pipeline / command on a GPU node), sim_settle.sbatch
├── tests/                    pytest; CPU-only, ~30 s (legacy_paths.py: old-repo paths, skip when absent)
├── tools/
│   └── legacy/               old-repo regression only (needs the old checkouts; README.md):
│                             old_repo.py (seed a run / list old outputs, OLD / ECHO2), comparison scripts, cpu_chain.sh
└── docs/
    ├── user_guide.md         method, install, quickstart, concepts, using the result in Eden, troubleshooting
    ├── pipeline.md           step reference: every step's inputs, outputs, options, old equivalent; file formats
    ├── structure.md          this file
    ├── setup.md              environment, conda envs, paths, Slurm
    ├── custom_capture.md     the episode folder (what `r2s check` validates), DROID preprocessing, own recordings
    ├── changes.md            behaviour changes, bugs found, verification results
    ├── DEVELOPING.md         rules for adding or porting a step
    └── porting/              old file -> new module map, old evaluator scene-assembly notes
```

## A run's output folder

Everything a run writes lives under `work` from `configs/paths.yaml`
(`$R2S_WORK_ROOT/runs/default`; `R2S_WORK=...` starts a separate run). Step 0 writes to `scene_data_out`
(default `<work>/scene_data/`) in the scene_generation layout, so other tools that
read that layout keep working. Once `scene_data/DROID/<s>/` exists in the run, steps
1-10 read that scene's step-0 files from there; otherwise from `paths.scene_data`
(`R2S_SCENE_DATA` pins the latter).

```
<work>/
├── scene_data/                       step 0 (only when you regenerate inputs)
│   ├── DROID/<s>/                    episode files, masks/, pcds/, bounding_boxes_and_masks/,
│   │                                 inpainted_depth.npy, mesh_sheet.glb
│   ├── scene_background/<s>/background.png
│   ├── input_mask/<s>/               LaMa inputs
│   └── assets/<s>/                   object meshes (image_k_segmented.glb, ...)
├── layout_manifest.tsv               one row per frozen layout (step 10)
└── scenes/<s>/
    ├── manifest.json                 what each step ran with, read and wrote
    ├── world/                        gaussian.ply, collider.glb, gaussian_trimmed.ply
    ├── align/                        aligned_pose*.npy (+ refined), review renders, log/,
    │                                 depth_check.json + .png (step 3b)
    ├── surface/                      operation_surface_plane_fit.json, operation_surface_shadow_v2.json
    ├── cropped_collision/            reachable_collider_*.glb (+ .json sidecar)
    ├── gen/                          step-0 scratch (LaMa intermediates, worker logs, replay copies)
    └── objects/
        ├── stock/object_poses.json   objects as the sim loads them (step 8a)
        ├── posefit/                  refined_poses.json, overlays/ (step 8b)
        ├── refine/stock/             stock settle (step 9)
        ├── refine/<tag>/             candidate.json, fit.log, review/, settle.json, settle.mp4, preview/, scores.json
        └── layout/                   candidate.json + freeze.json (step 10)
```

File names inside `scenes/<s>/` keep the old repo's names
(`aligned_pose_refined.npy`, `operation_surface_shadow_v2.json`, ...), so
`r2s export` only has to rearrange folders for the old evaluator.

## What stays outside the repo

Every location is a variable in `configs/local.env`.

| Variable | What | Used by |
|---|---|---|
| `R2S_EDEN` | [Eden](https://github.com/Kashu7100/Eden) checkout, branch `robotarena360`, read only | sim, renderer conventions |
| `R2S_WORK_ROOT` | `runs/` (one folder per run), `envs/eden` (the repo env: Eden's lock + gsplat), `cache/`, `logs/` | every step |
| `R2S_CONDA_ROOT` | `envs/{gemini,invsr,lama,sam3d-objects,match,moge}`, the model envs | step 0 |
| `R2S_SCENE_GENERATION`, `R2S_SAM3D_OBJECTS`, HF cache | model code and weights (`lama`, `InvSR`, `MINIMA`, `third_party`; SAM 3D Objects), read only | step 0 |
| `R2S_DATA` | existing step-0 outputs (`scene_data`), read only | steps 1-10 of scenes not imported into the run; default `inputs.episode` source |
| `R2S_SPLAT_SOURCE_DIR` | generated splats, `<s>.ply` | step 1a |
| `R2S_LAYOUT_REFINER` | `runs/<s>/<round>/`, agent-reviewed layouts | step 10 |
| `R2S_TASKS_DIR` | task notes (dims prior) | step 9 |
| `R2S_OLD_REPO`, `R2S_OLD_ECHO2` | the old repo and its assets, read only | `tools/legacy/` and the regression tests only |
