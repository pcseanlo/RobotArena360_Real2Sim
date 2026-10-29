# Inputs: DROID preprocessing and custom captures

The pipeline starts from an **episode folder**: one camera view of a static
tabletop scene, its metric depth, a calibration in the robot's base frame and
(optionally) the robot's joint states. Everything else, from masks to object
layouts, is produced by the pipeline's 22 steps. This page says exactly what that folder must
contain, what was done to DROID data before it reached step 0a, and what you need
to prepare for your own recording.

## 1. The episode folder (what step 0a reads)

`r2s run inputs.episode --scene <s> --src <folder>` links these files into the run;
`r2s check --scene <s>` validates them (`r2s check --scene <s> --src <folder>` before
importing). It prints one OK / WARN / FAIL line per item below and exits 1 on a FAIL.

| File | Required | Format | Used by |
|---|---|---|---|
| `images0/<n>.jpg` | yes, at least `0.jpg` and `1.jpg` | RGB frames, `n` = frame index; frame 1 is the anchor view | segmentation, inpainting, meshes, placement (1); object fits (0) |
| `depth/<n>.npy` | yes, at least `0.npy` and `1.npy` | `(H, W)` float32 **metres**, dense (no holes), same resolution and viewpoint as the RGB | object point clouds, placement, table depth fit, Method B depth term |
| `intrinsics.npy` | yes | `(3, 3)` K for that resolution | everything that projects |
| `extrinsics.npy` | yes | `(4, 4)` **world-to-camera**, OpenCV axes (x right, y down, z forward); world = robot base frame, z up, metres | everything that places geometry |
| `joints.npy` | for steps 9 (grasp term) and the sim reset pose | `(T, 7)` Franka joint angles (rad), one row per frame | settle runs, grasp anchor |
| `grippers.npy` | for step 9's grasp term | `(T,)` gripper command, > 0.5 = closed | grasp anchor |
| `tcp.npy` | for step 9's grasp term | `(T, 4, 4)` end-effector poses in the base frame (or `(T, 3)` positions) | grasp anchor |
| `lang.txt` | optional | task instruction, first line used | eval prompt, physics prompt context |

What `r2s check` tests:

| Item | OK | WARN | FAIL |
|---|---|---|---|
| frames | `images0/0.jpg`, `1.jpg`, `depth/0.npy`, `1.npy` present, one size | | missing, sizes differ |
| depth | 2-D float32 metres, the RGB's `(H, W)`, median in 0.05-20 m, ≤ 1 % holes | float64, > 1 % of pixels outside 0.05-20 m, 1-10 % holes | not float (mm?), wrong shape, median out of range, > 10 % holes |
| intrinsics | 3x3, `fx, fy > 0`, `\|2cx - W\|`, `\|2cy - H\|` ≤ 2 % of the side | principal point 2-10 % off (DROID: `2cy` = 370 for H = 360, 2.9 %) | not 3x3, `fx` or `fy` ≤ 0, > 10 % off |
| extrinsics | 4x4, orthonormal rotation (det 1), the optical axis crosses z = 0.3 .. −0.08 m inside the robot's workspace box | last row not `[0, 0, 0, 1]` | not 4x4, not a rotation, axis misses the workspace (with a hint when the inverse would pass: camera-to-world saved by mistake) |
| robot state | `joints` `(T, 7)` (or `joints_old.npy`), `grippers` `(T,)`, `tcp` `(T, 4, 4)` / `(T, 3)`, one `T` | missing (no grasp term, no recorded reset pose), lengths differ | wrong shapes |
| `lang.txt` | present | missing or empty | |

After the episode, `r2s check` lists every step with whether its inputs exist (with
its default options) and what the manifest says it last did; `--inputs-only` skips that.

Conventions the pipeline relies on:

- **Image size = 2 × principal point.** The simulator's cameras (Eden) put the
  principal point at the image centre, and every step derives the image size as
  `(round(2·cx), round(2·cy))`. Keep `cx ≈ W/2`, `cy ≈ H/2`; otherwise crop or pad
  the frames (and depth) so they are, and shift K accordingly. DROID's own K is off by
  ~5 px vertically (2·cy = 370 for a 360-px image), which the pose fit absorbs by
  resizing the target; larger offsets distort the fit.
- **The robot base is the world origin**, z up. The collision crop is a 1.1 m
  cylinder around the origin, the table search uses the Franka workspace
  (x 0.20-0.95 m, y ±0.55 m, z −0.08-0.85 m), and the support plane is expected
  near z = 0. A different robot or table layout needs `surface.plane --robot` /
  `--workspace-bounds` and `collision.crop --radius` adjusted.
- **The scene is static** between frames 0 and 1 and objects rest on one table.

## 2. What DROID data went through before step 0a

The DROID scenes in `scene_data/DROID/<s>/` were prepared outside this repo:

1. **Episode extraction** from the DROID release (exterior ZED camera, downsampled to
   640×360): RGB frames, dense metric depth per frame (float32 m, no holes),
   intrinsics, DROID's camera extrinsics (kept as `extrinsics0.npy`), joints, gripper
   state, TCP poses, `metadata_*.json` (task text → `lang.txt`).
2. **Camera refinement (required).** DROID's stored extrinsics are off by a few
   degrees and centimetres (droid820: 2.2° and 2.6 cm). Autocalib
   (`run_droid_seiredata.sh`, a differentiable robot render matched to the frames,
   using the joint states) rewrites `extrinsics.npy`. Every alignment and object
   placement downstream depends on this matrix, so do not skip it.

Nothing else is preprocessed; masks, background, meshes, depth fits and the mesh
sheet are all made by step 0.

## 3. Running a custom recording

### What you must prepare

| Item | Why | How |
|---|---|---|
| One sharp RGB view of the table (frames 0 and 1 may be the same image) | anchor for every step | any camera; avoid motion blur and strong glare |
| Dense metric depth at that view | object placement, table depth, metric scale of the world | RGB-D camera with hole filling, stereo, or a metric monocular depth model; save as float32 metres |
| Intrinsics K with a centred principal point | projection; Eden camera model | camera calibration; crop/pad to centre it |
| World-to-camera extrinsics in the robot base frame | every placement | hand-eye calibration, a calibration board placed in the base frame, or Autocalib if the robot is visible with joint states |
| Joint states (optional) | sim reset pose, Method B grasp term | from the robot log; without them use a default reset pose and drop the grasp weight (`objects.refine` weights) |
| `GEMINI_API_KEY` (or your own masks) | object detection and physics guesses | export the key; or write `masks/`, `result.json`, `bounding_boxes_and_masks/` yourself in the step-0 format and skip `inputs.segment` |
| A splat of `background.png` | the photoreal background | run your image-to-world generator on step 0's inpainted background (see 3.3) |
| Model environments and weights | step 0 | see docs/setup.md; `r2s doctor` lists what is missing |

### Running process

```bash
source scripts/env.sh
export R2S_WORK=$R2S_WORK_ROOT/runs/my_scene

r2s doctor                                               # env, paths, conda envs, weights, GPU
r2s run inputs.episode --scene my_scene --src /path/to/my_episode
r2s check --scene my_scene                               # validates the episode folder

r2s pipeline --scene my_scene --from inputs.segment --to inputs.mesh_sheet    # GPU node
#   -> <work>/scene_data/scene_background/my_scene/background.png
#   generate the splat from it, save as <splat_source_dir>/my_scene.ply
r2s pipeline --scene my_scene --from world.import --to objects.refine        # GPU node
#   inspect align/*.png (align/depth_check.json: fit against your depth sensor), surface/,
#   objects/refine/<tag>/check.png and settle.mp4
r2s run objects.freeze --scene my_scene --tag B_v1
r2s export --scene my_scene --to /path/to/eval     # or build the Eden scene with r2s.sim.scene.build_scene
```

Steps 1-10 pick up the run's own step-0 outputs automatically once
`inputs.episode` has imported the scene into the run (`r2s run` prints the inputs
root; no `R2S_SCENE_DATA` needed). A range with a GPU step stops at once on a node
without a GPU, before anything runs.

### What may be missing or need adapting

1. **Splat generator frame convention.** The base rotation of the splat into the
   sim frame (`SCALE_OPT_BASE_ROT`, `SCALE_OPT_ADJUSTMENT` in
   `r2s/common/transforms.py`) and the pose fit's start (`--init camera`: the world
   is generated around the input camera) were derived for World Labs Marble. A
   different generator needs its own two constants there — the only place they are
   defined — and possibly a different `--init`.
2. **Metric scale reference.** Step 3 fits the scale against `mesh_sheet.glb`,
   which comes from your depth. If depth is missing or not metric, the scale is
   wrong and every later step inherits it. Step 3b (`align.check`) compares the
   aligned collider with the raw sensor depth and reports the scale the sensor
   would prefer (`sensor_optimal_scale`, overall and on the table).
3. **Robot.** The sim side (settle runs, previews) and the grasp anchor use a DROID
   Franka with a Robotiq 2F-85 (`r2s/sim/robot.py`, `r2s/objects/refine/franka_fk.py`).
   Another robot needs those two modules replaced; the geometry steps (1-8b) do not
   depend on the robot except through the workspace bounds above.
4. **Table assumptions.** One dominant horizontal support surface inside the
   workspace. Multi-level surfaces, shelves or no table are outside what the plane
   and receiver fits handle.
5. **Object count and size.** Gemini + SAM3 work best on a handful of distinct,
   graspable objects; transparent or very thin objects segment and mesh poorly.
6. **Layout source for step 10.** Without a reviewed layout, freeze the Method B
   result (`--tag`) or pass `--from`.
