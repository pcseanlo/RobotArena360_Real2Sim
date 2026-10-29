# Step reference

One section per step, covering:

- what it does
- the options that matter
- the files it needs (it refuses to start if one is missing) and the files it writes
- how long it takes
- the old-repo command it replaces

The defaults are the values the old production launchers used, so a plain
`r2s run <step> --scene <s>` reproduces the reference pipeline.

## Running steps

```bash
source scripts/env.sh
r2s run <step> --scene <s> [options]            # one step (CPU / conda steps on any node)
scripts/slurm/submit.sh scripts/slurm/step.sbatch run <step> --scene <s> [options]   # the same on a GPU node
r2s run <step> --scene <s> --help               # every option of a step
```

Per-scene option values come from `configs/scenes.yaml`; a flag on the command line
wins. Each run is recorded in `<work>/scenes/<s>/manifest.json`: arguments,
`inputs_root`, input fingerprints, outputs, git revision, host, Slurm job and duration.
`r2s status --scene <s>` summarises the manifest.

**Where a step runs.** `r2s list` shows this for every step (`Step.needs`):

| `needs` | Meaning |
|---|---|
| `cpu` | repo env |
| `gpu` | repo env, needs a visible CUDA device |
| `conda:<env>` | CPU, with the model in its own conda env |
| `gpu+conda:<env>` | GPU, with the model in its own conda env |

**A range of steps** runs with `r2s pipeline`. `scripts/run_scene.sh SCENE [FIRST]
[LAST] [--skip-done]` is a thin wrapper around it.

```bash
r2s pipeline --scene <s> [--from STEP] [--to STEP] [--skip-done] [--include-optional] [--dry-run]
scripts/slurm/submit.sh scripts/slurm/step.sbatch pipeline --scene <s> --from world.import --to objects.refine
```

How `r2s pipeline` behaves:

- **Order and isolation.** Steps run in run order, each in its own process
  (`$R2S_PYTHON -m r2s run <step> --scene <s>`, with production defaults plus
  scenes.yaml). Genesis and torch state never leak between steps.
- **Failures.** The pipeline stops at the first failure and prints the command that
  resumes from it.
- **Optional steps.** `inputs.superres` is skipped unless it is named as `--from` /
  `--to` or `--include-optional` is set.
- **`--skip-done`.** Skips a step whose manifest status is `ok` and whose recorded
  outputs all still exist. Seeded files without a manifest record do not count as done.
- **Fail-fast checks.** Before anything runs, a range with a GPU step fails at once
  ("needs a GPU node") if `torch.cuda.is_available()` is false; this is one probe in
  a child process. A step whose conda env is missing also fails at once.

**Checks.** Both commands print OK / WARN / FAIL lines and exit 1 on a FAIL.

- `r2s doctor` reports the environment: imports, gsplat build, GPU, the Eden
  commit, paths, conda envs, model weights and `GEMINI_API_KEY`.
- `r2s check --scene <s>` validates the episode folder
  ([custom_capture.md](custom_capture.md) §1). It then lists which step inputs
  exist and what already ran.

**Path roots.** All three come from `configs/paths.yaml`, which reads this machine's
folders from `configs/local.env`:

| Short | Path | Notes |
|---|---|---|
| `IN` | `<root>/DROID/<s>/`, plus `scene_background/<s>/`, `assets/<s>/` and `input_mask/<s>/` next to it under `<root>` | step-0 data, read by steps 1-10 |
| `OUT` | `<scene_data_out>/DROID/<s>/` and its sibling folders | where step 0 writes (same layout) |
| `S` | `<work>/scenes/<s>/` | every output of steps 1-10 |

`<root>` is chosen per scene (`SceneLayout.inputs`):

- **After `inputs.episode`** has imported the scene into the run (so
  `<scene_data_out>/DROID/<s>/` exists), `<root>` is the run's own `scene_data_out`.
- **Otherwise** it is `scene_data` (`$R2S_DATA`), the existing scene_generation data.

Setting `R2S_SCENE_DATA=...` explicitly pins every read to that folder. `r2s run`
prints the root it used, and the manifest records it (`inputs_root`).

---

## Step 0 — inputs from a DROID episode

These steps build everything the later steps need from one recorded episode. They
read and write `OUT` only, so a fresh run is self-contained. Each model runs in its
own conda env through a worker script in `r2s/gen/workers/`. Model code and weights
stay in `scene_generation`, `sam-3d-objects` and the Hugging Face cache. The anchor
frame is frame 1 (`images0/1.jpg`).

### 0a `inputs.episode` — CPU

Links the raw episode into `OUT`, or copies it with `--copy`. Options: `--src DIR`
(default `<scene_data>/DROID/<s>/`), `--copy`, `--extra GLOB`.

|  | File | Content |
|---|---|---|
| In | `--src/images0/<n>.jpg`, `depth/<n>.npy` | RGB frames, metric depth (m) |
| In | `--src/intrinsics.npy`, `extrinsics.npy` | 3×3 K; 4×4 world-to-camera (OpenCV), Autocalib-refined |
| In | `--src/joints.npy`, `grippers.npy`, `tcp.npy` | (T, 7) joints, gripper state, end-effector poses |
| In (optional) | `lang.txt`, `joints_old.npy`, `metadata*.json` | task text, original joints, episode metadata |
| Out | `OUT/...` | the same files |

- Later outputs (`masks/`, `pcds/`, ...) are never imported. From this step on,
  steps 1-10 read this scene from `OUT`'s root.
- `r2s check --scene <s>` validates the imported folder. Use `--src DIR` to check a
  folder before importing it.
- Camera refinement (Autocalib, `run_droid_seiredata.sh`) and the DROID extraction
  happen outside this repo.

### 0c-1 `inputs.segment` — GPU, env `gemini`

Gemini lists and boxes the objects (gembox prompts); the robot and any human are
boxed too, but only for inpainting. SAM3 then segments every box.

Options:

- `--replay-gemini [JSON]` reruns SAM3 and the persistence step from saved Gemini
  answers, with no API call. The default answers are
  `<scene_data>/DROID/<s>/masks/gemini_boxes_raw.json` and `result.json`.
- `--replay-result`, `--model-name` (default `models/gemini-2.5-flash`),
  `--sam3-model`, `--device`, `--hf-online`.

A live run needs `GEMINI_API_KEY` in the environment.

|  | File | Content |
|---|---|---|
| In | `images0/1.jpg` (or `super.png`), `depth/1.npy`, calibration, `lang.txt` | anchor frame and depth |
| In | `GEMINI_API_KEY`, or a replay JSON | live run, or saved Gemini answers |
| Out | `masks/image_k_mask.png`, `masks/cropped/image_k_segmented.png` | per-object masked image and crop |
| Out | `masks/result.json` | `[{object_name, count, mass, friction, surface}]` |
| Out | `masks/gemini_boxes_raw.json`, `gemini_answers.json`, `gemini_boxes_debug.png` | raw Gemini answers (replayable) |
| Out | `bounding_boxes_and_masks/frame_00001_bounding_boxes_and_masks.json` | boolean mask per object |
| Out | `pcds/{k:04d}_frame1.ply`, `<scene_data_out>/input_mask/<s>/` | object point clouds; LaMa inputs |

Old: `segment_object_gembox_updated.py` via `pipeline_droid_test.sh ... gembox`.

### 0c-2 `inputs.physics` — CPU, env `gemini`

Gemini estimates each object's size, mass and friction from its crop and mask. The
step only makes API calls.

Options:

- `--replay [JSON]` validates and copies a saved file. The default is
  `<scene_data>/DROID/<s>/masks/physical_properties.json`.
- `--model-name`, `--delay`.

|  | File | Content |
|---|---|---|
| In | `masks/result.json`, `masks/cropped/`, `input_mask/<s>/` | object list and crops |
| Out | `masks/physical_properties.json` | `[{object_name, count, width, height, length, mass, friction, surface}]` |

Old: `prompt_physics_updated.py`.

### 0c-3 `inputs.superres` — GPU, env `invsr` (optional)

InvSR super-resolution of either the anchor frame (`--stage anchor` →
`images0/super.png`) or the object crops (`--stage masks`). The reference chain
never ran it, so `r2s pipeline` skips it unless it is named.

### 0c-4 `inputs.inpaint` — CPU, env `lama`

LaMa removes the objects, applying the masks one after another. Options: `--anchor`,
`--device` (default cpu), `--threads`, `--checkpoint`.

|  | File | Content |
|---|---|---|
| In | `input_mask/<s>/image_0000.png` + `*_mask.png` | anchor frame + binary masks |
| Out | `scene_background/<s>/background.png` | the empty-table image (pose-fit target, splat-generator input) |
| Out | `S/gen/inpaint/steps/` | intermediates |

Old: `lama/bin/predict.py` with hydra.

### 0c-5 `inputs.meshes` — GPU, env `sam3d-objects`

SAM 3D Objects reconstructs one textured mesh per object (seed 42). Options:
`--seed`, `--pipeline-config`, `--resize-h/-w`.

|  | File | Content |
|---|---|---|
| In | `images0/1.jpg`, the object masks listed in `result.json` (`image_k_mask.png` only) |  |
| Out | `assets/<s>/image_k_segmented{.glb,.ply,_scaled.glb,_mask.png}` | textured mesh, vertex-colour mesh, normalized mesh |
| Out | `assets/<s>/asset_info.json` | mesh paths (relative) |

Old: `src/pipeline/sam3d.py`.

### 0c-6 `inputs.place` — GPU, env `match`

Matches rendered views of each mesh to the frame (MINIMA + LightGlue) and solves the
object's scale and pose.

Options:

- `--seed 0`: the old runs were unseeded.
- `--background-object last|none`: `last` keeps the old quirk that the
  highest-index object is the background.
- `--background-depth raw|inpainted`: per scene in scenes.yaml.
- `--dataset`, `--checkpoint-dir`.

|  | File | Content |
|---|---|---|
| In | asset GLBs, `pcds/`, `masks/`, `images0/1.jpg`, `depth/1.npy` (or `inpainted_depth.npy`), calibration |  |
| Out | `masks/transformations.json` | per object: `scale`, `translation`, `rotation` (3×3), `original_rotation`, `quat_rotation` |
| Out | `assets/<s>/<k>/`, `assets/<s>/table.glb` | match debug folders; posed background-object mesh |

Old: the original `MatchFindRot.py`.

### 0c-7 `inputs.table_depth` — GPU, env `moge`

Depth-Anything V2 metric depth of the inpainted background, fitted to the real
depth. Option: `--device`.

|  | File | Content |
|---|---|---|
| In | `background.png`, `images0/1.jpg`, `depth/1.npy`, `input_mask/<s>/`, calibration |  |
| Out | `inpainted_depth.npy`, `inpainted_depth_fit.json` | (H, W) metric depth of the empty table; fit scale/shift |
| Out | `whole_mask.png`, `masks/image_N_mask.png`, `pcds/N_frame1.ply` | background mask and point cloud (N = number of objects; reruns overwrite) |

Old: `align_inpaint_updated.py`.

### 0c-8 `inputs.mesh_sheet` — GPU, env `match`

Wraps the metric depth into a triangle mesh, which is the scale reference for step 3.
It needs CUDA because SG's disparity edge filter and sheet warping default to
`device="cuda"`. Options: `--target-tris 200000`, `--beta`, `--alpha-threshold`,
`--depth-eps`, `--dataset`.

|  | File | Content |
|---|---|---|
| In | `inpainted_depth.npy`, `masks/image_N_mask.png`, `intrinsics.npy` |  |
| Out | `mesh_sheet.glb` | metric scene mesh in the camera frame |

Old: `sheet_wraping_updated.py --depth_source aligned`.

---

## Step 1 — Gaussian world and collider

The splat itself comes from an image-to-world generator run on
`scene_background/<s>/background.png`, outside this repo. Store the result as
`<splat_source_dir>/<s>.ply`.

The reference scenes use World Labs Marble: 2.2M gaussians, SH degree 1, downloaded
as PLY from the web UI. The optional client
`python -m r2s.world.providers.marble generate|status|download --scene <s>` does the
same through the API. It reads `WLT_API_KEY` and downloads `.spz`, which must be
converted to PLY.

### 1a `world.import` — CPU

Links the splat into the run (or copies it with `--copy`) and checks that it is a
3DGS PLY. Options: `--src PLY`, `--copy`.

|  | File | Content |
|---|---|---|
| In | `<splat_source_dir>/<s>.ply` | 3DGS PLY: `x y z`, `opacity`, `scale_0..2`, `rot_0..3`, `f_dc_*`, `f_rest_*` |
| Out | `S/world/gaussian.ply` | link or copy; gaussian count and SH degree go into the manifest |

### 1b `world.collider` — CPU

Poisson surface reconstruction of the splat, decimated to a collision mesh. The
mesher is `r2s/world/poisson_mesher.py`, the user's original `ply2glb.py` code,
unchanged. It runs in the repo env with Open3D 0.19.0; no external tool or conda env
is involved. Option: `--mode surface|planar`. scenes.yaml records which scenes
use which mode.

| Mode | Faces | Settings |
|---|---|---|
| `surface` (default) | about 200k | normals from the gaussian axes, Poisson depth 10 |
| `planar` | about 150k | opacity threshold 0.15, Poisson depth 9, stronger smoothing |

|  | File | Content |
|---|---|---|
| In | `S/world/gaussian.ply` |  |
| Out | `S/world/collider.glb` | y-up triangle mesh in the splat's own frame |

Takes about 4 min. Old: `ply2glb/run.sh <ply> <glb> <mode>`.

---

## Steps 2–3b — alignment

### 2 `align.pose` — GPU, 1–3 min

Renders the splat from the real camera with gsplat (Eden HEAD's conventions) and fits
x, y, z and yaw.

- **Loss:** `10·RGB-L2 + (1 − DINOv2 patch cosine)` against the inpainted
  background. The CMA-ES stage weights RGB at 0.1.
- **Search:** CMA-ES first (seed 42, up to 1000 evaluations), then 1500 Adam steps.
- **Start:** `--init camera`, so the translation starts at the camera position and
  the yaw at the camera heading + 90°. Every old result lies within ~0.2 m / ~5° of
  this, so no hand-tuned reruns are needed.

Options:

- `--init camera|cam_pos|bg_centroid`
- `--optimizer cma_then_adam|cma|adam`, `--max-iters`, `--cma-sigma0`, `--cma-max-evals`
- `--lr-trans 0.025`, `--lr-rot 0.01`
- `--rgb-weight-*` / `--dino-weight-*`
- `--rotate-xy` (also fits roll and pitch)
- `--bg 0.5`, `--seed`
- `--log` / `--log-every` (writes progress GIFs)

|  | File | Content |
|---|---|---|
| In | `S/world/gaussian.ply` | splat |
| In | `scene_background/<s>/background.png` | target image (resized to the camera's image size, `2·c`) |
| In | `IN/intrinsics.npy`, `IN/extrinsics.npy` | camera |
| Out | `S/align/aligned_pose.npy` | `[tx, ty, tz, yaw]` (or 6 values with `--rotate-xy`), scale-1 frame |
| Out | `S/align/aligned_pose_scene_scale.npy` | `1.0` (refined by step 3) |
| Out | `S/align/aligned_pose_relative_transform.npy` | `[tx, ty, tz, qw, qx, qy, qz]` |
| Out | `S/align/pose_fit_render.png`, `pose_fit_overlay.png`, `log/` | review images |

Old: the pose-fit script, run by hand.

### 3 `align.scale` — GPU, 10–15 min on the first run

Fits one scene scale.

- **Collider:** loaded as Eden builds it (CoACD convex pieces).
- **Metric:** a trimmed Chamfer distance to the metric scene mesh, inside the camera
  frustum and with occlusion culling.
- **Translation** follows `t(s) = s·t + (1 − s)·camera_position`.
- **Timing:** the first run per scene CoACD-decomposes the full collider; after
  that the result is cached and a run takes ~70 s.

Options:

- `--truncate-w 0.1 --truncate-h 0.3`: the fraction of the image cropped from each
  side. Per-scene values are in scenes.yaml.
- `--truncate-left/-right/-top/-bottom` for one side at a time.
- `--trim-quantile 0.8`, `--n-steps 400`, `--lr 5e-3`, `--s-min-ratio 0.3`, `--s-max-ratio 2.0`.
- `--metric-mesh PATH` (default `IN/mesh_sheet.glb`), `--viz-mask`, `--viewer`, `--cpu`.

|  | File | Content |
|---|---|---|
| In | `S/align/aligned_pose.npy`, `aligned_pose_scene_scale.npy` | step-2 pose |
| In | `S/world/collider.glb`, `S/world/gaussian.ply` | collider (fit), splat (snapshots) |
| In | `IN/mesh_sheet.glb` | metric reference. If missing, the largest-index segmented GLB is used and a warning goes into the manifest |
| In | `IN/intrinsics.npy`, `IN/extrinsics.npy` | camera |
| Out | `S/align/aligned_pose_refined.npy` | `[tx, ty, tz, yaw]` at the fitted scale: **the alignment every later step uses** |
| Out | `S/align/aligned_pose_scene_scale_refined.npy` | scalar scene scale |
| Out | `S/align/scale_init_snapshot.png`, `scale_refined_snapshot.png` | Eden camera renders before and after |

Old: `optimize.sh` → the scale-fit script, without its `while True` viewer loop.

### 3b `align.check` — CPU, seconds

Checks the alignment against the real sensor depth, independently of
`mesh_sheet.glb`. Nothing downstream reads the result, so this is a quality score
for the scene.

How it works:

1. `IN/depth/1.npy` is unprojected with the calibration (OpenCV world-to-camera).
2. Two kinds of pixel are left out: those on or next to objects (the union of
   `input_mask/<s>/*_mask.png`, dilated 5 px), and those whose camera ray misses
   the posed collider.
3. The collider is placed the same way surface.plane and the sim place it
   (`collider_world_transform`, GLB import rotation) and ray-cast from the camera
   (Embree).

Options: `--mask-dilate 5`, `--trim-q 0.8`, `--table-band 0.02`,
`--workspace-bounds X0 X1 Y0 Y1 Z0 Z1` (default: the scene robot's, as in
surface.plane), `--error-range 0.25`.

|  | File | Content |
|---|---|---|
| In | `IN/depth/1.npy`, `IN/images0/1.jpg`, calibration, `input_mask/<s>/` | sensor depth, object masks |
| In | `S/world/collider.glb`, `S/align/*_refined.npy` | the aligned collider |
| Out | `S/align/depth_check.json` | metrics below, for `all`, `workspace` and `table` pixels |
| Out | `S/align/depth_check.png` | frame with excluded / table pixels \| signed depth error (±25 cm) |

Metrics, each reported for three pixel sets:

- `all`: every kept pixel.
- `workspace`: pixels whose sensor point is inside the robot workspace box.
- `table`: pixels within ±2 cm of a plane fitted to the sensor's dominant horizontal
  surface in the workspace.

| Metric | Meaning |
|---|---|
| `trimmed_chamfer` | q = 0.8; sensor→collider point-to-mesh plus collider hit→nearest sensor point |
| `sensor_to_collider_median` | median sensor→collider distance |
| `depth_error_median` | collider − sensor depth; > 0 means the collider is behind the real surface |
| `sensor_optimal_scale` | s · median(z_sensor / z_collider); align.scale scales about the camera, so this is the scale at which the depths match |
| `table_height` | the sensor's table plane height vs the collider's (same estimator on the collider hits), and the scale that would match them |

This is the method of the `mesh_sheet_fill` depth-vs-collider analysis
([changes.md](changes.md)). On droid41's old alignment it reproduces that analysis
to < 1e-6: Chamfer 0.0629 m, table depth error +8.6 cm, sensor-optimal scale
1.069 (full frame) / 0.972 (table).

---

## Steps 4–6 — support surface

All three run on CPU in seconds. Run them in order: after a new plane fit, rerun
the trim and the receiver.

### 4 `surface.plane` — CPU

Finds the table in the aligned collider:

1. Sample points over the collider's area.
2. Compute local PCA normals and planarity.
3. Bin into height slabs and grow regions.
4. Fit a robust (Tukey) plane, snapped level.
5. Take a min-area rectangle as the footprint.

The camera picks which slab is the table.

Options:

- `--num-samples 400000`, `--plane-inlier-tol 0.02`, `--max-tilt-deg 15`
- `--robot franka|widowx`, `--workspace-bounds`, `--camera-frustum-crop`, `--camera-occlusion-cell-px 8`
- `--footprint-mode minrect|extent|inner-rectangle`
- `--viz-ply`, `--viewer` (opens viser and blocks)

|  | File | Content |
|---|---|---|
| In | `S/world/collider.glb` | collider |
| In | `S/align/aligned_pose_refined.npy`, `aligned_pose_scene_scale_refined.npy` | alignment |
| In | `IN/intrinsics.npy`, `IN/extrinsics.npy` | camera (slab selection) |
| Out | `S/surface/operation_surface_plane_fit.json` | `selected` table box (`center`, `quat_wxyz`, `size`, `plane_normal`, `plane_offset`, `support_boxes`), plus `candidates` and `stats` |
| Out (`--viz-ply`) | `S/surface/operation_surface_plane_fit.ply` | coloured points and candidate quads for inspection |

Old: `plane_fitting_plane_fit.sh`.

### 5 `surface.trim` — CPU

Removes floater gaussians near the table. A gaussian is removed when both hold:

- it is a floater: opacity < 0.3, or largest axis > 0.05 m;
- it is near the table: between 5 cm below and 15 cm above the plane, and within
  0.70 m of the table centre.

Options: `--min-opacity`, `--max-gaussian-scale`, `--plane-band-above/-below`,
`--workspace-radius`.

|  | File | Content |
|---|---|---|
| In | `S/world/gaussian.ply`, `S/surface/operation_surface_plane_fit.json`, refined alignment |  |
| Out | `S/world/gaussian_trimmed.ply` | binary PLY, same properties, fewer gaussians |

Old: `trim_gaussian_all.sh`.

### 6 `surface.receiver` — CPU

Traces the real table outline from the gaussian centres near the plane. It writes
two polygons plus the table's albedo:

- a physics polygon, padded 3 mm and lifted 15 mm;
- a lighting polygon, inset 6 mm.

Options:

- `--receiver-surface-lift 0.015`
- `--receiver-physics-boundary-padding 0.003`, `--receiver-lighting-boundary-inset 0.006`
- `--receiver-contour-concavity-ratio 0.12`
- `--viewer`

|  | File | Content |
|---|---|---|
| In | `S/surface/operation_surface_plane_fit.json`, `S/world/gaussian_trimmed.ply`, refined alignment |  |
| Out | `S/surface/operation_surface_shadow_v2.json` | `selected.footprints.physics` and `.lighting` (world-space polygon vertices + triangles), `material`, `fitted_plane` |

Old: the receiver refit inside `evaluate_pi_shadow_visual.sh` (`REFIT_RECEIVER=1`).

---

## Step 7 — collision crop

### 7 `collision.crop` — CPU

Bakes the posed, scaled collider into world coordinates. It then clips it to a
24-sided prism around the robot base: radius 1.10 m, z from −0.15 to 1.45 m.

Options: `--radius`, `--z-min`, `--z-max`, `--sides`, `--min-faces`, and
`--surface-mask`, which also writes a copy without the faces that duplicate the
table.

|  | File | Content |
|---|---|---|
| In | `S/world/collider.glb`, refined alignment |  |
| Out | `S/cropped_collision/reachable_collider_r1p100_zm0p150_1p450_s24.glb` | collider in world coordinates (z-up), ready to load at identity pose |
| Out | `…_s24.glb.json` | sidecar: source fingerprint, scale, root pose and crop settings (a mismatch triggers a rebuild) |

Old: built implicitly on the evaluator's first run; it is now an explicit step.

---

## Steps 8–10 — objects

### 8a `objects.dump` — CPU

Places every object exactly as the simulator loads it, without needing Genesis:

1. Take the pose from `transformations.json`.
2. Reproject it onto the support plane along the camera ray, correcting the scale
   by the depth ratio.
3. Convert the quaternion to the sim frame.
4. Apply the load-time collision guard, which lifts objects clear of the collider
   and onto the table.

Options:

- `--z-plane 0.02`, `--z-plane-offset 0.01`
- `--init-collision-clearance 0.005`, `--object-support-clearance 0.0005`
- `--support-query-rotation legacy|identity`: `legacy` (the default) reproduces the
  old guard bug; `identity` is the corrected guard ([changes.md](changes.md)).
- `--asset-override`, `--scale-override`, `--collision-override`

|  | File | Content |
|---|---|---|
| In | `IN/masks/transformations.json`, `masks/result.json`, `masks/physical_properties.json`, `IN/assets/<s>/image_k_segmented.glb` | object poses, names, physics, meshes |
| In | `IN/intrinsics.npy`, `IN/extrinsics.npy` | camera (reprojection) |
| In | `S/surface/operation_surface_plane_fit.json`, `operation_surface_shadow_v2.json` | support plane height |
| In | `S/cropped_collision/*.glb` (+ sidecar), refined alignment | collision guard |
| Out | `S/objects/stock/object_poses.json` | `z_plane`, `physics_top`, `reference_top`, and per object `safe_key`, `count`, `object_name`, `asset_file`, `scale`, `pos`, `quat_wxyz`, `aabb_min`, `aabb_max` |

Old: the `--object-pose-dump` of the evaluator's preview run.

### 8b `objects.posefit` — CPU

Moves each object in XY and yaw so that its rendered silhouette matches the real
frame-0 mask (IoU). It also levels tilted thin objects and drops floating ones onto
the table. It never changes scale. A fit is rejected if its IoU is below 0.3 or its
area ratio is outside 0.5–2.

Options: `--yaw off|sweep|c4`, `--yaw-step-deg 15`, `--min-iou 0.3`, `--no-level`,
`--no-drop`, `--max-overlap 0.25`, `--objects`.

|  | File | Content |
|---|---|---|
| In | `S/objects/stock/object_poses.json` | load poses |
| In | `IN/bounding_boxes_and_masks/frame_00001_*.json`, `IN/images0/0.jpg`, calibration | real masks and frame |
| Out | `S/objects/posefit/refined_poses.json` | per object: `status`, `apply`, `xy_delta_m`, `yaw_delta_deg`, `pos`, `quat_wxyz`, `iou_*`, `suggested_scale` |
| Out | `S/objects/posefit/overlays/` | review images |

Old: `object_pose_refine.py fit` in `object_pose_fit_scene.sh`.

### 9 `objects.refine` — GPU for the settle stages

Method B. For each object it fits scale (0.70–1.40×), XY, yaw and an upright
correction against the real masks, then checks the result by settling the layout in
physics.

The objective is a weighted sum:

| Term | Weight |
|---|---|
| IoU | 1 |
| area | 1 |
| dims prior | 0.2 |
| depth | 0.1 |
| grasp anchor | 0.25 |
| stability | 0.5 |

`--stage all|stock-settle|fit|candidate-settle|score` (default `all`) runs these
stages in order:

1. **stock-settle:** a 150-step free settle of the load layout.
2. **fit:** the CPU fit.
3. **candidate-settle:** 20 steps held upright, then free, 150 steps in total.
4. **score:** CPU scoring.

Other options: `--tag B_v1`, `--scales`, `--yaw`, `--orientations yaw|principal`,
`--no-upright`, `--settle-steps`, `--settle-hold-steps`. When present,
`<tag>/weights.json` and `<tag>/decisions.json` adjust the weights or pin the
orientations per object.

|  | File | Content |
|---|---|---|
| In | `S/objects/stock/object_poses.json`, `S/objects/posefit/refined_poses.json` (seed) |  |
| In | `IN/bounding_boxes_and_masks/`, `images0/0.jpg`, `depth/0.npy`, calibration | masks, image, depth term |
| In | `IN/masks/physical_properties.json`, task notes (`<tasks_dir>/<s>.py`) | size prior |
| In | `IN/grippers.npy`, `tcp.npy`, `joints.npy` | grasp anchor (gripper-close frames, forward kinematics) |
| In | `S/cropped_collision/*.glb`, everything the sim needs (splat, table, robot) | resting and settling |
| Out | `S/objects/refine/<tag>/candidate.json` | schema v2: per object `pos`, `quat_wxyz`, `scale_multiplier`, `scale_apply`, `orientation`, `upright`, `terms`; `scores` |
| Out | `S/objects/refine/{stock,<tag>}/settle.json`, `settle.mp4`, `preview/`, `check.png` | per object drift, drop, tilt, `unstable` (xy > 1 cm or tilt > 5°) |
| Out | `S/objects/refine/<tag>/scores.json`, `fit.log`, `review/` | mean/min IoU, objective, unstable count; residual table, overlays |

Timing: each settle stage takes 3–4 min on one GPU; `--stage fit` and `--stage
score` run on CPU.

A single settle or preview run outside the step:

```bash
scripts/slurm/submit.sh scripts/slurm/sim_settle.sbatch --scene <s> --out DIR \
  [--candidate C --pose-mode delta --mode upright --steps 150 --hold-steps 20]
```

Old: `asset_refine_batch.sh <scene> <tag>`.

### 10 `objects.freeze` — CPU

Pins the layout the evaluator uses: it copies one `candidate.json` byte for byte and
records its sha256. A candidate is accepted only next to the `settle.json` of its
own render; `--no-settle-check` skips that check.

Sources (pass exactly one):

- `--refiner-round ROUND`: a reviewed layout, from `<layout_refiner>/runs/<s>/<ROUND>/`.
  This is the production source, with the rounds per scene in scenes.yaml.
- `--tag TAG`: a Method B result.
- `--from PATH`: any candidate file.

`--note` adds free text to `freeze.json`.

|  | File | Content |
|---|---|---|
| In | the chosen `candidate.json` + its `settle.json` |  |
| Out | `S/objects/layout/candidate.json` | the frozen layout |
| Out | `S/objects/layout/freeze.json`, a row in `<work>/layout_manifest.tsv` | source, sha256, settle fingerprint, time |

Old: the freeze block of `submit_generic_home_1200_gpt6_yaw45.sh`
(`layout_eval/<run>/manifest.tsv`).

---

## File formats

These are the files that later steps and the simulator read. All world coordinates
are in the robot base frame, z up, metres; quaternions are `wxyz`.

| File | Format | Fields |
|---|---|---|
| `align/aligned_pose*.npy` | float64 array | `[tx, ty, tz, yaw]`, or `[tx, ty, tz, rx, ry, rz]`; radians |
| `align/aligned_pose_scene_scale*.npy` | 0-d float64 | scene scale `s` |
| `align/aligned_pose_relative_transform.npy` | float64 (7) | `[tx, ty, tz, qw, qx, qy, qz]`, the pose delta only |
| `align/depth_check.json` | JSON | `fitted_scale`, `pose`, `camera_pos`, `image_hw`, settings (`workspace_bounds`, `mask_dilate_px`, `trim_q`, `table_band_m`), pixel fractions (`object_mask_frac`, `valid_frac`, `workspace_frac`, `table_frac`), `all` / `workspace` / `table` metric blocks, `table_height`, `sensor_optimal_scale` |
| `surface/operation_surface_plane_fit.json` | JSON v1 | `transform` (placement provenance), `parameters`, `stats`, `selected {center, quat_wxyz, size, plane_normal, plane_offset, basis_u, basis_v, support_boxes}`, `candidates[]` |
| `surface/operation_surface_shadow_v2.json` | JSON v2 | `selected {footprints: {physics, lighting} {vertices_world, triangles, thickness, clearance}, material, fitted_plane, surface_gaussian_prior}`, `reference_surface`, `stats` |
| `cropped_collision/*.glb.json` | JSON v4 | source path / mtime / size, scale, root pose, crop radius / z band / sides |
| `objects/stock/object_poses.json` | JSON v1 | `z_plane`, `physics_top`, `reference_top`, `objects[] {safe_key, count, object_name, asset_file, scale, pos, quat_wxyz, aabb_min, aabb_max}` |
| `objects/posefit/refined_poses.json` | JSON v1 | `settings`, `objects[] {status, apply, xy_delta_m, yaw_delta_deg, pos, quat_wxyz, base_*, iou_*, scale, suggested_scale}` |
| `objects/refine/<tag>/candidate.json` | JSON v2 | `candidate_id`, `method`, `parent`, `settings`, `objects[] {safe_key, pos, quat_wxyz, scale_multiplier, scale_apply, orientation, upright, terms}`, `scores` |
| `objects/refine/<tag>/settle.json` | JSON v1 | `steps`, `sample_every`, `objects[] {commanded, settled, released, xy_drift_m, z_drop_m, tilt_deg, yaw_drift_deg, residual_motion_m, settled, unstable, topples_after_release, samples}` |
| `objects/layout/freeze.json` | JSON | `source_kind`, `source_id`, `source`, `sha256`, `settle` fingerprint, `frozen_at`, `warnings` |
| `manifest.json` | JSON v1 | `steps {<step>: {status, started, seconds, args, inputs_root, inputs[], outputs[], git_rev, host, slurm_job, argv}}` |

Step-0 files keep the layout of the original asset pipeline
(`DROID/<s>/masks/...`, `scene_background/<s>/`, `assets/<s>/`), so tools written
against it can read a run's `scene_data/` directly.

---

## Handing a scene to the evaluator

```bash
r2s export --scene droid2 --to /path/to/eval_layout [--copy]
```

This writes symlinks (or copies, with `--copy`) for `optimization_results/<s>/`,
`echo2/<s>/` and `layout/<s>/candidate.json`, in the old evaluator's layout.

- Pass `--scene_scale` explicitly to the old evaluator.
- On Eden HEAD, the splat entity needs `PCDOptions(scale=s, pos=t, quat=q)`, not
  `pos=t/s` (see [changes.md](changes.md)).
- [user_guide.md](user_guide.md#using-the-result-in-eden) maps each output file to
  its Eden option.
