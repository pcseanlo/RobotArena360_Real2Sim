# Old file → new step / module

The only place the old repos' file names appear (besides tests and
`tools/legacy/` scripts that must open the old files to compare against them).

## scene_gaussian_alignment

| Old file | New | Notes |
|---|---|---|
| `world_lab_pose_optimization.py` | `align.pose` → `r2s/align/pose_fit.py`, `r2s/align/render.py` | gsplat renderer, camera-derived start |
| `world_lab_scale_optimization.py`, `optimize.sh` | `align.scale` → `r2s/align/scale_fit.py`, `scale_core.py` | numerics verbatim |
| `world_lab_optimization.py`, `world_lab_scale_and_pose_optimization.py`, `world_lab_align_scale_with_precomputed_mesh.py`, `world_lab_with_objects.py`, `scenes/modern_kitchen.py` | — | superseded (modern-kitchen era) |
| `world_labs_marble.py`, `generate_marble.sh` | `r2s/world/providers/marble.py` (optional tool) | key from `WLT_API_KEY` |
| `trim_gaussian_ply.py`, `trim_gaussian*.sh` | `surface.trim` → `r2s/surface/trim.py` | |
| `operation_surface_plane_fit.py`, `plane_fitting_plane_fit.sh` | `surface.plane` → `r2s/surface/plane_fit.py` | |
| `operation_surface_precompute.py` (helpers only) | `r2s/surface/geometry.py`, `r2s/common/*` | legacy fit modes dropped |
| `operation_surface_render_receiver_fit.py` (polygon mode) | `surface.receiver` → `r2s/surface/receiver_fit.py` | other modes dropped |
| `operation_surface_runtime.py` | `r2s/surface/runtime.py` | |
| `operation_surface_from_support_points.py`, `operation_surface_plan_fit.py`, `plane_fitting*.sh` (others), `show_receiver_viser.sh` | — / `--viewer` | dormant experiments |
| `collision_mesh_crop.py` | `collision.crop` → `r2s/collision/crop.py` | now an explicit step |
| `init_collision_guard.py` | `r2s/collision/support.py` | |
| `object_collision_profiles.py` | `r2s/collision/profiles.py` | |
| `object_pose_override.py`, `object_scale_override.py`, `object_asset_override.py`, `object_collision_override.py` | `r2s/objects/overrides.py` | |
| `asset_init_randomization.py` (placement types only) | `r2s/objects/placement_types.py` | eval-time randomization stays out |
| evaluator `load_scene_objects` + `--object-pose-dump` | `objects.dump` → `r2s/objects/dump.py` | CPU, no Genesis |
| `object_pose_refine.py`, `object_pose_fit_scene.sh` | `objects.posefit` → `r2s/objects/pose_fit.py` | |
| `object_settle_probe.py` | `r2s/objects/settle.py` | |
| `asset_refine_{common,fit,metrics,harness}.py`, `asset_refine_batch.sh`, `asset_refine_scene.sh` | `objects.refine` → `r2s/objects/refine/` | |
| `ee_error_from_chunks.py` (Franka FK only) | `r2s/objects/refine/franka_fk.py` | |
| evaluator scene assembly for preview/settle runs | `r2s/sim/` | Eden HEAD |
| freeze block of `submit_generic_home_1200_gpt6_yaw45.sh` | `objects.freeze` → `r2s/objects/freeze.py` | |
| `world_lab_evaluate_with_scene.py` (policy eval), Nyx / shadow / visual-calibration modules, sysid, gap analysis, reporting | — | out of scope |

## scene_generation (SG)

| Old file | New | Notes |
|---|---|---|
| `bash_scripts/pipeline_droid_test.sh`, `pipeline_droid.bash` | `inputs.*` steps | one step per sub-step |
| `src/pipeline/segment_object_gembox_updated.py` (+ `src/utils/gemini_utils.py`, `dataloader_utils*.py`) | `inputs.segment` → `r2s/gen/segment.py`, `workers/segment_gembox.py`, `sg_glue/` | replayable |
| `src/pipeline/prompt_physics_updated.py`, `src/utils/physics_utils.py` | `inputs.physics` → `r2s/gen/physics.py` | |
| InvSR call in the driver | `inputs.superres` (optional) | |
| `lama/bin/predict.py` (modified) | `inputs.inpaint` → `workers/lama_inpaint.py` | no hydra |
| `src/pipeline/sam3d.py` | `inputs.meshes` → `workers/sam3d_meshes.py` | |
| `src/pipeline/MatchFindRot.py` (+ matching/mini/simulator utils) | `inputs.place` → `workers/match_place.py`, `sg_glue/` | seeded |
| `src/pipeline/align_inpaint_updated.py` | `inputs.table_depth` → `workers/align_table_depth.py` | idempotent |
| `src/pipeline/sheet_wraping_updated.py` | `inputs.mesh_sheet` → `workers/sheet_wrap.py` | |

## Other repos

| Old | New |
|---|---|
| `ply2glb/run.sh`, `ply2glb.py` | `world.collider` → `r2s/world/collider.py`, `r2s/world/workers/ply2glb.py` |
| `scene_layout_refiner/runs/<s>/<round>/candidate.json` | read by `objects.freeze --refiner-round` |
| `Autocalib/run_droid_seiredata.sh` | stays external (camera refinement before step 0a) |
