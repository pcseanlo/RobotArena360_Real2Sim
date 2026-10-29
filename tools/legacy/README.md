# Old-repo regression tools

Scripts that verified the port against the old repos. **They are not needed to
use the pipeline**: nothing in `r2s/` imports them, and every test that touches the
old checkouts skips when they are absent.

They need read-only checkouts of the old code and data, set in `configs/local.env`:

| What | Variable |
|---|---|
| old repo `scene_gaussian_alignment` (`optimization_results/`, `asset_refine/`, `object_pose_fit/`, its scripts) | `R2S_OLD_REPO` |
| Eden's old `echo2` assets (splat, collider, trimmed splat per scene) | `R2S_OLD_ECHO2` |
| `scene_generation` and its `data/` (old step-0 outputs) | `R2S_SCENE_GENERATION`, `R2S_DATA` (`paths.scene_generation`, `paths.scene_data`) |
| how the old outputs spelled the same data root, when it was mounted elsewhere | `R2S_OLD_PATH_ALIASES="OLD_PREFIX=NEW_PREFIX[;...]"` |

`old_repo.py` holds the one definition of `OLD`, `ECHO2`, `OLDCODE_BASELINE`
(`$R2S_WORK_ROOT/runs/verify_oldcode`) and `PATH_ALIASES`; the other scripts import
it as a sibling (`from old_repo import OLD`; run them as `python tools/legacy/<x>.py`).
An unset variable becomes the path `<R2S_OLD_REPO unset>`, so errors and skipped
tests name it. Results go under the work folder (`R2S_WORK`), never into the repo or
the old checkouts.

## Seeding

| Script | What it does |
|---|---|
| `old_repo.py seed --scene S --through STEP [--run NAME]` | fills a run with the old outputs of every step up to `STEP` (small files copied, PLY/GLB symlinked) |
| `old_repo.py reference --scene S` | prints old output -> new output for every step |
| `old_repo.py where` | prints `OLD=... ECHO2=...` for shell scripts (`eval`) |
| `gen_b_seed.py --scene S` | seeds `scene_data_out` with the old upstream inputs of 0c-5 .. 0c-8 |

## Comparisons

| Script | Compares |
|---|---|
| `cpu_chain.sh S [RUN]` | seeds through `align.scale`, runs steps 4 -> 8b and 9 (fit) on CPU, compares each output with the old repo (plane / trim / receiver / crop / dump / posefit / B_v1 fit) |
| `gen_a_compare.py` | regenerated 0a, 0c-1, 0c-2, 0c-4 outputs vs the old `scene_generation/data` files |
| `gen_b_compare.py` | new 0c-5 .. 0c-8 and 1b outputs (meshes, placements, table depth, mesh sheet, collider) vs production |
| `align_pose_check.py` (GPU) | a new pose fit vs the old `aligned_pose.npy`: fit loss of both, translation / yaw difference, review image |
| `align_scale_sweep.py` (GPU) | scale fit over a truncation grid vs the old refined scale (how the per-scene truncations were recovered) |
| `surface_compare.py OLD NEW ...` | surface JSON descriptors and PLYs, numerically |
| `objects_compare.py json/mesh/log/batch` | object / collision outputs: recursive JSON diff, GLB arrays, fit logs |
| `objects_oldcode_baseline.py` | runs the old repo's own code (crop, posefit, fit, score) on the production inputs, as the baseline for steps 7, 8b, 9 |
| `objects_dump_oldcode.py` | runs the old evaluator's object loading (extracted with `ast`) as the baseline for `objects.dump` |
| `objects_dump_verify.py` | `objects.dump` vs that baseline and the on-disk old dumps |
| `objects_verify.py` | the new collision / object steps on a seeded run vs the old-code baseline |
| `sim_settle_compare.py` | a new `settle.json` vs the old one (drift, tilt, verdicts, settled pose) |
| `e2e_compare.py` | an end-to-end new-code run vs the old outputs (alignment, table, object load poses, fit, settle) |
| `sim_alignment_check.py` (GPU) | the Eden HEAD sim scene (splat render, collider, objects) vs where r2s places them |
| `sim_glb_frames.py` (GPU) | which rotation Eden HEAD applies to a GLB for a given `up` / `front` |

Verification results are summarised in [docs/changes.md](../../docs/changes.md).
