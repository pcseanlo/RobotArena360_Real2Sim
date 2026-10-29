# Changes from the old repos, bugs found, verification

Old sources: `scene_gaussian_alignment` ("OLD"), `scene_generation` ("SG"),
`ply2glb`, `scene_layout_refiner`. All were read only during the port.

## Verification against the old outputs

Each step was run on the old repo's inputs for that step (seeded with
`tools/legacy/old_repo.py seed`) and compared with (a) what today's old code produces and
(b) the files on disk. Reports are under `$R2S_WORK_ROOT/runs/verify*/`; the scripts are
in `tools/legacy/` (README there).

| Step | droid2 | droid820 | Notes |
|---|---|---|---|
| 0a episode | identical | identical | |
| 0c-1 segment (replay) | masks pixel-identical | masks pixel-identical; 3 point clouds +1/2/4 points | Open3D's threaded outlier filter dropped points at random; now single-threaded. SAM3 output is GPU-model sensitive (bit-exact on the original GPU type) |
| 0c-2 physics (replay) | byte-identical | byte-identical | |
| 0c-4 inpaint | byte-identical | byte-identical | LaMa is CPU-family sensitive (±1 grey level on another CPU family) |
| 0c-5 meshes | ≤5e-5 of bbox (PLY) | ≤5e-5 of bbox (PLY) | GPU nondeterminism; decimated GLB within ~1% |
| 0c-6 place | Chamfer 0.4-10 mm | Chamfer 0.8-4.1 mm | old runs were unseeded; differences are within the old seed-to-seed spread. Now seeded |
| 0c-7 table depth | bit-identical | bit-identical | |
| 0c-8 mesh sheet | identical | identical | |
| 1b collider | 94% vertices exact, p95 0.05 mm | 92% exact, p95 5.5 mm | Open3D Poisson multithreading; then in the old `ply2glb` env, now the repo env (see below) |
| HEAD splat render | PSNR 82.5 dB vs the pose-fit renderer | PSNR 82.0 dB | Eden HEAD's own camera render of the placed splat |
| HEAD collider | crop vertices exactly on the transform | same | built CoACD hulls within 3.8 mm median of the surface |
| 2 pose fit | Δt 0.1 cm, Δyaw 0.02° | Δt 1.2 cm, Δyaw 0.58° | new renderer (gsplat); new loss ≤ old on 6/6 scenes (also droid4/34/69/1034: ≤3.3 cm, ≤0.72°) |
| 3 scale fit | s −0.062% (needs truncate 0.45/0.30, recovered by sweep) | s −0.016% (default 0.1/0.3) | from the old aligned pose; collider sampled from the CoACD hulls Genesis builds at HEAD, as before |
| 4 plane fit | identical | identical | droid2's 05-15 file differs by 1 ulp in a ranking score (old numpy build) |
| 5 trim | byte-identical | byte-identical | |
| 6 receiver | identical | identical | |
| 7 crop | byte-identical GLB | byte-identical GLB | |
| 8a dump | identical (stock + B_v1) | identical (stock + B_v1) | also droid1034 |
| 8b posefit | bit-identical to old code | bit-identical to old code | old code vs disk ≤ 2.7e-13 |
| 9 fit + score | bit-identical to old code | bit-identical to old code | the on-disk B_v1 candidates were made by an older fit algorithm; not reproducible by today's old code either |
| 9 settle | B_v1: 4/5 objects match (≤0.06 cm, ≤1.7°); **white cup flips unstable→stable** | B_v1: all 5 within 0.02 cm / 0.4°, same verdicts (red block topples 35.5° vs 35.9°); stock: **rubik's cube flips unstable→stable** | Genesis 1.4.4 vs the old version; old stock runs also loaded some objects rotated (see porting/sim_assembly.md) |
| 10 freeze | byte-identical, same sha256 | byte-identical, same sha256 | also droid41 |

### End-to-end (droid820, new code only)

`world.import` → `objects.freeze` (then `scripts/run_scene.sh`, now `r2s pipeline`)
ran all 12 steps from the old splat and the existing step-0 data. It took 27 min on
one L40S: refine 15 min, collider 4 min, scale 6 min. Every later step consumed only
new outputs, so differences compound, and the scene still lands within about 1 cm
of the old one. `tools/legacy/e2e_compare.py` reproduces the table.

| Quantity | Old | New | Δ |
|---|---|---|---|
| Scene scale | 0.75324 | 0.75884 | +0.74% |
| Splat translation | (0.335, 0.370, 0.287) | (0.341, 0.376, 0.283) | 0.87 cm |
| Yaw | 69.34° | 68.77° | −0.58° |
| Table plane centre | (0.675, 0.000, 0.000) | (0.685, −0.004, −0.005) | 1.22 cm |
| Physics table top / `z_plane` | 0.0295 m | 0.0241 m | −0.54 cm |
| Object load positions (5) | | | 0.7-1.8 cm, scale ×1.019 |
| Method B fit vs today's old code | | | positions 0.7-1.4 cm; scale multipliers equal for 4/5 (yellow block 0.98 vs 1.05); cube yaws equal; bowl yaw +39° (round object) |
| B_v1 settle verdicts | red block topples, rest stable | same | identical |
| Stock settle verdicts | all 5 unstable | rubik's cube stable | the known Genesis 1.4.4 flip |

### Collider in the repo env (droid2, droid37)

`world.collider` used to run the copied mesher in the old `ply2glb` conda env
(Python 3.10, Open3D 0.19.0). It now runs in the repo env (Python 3.12, Open3D
0.19.0; numpy, scipy and trimesh at the lock's versions).

- **Where:** `$R2S_WORK_ROOT/runs/collider_repo_env/`. `compare/report_surface.json`
  holds the numbers; `check.sh` and `compare_surfaces.py` reproduce them.
- **Method:** the same splat and preset were meshed in both envs. Distances are
  two-sided point-to-surface over 300k surface samples per side.
- **Baseline:** Open3D's multithreaded Poisson is not bit-reproducible, so two runs
  in the old env are the reference spread.

| Pair | Vertices / faces | Same vertices | Distance p95 / p99 / max | > 5 mm |
|---|---|---|---|---|
| droid2 surface: old env, run A vs run B | 102981 / 200013 vs 200012 | 75% | 0.25 / 1.3 / 25 mm | 0.015% |
| droid2 surface: old env A vs repo env | 102981 vs 102984 / 200013 vs 200012 | 83% | 0.07 / 1.0 / 20 mm | 0.010% |
| droid2 surface: old env B vs repo env | 102981 vs 102984 / 200012 | 80% | 0.15 / 1.1 / 12 mm | 0.008% |
| droid37 planar: old env vs the `world.collider` step | 76188 / 150004 vs 150003 | 76% | 0.15 / 0.67 / 9.6 mm | 0.001% |

The repo env is within the old env's own run-to-run spread, so this change does not
alter the colliders.

### Mesh-sheet fill (droid8, droid32, droid37, droid41)

These four scenes never had a `mesh_sheet.glb`, so their old scale fits fell back to
the largest-index segmented object mesh. They all had `inpainted_depth.npy`, so
`inputs.mesh_sheet` was run for them in a separate run. `align.scale` was then
rerun from the old pose, and both alignments were scored against the raw sensor
depth with the method that became `align.check`.

- **Where:** `$R2S_WORK_ROOT/runs/mesh_sheet_fill/`, with the analysis in
  `analysis/depth_vs_collider_*.json` and `summary_table_heights.json`.
- **Metrics:** Chamfer and median are sensor ↔ collider, in cm. Table height is the
  table plane's z in cm, sensor vs collider.

| Scene | Scale old → new | Trimmed Chamfer | Sensor → collider median | Table height: sensor / old / new | Verdict |
|---|---|---|---|---|---|
| droid41 | 1.1129 → 1.0665 (−4.2%) | 6.29 → 4.63 | 7.21 → 4.98 | −1.0 / −6.9 / −5.1 | better on every metric |
| droid8 | 0.9715 → 0.9953 (+2.5%) | 8.27 → 9.08 | 8.45 → 9.60 | 2.4 / 11.2 / 10.4 | mixed; the 8 cm table error is the collider's shape, not its scale |
| droid37 | 0.8571 → 0.8400 (−2.0%) | 4.02 → 4.29 | 4.33 → 4.83 | −7.1 / −6.0 / −5.4 | slightly worse |
| droid32 | 0.8584 → 0.8569 (−0.2%) | 5.75 → 5.71 | 7.66 → 7.63 | −3.0 / −3.6 / −3.5 | unchanged |

Only droid41 is worth adopting. It has not been switched: `configs/scenes.yaml`
still freezes droid41's refiner round 5, which was reviewed against the old
alignment. Adopting the new scale means rerunning steps 4-10 for droid41 and
reviewing its layout again.

## Behaviour changes

- **Eden HEAD splat placement.** At `robotarena360` HEAD splats render through
  gsplat with `world = R (s x) + pos`; the old 2DGS renderer did
  `world = s (R x + pos)`, which is why the old code passed `pos = t/s`. Anything
  building the splat entity on HEAD must pass `PCDOptions(scale=s, pos=t, quat=q)`.
  The old evaluator's `legacy` splat path still passes `t/s`.
- **Pose fit renderer.** gsplat 3DGS (what the evaluator renders at HEAD) instead
  of the removed 2DGS `diff_surfel` renderer; no Genesis scene is built.
- **Pose fit start.** `--init camera`: translation = camera position, yaw = camera
  heading + 90°. Image-to-world generators build the world around the input camera, so every old
  result is within ~0.2 m / ~5° of this. The old start (yaw 0) needed hand-tuned
  reruns (up to ~24 per scene).
- **Pose fit "best" snapshot** is taken before the optimizer step (the old code
  paired a loss with the next step's parameters).
- **No silent fallbacks.** A missing scale file, refined pose, trimmed splat,
  collider or crop is an error. Old fallbacks: scale 0.5, unrefined pose at scale 1,
  untrimmed splat, v1 rectangles as table physics, objects resting on the plane.
- **Explicit paths.** No absolute machine paths in the code, no Eden-asset lookups, no
  "file next to the result" guessing; every root comes from `configs/paths.yaml`,
  which reads this machine's `configs/local.env`.
- **Reproducibility.** Every run writes its arguments, input fingerprints and
  outputs to the scene manifest. Per-scene settings that only lived in shell
  history, log-folder names or file evidence are in `configs/scenes.yaml`.
- **Step 0:** reads/writes only `scene_data_out` (never SG's `data/`); no hydra
  cwd tricks for LaMa; SAM 3D only sees object masks (not `gemini_boxes_debug.png`
  or the background mask); table depth is idempotent (fixed index); placement is
  seeded; `transformations.json` is rebuilt fresh; the Gemini key comes only from
  `GEMINI_API_KEY`, and both Gemini steps can replay saved answers.
- **Machine settings in one file.** Every folder outside the repo and the Slurm
  partition / QOS / GPU type come from the gitignored `configs/local.env`;
  `configs/paths.yaml`, the scripts, tests and docs hold no machine paths.
  `scripts/slurm/submit.sh` adds the Slurm settings to `sbatch`. `r2s doctor`
  checks that the Eden checkout contains the verified `robotarena360` commit.
- **Collider mesher in the repo.** `world.collider` no longer needs the user's
  ply2glb tool or its `ply2glb` conda env. Its mesher is
  `r2s/world/poisson_mesher.py`: the original `ply2glb.py` code, unchanged and pinned
  by a test. It runs in the repo env, to which `setup_env.sh` adds Open3D 0.19.0
  without changing any locked package. The results match within Poisson's own
  run-to-run spread (see "Collider in the repo env" above).
- **Collision crop** is its own step instead of a side effect of the first eval.
- **Input root per scene.** Steps 1-10 read a scene's step-0 files from the run's
  own `scene_data_out` once `inputs.episode` imported it, else from `scene_data`
  (`R2S_SCENE_DATA` pins it); the manifest records which (`inputs_root`).
- **New check step `align.check`** (3b): the aligned collider against the real
  sensor depth (Chamfer, depth error, table height, sensor-optimal scale). It
  writes only `align/depth_check.*`; nothing downstream reads it.
- **`inputs.mesh_sheet` is a GPU step**: SG's sheet-warping helpers default to
  `device="cuda"` (the old docs called it CPU).
- **Removed legacy:** superseded alignment scripts (modern-kitchen era), legacy
  surface fitters (precompute simple/legacy, support points, plan fit, receiver v1,
  one-plane), receiver modes other than polygon, and every `while True` viewer loop.

## Bugs found in the old code (kept for parity, with a switch)

1. **Load-time collision guard queried a rotated collider.** The cropped
   collider is saved z-up, but the guard built its query with
   `GLB_WORLDSPACE_LOAD_QUAT_WXYZ` (−90° about X). Objects whose footprint hits
   the rotated mesh start too high: droid2's black bowl by 3.95 cm, droid1034's
   bowl by 1.89 cm; other objects in droid2/droid820/droid1034 are unaffected.
   `objects.dump --support-query-rotation identity` fixes it; `legacy` (default)
   reproduces the old dumps.
2. **`z_plane` floor of 0.02 m.** The support plane is `max(0.02, plane_top + 0.01,
   physics_top)`, so in 13 of 23 scenes objects start up to ~6 cm above the real
   table and drop. Kept (it is what every layout was fitted with); `--z-plane`
   changes it.
3. **Mapping files overwritten.** The old Marble API script opened its mapping files
   with `w+`, keeping only the last scene and saving `world_id` as null. Fixed in
   the optional `python -m r2s.world.providers.marble` tool (one record per scene).
4. **Settle step count.** The production eval passed `--object-settle-steps 150`
   with hold 20 (130 free steps) while the refiner rendered 150 free steps.
5. **Gripper pad geometry** depended on the renderer: the fingertip box rewrite
   only happened on the Nyx path. r2s always applies it.
6. **Arm asset changed upstream.** Old Eden's `FrankaResearch3` loaded the
   hand-less Panda file; at HEAD it is a real FR3. r2s pins the old file.

## Security

The old `generate_marble.sh` and SG's `configs/config_DROID.yaml` / `sam3d.py`
hold plain-text API keys. Nothing in this repo reads or stores keys; use
`GEMINI_API_KEY` / `WLT_API_KEY` environment variables.
