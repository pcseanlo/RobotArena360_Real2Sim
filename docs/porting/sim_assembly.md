# Porting notes: sim scene assembly (old evaluator → Eden HEAD)

How the old evaluator (see docs/porting/old_to_new.md,
"OLD" below, line numbers refer to it) assembled the scene for the object steps,
and what changes on Eden HEAD (`robotarena360`, f7ab779). Used by
`r2s/sim/` (settle runs, previews) and `r2s/objects/dump.py`.

## What the object steps used it for

The "preview run" (`asset_refine_scene.sh` → `evaluate_pi_shadow_visual.sh` →
`evaluate_with_nyxsplat.sh` → evaluator, `REQUIRE_POLICY_SERVER=0`) built the full
eval scene, wrote `--object-pose-dump object_poses.json`, applied an optional
pose/scale override, ran the settle probe (`--object-settle-dump settle.json`,
`settle.mp4`), saved `--preview-views-out` images, and exited before any policy.

## Findings that change behaviour

1. **`object_poses.json` needs no Genesis.** It is written after
   `load_scene_objects` and before `en.init`; its AABBs come from trimesh
   vertices (`ObjectPlacementSpec.base_aabb`). → `objects.dump` is a CPU step.
2. **Old bug in the load-time collision guard.** The cropped collider is saved in
   world (z-up) coordinates, but `BackgroundSupportQuery` was built from it with
   `root_quat = GLB_WORLDSPACE_LOAD_QUAT_WXYZ = (0.70711, -0.70711, 0, 0)`
   (OLD 6574-6584), i.e. the world mesh rotated -90° about X. Only objects where
   that rotated mesh fakes a penetration move: droid2's black bowl starts 3.95 cm
   too high (z 0.0911 vs 0.0516 with the correct identity rotation), droid1034's
   bowl 1.89 cm; every other object in droid2/droid820/droid1034 is unchanged
   because the floor lift onto `z_plane + 0.0005` dominates. r2s keeps the old
   behaviour by default for parity (`objects.dump --support-query-rotation legacy`)
   and offers `identity` as the fix.
3. **Splat placement convention changed.** Old renderer: `world = s·(R x + pos)`,
   so the old code passed `pos = t/s`. HEAD gsplat (`BatchSplatRenderer`):
   `world = R (s x) + pos` → pass `PCDOptions(scale=s, pos=t, quat=q)`.
4. **Arm asset changed.** OLD's `FrankaResearch3` loaded the hand-less Panda
   (`arms/franka_research3/fr3.xml`); at HEAD `FrankaResearch3` loads a real FR3
   (commit 529a8cf5, different limits/friction). Pin the old file explicitly.
5. **Root-pose frame changed (f01aa5f2).** HEAD writes default root pos and quat
   `relative=True` (mesh origin). Don't port OLD's post-build
   `set_quat(..., relative=False)` (7517-7525).
6. **Gripper pad collision.** OLD's cached Robotiq MJCF (2663-2790, applied on
   the Nyx path) replaced each 5-sphere fingertip cluster with one box
   (`name=<pad>_box type=box class=pad_sphere pos="0.044258 0 0.1305"
   size="0.00155 0.01055 0.01555"`). The gsplat path skipped it; r2s applies it.

## Effective settings of the preview run

- Scene: `--scene_type DROID`, robot franka (DROID), timing profile `current`:
  `sim_dt 0.004`, `sim_substeps 4`, `decimation 5`; `noslip_iterations 5`,
  `ls_iterations 100`, `iterations 100`, `max_collision_pairs 2000`.
- Pose/scale: `aligned_pose_refined.npy` + `aligned_pose_scene_scale_refined.npy`.
- Support plane: `z_plane = max(0.02, reference_top + 0.01, physics_top)`;
  `reference_top` = max z of the 8 corners of `plane_fit.selected` box (2346-2377);
  `physics_top` = max finite z of `shadow_v2.selected.footprints.physics.vertices_world`
  (2380-2391).
- Init guard: clearance 0.005 (penetration), object support clearance 0.0005,
  footprint margin 0.015, surface normal z 0.75, bottom slab 0.02,
  `background_support_max_z = z_plane + 0.005`, then floor lift
  `max(0, z_plane + 0.0005 - (min_z + lift))`.
- Collision crop: `always`, r 1.10, z [-0.15, 1.45], 24 sides, cache
  `cropped_collision/`.
- Operation surface: `shadow_v2.json`, mode `physics_polygon`, thickness 0.03,
  contact `default`.
- Objects: collision profile `detailed` (`decimate=False, face_num=1000,
  aggressiveness=2, decompose=0.05, coacd=0.03`), contact `default`, material
  `auto`, mass floor 0.03 (mass = max(phys.mass or 0.1, 0.03); friction default 0.5).
- Gripper: collision profile `high-fidelity-eef`; arm gains `pi-sysid`:
  kp (624.04, 2403.96, 1064.94, 2146.25, 366.29, 1568.66, 1015.37),
  kd (240.57, 799.99, 356.44, 799.47, 117.23, 446.11, 363.74).
- Reset pose: `--droid-reset-profile recorded`: first row of
  `DROID_original/<s>/joints_old.npy`, else `DROID_original/<s>/joints.npy`
  (if its rows match `video1.mp4`), else `DROID/<s>/joints.npy`.

## Ordered assembly (OLD line refs)

1. Calibration (`load_calibration` 2179-2197): `T_c2w = inv(extr)`,
   `T_c2w[:3,:3] @= diag(1,-1,-1)`, `W = round(2cx)`, `H = round(2cy)`,
   `fov_y = 2 atan(H / 2fy)`, `pos = t`, `lookat = pos - R[:,2]`, `up = R[:,1]`.
2. Base quaternions (6443-6459): `orig_splat_quat = ADJ ⊗ BASE`,
   `orig_collider_quat = ADJ`, `q_rel = Rz(rz)` (or `Rz Ry Rx`),
   `optimised_* = q_rel ⊗ orig_*`; `collider_pos = t`.
3. z_plane (6476).
4. Collider crop (6488-6569, `collision_mesh_crop.py:381-530`): load GLB with
   trimesh, `align_up_and_front((0,1,0),(1,0,0))`, scale s, rotate by
   `optimised_collider_quat`, translate t; clip to the 24-gon cylinder (apothem
   `r cos(pi/24)`) and z band; export GLB + `.json` (meta v4). Load the cropped
   GLB with `scale=1, pos=0, quat=GLB_WORLDSPACE_LOAD_QUAT_WXYZ, up=(0,0,1),
   front=(1,0,0)` (cancels Genesis's +90° X conversion of y-up GLBs).
5. Init query (6571-6593) — see finding 2.
6. `load_scene_objects` (3111-3439), per key of `masks/transformations.json`:
   - `count = int(key.split("_")[1])`; asset override may swap the GLB.
   - `reproject_to_plane` (2313-2343): project `translation` through K and the raw
     extrinsics, intersect the ray with `z = z_plane`, `scale_factor = Zcam_new / Zcam_orig`.
   - `scale = attr.scale * scale_factor * mult` (mult clamped [0.2, 5]).
   - `q = scipy(R).wxyz`; unless R is identity: `q = (.7071067811865476,-.7071067811865476,0,0) ⊗ q;
     q = (0,0,.7071,.7071) ⊗ q; q = q ⊗ (0,0,-.7071,-.7071)` (3212-3221), not renormalised.
   - Guard (3238-3313): `mesh_world_vertices_and_contact_points` (import rot
     `align_up_and_front((0,1,0),(1,0,0))`, slab 0.02), `corrected_init_position(...)`,
     floor lift.
   - `safe_key = "obj_" + key.replace("-", "_")`.
   - `ObjectPlacementSpec.from_asset(asset_file, scale, base_pos, q)` (1500-vertex
     subsample keeping extremes → exact AABB).
   - Entity: `EntityOptions(file, scale, default_root_pos, default_root_quat,
     is_fixed_base=False, convexify=True, decimate..., coacd..., surface=...)`;
     mass/friction set after build.
7. Dump (6720 → `object_pose_override.py:84-125`):
   `{version: 1, scene, scene_meta_dir, z_plane, physics_top, reference_top, objects: [{safe_key, count,
   object_name, asset_file, scale, pos, quat_wxyz, aabb_min, aabb_max}]}`.
8. Override resolution (6731): `absolute` keeps file poses (revert target = this
   run's base pose); `delta`: `pos = own + (dx, dy, 0)`,
   `quat = (q_fit ⊗ conj(q_file_load)) ⊗ own_quat`.
9. Eden setup (6766-7380):
   - `en.init(gs.gpu, debug=True, performance_mode=True)`.
   - Collider entity: cropped GLB, `scale=1, up=(0,0,1), front=(1,0,0), pos=0,
     quat=GLB_WORLDSPACE_LOAD_QUAT_WXYZ, is_fixed_base=True, convexify=True,
     coacd_threshold=0.01, decimate_face_num=5000, decompose_object_error_threshold=0.0,
     decimate_aggressiveness=2, coacd_preprocess_resolution=150, vis_mode="collision",
     material=RigidMaterialOptions(sdf_cell_size=0.05)`.
   - Physics surface: `load_operation_surface_boxes(shadow_v2, mode="physics_polygon",
     visualization=False, collision=True, collision_thickness=0.03)` → OBJ prism of
     the physics footprint (`EntityOptions(up=(0,0,1), is_fixed_base=True,
     decimate=False, convexify=True, decompose=0.0, coacd=0.03, coacd_res=150)`).
   - Ground plane: `PlaneOptions(default_root_pos=(tx, ty, tz - 2.0))`.
   - Robot (3832-4019): `Droid(arm=FrankaResearch3(default_dofs_pos=q0, kp, kd,
     material=RigidMaterialOptions(gravity_compensation=1.0), file=arms/franka_research3/fr3.xml (old hand-less Panda)),
     gripper=Robotiq2f85(batch_fixed_verts=True, is_fixed_base=False, decimate=False,
     gravity_compensation=1.0, collision_link_patterns=["base", ".*_driver",
     ".*_spring_link", ".*_follower", ".*pad"], default_geoms_sol_params={".*pad":
     (0.002, 1.0, 0.95, 0.999, 0.001, 0.5, 2.0)}, file=<2f85.xml with pad boxes>))`,
     base at origin, no mount yaw.
   - Scene order: background, plane, objects, surfaces, robot last.
   - Env options: `num_envs=1, episode_length_s=133.04, sim_dt=0.004, sim_substeps=4,
     decimation=5, noslip_iterations=5, ls_iterations=100, iterations=100,
     max_collision_pairs=2000`.
   - Lights (gsplat path, s=1.0): `ambient_light=0.5`, directional (-1,-1,-1) 4.0,
     (1,1,-1) 2.0, (0,0,1) 1.5.
   - Cameras: `camera` from calibration (`background_entities=["background","plane"]`);
     `wrist_camera` intrinsics `[[184.5,0,160],[0,184.5,90],[0,0,1]]` attached to
     `gripper.base`, offset (0.031, 0.06, -0.025), quat (0,0,1,0).
   - Splat (HEAD): `PCDOptions(path=gaussian_trimmed.ply, scale=s, pos=t, quat=optimised_splat_quat)`.
10. `env.build()`; `env.reset_state = env.scene.get_state()`; `env.reset()`;
    per object `set_mass`, `set_friction`.
11. Preview (7883-7899): `env.reset()`; apply override with `set_pos/set_quat(relative=True)`,
    `env.step(None)` ×5; `disp = |Δxy|` (use `max(disp, |Δz|)` if
    `|Δz| > max(0.05, 5·maxdisp)`); if `disp > maxdisp` revert to the load pose and
    step ×5 more. Then the settle probe, then render `camera.png` and
    `wrist_camera.png` (uint8, imageio).
12. Settle probe (`object_settle_probe.py:353-447`): commanded = `get_pos/get_quat(relative=True)`
    after the override; N steps of `env.step(None)`; while step ≤ hold
    (modes gentle/upright): upright → `set_pos((cmd_x, cmd_y, cur_z))`, `set_quat(cmd)`,
    `zero_all_dofs_velocity()`; gentle → zero velocity only; record released pose at
    step == hold; sample every 5 steps and the last; metrics `xy_drift`, `z_drop`,
    `tilt = acos(R[2,2])`, `yaw = atan2(R[1,0], R[0,0])` of `R_to R_from^T`,
    `residual = |p_last - p(last sample ≤ last-10)|`; settled if residual ≤ 0.001;
    unstable if xy > 0.01 or tilt > 5° or not settled; topples if tilt after release > 5°.
    Video: camera frame before step 1 + one per step, mp4 15 fps.
13. `check.png`: `object_pose_refine.py check-render` (661-675) — contours of the first
    `bounding_boxes_and_masks/frame_*.json` on the render.

## Found while porting to HEAD

- **GLB up/front at HEAD.** HEAD maps a GLB loaded with `up=(0,1,0), front=(1,0,0)`
  to Rx(180°), so the old evaluator's *uncropped*-collider options would come out
  flipped. r2s never builds that entity (it always loads the baked crop); the
  scale-fit entity frames were checked unchanged (`tools/legacy/sim_glb_frames.py`).
- **Old stock settles loaded some objects rotated.** With old Eden's root-pose
  writes, droid820's bowl started 38° and rubik's cube 41° away from the dump pose
  (droid2: bowl 22°, cup 10°, tiger 12°, wheel 7°). At HEAD objects sit exactly at
  the dump pose (≤ 7e-8 m), so stock settles differ by design.
- **Old B_v1 renders used `delta` pose mode**; the launcher default became
  `absolute` on 2026-09-12, after them. Compare old B_v1 settles with
  `--pose-mode delta`.
- **Robot hold.** After each reset r2s sets explicit PD targets at the reset pose
  (the old evaluator held the reset action through `env.step(None)`).
- **Solver tolerance** stays at the HEAD default (Genesis changed what the old
  literal 1e-5 means).
- **TF32.** `en.init` enables TF32 matmuls, which `torch.cdist` in the scale fit
  uses; Eden has done so since Nov 2025, so the old runs had it too.
