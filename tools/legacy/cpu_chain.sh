#!/usr/bin/env bash
# CPU regression chain in the repo env: seed the old alignment + world files, run
# steps 4 -> 8b -> 9(fit) with production defaults, compare every output with the
# old repo. The old-code B_v1 baseline (made by objects_oldcode_baseline.py) is used
# for step 9 because the on-disk B_v1 candidates predate today's fit algorithm.
#
#   bash tools/legacy/cpu_chain.sh droid2 [RUN_NAME]
#
# Needs the old checkouts: R2S_OLD_REPO / R2S_OLD_ECHO2 (and R2S_OLD_PATH_ALIASES) in configs/local.env.
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/env.sh
scene="$1"; run="${2:-verify_cpu_chain}"
export R2S_WORK="$R2S_WORK_ROOT/runs/$run"
eval "$("$R2S_PYTHON" tools/legacy/old_repo.py where)"   # OLD=... ECHO2=...
BASE="$R2S_WORK_ROOT/runs/verify_oldcode/$scene/asset_refine/$scene"
S="$R2S_WORK/scenes/$scene"

rm -rf "$S"
"$R2S_PYTHON" tools/legacy/old_repo.py seed --scene "$scene" --through align.scale >/dev/null
for step in surface.plane surface.trim surface.receiver collision.crop objects.dump objects.posefit; do
  "$R2S_PYTHON" -m r2s run "$step" --scene "$scene" >/dev/null
done
"$R2S_PYTHON" -m r2s run objects.refine --scene "$scene" --stage fit --tag B_v1 >/dev/null

cmp_json() { "$R2S_PYTHON" tools/legacy/objects_compare.py json "$1" "$2" "${@:3}" 2>&1 | grep -E "max_abs:|!=|only in" | head -2 | tr "\n" " "; }
echo "== $scene ($R2S_WORK)"
echo "plane    $("$R2S_PYTHON" tools/legacy/surface_compare.py "$OLD/optimization_results/$scene/operation_surface_plane_fit.json" "$S/surface/operation_surface_plane_fit.json" --tol 1e-9 2>&1 | tail -1)"
echo "trim     $("$R2S_PYTHON" tools/legacy/surface_compare.py "$ECHO2/$scene/gaussian_trimmed.ply" "$S/world/gaussian_trimmed.ply" 2>&1 | tail -1)"
echo "receiver $("$R2S_PYTHON" tools/legacy/surface_compare.py "$OLD/optimization_results/$scene/operation_surface_shadow_v2.json" "$S/surface/operation_surface_shadow_v2.json" 2>&1 | tail -1)"
echo "crop     $("$R2S_PYTHON" tools/legacy/objects_compare.py mesh "$OLD/optimization_results/$scene/cropped_collision/reachable_collider_r1p100_zm0p150_1p450_s24.glb" "$S/cropped_collision/reachable_collider_r1p100_zm0p150_1p450_s24.glb" 2>&1 | grep -E "max_abs|vertices|faces" | tr -s " " | tr "\n" " ")"
echo "dump     $(cmp_json "$OLD/asset_refine/$scene/stock/object_poses.json" "$S/objects/stock/object_poses.json")"
echo "posefit  $(cmp_json "$OLD/object_pose_fit/$scene/refined_poses.json" "$S/objects/posefit/refined_poses.json")"
if [[ -f "$BASE/B_v1/candidate.json" ]]; then
  echo "fit B_v1 $(cmp_json "$BASE/B_v1/candidate.json" "$S/objects/refine/B_v1/candidate.json" --ignore parent)"
fi
