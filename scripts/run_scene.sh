#!/usr/bin/env bash
# Compatibility wrapper for `r2s pipeline` (all logic lives there):
#
#   bash scripts/run_scene.sh SCENE [FIRST_STEP] [LAST_STEP] [--skip-done]
#   == r2s pipeline --scene SCENE [--from FIRST_STEP] [--to LAST_STEP] [--skip-done]
#
#   bash scripts/run_scene.sh droid2 align.pose collision.crop
#   sbatch scripts/slurm/step.sbatch pipeline --scene droid2 --from align.pose --to objects.refine
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"

scene="${1:?usage: run_scene.sh SCENE [FIRST_STEP] [LAST_STEP] [--skip-done]}"; shift
skip=(); steps=()
for a in "$@"; do if [[ "$a" == --skip-done ]]; then skip=(--skip-done); else steps+=("$a"); fi; done
exec "$R2S_PYTHON" -m r2s pipeline --scene "$scene" ${steps[0]:+--from "${steps[0]}"} ${steps[1]:+--to "${steps[1]}"} \
  "${steps[@]:2}" "${skip[@]}"
