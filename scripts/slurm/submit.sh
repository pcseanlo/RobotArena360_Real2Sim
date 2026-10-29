#!/usr/bin/env bash
# sbatch with this machine's Slurm settings from configs/local.env (R2S_SLURM_PARTITION,
# _QOS, _ACCOUNT, _GRES; empty = the cluster default), logging to
# $R2S_WORK_ROOT/logs/r2s_<job name>_<job id>.log. Arguments go to sbatch unchanged,
# so sbatch options come before the job script.
#
#   scripts/slurm/submit.sh scripts/slurm/step.sbatch run align.pose --scene droid2
#   R2S_WORK=/path/to/run scripts/slurm/submit.sh scripts/slurm/step.sbatch pipeline --scene droid2 --from align.pose
#   scripts/slurm/submit.sh --time=8:00:00 scripts/slurm/step.sbatch run objects.refine --scene droid2
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../env.sh"
: "${R2S_WORK_ROOT:?set R2S_WORK_ROOT in configs/local.env}"
mkdir -p "$R2S_WORK_ROOT/logs"
cd "$R2S_REPO"   # the job starts here (and the job script paths are relative to it)
exec sbatch \
  ${R2S_SLURM_PARTITION:+--partition="$R2S_SLURM_PARTITION"} \
  ${R2S_SLURM_QOS:+--qos="$R2S_SLURM_QOS"} \
  ${R2S_SLURM_ACCOUNT:+--account="$R2S_SLURM_ACCOUNT"} \
  ${R2S_SLURM_GRES:+--gres="$R2S_SLURM_GRES"} \
  --output="$R2S_WORK_ROOT/logs/r2s_%x_%j.log" \
  "$@"
