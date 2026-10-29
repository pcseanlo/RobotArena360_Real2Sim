#!/usr/bin/env bash
# Run one r2s step (or any command) on a GPU node.
#
#   scripts/slurm/submit.sh scripts/slurm/step.sbatch run align.pose --scene droid2
#   R2S_WORK=/path/to/run scripts/slurm/submit.sh scripts/slurm/step.sbatch run align.pose --scene droid2
#   scripts/slurm/submit.sh scripts/slurm/step.sbatch pipeline --scene droid2 --from align.pose --to objects.refine
#   scripts/slurm/submit.sh scripts/slurm/step.sbatch --cmd "python tools/legacy/align_pose_check.py --scene droid2"
#
# Submit through submit.sh, which adds this machine's partition / QOS / GPU type
# (configs/local.env) and the log file $R2S_WORK_ROOT/logs/r2s_<name>_<jobid>.log.
#
#SBATCH --job-name=r2s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00

set -euo pipefail
cd "${R2S_REPO:-$SLURM_SUBMIT_DIR}"
source scripts/env.sh
echo "[r2s] host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?} work=${R2S_WORK:-default} args=$*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

if [[ "${1:-}" == "--cmd" ]]; then
  shift
  eval "$*"
else
  "$R2S_PYTHON" -m r2s "$@"
fi
