#!/usr/bin/env bash
# One preview / settle run in the sim (r2s.sim.settle), on a GPU node.
#
#   scripts/slurm/submit.sh scripts/slurm/sim_settle.sbatch --scene droid820 --out RUN/objects/refine/stock --mode free --steps 150
#   R2S_WORK=/path/to/run scripts/slurm/submit.sh scripts/slurm/sim_settle.sbatch --scene droid820 \
#       --out DIR --candidate DIR/candidate.json --dump DIR/object_poses.json --mode upright --steps 150 --hold-steps 20
#
# The refine stages themselves go through the generic step job:
#   scripts/slurm/submit.sh scripts/slurm/step.sbatch run objects.refine --scene droid820 --stage stock-settle
#
# Submit through submit.sh, which adds this machine's partition / QOS / GPU type
# (configs/local.env) and the log file $R2S_WORK_ROOT/logs/r2s_<name>_<jobid>.log.
#
#SBATCH --job-name=r2s_settle
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00

set -euo pipefail
cd "${R2S_REPO:-$SLURM_SUBMIT_DIR}"
source scripts/env.sh
echo "[r2s] host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?} work=${R2S_WORK:-default} args=$*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
"$R2S_PYTHON" -m r2s.sim.settle "$@"
