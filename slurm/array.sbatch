#!/bin/bash
# submit from the repo root (venv in .venv, outputs in logs/):
#   sbatch -A <account> --array=0-114 slurm/array.sbatch mappo <run_id>
#   sbatch -A <account> --array=0-4   slurm/array.sbatch cleanrl <run_id>
#SBATCH --job-name=pareto-marl
#SBATCH --partition=bem2-cpu-short
#SBATCH --cpus-per-task=1
#SBATCH --mem=4gb
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail

mode="${1:?usage: array.sbatch <mappo|cleanrl> <run_id> [total_steps]}"
run_id="${2:?usage: array.sbatch <mappo|cleanrl> <run_id> [total_steps]}"
total_steps="${3:-1000000}"
index="${SLURM_ARRAY_TASK_ID:?submit with --array}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONNOUSERSITE=1
export WANDB_DIR="$TMPDIR"

repo="${SLURM_SUBMIT_DIR:?run under sbatch}"
# PD of this service is near its file quota: only result JSONs go there
out="/lustre/pd03/${SLURM_JOB_ACCOUNT}/pareto_marl/results/$run_id"
# shellcheck source=/dev/null  # venv is built on the cluster by uv sync
source "$repo/.venv/bin/activate"

copy_out() {
  mkdir -p "$out"
  cp "$TMPDIR"/results/*.json "$out/" 2>/dev/null || true
  echo "finished $(date -Is) -> $out"
}
trap copy_out EXIT

echo "task $mode $index run_id=$run_id commit=$(git rev-parse --short HEAD)"
mkdir -p "$TMPDIR/results"

case "$mode" in
mappo)
  python -m pareto_marl.run --index "$index" --run-id "$run_id" \
    --total-steps "$total_steps" --out "$TMPDIR/results"
  ;;
cleanrl)
  cd "$TMPDIR"
  python "$repo/reference/cleanrl_ppo_continuous_action.py" \
    --env-id Ant-v5 --seed "$index" --total-timesteps "$total_steps" \
    --num-envs 8 --num-steps 256 --no-cuda --track \
    --wandb-project-name marl-partition-moo --wandb-group "$run_id" \
    --curve-path "$TMPDIR/results/cleanrl-s$index.json"
  ;;
*)
  echo "unknown mode '$mode': expected mappo or cleanrl" >&2
  exit 2
  ;;
esac
