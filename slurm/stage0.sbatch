#!/bin/bash
# one array task = one stage-0 program (5 seeds); submit from the repo root
# (venv in .venv, logs in logs/), account with -A:
#   sbatch -A <account> --array=0-15 slurm/stage0.sbatch run <run_id> [steps]
#   sbatch -A <account> --array=0-15 --time=0:30:00 \
#     slurm/stage0.sbatch bench [iterations]
#SBATCH --job-name=stage0
#SBATCH --partition=lem-gpu
#SBATCH --gres=gpu:hopper:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail

usage="usage: stage0.sbatch run <run_id> [total_steps] | bench [iterations]"
mode="${1:?$usage}"
index="${SLURM_ARRAY_TASK_ID:?submit with --array}"
repo="${SLURM_SUBMIT_DIR:?run under sbatch}"

export PYTHONNOUSERSITE=1
export WANDB_DIR="$TMPDIR"
# shellcheck source=/dev/null  # venv is built on the cluster by uv sync
source "$repo/.venv/bin/activate"
cd "$repo"

echo "task $mode $index on $(hostname) commit=$(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

case "$mode" in
run)
  run_id="${2:?$usage}"
  total_steps="${3:-3000000}"
  # PD of this grant is near its file quota: only result JSONs go there
  out="/lustre/pd03/${SLURM_JOB_ACCOUNT}/pareto_marl/results/$run_id"
  copy_out() {
    if compgen -G "$TMPDIR/results/*.json" >/dev/null; then
      mkdir -p "$out"
      cp "$TMPDIR"/results/*.json "$out/"
    fi
    echo "finished $(date -Is) -> $out"
  }
  trap copy_out EXIT
  mkdir -p "$TMPDIR/results"
  python -m pareto_marl.stage0 run --index "$index" --run-id "$run_id" \
    --total-steps "$total_steps" --out "$TMPDIR/results"
  ;;
bench)
  python -m pareto_marl.stage0 bench --index "$index" \
    --iterations "${2:-3}"
  ;;
*)
  echo "unknown mode '$mode': $usage" >&2
  exit 2
  ;;
esac
