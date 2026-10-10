#!/bin/bash
# stage 0 on CPU, one core per run; submit from the repo root (venv in
# .venv, logs in logs/), account with -A. Array index = task 0..79:
# program index // 5 (configs/stage0_indices.json), seed index % 5.
#   sbatch -A <account> --array=0-79 slurm/stage0_cpu.sbatch <run_id> [steps]
# --time: n_segs 16 joint (K=64) ran 438 env steps/s on one M4 core, i.e.
# 1.9 h for 3M steps; Bem2 is ~5x slower per core (v2: 1.15 h per 3M-step
# run), so ~9.5 h; 24 h leaves a >2x margin
#SBATCH --job-name=stage0cpu
#SBATCH --partition=bem2-cpu-short
#SBATCH --cpus-per-task=1
#SBATCH --mem=4gb
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail

usage="usage: stage0_cpu.sbatch <run_id> [total_steps]"
run_id="${1:?$usage}"
total_steps="${2:-3000000}"
index="${SLURM_ARRAY_TASK_ID:?submit with --array}"
repo="${SLURM_SUBMIT_DIR:?run under sbatch}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONNOUSERSITE=1
export WANDB_DIR="$TMPDIR"
# shellcheck source=/dev/null  # venv is built on the cluster by uv sync
source "$repo/.venv/bin/activate"
cd "$repo"

# PD of this grant is near its file quota: only result JSONs go there
out="/lustre/pd03/${SLURM_JOB_ACCOUNT}/pareto_marl/results/$run_id"

# shellcheck disable=SC2329  # invoked by the EXIT trap
copy_out() {
  if compgen -G "$TMPDIR/results/*.json" >/dev/null; then
    mkdir -p "$out"
    cp "$TMPDIR"/results/*.json "$out/"
  fi
  echo "finished $(date -Is) -> $out"
}
trap copy_out EXIT

echo "task $index on $(hostname) run_id=$run_id" \
  "commit=$(git rev-parse --short HEAD)"
mkdir -p "$TMPDIR/results"
python -m pareto_marl.stage0_cpu run --index "$index" --run-id "$run_id" \
  --total-steps "$total_steps" --out "$TMPDIR/results"
