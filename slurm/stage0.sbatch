#!/bin/bash
# stage 0 on one H100 per array task; submit from the repo root (venv in
# .venv, logs in logs/), account with -A. Modes:
#   run       array index = n_segs index 0..3: its 4 conditions (programs
#             4i..4i+3) run as 4 concurrent processes on the one GPU
#   run-one   array index = program 0..15, one process (reruns)
#   bench     as run, prints one JSON line per program, writes no results
#   bench-one as run-one
#   sbatch -A <account> --array=0-3 slurm/stage0.sbatch run <run_id> [steps]
#   sbatch -A <account> --array=0-3 --time=0:30:00 \
#     slurm/stage0.sbatch bench [iterations]
#SBATCH --job-name=stage0
#SBATCH --partition=lem-gpu
#SBATCH --gres=gpu:hopper:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail

usage="usage: stage0.sbatch run|run-one <run_id> [total_steps]"
usage+=" | bench|bench-one [iterations]"
mode="${1:?$usage}"
index="${SLURM_ARRAY_TASK_ID:?submit with --array}"
repo="${SLURM_SUBMIT_DIR:?run under sbatch}"

export PYTHONNOUSERSITE=1
export WANDB_DIR="$TMPDIR"
# the env is latency-bound, so 4 programs share one GPU: allocate on demand
# and cap each process so 4 fit
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.22
# shellcheck source=/dev/null  # venv is built on the cluster by uv sync
source "$repo/.venv/bin/activate"
cd "$repo"

echo "task $mode $index on $(hostname) commit=$(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# shellcheck disable=SC2329  # invoked by the EXIT trap in run modes
copy_out() {
  if compgen -G "$TMPDIR/results/*.json" >/dev/null; then
    mkdir -p "$out"
    cp "$TMPDIR"/results/*.json "$out/"
  fi
  echo "finished $(date -Is) -> $out"
}

programs=()
case "$mode" in
run | bench)
  # 4 conditions per n_segs, program index = 4 * size index + condition
  for condition in 0 1 2 3; do
    programs+=("$((4 * index + condition))")
  done
  ;;
run-one | bench-one)
  programs=("$index")
  ;;
*)
  echo "unknown mode '$mode': $usage" >&2
  exit 2
  ;;
esac

case "$mode" in
run | run-one)
  run_id="${2:?$usage}"
  # PD of this grant is near its file quota: only result JSONs go there
  out="/lustre/pd03/${SLURM_JOB_ACCOUNT}/pareto_marl/results/$run_id"
  trap copy_out EXIT
  mkdir -p "$TMPDIR/results"
  args=(run --run-id "$run_id" --total-steps "${3:-3000000}"
    --out "$TMPDIR/results")
  ;;
bench | bench-one)
  args=(bench --iterations "${2:-3}")
  ;;
esac

pids=()
logs=()
for program in "${programs[@]}"; do
  log="logs/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${index}_p${program}.out"
  python -m pareto_marl.stage0 "${args[@]}" --index "$program" >"$log" 2>&1 &
  pids+=("$!")
  logs+=("$log")
done

status=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "program ${programs[$i]} failed, see ${logs[$i]}" >&2
    status=1
  fi
  tail -n 5 "${logs[$i]}"
done
exit "$status"
