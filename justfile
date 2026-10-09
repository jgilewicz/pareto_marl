results := "results"

check:
    uv run ruff check .
    uv run ruff format --check .
    uv run ty check

design:
    uv run python -m pareto_marl.design

# 20k-step MAPPO run for task 0 (K=1, seed 0), no wandb
smoke:
    WANDB_MODE=disabled uv run python -m pareto_marl.run --index 0 \
        --run-id smoke --total-steps 20480 --out {{results}}/smoke

analyze dir:
    uv run python -m pareto_marl.analysis {{dir}}

# --- WCSS login node, run from the repo in $HOME ---

venv account:
    UV_PROJECT_ENVIRONMENT=/lustre/pd03/{{account}}/pareto_marl/venv \
        uv sync --frozen --no-dev

# first, middle and last task at 50k steps
smoke-wcss account:
    mkdir -p logs
    sbatch -A {{account}} --array=0,57,114 --time=0:30:00 \
        slurm/array.sbatch mappo smoke 51200
    sbatch -A {{account}} --array=0 --time=0:30:00 \
        slurm/array.sbatch cleanrl smoke 51200

submit account run_id:
    mkdir -p logs
    sbatch -A {{account}} --array=0-114 slurm/array.sbatch mappo {{run_id}}
    sbatch -A {{account}} --array=0-4 slurm/array.sbatch cleanrl {{run_id}}

sync account run_id:
    wandb sync /lustre/pd03/{{account}}/pareto_marl/wandb/{{run_id}}/offline-run-*
