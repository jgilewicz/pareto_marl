account := "hpc-danbor2008-1756464546"
pd := "/lustre/pd03/" + account + "/pareto_marl"

check:
    uv run ruff check .
    uv run ruff format --check .
    uv run ty check
    shellcheck slurm/*.sbatch
    shfmt -i 2 -d slurm/*.sbatch

# agent indices from MaMuJoCo, run locally and commit (never on the cluster)
indices:
    uv run python -m pareto_marl.stage0 indices

# 2 iterations of program 0 (n_segs 2, single, 5 seeds), no wandb
smoke:
    WANDB_MODE=disabled uv run python -m pareto_marl.stage0 run --index 0 \
        --run-id smoke --total-steps 4096 --out results/smoke

# compile time and env steps/s of one program, writes nothing
bench index="15" iterations="3":
    uv run python -m pareto_marl.stage0 bench --index {{index}} \
        --iterations {{iterations}}

analyze dir:
    uv run python -m pareto_marl.analysis {{dir}}

# --- WCSS login node, from the repo root in $HOME ---

venv:
    uv sync --frozen --no-dev

# GPU throughput gate: one bench task per program, output in logs/
bench-wcss array="0-15" iterations="3":
    mkdir -p logs
    sbatch -A {{account}} --array={{array}} --time=0:30:00 \
        slurm/stage0.sbatch bench {{iterations}}

# smallest and largest program, 2 iterations, JSON to PD results/smoke
smoke-wcss:
    mkdir -p logs
    sbatch -A {{account}} --array=0,15 --time=0:30:00 \
        slurm/stage0.sbatch run smoke 4096

# set time from the bench-wcss estimate (slowest program + margin)
submit run_id time="12:00:00":
    mkdir -p logs
    sbatch -A {{account}} --array=0-15 --time={{time}} \
        slurm/stage0.sbatch run {{run_id}}

results run_id:
    ls {{pd}}/results/{{run_id}} | wc -l
