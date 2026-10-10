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

# 2 iterations of task 0 (n_segs 2 single) and 75 (n_segs 16 joint), no wandb
smoke-cpu:
    for i in 0 75; do WANDB_MODE=disabled uv run python -m \
        pareto_marl.stage0_cpu run --index $i --run-id smoke \
        --total-steps 4096 --out results/smoke; done

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

# stage 0 of record: 80 one-core runs, task i = program i // 5, seed i % 5
submit-cpu run_id time="24:00:00":
    mkdir -p logs
    sbatch -A {{account}} --array=0-79 --time={{time}} \
        slurm/stage0_cpu.sbatch {{run_id}}

# single tasks (0..79, e.g. "17,75") into the same run_id
rerun-cpu run_id tasks time="24:00:00":
    mkdir -p logs
    sbatch -A {{account}} --array={{tasks}} --time={{time}} \
        slurm/stage0_cpu.sbatch {{run_id}}

# --- MJX/GPU path: too slow at 8 envs per seed, kept for reference ---

# GPU gate: per n_segs its 4 programs packed on one GPU, JSON lines in logs/
bench-wcss array="0-3" iterations="3":
    mkdir -p logs
    sbatch -A {{account}} --array={{array}} --time=0:30:00 \
        slurm/stage0.sbatch bench {{iterations}}

# n_segs 2 and 16 packed, 2 iterations, JSON to PD results/smoke
smoke-wcss:
    mkdir -p logs
    sbatch -A {{account}} --array=0,3 --time=1:00:00 \
        slurm/stage0.sbatch run smoke 4096

# array index = n_segs index; set time per n_segs from bench-wcss
submit run_id array="0-3" time="60:00:00":
    mkdir -p logs
    sbatch -A {{account}} --array={{array}} --time={{time}} \
        slurm/stage0.sbatch run {{run_id}}

# single programs (0..15, e.g. "6,13"), one process per GPU
rerun run_id programs time="60:00:00":
    mkdir -p logs
    sbatch -A {{account}} --array={{programs}} --time={{time}} \
        slurm/stage0.sbatch run-one {{run_id}}

results run_id:
    ls {{pd}}/results/{{run_id}} | wc -l
