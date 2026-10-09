# pareto-marl

Does the partition of Ant's 8 actuators into MAPPO agents change the return
more than seed noise? GO / NO-GO experiment, design in `plan.md`.

## Layout

- `pareto_marl/partition.py` — canonical partitions, enumeration, sampling,
  `PartitionSpec` (obs/action indices from MaMuJoCo, actor param count)
- `pareto_marl/design.py` — writes `configs/design.json` (23 partitions × 5 seeds)
- `pareto_marl/mappo.py` — MAPPO = CleanRL PPO with K actors, shared critic
- `pareto_marl/run.py` — one SLURM array task → JSON + wandb run
- `pareto_marl/analysis.py` — ANOVA + ICC per K, verdict, table, plot
- `reference/cleanrl_ppo_continuous_action.py` — CleanRL ported to gymnasium 1.x
- `slurm/array.sbatch` — array task, `mappo` or `cleanrl` mode

## Local

- `uv sync`
- `just check` — ruff, format, ty
- `just smoke` — 20k-step run of task 0

## WCSS

- once, on the login node: install `uv`, clone into `$HOME`, `just venv <account>`
- `just smoke-wcss <account>` — 3 MAPPO + 1 CleanRL tasks at 50k steps
- `just submit <account> <run_id>` — 115 MAPPO + 5 CleanRL tasks
- results: `/lustre/pd03/<account>/pareto_marl/results/<run_id>/*.json`
- `just sync <account> <run_id>` — upload offline wandb runs
- `just analyze <results_dir>` — writes `verdict.json`, `partitions.csv`,
  `front.png`, prints GO / NO-GO
