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
- `pareto_marl/report.py` + `report_template.html` — self-contained HTML report
- `reference/cleanrl_ppo_continuous_action.py` — CleanRL ported to gymnasium 1.x
- `slurm/array.sbatch` — array task, `mappo` or `cleanrl` mode

### JAX/MJX port (stage 0, `plan_stage0.md`)

- functional JAX, no classes for logic: `envs/`, `networks/`, `losses/`,
  `training/`, `utils/`
- `pareto_marl/envs/contract.py` — env contract shared by all JAX envs
  - `EnvSpec`: static `obs_dim`, `action_dim`, `episode_length` + `params`
    (env-specific pytree, e.g. `mjx.Model`)
  - `EnvState`: opaque `physics`, raw float32 `obs`, `reward`, `done`,
    `metrics` (`x_velocity`), episode return/length bookkeeping
  - `reset(spec, key)`, `step(spec, state, action, key)`: single env, the
    trainer batches with `jax.vmap`
  - `finish_step`: truncation, bookkeeping, gymnasium `SAME_STEP` autoreset
- `pareto_marl/envs/dummy.py` — damped point mass on the contract, for
  developing the trainer without MJX

## Local

- `uv sync` — JAX CPU on macOS, `jax[cuda12]` on Linux
- `just check` — ruff, format, ty
- `just smoke` — 20k-step run of task 0

## WCSS (grant `hpc-danbor2008-1756464546`, set in `justfile`)

- repo in `~/workspace/pareto_marl`, `just venv` builds `.venv` there
  (PD of this grant is near its file quota, so no venv on PD)
- `just smoke-wcss` — 3 MAPPO + 1 CleanRL tasks at 50k steps
- `just submit <run_id>` — 115 MAPPO + 5 CleanRL tasks, wandb online
- results: `/lustre/pd03/hpc-danbor2008-1756464546/pareto_marl/results/<run_id>/`
- `just analyze <results_dir>` — writes `verdict.json`, `partitions.csv`,
  `front.png`, prints GO / NO-GO
- `just report <results_dir> <out.html>` — HTML report
