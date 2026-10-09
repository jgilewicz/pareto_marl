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
  - `EnvState`: opaque `physics`, raw float32 `obs`, `reward`,
    `terminated`, `done` (terminated or truncated),
    `metrics` (`x_velocity`), episode return/length bookkeeping
  - `reset(spec, key)`, `step(spec, state, action, key)`: single env, the
    trainer batches with `jax.vmap`
  - `finish_step`: truncation, bookkeeping, gymnasium `SAME_STEP` autoreset
- `pareto_marl/envs/dummy.py` — damped point mass on the contract, for
  developing the trainer without MJX
- `pareto_marl/envs/many_segment_ant.py` — ManySegmentAnt (MaMuJoCo
  `gen_asset`, 4 actuators per segment) on MJX
  - `make_spec(n_segs)`: `params` = `AntParams` (mjx model, init qpos/qvel)
  - task = `Ant-v5` on that XML without contact cost: frame_skip 5, obs
    `qpos[2:]`, `qvel`, reward x_velocity(`torso_0`) + 1 − 0.5·‖a‖²,
    terminated when z ∉ [0.2, 1] or state not finite
  - MJX solver budget `iterations=10`, `ls_iterations=20` (XML: 100/50)
- `pareto_marl/partition.py` (ManySegmentAnt)
  - `structured_partitions(n_segs)`: `single`, `segment`, `leg`, `joint`
  - `make_many_segment_env(x)`, `PartitionSpec.build_many_segment(x)`:
    MaMuJoCo agent obs/action indices, with the ManySegmentAnt nodes fixed
- `gates/` — port gates, run once from the repo root, output goes in the PR
  - `uv run python gates/physics.py` — MJX vs Ant-v5 (MuJoCo C), n_segs 2, 16
  - `uv run python gates/indices.py` — agent indices vs MuJoCo names
  - `uv run python gates/throughput.py` — `jit(vmap(step))`, 8 envs
- `pareto_marl/networks/actor_critic.py` — K actors `vmap`ped over a
  leading agent axis, obs/actions padded to the partition max and masked;
  shared critic; hand-written per-agent Gaussian log-prob/entropy
  - `build_layout(obs_idx, act_idx, obs_dim, action_dim)` — same index
    lists as `PartitionSpec`; checks each actuator is owned exactly once
- `pareto_marl/losses/ppo.py` — MAPPO loss (per-agent clip, sum over agents)
- `pareto_marl/utils/normalize.py` — gymnasium 1.4 `NormalizeObservation` /
  `NormalizeReward` + clip ±10, per env
- `pareto_marl/training/mappo.py` — `PPOConfig` (v2 values), `train(setup,
  spec, seed)` jit/vmap-able over seeds, `train_seeds` = one compiled
  program for several seeds + compile time and steps/s
  - output: eval mean/std raw return, eval `x_velocity`, per-iteration
    sum/count of finished episodes' raw returns, mean losses per iteration

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
