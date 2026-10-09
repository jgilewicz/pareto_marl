# pareto-marl

Is there an actuator count at which splitting the body into MAPPO agents
learns better than one agent? Stage 0 on ManySegmentAnt (8–64 actuators),
design and pre-registered GO / NO-GO rule in `plan_stage0.md`.

- v1/v2 (Ant-v5, 8 actuators, torch, 23 partitions): NO-GO, records and
  reports in `experiments/`; code reproducible from commit `52a7a0a` (v2)
  and `a3d54a7` (v1)

## Layout

- functional JAX, no classes for logic: `envs/`, `networks/`, `losses/`,
  `training/`, `utils/`
- `pareto_marl/design.py` — stage-0 design: `SIZES` n_segs 2/4/8/16,
  `CONDITIONS` single/segment/leg/joint, `SEEDS` 0–4, T = 3M;
  program index = 4 · size index + condition index (0..15)
- `configs/stage0_indices.json` — per program: n_segs, actuators,
  condition, partition key, K, actor params, per-agent obs and action
  indices (from MaMuJoCo, built once by `just indices`, committed)
- `pareto_marl/stage0.py` — CLI
  - `indices` — writes `configs/stage0_indices.json`
  - `run --index i --out d --run-id r [--total-steps]` — program i
    (default `$SLURM_ARRAY_TASK_ID`), seeds 0–4 in one `jit(vmap(train))`;
    one JSON per seed, then one wandb run per seed
  - `bench --index i [--iterations 3]` — compile time of the full program
    and env steps/s of `iterations` training iterations (no eval), plus an
    estimate of the hours for T; prints one JSON line, writes nothing
  - `log <json>...` — (re)log result JSONs to wandb
- result JSON `stage0-n{n_segs}-{condition}-s{seed}.json`: n_segs,
  actuators, condition, partition_key, k, actor_params, seed, seeds (the
  vmap batch), eval_return, eval_return_std, eval_x_velocity,
  learning_curve `[[global_step, mean train return], ...]`, steps_per_s
  (whole program, 5 seeds), compile_s, train_s, wall_s, device, run_id,
  git_commit, config
- wandb: project `marl-partition-moo`, group = run_id, job_type `stage0`,
  name `stage0-n{n_segs}-{condition}-s{seed}`, `train/episodic_return` vs
  `global_step`, eval metrics in the summary
- `pareto_marl/analysis.py` — plan §1: one-sided Welch split > single per
  n_segs × split (12), Holm, GO iff a comparison at ≥ 32 actuators has
  p_holm < 0.05; writes `verdict.json`, `stage0.csv`, `scaling.png`,
  `curves.png`
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
- `pareto_marl/partition.py`
  - `structured_partitions(n_segs)`: `single`, `segment`, `leg`, `joint`
  - `make_many_segment_env(x)`, `PartitionSpec.build(x)`: MaMuJoCo agent
    obs/action indices, with the ManySegmentAnt nodes fixed
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
  program for several seeds + compile time and steps/s, `benchmark`
  - output: eval mean/std raw return, eval `x_velocity`, per-iteration
    sum/count of finished episodes' raw returns, mean losses per iteration
- `slurm/stage0.sbatch` — one array task per program, `run` or `bench`
- `gates/` — port gates, run once from the repo root, output goes in the PR
  - `uv run python gates/physics.py` — MJX vs Ant-v5 (MuJoCo C), n_segs 2, 16
  - `uv run python gates/indices.py` — agent indices vs MuJoCo names
  - `uv run python gates/throughput.py` — `jit(vmap(step))`, 8 envs

## Local

- `uv sync` — JAX CPU on macOS, `jax[cuda12]` on Linux
- `just check` — ruff, format, ty, shellcheck, shfmt
- `just indices` — rebuild `configs/stage0_indices.json` (commit it)
- `just smoke` — 2 iterations of program 0, no wandb
- `just bench [index] [iterations]` — compile time and env steps/s
- `just analyze <results_dir>` — prints the table, the 12 comparisons and
  GO / NO-GO

## WCSS (grant `hpc-danbor2008-1756464546`, set in `justfile`)

- repo in `~/workspace/pareto_marl`, `just venv` builds `.venv` there
  (PD of this grant is near its file quota, so no venv on PD); the login
  node has no AVX: build the venv there, never import
- `slurm/stage0.sbatch`: `lem-gpu`, 1 × H100, 4 cores, 32 GB, 12 h default
- `just bench-wcss [array] [iterations]` — GPU throughput gate, one task per
  program, JSON line per task in `logs/`
- `just smoke-wcss` — programs 0 and 15 at 2 iterations, to PD `results/smoke`
- `just submit <run_id> [time]` — 16 tasks, wandb online
- results: `/lustre/pd03/hpc-danbor2008-1756464546/pareto_marl/results/<run_id>/`
  (80 JSONs, copied from `$TMPDIR` by the EXIT trap)
- wandb failure after training: JSONs are written first; the task exits 1,
  re-log with `python -m pareto_marl.stage0 log <json>...`
