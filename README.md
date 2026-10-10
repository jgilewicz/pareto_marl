# pareto-marl

Is there an actuator count at which splitting the body into MAPPO agents
learns better than one agent? Stage 0 on ManySegmentAnt (8–64 actuators),
design and pre-registered GO / NO-GO rule in `plan_stage0.md`.

- v1/v2 (Ant-v5, 8 actuators, torch, 23 partitions): NO-GO, records and
  reports in `experiments/`; code reproducible from commit `52a7a0a` (v2)
  and `a3d54a7` (v1)

## Stage-0 runner of record: CPU

- MuJoCo C + torch MAPPO (v2 algorithm), 80 independent one-core runs in
  one SLURM array (`slurm/stage0_cpu.sbatch`, `just submit-cpu`)
- MJX/GPU path (below) is too slow at 8 envs per seed: latency-bound,
  sequential PPO updates; the s0v1 run was cancelled after 12.7 h with no
  results. MuJoCo C on one M4 core (random actions): 8064 / 4883 / 2289 /
  1542 env steps/s for n_segs 2 / 4 / 8 / 16. Code kept, not deleted.
- `pareto_marl/mappo_torch.py` — v2 MAPPO (`52a7a0a`) for any actuator
  count: `Agents(obs_idx, act_idx)` per-agent index lists, CleanRL wrappers,
  `SAME_STEP` autoreset, critic built first, per-agent ratios summed, one
  Adam, one global clip, v2 hyperparameters, T = 3M; checks at start that
  the indices cover the env's obs dim and own each actuator once
- env: `gymnasium.make("Ant-v5", xml_file=<gen_asset(n_segs)>,
  include_cfrc_ext_in_observation=False)`, other Ant-v5 defaults (reward
  keeps the contact cost); XML written to the process's own temp dir
- `pareto_marl/stage0_cpu.py` — CLI
  - `run --index i --out d --run-id r [--total-steps]` — task i ∈ 0..79
    (default `$SLURM_ARRAY_TASK_ID`): program i // 5 of
    `configs/stage0_indices.json`, seed i % 5; one JSON, then wandb
  - `log <json>...` — (re)log result JSONs to wandb
- result JSON `stage0-n{n_segs}-{condition}-s{seed}.json` (read by
  `analysis.py`): n_segs, actuators, condition, partition_key, k,
  actor_params, seed, eval_return, eval_return_std, eval_x_velocity,
  learning_curve `[[global_step, mean train return], ...]` per iteration,
  steps_per_s (train loop), compile_s (0), train_s, wall_s, device,
  run_id, git_commit, config
- wandb: `marl-partition-moo`, group = run_id, job_type `stage0-cpu`, name
  `stage0cpu-n{n_segs}-{condition}-s{seed}`
- local full-loop env steps/s (M4, 1 core, 2 iterations), single /
  segment / leg / joint:
  - n_segs 2: 4077 / 3720 / 3141 / 2440
  - n_segs 4: 3086 / 2486 / 1979 / 1444
  - n_segs 8: 1813 / 1382 / 1082 / 777
  - n_segs 16: 1255 / 867 / 638 / 438
- `--time=24:00:00`: n_segs 16 joint 1.9 h on M4, Bem2 ~5× slower per core
  (v2: 1.15 h per 3M-step run) → ~9.5 h, >2× margin; whole array
  ~52 core-h on M4 → ~260 CPU-h on Bem2

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
    and env steps/s of `iterations` full training iterations (rollout +
    update, no eval), estimated hours for T, fails on non-finite params;
    prints one JSON line, writes nothing
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
- `slurm/stage0.sbatch` — one H100 per array task; `run`/`bench`: array
  index = n_segs index, its 4 programs as 4 concurrent processes on the
  GPU (env is latency-bound); `run-one`/`bench-one`: array index = program
- `gates/` — port gates, run once from the repo root, output goes in the PR
  - `uv run python gates/physics.py` — MJX vs Ant-v5 (MuJoCo C), n_segs 2, 16
  - `uv run python gates/indices.py` — agent indices vs MuJoCo names
  - `uv run python gates/throughput.py` — `jit(vmap(step))`, 8 envs

## Local

- `uv sync` — JAX CPU on macOS, `jax[cuda12]` on Linux
- `just check` — ruff, format, ty, shellcheck, shfmt
- `just smoke-cpu` — 2 iterations of tasks 0 and 75, no wandb
- `just indices` — rebuild `configs/stage0_indices.json` (commit it)
- `just smoke` — 2 iterations of program 0, no wandb
- `just bench [index] [iterations]` — compile time and env steps/s
- `just analyze <results_dir>` — prints the table, the 12 comparisons and
  GO / NO-GO

## WCSS (grant `hpc-danbor2008-1756464546`, set in `justfile`)

- repo in `~/workspace/pareto_marl`, `just venv` builds `.venv` there
  (PD of this grant is near its file quota, so no venv on PD); the login
  node has no AVX: build the venv there, never import
- `slurm/stage0_cpu.sbatch`: `bem2-cpu-short`, 1 core, 4 GB, 24 h;
  `OMP_NUM_THREADS=1`, results in `$TMPDIR` copied by the EXIT trap to
  PD `results/<run_id>/`, logs `logs/stage0cpu_<job>_<task>.out`
- `just submit-cpu <run_id> [time=24:00:00]` — tasks 0–79, wandb online
- `just rerun-cpu <run_id> <tasks> [time]` — single tasks, same run_id
- MJX/GPU path, kept for reference:
- `slurm/stage0.sbatch`: `lem-gpu`, 1 × H100, 4 cores, 64 GB, 12 h default;
  `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `MEM_FRACTION=0.22` so 4 JAX
  processes share the GPU; per-program logs `logs/stage0_<job>_<task>_p<i>.out`;
  any failed program fails the task, results still copied
- `just bench-wcss [array] [iterations]` — GPU gate, 4 packed programs per
  n_segs, one JSON line per program
- `just smoke-wcss` — n_segs 2 and 16 packed, 2 iterations, to PD
  `results/smoke`
- `just submit <run_id> [array] [time=60:00:00]` — 4 tasks (one per
  n_segs), wandb online; `lem-gpu` resolves to `lem-gpu-short` (3 d cap)
- `just rerun <run_id> <programs> [time=60:00:00]` — single programs
  (`run-one`)
- no checkpoint: JSONs are written only when training ends, so a task that
  hits its time limit loses all its programs; size `time` from the bench
- results: `/lustre/pd03/hpc-danbor2008-1756464546/pareto_marl/results/<run_id>/`
  (80 JSONs, copied from `$TMPDIR` by the EXIT trap)
- wandb failure after training: JSONs are written first; the task exits 1,
  re-log with `python -m pareto_marl.stage0 log <json>...`
