# pareto-marl

- Goal: GO / NO-GO on whether the actuator partition matters (plan.md §3).
- Python 3.13, `uv`, exact pins; torch from the PyTorch CPU index.
- No tests by decision of the user; verify with `just check` and `just smoke`.
- `mappo.py` mirrors CleanRL `ppo_continuous_action`; for K=1 it must stay
  equivalent (same wrappers, SAME_STEP autoreset, critic built first).
- Per-agent PPO ratios and clipping, summed over agents; one Adam, one
  global grad-norm clip.
- `configs/design.json` is generated once (`just design`) and committed; array
  task i → partition i // 5, seed i % 5.
- `gymnasium_robotics` prints an Adroit notice on import; `partition.py`
  silences it with `redirect_stderr`.
- `reference/` is vendored CleanRL, excluded from ruff and ty.
- JAX/MJX port (stage 0): functional JAX, no classes for logic; flax linen
  (pure `init`/`apply`, params are pytrees, `vmap` over agents and seeds).
- Every JAX env implements `envs/contract.py`: `reset(spec, key)`,
  `step(spec, state, action, key)` → `EnvState`, single env; build the result
  of `step` with `finish_step` so truncation, episode stats and autoreset
  stay identical across envs.
- Autoreset = gymnasium `SAME_STEP`: the final step returns its own reward,
  metrics and `done=1`, but the obs of the new episode. `done` = terminated
  or truncated (no bootstrapping on truncation, as CleanRL).
- `terminated` is exposed separately: gymnasium `NormalizeReward` resets its
  discounted return on `terminated` only, not on truncation.
- Env `step` clips actions to [-1, 1] (CleanRL `ClipAction`); obs and reward
  are raw, normalization and reward scaling are the trainer's job.
- `returned_episode_return/length` are valid only where
  `returned_episode == 1` (raw reward, like `RecordEpisodeStatistics`).
- `import mujoco.mjx` prints `Failed to import warp` to stdout (unused warp
  backend); silence it with `contextlib.redirect_stdout`.
- jax is pinned per platform: CPU on darwin, `jax[cuda12]` on linux (H100).
- WCSS: grant `hpc-danbor2008-1756464546`, `bem2-cpu-short`, 1 core per
  task, wandb online. Venv in the repo `.venv` in `$HOME`: the grant's PD is
  near its file quota, only result JSONs go to PD.
- ManySegmentAnt: gymnasium-robotics 1.4.2 `get_parts_and_edges` has wrong
  qpos/qvel ids (all but the last segment point into the root joint),
  act_ids that swap the two legs of a segment, and deepcopied
  inter-segment edges (one-way, duplicated neighbours); `partition.py`
  `many_segment_graph` fixes the nodes and edges before passing them as
  `agent_factorization`. `gates/indices.py` checks against MuJoCo names.
- `many_segment_ant.py` reads `x_velocity` from `xpos` of `torso_0` like
  `Ant-v5` (no `mj_forward` after the step: xpos of the last RK4 stage), so
  `reset` runs one `mjx.forward`.
- Pyramidal cone kept (`Ant-v5`): MJX matches MuJoCo C exactly until the
  first contact, then diverges; MJX and C pick different tangent frames for
  capsule–plane contacts, so the friction pyramid rotates. With elliptic
  cones both agree (gate 1), so the integration itself matches.
- Port gates are scripts in `gates/`, run once, output in the PR (no pytest).
- JAX MAPPO = torch `mappo.py` (v2); actors padded to the partition max:
  padded obs inputs are zeroed, padded action slots are masked in
  log-prob/entropy and never gathered into the env action, so padded
  params get zero grads and stay zero. Init each actor at its true size,
  then zero-pad (orthogonal init of the padded shape differs).
- Adv. normalization uses `std(ddof=1)` (torch `.std()`).
- Obs stats skip gymnasium's update with the terminal obs on done steps:
  the contract exposes only the reset obs.
- A seed's result depends on the vmap batch it runs in (ulp-level diffs in
  batched matmuls grow chaotically); the same call is bitwise reproducible
  on CPU, so keep the seed set per compiled program fixed.
