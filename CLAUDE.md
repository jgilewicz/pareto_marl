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
