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
- WCSS: `bem2-cpu-short`, 1 core per task, results in PD
  (`/lustre/pd03/<account>/pareto_marl`), wandb offline by default.
