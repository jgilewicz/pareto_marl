# pareto-marl

- Goal: GO / NO-GO of stage 0 (`plan_stage0.md` §1, pre-registered: do not
  change the rule in `analysis.py`).
- Python 3.13, `uv`, exact pins.
- No tests by decision of the user; verify with `just check`, `just smoke`
  and the scripts in `gates/`.
- v1/v2 (torch, Ant-v5) are gone from the tree; reproducible from commit
  `52a7a0a` (v2) / `a3d54a7` (v1); `experiments/` keeps their records.
- `configs/stage0_indices.json` is built once locally (`just indices`) and
  committed. MaMuJoCo writes and deletes its XML inside site-packages, so
  concurrent jobs must never call it: the runner only reads the JSON.
- Program index i → n_segs `SIZES[i // 4]`, condition `CONDITIONS[i % 4]`
  (`design.py`); seeds 0–4 always run together in one `jit(vmap(train))`.
- `stage0 run` writes every seed's JSON before any wandb call; a wandb
  failure exits 1 with the JSONs intact (`stage0 log` re-logs them).
- `analysis.py` imports only `design.py`, not the JAX runner.
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
- `gymnasium_robotics` prints an Adroit notice on import; silence it with
  `contextlib.redirect_stderr`.
- `import mujoco.mjx` prints `Failed to import warp` to stdout (unused warp
  backend); silence it with `contextlib.redirect_stdout`.
- jax is pinned per platform: CPU on darwin, `jax[cuda12]` on linux (H100).
- WCSS: grant `hpc-danbor2008-1756464546`, `lem-gpu`, `--gres=gpu:hopper:1`
  (only accepted form), `--cpus-per-task` stated and ≤ 16 (4: the GPU does
  the work, cores are billed), wandb online. Venv in the repo `.venv` in
  `$HOME`: the grant's PD is near its file quota, only result JSONs go to
  PD. The login node has no AVX: never import JAX there.
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
- MJX solver budget: `load_model` sets `opt.iterations=10`,
  `ls_iterations=20` (XML: 100/50). Float32 Newton never reaches the XML
  tolerance 1e-8, so MJX always ran all 100/50. One-step qvel vs 100/50 over
  100 random steps: max |Δ| 1.9e-6 (n_segs 2), 1.3e-5 (16); 4/8 deviates
  (4.5 at 16). ~2× env steps/s on CPU.
- JAX MAPPO = torch `mappo.py` of v2 (`52a7a0a`); actors padded to the partition max:
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
- Port gates are scripts in `gates/`, run once, output in the PR (no pytest).
