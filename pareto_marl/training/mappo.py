import dataclasses
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import cast

import jax
import jax.numpy as jnp
import optax
from flax import struct

from pareto_marl.envs.contract import EnvSpec, EnvState, ResetFn, StepFn
from pareto_marl.losses.ppo import Batch, mappo_loss
from pareto_marl.networks.actor_critic import (
    AgentLayout,
    Params,
    actor_mean,
    critic_value,
    init_params,
    log_prob,
    sample_action,
    to_env_action,
)
from pareto_marl.utils.normalize import (
    RewardScaler,
    RunningMeanStd,
    init_reward_scaler,
    init_rms,
    normalize_obs,
    scale_reward,
    update_rms,
)

EVAL_SEED_OFFSET = 10_000


@dataclass(frozen=True)
class PPOConfig:
    total_steps: int = 3_000_000
    num_envs: int = 8
    num_steps: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    update_epochs: int = 10
    num_minibatches: int = 32
    clip_coef: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    max_grad_norm: float = 0.5
    eval_episodes: int = 10

    def __post_init__(self) -> None:
        if self.batch_size % self.num_minibatches:
            raise ValueError(
                f"batch {self.batch_size} = num_envs * num_steps is not "
                f"divisible by num_minibatches={self.num_minibatches}"
            )
        if self.num_iterations < 1:
            raise ValueError(
                f"total_steps={self.total_steps} < one batch "
                f"({self.batch_size}): raise total_steps"
            )

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches

    @property
    def num_iterations(self) -> int:
        return self.total_steps // self.batch_size


@dataclass(frozen=True)
class Setup:
    cfg: PPOConfig
    layout: AgentLayout
    reset: ResetFn
    step: StepFn


@struct.dataclass
class Runner:
    params: Params
    opt_state: optax.OptState
    env_state: EnvState
    obs_rms: RunningMeanStd
    reward_scaler: RewardScaler
    key: jax.Array


@struct.dataclass
class Step:
    obs: jax.Array
    actions: jax.Array
    logprobs: jax.Array
    values: jax.Array
    rewards: jax.Array
    dones: jax.Array
    episode_return: jax.Array
    episode: jax.Array


@struct.dataclass
class TrainOutput:
    eval_return: jax.Array
    eval_return_std: jax.Array
    eval_x_velocity: jax.Array
    # per iteration: sum and count of finished episodes' raw returns
    curve_return_sum: jax.Array
    curve_episodes: jax.Array
    losses: dict[str, jax.Array]


def make_optimizer(cfg: PPOConfig) -> optax.GradientTransformation:
    updates_per_iteration = cfg.update_epochs * cfg.num_minibatches

    # CleanRL anneals once per iteration, constant within it
    def learning_rate(count: jax.typing.ArrayLike) -> jax.Array:
        iteration = jnp.asarray(count) // updates_per_iteration
        return cfg.learning_rate * (1.0 - iteration / cfg.num_iterations)

    return optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adam(learning_rate, eps=1e-5),
    )


def _rollout_step(
    setup: Setup, spec: EnvSpec, runner: Runner
) -> tuple[Runner, Step]:
    cfg, layout, params = setup.cfg, setup.layout, runner.params
    key, act_key, step_key = jax.random.split(runner.key, 3)
    # stats were updated with this obs when it arrived, as in the wrapper
    obs = jax.vmap(normalize_obs)(runner.obs_rms, runner.env_state.obs)
    mean = actor_mean(layout, params, obs)
    action = sample_action(params, mean, act_key)
    env_state = jax.vmap(setup.step, in_axes=(None, 0, 0, 0))(
        spec,
        runner.env_state,
        to_env_action(layout, action),
        jax.random.split(step_key, cfg.num_envs),
    )
    scaler, reward = jax.vmap(scale_reward, in_axes=(0, 0, 0, None))(
        runner.reward_scaler, env_state.reward, env_state.terminated, cfg.gamma
    )
    step = Step(
        obs=obs,
        actions=action,
        logprobs=log_prob(layout, params, mean, action),
        values=critic_value(params, obs),
        rewards=reward,
        dones=env_state.done,
        episode_return=env_state.returned_episode
        * env_state.returned_episode_return,
        episode=env_state.returned_episode,
    )
    runner = dataclasses.replace(
        runner,
        env_state=env_state,
        obs_rms=jax.vmap(update_rms)(runner.obs_rms, env_state.obs),
        reward_scaler=scaler,
        key=key,
    )
    return runner, step


def _gae(cfg: PPOConfig, steps: Step, last_value: jax.Array) -> jax.Array:
    # dones[t] is the done returned by step t (CleanRL's dones[t + 1]):
    # no bootstrap across terminated or truncated
    def backward(
        carry: tuple[jax.Array, jax.Array], t: Step
    ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
        last_gae, next_value = carry
        nonterminal = 1.0 - t.dones
        delta = t.rewards + cfg.gamma * next_value * nonterminal - t.values
        last_gae = delta + cfg.gamma * cfg.gae_lambda * nonterminal * last_gae
        return (last_gae, t.values), last_gae

    init = (jnp.zeros_like(last_value), last_value)
    _, advantages = jax.lax.scan(backward, init, steps, reverse=True)
    return advantages


def _update(
    setup: Setup,
    train_state: tuple[Params, optax.OptState],
    batch: Batch,
    key: jax.Array,
) -> tuple[tuple[Params, optax.OptState], dict[str, jax.Array]]:
    cfg = setup.cfg
    optimizer = make_optimizer(cfg)
    grad_fn = jax.grad(mappo_loss, has_aux=True)

    def minibatch(
        state: tuple[Params, optax.OptState], mb: Batch
    ) -> tuple[tuple[Params, optax.OptState], dict[str, jax.Array]]:
        params, opt_state = state
        grads, stats = grad_fn(params, setup.layout, mb, cfg)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = cast(Params, optax.apply_updates(params, updates))
        return (params, opt_state), stats

    def epoch(
        state: tuple[Params, optax.OptState], epoch_key: jax.Array
    ) -> tuple[tuple[Params, optax.OptState], dict[str, jax.Array]]:
        perm = jax.random.permutation(epoch_key, cfg.batch_size)
        shuffled = jax.tree.map(
            lambda x: x[perm].reshape(
                cfg.num_minibatches, cfg.minibatch_size, *x.shape[1:]
            ),
            batch,
        )
        return jax.lax.scan(minibatch, state, shuffled)

    keys = jax.random.split(key, cfg.update_epochs)
    train_state, stats = jax.lax.scan(epoch, train_state, keys)
    return train_state, jax.tree.map(jnp.mean, stats)


def _iteration(
    setup: Setup, spec: EnvSpec, runner: Runner
) -> tuple[Runner, tuple[jax.Array, jax.Array, dict[str, jax.Array]]]:
    cfg = setup.cfg
    runner, steps = jax.lax.scan(
        lambda r, _: _rollout_step(setup, spec, r),
        runner,
        length=cfg.num_steps,
    )
    obs = jax.vmap(normalize_obs)(runner.obs_rms, runner.env_state.obs)
    advantages = _gae(cfg, steps, critic_value(runner.params, obs))
    batch = jax.tree.map(
        lambda x: x.reshape(cfg.batch_size, *x.shape[2:]),
        Batch(
            obs=steps.obs,
            actions=steps.actions,
            logprobs=steps.logprobs,
            advantages=advantages,
            returns=advantages + steps.values,
            values=steps.values,
        ),
    )
    key, update_key = jax.random.split(runner.key)
    (params, opt_state), stats = _update(
        setup, (runner.params, runner.opt_state), batch, update_key
    )
    runner = dataclasses.replace(
        runner, params=params, opt_state=opt_state, key=key
    )
    curve = (steps.episode_return.sum(), steps.episode.sum())
    return runner, (*curve, stats)


def _eval_episode(
    setup: Setup,
    spec: EnvSpec,
    params: Params,
    obs_rms: RunningMeanStd,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    reset_key, step_key = jax.random.split(key)

    def body(
        carry: tuple[EnvState, jax.Array], step_key: jax.Array
    ) -> tuple[tuple[EnvState, jax.Array], tuple[jax.Array, ...]]:
        state, alive = carry
        obs = normalize_obs(obs_rms, state.obs)
        mean = actor_mean(setup.layout, params, obs)
        action = to_env_action(setup.layout, mean)
        state = setup.step(spec, state, action, step_key)
        out = (alive * state.reward, alive * state.metrics["x_velocity"], alive)
        return (state, alive * (1.0 - state.done)), out

    init = (setup.reset(spec, reset_key), jnp.ones(()))
    keys = jax.random.split(step_key, spec.episode_length)
    _, (rewards, velocities, alive) = jax.lax.scan(body, init, keys)
    return rewards.sum(), velocities.sum() / alive.sum()


def evaluate(
    setup: Setup,
    spec: EnvSpec,
    params: Params,
    obs_rms: RunningMeanStd,
    seed: jax.Array,
) -> dict[str, jax.Array]:
    key = jax.random.key(seed + EVAL_SEED_OFFSET)
    keys = jax.random.split(key, setup.cfg.eval_episodes)
    episode = partial(_eval_episode, setup, spec, params, obs_rms)
    returns, velocities = jax.vmap(episode)(keys)
    return {
        "eval_return": returns.mean(),
        "eval_return_std": returns.std(),
        "eval_x_velocity": velocities.mean(),
    }


def _init_runner(setup: Setup, spec: EnvSpec, key: jax.Array) -> Runner:
    n = setup.cfg.num_envs
    key, init_key, reset_key = jax.random.split(key, 3)
    params = init_params(setup.layout, spec.obs_dim, init_key)
    env_state = jax.vmap(setup.reset, in_axes=(None, 0))(
        spec, jax.random.split(reset_key, n)
    )

    def per_env[T](tree: T) -> T:
        return jax.tree.map(lambda x: jnp.broadcast_to(x, (n, *x.shape)), tree)

    obs_rms = per_env(init_rms((spec.obs_dim,)))
    return Runner(
        params=params,
        opt_state=make_optimizer(setup.cfg).init(params),
        env_state=env_state,
        obs_rms=jax.vmap(update_rms)(obs_rms, env_state.obs),
        reward_scaler=per_env(init_reward_scaler()),
        key=key,
    )


def _train_iterations(setup: Setup, spec: EnvSpec, seed: jax.Array) -> Params:
    runner = _init_runner(setup, spec, jax.random.key(seed))
    runner, _ = jax.lax.scan(
        lambda r, _: _iteration(setup, spec, r),
        runner,
        length=setup.cfg.num_iterations,
    )
    return runner.params


def train(setup: Setup, spec: EnvSpec, seed: jax.Array) -> TrainOutput:
    runner = _init_runner(setup, spec, jax.random.key(seed))
    runner, (curve_sum, curve_episodes, losses) = jax.lax.scan(
        lambda r, _: _iteration(setup, spec, r),
        runner,
        length=setup.cfg.num_iterations,
    )
    env0_rms = jax.tree.map(lambda x: x[0], runner.obs_rms)
    result = evaluate(setup, spec, runner.params, env0_rms, seed)
    return TrainOutput(
        **result,
        curve_return_sum=curve_sum,
        curve_episodes=curve_episodes,
        losses=losses,
    )


def _compile[T](
    fn: Callable[[EnvSpec, jax.Array], T], spec: EnvSpec, seeds: jax.Array
) -> tuple[Callable[[EnvSpec, jax.Array], T], float]:
    start = time.perf_counter()
    compiled = jax.jit(jax.vmap(fn, in_axes=(None, 0))).lower(spec, seeds)
    compiled = compiled.compile()
    return compiled, time.perf_counter() - start


def _timed_run[T](
    setup: Setup,
    fn: Callable[[EnvSpec, jax.Array], T],
    spec: EnvSpec,
    seeds: Sequence[int],
) -> tuple[T, dict[str, float]]:
    seed_arr = jnp.asarray(seeds, dtype=jnp.int32)
    compiled, compile_s = _compile(fn, spec, seed_arr)
    start = time.perf_counter()
    out = jax.block_until_ready(compiled(spec, seed_arr))
    run_s = time.perf_counter() - start
    cfg = setup.cfg
    env_steps = len(seeds) * cfg.num_iterations * cfg.batch_size
    timing = {
        "compile_s": compile_s,
        "run_s": run_s,
        "steps_per_s": env_steps / run_s,
    }
    return out, timing


def train_seeds(
    setup: Setup, spec: EnvSpec, seeds: Sequence[int]
) -> tuple[TrainOutput, dict[str, float]]:
    return _timed_run(setup, partial(train, setup), spec, seeds)


def benchmark(
    setup: Setup, spec: EnvSpec, seeds: Sequence[int], iterations: int
) -> dict[str, float]:
    # compile the full program, but time only `iterations` training
    # iterations: a short run would be dominated by the 1000-step eval
    seed_arr = jnp.asarray(seeds, dtype=jnp.int32)
    _, compile_s = _compile(partial(train, setup), spec, seed_arr)
    short_cfg = dataclasses.replace(
        setup.cfg, total_steps=iterations * setup.cfg.batch_size
    )
    short = dataclasses.replace(setup, cfg=short_cfg)
    _, timing = _timed_run(
        short, partial(_train_iterations, short), spec, seeds
    )
    return {
        "compile_s": compile_s,
        "iterations_compile_s": timing["compile_s"],
        "iterations_run_s": timing["run_s"],
        "steps_per_s": timing["steps_per_s"],
    }
