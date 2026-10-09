from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class EnvSpec:
    obs_dim: int = struct.field(pytree_node=False)
    action_dim: int = struct.field(pytree_node=False)
    episode_length: int = struct.field(pytree_node=False)
    # env-specific pytree (e.g. mjx.Model); traced when spec is a jit argument
    params: Any = None


@struct.dataclass
class EnvState:
    physics: Any
    obs: jax.Array
    reward: jax.Array
    terminated: jax.Array
    done: jax.Array
    metrics: dict[str, jax.Array]
    episode_return: jax.Array
    episode_length: jax.Array
    returned_episode: jax.Array
    returned_episode_return: jax.Array
    returned_episode_length: jax.Array


@struct.dataclass
class Transition:
    physics: Any
    obs: jax.Array
    reward: jax.Array
    terminated: jax.Array
    metrics: dict[str, jax.Array]


ResetFn = Callable[[EnvSpec, jax.Array], EnvState]
StepFn = Callable[[EnvSpec, EnvState, jax.Array, jax.Array], EnvState]


def initial_state(
    physics: Any, obs: jax.Array, metrics: dict[str, jax.Array]
) -> EnvState:
    zero = jnp.zeros((), jnp.float32)
    zero_steps = jnp.zeros((), jnp.int32)
    return EnvState(
        physics=physics,
        obs=obs.astype(jnp.float32),
        reward=zero,
        terminated=zero,
        done=zero,
        metrics=metrics,
        episode_return=zero,
        episode_length=zero_steps,
        returned_episode=zero,
        returned_episode_return=zero,
        returned_episode_length=zero_steps,
    )


def finish_step(
    spec: EnvSpec, state: EnvState, t: Transition, fresh: EnvState
) -> EnvState:
    reward = t.reward.astype(jnp.float32)
    episode_return = state.episode_return + reward
    episode_length = state.episode_length + 1
    truncated = episode_length >= spec.episode_length
    done = jnp.logical_or(t.terminated, truncated)
    # gymnasium SAME_STEP autoreset (CleanRL v1/v2): the final step returns its
    # own reward and metrics with done=1, but the obs of the reset episode
    physics, obs = jax.tree.map(
        lambda reset, cont: jnp.where(done, reset, cont),
        (fresh.physics, fresh.obs),
        (t.physics, t.obs.astype(jnp.float32)),
    )
    return EnvState(
        physics=physics,
        obs=obs,
        reward=reward,
        terminated=t.terminated.astype(jnp.float32),
        done=done.astype(jnp.float32),
        metrics=t.metrics,
        episode_return=jnp.where(done, 0.0, episode_return),
        episode_length=jnp.where(done, 0, episode_length),
        returned_episode=done.astype(jnp.float32),
        returned_episode_return=jnp.where(
            done, episode_return, state.returned_episode_return
        ),
        returned_episode_length=jnp.where(
            done, episode_length, state.returned_episode_length
        ),
    )
