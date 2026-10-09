import jax
import jax.numpy as jnp
from flax import struct

from pareto_marl.envs.contract import (
    EnvSpec,
    EnvState,
    Transition,
    finish_step,
    initial_state,
)

DT = 0.05
DAMPING = 1.0
RESET_NOISE = 0.1
CTRL_COST = 0.1
MAX_SPREAD = 0.5


@struct.dataclass
class PointMass:
    pos: jax.Array
    vel: jax.Array


def make_spec(obs_dim: int, action_dim: int, episode_length: int) -> EnvSpec:
    if obs_dim < 2 * action_dim:
        raise ValueError(
            f"obs_dim={obs_dim} < 2 * action_dim={2 * action_dim}: the obs "
            "holds position and velocity per action dim, pick a larger obs_dim"
        )
    if action_dim < 1 or episode_length < 1:
        raise ValueError(
            f"action_dim={action_dim} and episode_length={episode_length} "
            "must both be >= 1"
        )
    return EnvSpec(
        obs_dim=obs_dim, action_dim=action_dim, episode_length=episode_length
    )


def _observe(spec: EnvSpec, mass: PointMass, key: jax.Array) -> jax.Array:
    n_distractors = spec.obs_dim - 2 * spec.action_dim
    distractors = jax.random.normal(key, (n_distractors,))
    return jnp.concatenate([mass.pos, mass.vel, distractors])


def reset(spec: EnvSpec, key: jax.Array) -> EnvState:
    pos_key, vel_key, obs_key = jax.random.split(key, 3)
    shape = (spec.action_dim,)
    mass = PointMass(
        pos=jax.random.uniform(pos_key, shape, minval=-1.0, maxval=1.0)
        * RESET_NOISE,
        vel=jax.random.uniform(vel_key, shape, minval=-1.0, maxval=1.0)
        * RESET_NOISE,
    )
    metrics = {"x_velocity": jnp.zeros((), jnp.float32)}
    return initial_state(mass, _observe(spec, mass, obs_key), metrics)


def step(
    spec: EnvSpec, state: EnvState, action: jax.Array, key: jax.Array
) -> EnvState:
    obs_key, reset_key = jax.random.split(key)
    action = jnp.clip(action, -1.0, 1.0)
    mass: PointMass = state.physics
    vel = (1.0 - DAMPING * DT) * mass.vel + DT * action
    pos = mass.pos + DT * vel
    x_velocity = jnp.mean(vel)
    reward = 1.0 + x_velocity - CTRL_COST * jnp.sum(action**2)
    # "falls over" when coordinates drift apart, like Ant's unhealthy torso
    terminated = jnp.max(pos) - jnp.min(pos) > MAX_SPREAD
    moved = PointMass(pos=pos, vel=vel)
    transition = Transition(
        physics=moved,
        obs=_observe(spec, moved, obs_key),
        reward=reward,
        terminated=terminated,
        metrics={"x_velocity": x_velocity},
    )
    return finish_step(spec, state, transition, reset(spec, reset_key))
