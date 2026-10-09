import contextlib
import io
import os
import tempfile

import jax
import jax.numpy as jnp
from flax import struct

# compiled extension with no stubs, re-exported by a star import
from mujoco import MjModel  # ty: ignore[unresolved-import]

from pareto_marl.envs.contract import (
    EnvSpec,
    EnvState,
    Transition,
    finish_step,
    initial_state,
)

# package prints an Adroit reward-version notice on import; Adroit is unused
with contextlib.redirect_stderr(io.StringIO()):
    from gymnasium_robotics.envs.multiagent_mujoco.many_segment_ant import (
        gen_asset,
    )
# mjx prints "Failed to import warp" on import; the warp backend is unused
with contextlib.redirect_stdout(io.StringIO()):
    from mujoco import mjx

EPISODE_LENGTH = 1000
FRAME_SKIP = 5
RESET_NOISE = 0.1
CTRL_COST = 0.5
HEALTHY_REWARD = 1.0
HEALTHY_Z = (0.2, 1.0)
MAIN_BODY = "torso_0"


@struct.dataclass
class AntParams:
    model: mjx.Model
    init_qpos: jax.Array
    init_qvel: jax.Array
    main_body: int = struct.field(pytree_node=False)
    dt: float = struct.field(pytree_node=False)


def load_model(n_segs: int) -> MjModel:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, f"many_segment_ant_{n_segs}.xml")
        gen_asset(n_segs, path)
        return MjModel.from_xml_path(path)


def make_spec(n_segs: int) -> EnvSpec:
    if n_segs < 1:
        raise ValueError(f"n_segs={n_segs} must be >= 1")
    model = load_model(n_segs)
    params = AntParams(
        model=mjx.put_model(model),
        init_qpos=jnp.asarray(model.qpos0, jnp.float32),
        init_qvel=jnp.zeros(model.nv, jnp.float32),
        main_body=model.body(MAIN_BODY).id,
        dt=float(model.opt.timestep) * FRAME_SKIP,
    )
    return EnvSpec(
        obs_dim=model.nq - 2 + model.nv,
        action_dim=model.nu,
        episode_length=EPISODE_LENGTH,
        params=params,
    )


def _observe(data: mjx.Data) -> jax.Array:
    return jnp.concatenate([data.qpos[2:], data.qvel])


def _is_healthy(data: mjx.Data) -> jax.Array:
    finite = jnp.all(jnp.isfinite(data.qpos)) & jnp.all(jnp.isfinite(data.qvel))
    z = data.qpos[2]
    return finite & (z >= HEALTHY_Z[0]) & (z <= HEALTHY_Z[1])


def reset(spec: EnvSpec, key: jax.Array) -> EnvState:
    params: AntParams = spec.params
    pos_key, vel_key = jax.random.split(key)
    qpos = params.init_qpos + jax.random.uniform(
        pos_key, params.init_qpos.shape, minval=-RESET_NOISE, maxval=RESET_NOISE
    )
    qvel = params.init_qvel + RESET_NOISE * jax.random.normal(
        vel_key, params.init_qvel.shape
    )
    data = mjx.make_data(params.model).replace(qpos=qpos, qvel=qvel)
    # Ant-v5 set_state runs mj_forward: xpos of the main body must be current
    # for the x_velocity of the first step
    data = mjx.forward(params.model, data)
    metrics = {"x_velocity": jnp.zeros((), jnp.float32)}
    return initial_state(data, _observe(data), metrics)


def step(
    spec: EnvSpec, state: EnvState, action: jax.Array, key: jax.Array
) -> EnvState:
    params: AntParams = spec.params
    action = jnp.clip(action, -1.0, 1.0)
    before: mjx.Data = state.physics
    data = jax.lax.fori_loop(
        0,
        FRAME_SKIP,
        lambda _, d: mjx.step(params.model, d),
        before.replace(ctrl=action),
    )
    # Ant-v5 reads xpos after mj_step without a new mj_forward: kinematics of
    # the last integrator stage, not of the final qpos
    x_before = before.xpos[params.main_body, 0]
    x_velocity = (data.xpos[params.main_body, 0] - x_before) / params.dt
    healthy = _is_healthy(data)
    reward = (
        x_velocity
        + HEALTHY_REWARD * healthy
        - CTRL_COST * jnp.sum(jnp.square(action))
    )
    transition = Transition(
        physics=data,
        obs=_observe(data),
        reward=reward,
        terminated=~healthy,
        metrics={"x_velocity": x_velocity},
    )
    return finish_step(spec, state, transition, reset(spec, key))
