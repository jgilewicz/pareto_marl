import dataclasses
import time

import jax
import jax.numpy as jnp

from pareto_marl.envs import many_segment_ant
from pareto_marl.envs.contract import EnvSpec
from pareto_marl.envs.many_segment_ant import load_model, mjx

SIZES = (2, 4, 8, 16)
N_ENVS = 8
N_CALLS = 200
CONES = {"pyramidal": 0, "elliptic": 1}


def spec_with_cone(n_segs: int, cone: int) -> EnvSpec:
    spec = many_segment_ant.make_spec(n_segs)
    model = load_model(n_segs)
    model.opt.cone = cone
    params = dataclasses.replace(spec.params, model=mjx.put_model(model))
    return dataclasses.replace(spec, params=params)


def measure(spec: EnvSpec) -> tuple[float, float]:
    reset = jax.jit(jax.vmap(many_segment_ant.reset, in_axes=(None, 0)))
    step = jax.jit(jax.vmap(many_segment_ant.step, in_axes=(None, 0, 0, 0)))
    key = jax.random.key(0)
    state = reset(spec, jax.random.split(key, N_ENVS))
    action = jnp.zeros((N_ENVS, spec.action_dim))
    keys = jax.random.split(key, N_ENVS)
    start = time.perf_counter()
    compiled = step.lower(spec, state, action, keys).compile()
    compile_s = time.perf_counter() - start
    actions = jax.random.uniform(
        key, (N_CALLS, N_ENVS, spec.action_dim), minval=-1.0, maxval=1.0
    )
    state = jax.block_until_ready(compiled(spec, state, actions[0], keys))
    start = time.perf_counter()
    for action in actions:
        state = compiled(spec, state, action, keys)
    jax.block_until_ready(state)
    steps_per_s = N_CALLS * N_ENVS / (time.perf_counter() - start)
    return compile_s, steps_per_s


def main() -> None:
    print(f"jit(vmap(step)) over {N_ENVS} envs, {jax.devices()[0]}")
    print("n_segs  actuators  cone        compile s   env steps/s")
    for n_segs in SIZES:
        for name, cone in CONES.items():
            compile_s, steps_per_s = measure(spec_with_cone(n_segs, cone))
            print(
                f"{n_segs:6d}  {4 * n_segs:9d}  {name:10s}  {compile_s:9.1f}"
                f"   {steps_per_s:11.0f}"
            )


if __name__ == "__main__":
    main()
