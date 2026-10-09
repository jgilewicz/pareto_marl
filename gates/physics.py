import dataclasses
import os
import tempfile
from typing import Any

import gymnasium
import jax
import numpy as np

from pareto_marl.envs import many_segment_ant
from pareto_marl.envs.many_segment_ant import gen_asset, load_model, mjx

SIZES = (2, 16)
N_STEPS = 100
EARLY = 10
TOL = 1e-3
CONES = {"pyramidal": 0, "elliptic": 1}


def make_reference(n_segs: int) -> Any:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, f"many_segment_ant_{n_segs}.xml")
        gen_asset(n_segs, path)
        env = gymnasium.make(
            "Ant-v5",
            xml_file=path,
            include_cfrc_ext_in_observation=False,
            contact_cost_weight=0.0,
        )
    env.reset(seed=0)
    return env.unwrapped


def rollout(n_segs: int, one_step: bool, cone: str) -> list[dict[str, float]]:
    spec = many_segment_ant.make_spec(n_segs)
    model = load_model(n_segs)
    model.opt.cone = CONES[cone]
    params = dataclasses.replace(spec.params, model=mjx.put_model(model))
    spec = dataclasses.replace(spec, params=params)
    step = jax.jit(many_segment_ant.step)
    ref = make_reference(n_segs)
    ref.model.opt.cone = CONES[cone]
    key = jax.random.key(n_segs)
    key, reset_key = jax.random.split(key)
    state = jax.jit(many_segment_ant.reset)(spec, reset_key)
    actions = np.random.default_rng(n_segs).uniform(
        -1, 1, (N_STEPS, spec.action_dim)
    )
    ref.set_state(
        np.asarray(state.physics.qpos, np.float64),
        np.asarray(state.physics.qvel, np.float64),
    )
    rows = []
    for action in actions:
        if one_step:
            ref.set_state(
                np.asarray(state.physics.qpos, np.float64),
                np.asarray(state.physics.qvel, np.float64),
            )
        key, step_key = jax.random.split(key)
        state = step(spec, state, action.astype(np.float32), step_key)
        obs, reward, terminated, _, info = ref.step(action)
        rows.append(
            {
                "qpos": float(np.abs(state.physics.qpos - ref.data.qpos).max()),
                "qvel": float(np.abs(state.physics.qvel - ref.data.qvel).max()),
                "obs": float(np.abs(state.obs - obs).max()),
                "reward": abs(float(state.reward) - reward),
                "x_velocity": abs(
                    float(state.metrics["x_velocity"]) - info["x_velocity"]
                ),
                "terminated": float(state.terminated) != float(terminated),
                "done": float(state.done),
                "x_mjx": float(state.physics.qpos[0]),
                "x_ref": float(ref.data.qpos[0]),
                "z_mjx": float(state.physics.qpos[2]),
                "z_ref": float(ref.data.qpos[2]),
            }
        )
    return rows


def report(n_segs: int, one_step: bool, cone: str) -> bool:
    rows = rollout(n_segs, one_step, cone)
    mode = "one-step (C reset to MJX state)" if one_step else "open-loop"
    print(f"\nn_segs={n_segs} {cone} cone, {mode}")
    print(
        "step  max|dqpos|  max|dqvel|  max|dobs|   |dreward|   |dx_vel|"
        "    x mjx/ref        z mjx/ref"
    )
    for t, r in enumerate(rows, start=1):
        if t <= EARLY or t % 10 == 0:
            print(
                f"{t:4d}  {r['qpos']:.2e}    {r['qvel']:.2e}    "
                f"{r['obs']:.2e}    {r['reward']:.2e}    "
                f"{r['x_velocity']:.2e}   {r['x_mjx']:+.3f}/{r['x_ref']:+.3f}"
                f"    {r['z_mjx']:.3f}/{r['z_ref']:.3f}"
            )
    early = rows[:EARLY]
    early_q = max(r["qpos"] for r in early)
    early_v = max(r["qvel"] for r in early)
    flips = sum(r["terminated"] for r in rows)
    resets = sum(r["done"] for r in rows)
    print(
        f"first {EARLY} steps: max|dqpos|={early_q:.2e} "
        f"max|dqvel|={early_v:.2e}; all {N_STEPS}: "
        f"max|dqpos|={max(r['qpos'] for r in rows):.2e} "
        f"max|dqvel|={max(r['qvel'] for r in rows):.2e} "
        f"max|dreward|={max(r['reward'] for r in rows):.2e}; "
        f"terminated mismatches={flips}, MJX resets={resets:.0f}"
    )
    return early_q < TOL and early_v < TOL and not flips and not resets


def main() -> None:
    for cone in CONES:
        ok = True
        for n_segs in SIZES:
            for one_step in (False, True):
                ok &= report(n_segs, one_step, cone)
        print(f"\ngate 1, {cone} cone:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
