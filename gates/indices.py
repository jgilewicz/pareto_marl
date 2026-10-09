import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import numpy as np

from pareto_marl.envs import many_segment_ant
from pareto_marl.partition import (
    ANT_KWARGS,
    LOCAL_CATEGORIES,
    PartitionSpec,
    make_ma_env,
    make_many_segment_env,
    mamujoco_v1,
    parse_key,
    structured_partitions,
)

SIZES = (2, 4, 8, 16)
N_STEPS = 5
ROOT_LABELS = [f"root.qpos{i}" for i in range(2, 7)] + [
    f"root.qvel{i}" for i in range(6)
]

Graph = dict[str, set[str]]


def _connect(graph: Graph, joints: list[str]) -> None:
    for a in joints:
        graph.setdefault(a, set()).update(b for b in joints if b != a)


def segment_graph(n_segs: int) -> Graph:
    graph: Graph = {}
    for s in range(n_segs):
        _connect(graph, [f"hip1_{s}", f"ankle1_{s}"])
        _connect(graph, [f"hip2_{s}", f"ankle2_{s}"])
        _connect(graph, [f"hip1_{s}", f"hip2_{s}"])
        if s:
            hips = [f"hip{leg}_{t}" for t in (s - 1, s) for leg in (1, 2)]
            _connect(graph, hips)
    return graph


def ant_graph() -> Graph:
    graph: Graph = {}
    for leg in range(1, 5):
        _connect(graph, [f"hip_{leg}", f"ankle_{leg}"])
    _connect(graph, [f"hip_{leg}" for leg in range(1, 5)])
    return graph


def state_labels(model: Any) -> np.ndarray:
    qpos = [""] * model.nq
    qvel = [""] * model.nv
    for j in range(model.njnt):
        name = model.joint(j).name
        free = model.jnt_type[j] == 0
        for i in range(7 if free else 1):
            qpos[model.jnt_qposadr[j] + i] = f"{name}.qpos" + (
                str(i) if free else ""
            )
        for i in range(6 if free else 1):
            qvel[model.jnt_dofadr[j] + i] = f"{name}.qvel" + (
                str(i) if free else ""
            )
    return np.array(qpos[2:] + qvel)


def check_names(
    env: Any, x: tuple[int, ...], graph: Graph, joint_of: Callable[[str], str]
) -> list[str]:
    model = env.single_agent_env.unwrapped.model
    labels = state_labels(model)
    actuator_joint = [
        model.joint(model.actuator_trnid[i, 0]).name for i in range(model.nu)
    ]
    errors = []
    for k, agent in enumerate(env.possible_agents):
        nodes = env.agent_action_partitions[k]
        own = [joint_of(node.label) for node in nodes]
        acts = [node.act_ids for node in nodes]
        if [actuator_joint[a] for a in acts] != own:
            errors.append(f"{agent}: actuators {acts} do not drive {own}")
        if sorted(acts) != [i for i, v in enumerate(x) if v == k]:
            errors.append(f"{agent}: actuators {acts} != block {k} of {x}")
        seen = set().union(*(graph[j] for j in own)) - set(own)
        expected = (
            [f"{j}.{c}" for j in own for c in ("qpos", "qvel")]
            + [f"{j}.qpos" for j in seen]
            + ROOT_LABELS
        )
        got = labels[env.observation_factorization[agent]].tolist()
        if sorted(got) != sorted(expected):
            errors.append(f"{agent}: observes {got}, expected {expected}")
    covered = sorted(n.act_ids for p in env.agent_action_partitions for n in p)
    if covered != list(range(model.nu)):
        errors.append(f"actuators covered {covered}")
    return errors


def check_values(env: Any, rng: np.random.Generator) -> list[str]:
    obs, _ = env.reset(seed=int(rng.integers(1 << 31)))
    for _ in range(N_STEPS):
        actions = {
            a: rng.uniform(-1, 1, env.action_space(a).shape)
            for a in env.possible_agents
        }
        obs, *_ = env.step(actions)
    data = env.single_agent_env.unwrapped.data
    state = env.state()
    errors = []
    if not np.array_equal(state, np.concatenate([data.qpos[2:], data.qvel])):
        errors.append("state() != concat(qpos[2:], qvel)")
    for agent, idx in env.observation_factorization.items():
        if not np.array_equal(obs[agent], state[idx]):
            errors.append(f"{agent}: obs != state()[obs_idx]")
    return errors


def check_layout(env: Any, spec: Any, key: jax.Array) -> list[str]:
    reset = many_segment_ant.reset(spec, key)
    data = reset.physics
    env.single_agent_env.unwrapped.set_state(
        np.asarray(data.qpos, np.float64), np.asarray(data.qvel, np.float64)
    )
    state = env.state().astype(np.float32)
    if not np.array_equal(state, np.asarray(reset.obs)):
        return ["MaMuJoCo state() != MJX obs at the same physics state"]
    return []


def many_segment_gate(rng: np.random.Generator) -> bool:
    ok = True
    for n_segs in SIZES:
        spec = many_segment_ant.make_spec(n_segs)
        graph = segment_graph(n_segs)
        for name, x in structured_partitions(n_segs).items():
            env = make_many_segment_env(x)
            p = PartitionSpec.from_env(x, env)
            key = jax.random.key(int(rng.integers(1 << 31)))
            errors = (
                check_names(env, x, graph, lambda label: label)
                + check_values(env, rng)
                + check_layout(env, spec, key)
            )
            env.close()
            ok &= not errors
            print(
                f"n_segs={n_segs:2d} {name:7s} K={p.k:2d} "
                f"obs_dim={min(p.obs_dims)}..{max(p.obs_dims)} "
                f"act_dim={min(p.act_dims)}..{max(p.act_dims)} "
                + ("PASS" if not errors else "FAIL " + "; ".join(errors))
            )
    return ok


def unpatched_control() -> None:
    for n_segs in SIZES:
        env = mamujoco_v1.parallel_env(
            "ManySegmentAnt",
            f"{n_segs}x1",
            agent_obsk=1,
            local_categories=LOCAL_CATEGORIES,
            **ANT_KWARGS,
        )
        x = structured_partitions(n_segs)["segment"]
        errors = check_names(env, x, segment_graph(n_segs), lambda s: s)
        env.close()
        wrong_act = sum("do not drive" in e for e in errors)
        wrong_obs = sum("observes" in e for e in errors)
        print(
            f"control, unpatched MaMuJoCo n_segs={n_segs:2d} segment: "
            f"of {n_segs} agents {wrong_act} drive the wrong actuators, "
            f"{wrong_obs} observe the wrong joints"
        )


def ant_gate() -> bool:
    design = json.loads(Path("configs/design.json").read_text())
    keys = [p["x"] for p in design["partitions"]]
    failed = []
    for key in keys:
        x = parse_key(key)
        env = make_ma_env(x)
        errors = check_names(
            env, x, ant_graph(), lambda s: re.sub(r"(\d)$", r"_\1", s)
        )
        env.close()
        if errors:
            failed.append(f"{key}: {'; '.join(errors)}")
    passed = len(keys) - len(failed)
    print(f"Ant v1/v2 path: {passed}/{len(keys)} design partitions PASS names")
    for line in failed:
        print("  FAIL", line)
    return not failed


def main() -> None:
    rng = np.random.default_rng(0)
    ok = many_segment_gate(rng)
    ok &= ant_gate()
    unpatched_control()
    print("gate 2:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
