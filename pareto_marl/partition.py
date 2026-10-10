import contextlib
import io
from dataclasses import dataclass
from typing import Any, Self

import numpy as np

# package prints an Adroit reward-version notice on import; Adroit is unused
with contextlib.redirect_stderr(io.StringIO()):
    from gymnasium_robotics import mamujoco_v1
    from gymnasium_robotics.envs.multiagent_mujoco.obsk import (
        HyperEdge,
        Node,
        get_parts_and_edges,
    )

# ManySegmentAnt actuators per segment, XML order: hip1, ankle1, hip2, ankle2
SEGMENT_ACTUATORS = 4
HIDDEN = 64
# contact forces are left out of every observation
ANT_KWARGS: dict[str, Any] = {"include_cfrc_ext_in_observation": False}
LOCAL_CATEGORIES = [["qpos", "qvel"], ["qpos"]]

Partition = tuple[int, ...]


def key_of(x: Partition) -> str:
    return "-".join(str(v) for v in x)


def n_blocks(x: Partition) -> int:
    return max(x) + 1


def mlp_params(obs_dim: int, act_dim: int) -> int:
    first = obs_dim * HIDDEN + HIDDEN
    second = HIDDEN * HIDDEN + HIDDEN
    head = HIDDEN * act_dim + act_dim
    log_std = act_dim
    return first + second + head + log_std


def structured_partitions(n_segs: int) -> dict[str, Partition]:
    m = SEGMENT_ACTUATORS * n_segs
    return {
        "single": tuple(0 for _ in range(m)),
        "segment": tuple(i // SEGMENT_ACTUATORS for i in range(m)),
        "leg": tuple(i // 2 for i in range(m)),
        "joint": tuple(range(m)),
    }


def many_segment_graph(
    n_segs: int,
) -> tuple[list[Node], list[HyperEdge], list[Node]]:
    parts, edges, globals_ = get_parts_and_edges(
        "ManySegmentAnt", f"{n_segs}x1"
    )
    nodes = [node for part in parts for node in part]
    # gymnasium-robotics 1.4.2 bugs: qpos/qvel ids of all but the last segment
    # point into the root joint, act_ids swap the two legs of a segment, and
    # inter-segment edges hold deepcopies (one-way, duplicated neighbours)
    for i, node in enumerate(nodes):
        segment, joint = divmod(i, SEGMENT_ACTUATORS)
        node.qpos_ids = node.qvel_ids = (
            -SEGMENT_ACTUATORS * (n_segs - segment) + joint
        )
        node.act_ids = i
    by_label = {node.label: node for node in nodes}
    edges = [HyperEdge(*(by_label[n.label] for n in e.nodes)) for e in edges]
    return nodes, edges, globals_


# MaMuJoCo writes and deletes its XML inside site-packages: never call this
# from concurrent jobs, read configs/stage0_indices.json instead
def make_many_segment_env(x: Partition, obsk: int = 1) -> Any:
    n_segs, rest = divmod(len(x), SEGMENT_ACTUATORS)
    if rest or not n_segs:
        raise ValueError(
            f"partition of length {len(x)} does not cover whole segments: "
            f"ManySegmentAnt has {SEGMENT_ACTUATORS} actuators per segment"
        )
    nodes, edges, globals_ = many_segment_graph(n_segs)
    blocks = [
        tuple(nodes[i] for i in range(len(x)) if x[i] == k)
        for k in range(n_blocks(x))
    ]
    return mamujoco_v1.parallel_env(
        "ManySegmentAnt",
        f"{n_segs}x1",
        agent_obsk=obsk,
        agent_factorization={
            "partition": blocks,
            "edges": edges,
            "globals": globals_,
        },
        local_categories=LOCAL_CATEGORIES,
        **ANT_KWARGS,
    )


@dataclass(frozen=True)
class PartitionSpec:
    x: Partition
    obs_idx: list[np.ndarray]
    act_idx: list[np.ndarray]

    @classmethod
    def build(cls, x: Partition, obsk: int = 1) -> Self:
        return cls.from_env(x, make_many_segment_env(x, obsk))

    @classmethod
    def from_env(cls, x: Partition, env: Any) -> Self:
        obs_idx = [
            np.asarray(env.observation_factorization[a])
            for a in env.possible_agents
        ]
        act_idx = [
            np.array([node.act_ids for node in part])
            for part in env.agent_action_partitions
        ]
        env.close()
        return cls(x=x, obs_idx=obs_idx, act_idx=act_idx)

    @property
    def k(self) -> int:
        return len(self.act_idx)

    @property
    def obs_dims(self) -> list[int]:
        return [len(idx) for idx in self.obs_idx]

    @property
    def act_dims(self) -> list[int]:
        return [len(idx) for idx in self.act_idx]

    @property
    def actor_params(self) -> int:
        return sum(
            mlp_params(o, a)
            for o, a in zip(self.obs_dims, self.act_dims, strict=True)
        )
