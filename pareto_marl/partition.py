import contextlib
import io
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Self

import numpy as np

# package prints an Adroit reward-version notice on import; Adroit is unused
with contextlib.redirect_stderr(io.StringIO()):
    from gymnasium_robotics import mamujoco_v1
    from gymnasium_robotics.envs.multiagent_mujoco.obsk import (
        get_parts_and_edges,
    )

N_ACTUATORS = 8
HIDDEN = 64

Partition = tuple[int, ...]


def canon(x: Sequence[int]) -> Partition:
    remap: dict[int, int] = {}
    for v in x:
        remap.setdefault(int(v), len(remap))
    return tuple(remap[int(v)] for v in x)


def key_of(x: Sequence[int]) -> str:
    return "".join(str(v) for v in canon(x))


def parse_key(key: str) -> Partition:
    x = tuple(int(c) for c in key)
    if len(x) != N_ACTUATORS or canon(x) != x:
        raise ValueError(
            f"invalid partition key {key!r}: expected {N_ACTUATORS} digits "
            "in canonical form, e.g. '00112233'"
        )
    return x


def n_blocks(x: Partition) -> int:
    return max(x) + 1


def all_partitions(m: int = N_ACTUATORS) -> Iterator[Partition]:
    def extend(prefix: list[int]) -> Iterator[Partition]:
        if len(prefix) == m:
            yield tuple(prefix)
            return
        for v in range(max(prefix) + 2):
            yield from extend([*prefix, v])

    yield from extend([0])


def sample_partitions(
    k: int, n: int, rng: np.random.Generator, exclude: set[Partition]
) -> list[Partition]:
    pool = [
        x for x in all_partitions() if n_blocks(x) == k and x not in exclude
    ]
    if n > len(pool):
        raise ValueError(
            f"requested {n} partitions with K={k}, only {len(pool)} available"
        )
    picks = rng.choice(len(pool), size=n, replace=False)
    return [pool[i] for i in sorted(picks)]


def mlp_params(obs_dim: int, act_dim: int) -> int:
    first = obs_dim * HIDDEN + HIDDEN
    second = HIDDEN * HIDDEN + HIDDEN
    head = HIDDEN * act_dim + act_dim
    log_std = act_dim
    return first + second + head + log_std


def make_ma_env(x: Partition, obsk: int = 1) -> Any:
    parts, edges, globals_ = get_parts_and_edges("Ant", None)
    nodes = list(parts[0])
    blocks = [
        tuple(nodes[i] for i in range(N_ACTUATORS) if x[i] == k)
        for k in range(n_blocks(x))
    ]
    factorization = {"partition": blocks, "edges": edges, "globals": globals_}
    return mamujoco_v1.parallel_env(
        "Ant", "custom", agent_obsk=obsk, agent_factorization=factorization
    )


@dataclass(frozen=True)
class PartitionSpec:
    x: Partition
    obs_idx: list[np.ndarray]
    act_idx: list[np.ndarray]

    @classmethod
    def build(cls, x: Partition, obsk: int = 1) -> Self:
        env = make_ma_env(x, obsk)
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

    def split_obs(self, global_obs: np.ndarray) -> list[np.ndarray]:
        return [global_obs[..., idx] for idx in self.obs_idx]

    def merge_actions(self, local_actions: Sequence[np.ndarray]) -> np.ndarray:
        batch = local_actions[0].shape[:-1]
        merged = np.zeros((*batch, N_ACTUATORS), dtype=np.float32)
        for idx, act in zip(self.act_idx, local_actions, strict=True):
            merged[..., idx] = act
        return merged
