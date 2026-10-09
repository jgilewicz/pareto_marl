import math
from dataclasses import dataclass
from typing import Any, cast

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

HIDDEN = 64
LOG_2PI = math.log(2 * math.pi)

Params = dict[str, Any]


class MLP(nn.Module):
    out_dim: int
    out_std: float

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        hidden_init = nn.initializers.orthogonal(math.sqrt(2))
        for _ in range(2):
            x = nn.tanh(nn.Dense(HIDDEN, kernel_init=hidden_init)(x))
        head_init = nn.initializers.orthogonal(self.out_std)
        return nn.Dense(self.out_dim, kernel_init=head_init)(x)


@dataclass(frozen=True)
class AgentLayout:
    obs_dims: tuple[int, ...]
    act_dims: tuple[int, ...]
    # [K, max obs]: indices into the full obs, padding points at 0
    obs_idx: np.ndarray
    obs_mask: np.ndarray
    # [K, max act]
    act_mask: np.ndarray
    # [action_dim]: position of each actuator in the flat [K * max act] action
    act_gather: np.ndarray

    @property
    def k(self) -> int:
        return len(self.act_dims)

    @property
    def max_obs(self) -> int:
        return self.obs_idx.shape[1]

    @property
    def max_act(self) -> int:
        return self.act_mask.shape[1]


def _pad_rows(
    idx: list[np.ndarray], width: int
) -> tuple[np.ndarray, np.ndarray]:
    padded = np.zeros((len(idx), width), dtype=np.int32)
    mask = np.zeros((len(idx), width), dtype=np.float32)
    for k, row in enumerate(idx):
        padded[k, : len(row)] = row
        mask[k, : len(row)] = 1.0
    return padded, mask


def build_layout(
    obs_idx: list[np.ndarray],
    act_idx: list[np.ndarray],
    obs_dim: int,
    action_dim: int,
) -> AgentLayout:
    if len(obs_idx) != len(act_idx):
        raise ValueError(
            f"{len(obs_idx)} obs index sets vs {len(act_idx)} action index "
            "sets: pass one of each per agent"
        )
    flat_obs = np.concatenate(obs_idx)
    if flat_obs.min() < 0 or flat_obs.max() >= obs_dim:
        raise ValueError(f"obs indices outside [0, {obs_dim})")
    owned = np.sort(np.concatenate(act_idx))
    if not np.array_equal(owned, np.arange(action_dim)):
        raise ValueError(
            f"action indices must cover each of {action_dim} actuators "
            f"exactly once, got sorted {owned.tolist()}"
        )
    obs_pad, obs_mask = _pad_rows(obs_idx, max(len(i) for i in obs_idx))
    max_act = max(len(i) for i in act_idx)
    _, act_mask = _pad_rows(act_idx, max_act)
    act_gather = np.empty(action_dim, dtype=np.int32)
    for k, row in enumerate(act_idx):
        act_gather[row] = k * max_act + np.arange(len(row))
    return AgentLayout(
        obs_dims=tuple(len(i) for i in obs_idx),
        act_dims=tuple(len(i) for i in act_idx),
        obs_idx=obs_pad,
        obs_mask=obs_mask,
        act_mask=act_mask,
        act_gather=act_gather,
    )


def _init_actor(
    layout: AgentLayout, obs_dim: int, act_dim: int, key: jax.Array
) -> Params:
    # init at the true size, then zero-pad: orthogonal init of the padded
    # shape would differ, and zero rows/cols get zero grads, so they stay 0
    params = MLP(act_dim, 0.01).init(key, jnp.zeros(obs_dim))["params"]
    first, head = params["Dense_0"], params["Dense_2"]
    obs_pad = layout.max_obs - obs_dim
    act_pad = layout.max_act - act_dim
    return {
        "Dense_0": {
            "kernel": jnp.pad(first["kernel"], ((0, obs_pad), (0, 0))),
            "bias": first["bias"],
        },
        "Dense_1": params["Dense_1"],
        "Dense_2": {
            "kernel": jnp.pad(head["kernel"], ((0, 0), (0, act_pad))),
            "bias": jnp.pad(head["bias"], (0, act_pad)),
        },
    }


def init_params(layout: AgentLayout, obs_dim: int, key: jax.Array) -> Params:
    critic_key, actor_key = jax.random.split(key)
    actor_keys = jax.random.split(actor_key, layout.k)
    actors = [
        _init_actor(layout, o, a, k)
        for o, a, k in zip(
            layout.obs_dims, layout.act_dims, actor_keys, strict=True
        )
    ]
    critic = MLP(1, 1.0).init(critic_key, jnp.zeros(obs_dim))["params"]
    return {
        "actor": jax.tree.map(lambda *xs: jnp.stack(xs), *actors),
        "log_std": jnp.zeros((layout.k, layout.max_act)),
        "critic": critic,
    }


def actor_mean(
    layout: AgentLayout, params: Params, obs: jax.Array
) -> jax.Array:
    local = obs[..., layout.obs_idx] * layout.obs_mask
    mlp = MLP(layout.max_act, 0.01)

    def apply(p: Params, x: jax.Array) -> jax.Array:
        return cast(jax.Array, mlp.apply({"params": p}, x))

    return jax.vmap(apply, in_axes=(0, -2), out_axes=-2)(params["actor"], local)


def critic_value(params: Params, obs: jax.Array) -> jax.Array:
    value = MLP(1, 1.0).apply({"params": params["critic"]}, obs)
    return cast(jax.Array, value)[..., 0]


def sample_action(params: Params, mean: jax.Array, key: jax.Array) -> jax.Array:
    noise = jax.random.normal(key, mean.shape)
    return mean + jnp.exp(params["log_std"]) * noise


def log_prob(
    layout: AgentLayout, params: Params, mean: jax.Array, action: jax.Array
) -> jax.Array:
    log_std = params["log_std"]
    z = (action - mean) * jnp.exp(-log_std)
    per_dim = -0.5 * z**2 - log_std - 0.5 * LOG_2PI
    return (per_dim * layout.act_mask).sum(-1)


def entropy(layout: AgentLayout, params: Params) -> jax.Array:
    per_dim = 0.5 + 0.5 * LOG_2PI + params["log_std"]
    return (per_dim * layout.act_mask).sum(-1)


def to_env_action(layout: AgentLayout, action: jax.Array) -> jax.Array:
    flat = action.reshape(*action.shape[:-2], layout.k * layout.max_act)
    return flat[..., layout.act_gather]
