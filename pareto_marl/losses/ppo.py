from typing import Protocol

import jax
import jax.numpy as jnp
from flax import struct

from pareto_marl.networks.actor_critic import (
    AgentLayout,
    Params,
    actor_mean,
    critic_value,
    entropy,
    log_prob,
)


class LossConfig(Protocol):
    @property
    def clip_coef(self) -> float: ...
    @property
    def vf_coef(self) -> float: ...
    @property
    def ent_coef(self) -> float: ...


@struct.dataclass
class Batch:
    obs: jax.Array
    # padded per-agent actions [B, K, max act]
    actions: jax.Array
    logprobs: jax.Array
    advantages: jax.Array
    returns: jax.Array
    values: jax.Array


def mappo_loss(
    params: Params, layout: AgentLayout, mb: Batch, cfg: LossConfig
) -> tuple[jax.Array, dict[str, jax.Array]]:
    mean = actor_mean(layout, params, mb.obs)
    newlogprob = log_prob(layout, params, mean, mb.actions)
    logratio = newlogprob - mb.logprobs
    ratio = jnp.exp(logratio)
    adv = mb.advantages
    # torch .std() is the unbiased estimator
    adv = ((adv - adv.mean()) / (adv.std(ddof=1) + 1e-8))[:, None]
    clipped = jnp.clip(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
    # per-agent clipped surrogate, summed over agents (MAPPO)
    pg_loss = jnp.maximum(-adv * ratio, -adv * clipped).mean(0).sum()

    newvalue = critic_value(params, mb.obs)
    v_clipped = mb.values + jnp.clip(
        newvalue - mb.values, -cfg.clip_coef, cfg.clip_coef
    )
    v_loss = (
        0.5
        * jnp.maximum(
            (newvalue - mb.returns) ** 2, (v_clipped - mb.returns) ** 2
        ).mean()
    )
    # state-independent std: the batch mean equals the per-sample entropy
    ent = entropy(layout, params).sum()

    loss = pg_loss - cfg.ent_coef * ent + cfg.vf_coef * v_loss
    stats = {
        "policy_loss": pg_loss,
        "value_loss": v_loss,
        "entropy": ent,
        "approx_kl": jax.lax.stop_gradient(((ratio - 1) - logratio).mean()),
        "clipfrac": jax.lax.stop_gradient(
            (jnp.abs(ratio - 1) > cfg.clip_coef).mean()
        ),
    }
    return loss, stats
