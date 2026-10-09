import jax
import jax.numpy as jnp
from flax import struct

EPSILON = 1e-8
OBS_CLIP = 10.0
REWARD_CLIP = 10.0


@struct.dataclass
class RunningMeanStd:
    mean: jax.Array
    var: jax.Array
    count: jax.Array


@struct.dataclass
class RewardScaler:
    rms: RunningMeanStd
    discounted: jax.Array


def init_rms(shape: tuple[int, ...]) -> RunningMeanStd:
    return RunningMeanStd(
        mean=jnp.zeros(shape),
        var=jnp.ones(shape),
        count=jnp.asarray(1e-4),
    )


def update_rms(rms: RunningMeanStd, x: jax.Array) -> RunningMeanStd:
    # gymnasium RunningMeanStd.update on a batch of one sample (batch_var 0)
    delta = x - rms.mean
    total = rms.count + 1.0
    m2 = rms.var * rms.count + delta**2 * rms.count / total
    return RunningMeanStd(
        mean=rms.mean + delta / total, var=m2 / total, count=total
    )


def normalize_obs(rms: RunningMeanStd, obs: jax.Array) -> jax.Array:
    scaled = (obs - rms.mean) / jnp.sqrt(rms.var + EPSILON)
    return jnp.clip(scaled, -OBS_CLIP, OBS_CLIP)


def init_reward_scaler() -> RewardScaler:
    return RewardScaler(rms=init_rms(()), discounted=jnp.zeros(()))


def scale_reward(
    scaler: RewardScaler,
    reward: jax.Array,
    terminated: jax.Array,
    gamma: float,
) -> tuple[RewardScaler, jax.Array]:
    # gymnasium 1.4 NormalizeReward: discounts with (1 - terminated), so the
    # running return carries over truncations; no reset() hook either
    discounted = scaler.discounted * gamma * (1.0 - terminated) + reward
    rms = update_rms(scaler.rms, discounted)
    scaled = reward / jnp.sqrt(rms.var + EPSILON)
    clipped = jnp.clip(scaled, -REWARD_CLIP, REWARD_CLIP)
    return RewardScaler(rms=rms, discounted=discounted), clipped
