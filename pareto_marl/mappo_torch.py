import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions.normal import Normal

ENV_ID = "Ant-v5"
OBS_CLIP = 10.0
REWARD_CLIP = 10.0
EVAL_SEED_OFFSET = 10_000

LogFn = Callable[[dict[str, float], int], None]


@dataclass(frozen=True)
class PPOConfig:
    total_steps: int = 3_000_000
    num_envs: int = 8
    num_steps: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    update_epochs: int = 10
    num_minibatches: int = 32
    clip_coef: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    max_grad_norm: float = 0.5
    eval_episodes: int = 10

    def __post_init__(self) -> None:
        if self.batch_size % self.num_minibatches:
            raise ValueError(
                f"batch {self.batch_size} = num_envs * num_steps is not "
                f"divisible by num_minibatches={self.num_minibatches}"
            )
        if self.num_iterations < 1:
            raise ValueError(
                f"total_steps={self.total_steps} < one batch "
                f"({self.batch_size}): raise total_steps"
            )

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches

    @property
    def num_iterations(self) -> int:
        return self.total_steps // self.batch_size


@dataclass(frozen=True)
class Agents:
    obs_idx: list[np.ndarray]
    act_idx: list[np.ndarray]

    @property
    def k(self) -> int:
        return len(self.act_idx)

    @property
    def action_dim(self) -> int:
        return sum(len(idx) for idx in self.act_idx)

    def check(self, obs_dim: int, action_dim: int, env_name: str) -> None:
        observed = np.unique(np.concatenate(self.obs_idx))
        if not np.array_equal(observed, np.arange(obs_dim)):
            raise ValueError(
                f"agent obs indices cover state dims {observed.min()}.."
                f"{observed.max()} ({len(observed)} dims), but {env_name} "
                f"has obs dim {obs_dim}: the indices were built for another "
                "model or obs layout; rebuild them with `just indices`"
            )
        owned = np.sort(np.concatenate(self.act_idx))
        if not np.array_equal(owned, np.arange(action_dim)):
            raise ValueError(
                f"agent action indices {owned.tolist()} do not own each of "
                f"the {action_dim} actuators of {env_name} exactly once; "
                "rebuild them with `just indices`"
            )


def layer_init(layer: nn.Linear, std: float = np.sqrt(2)) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, 0.0)
    return layer


def mlp(in_dim: int, out_dim: int, out_std: float) -> nn.Sequential:
    return nn.Sequential(
        layer_init(nn.Linear(in_dim, 64)),
        nn.Tanh(),
        layer_init(nn.Linear(64, 64)),
        nn.Tanh(),
        layer_init(nn.Linear(64, out_dim), std=out_std),
    )


class MultiAgentActor(nn.Module):
    def __init__(self, agents: Agents) -> None:
        super().__init__()
        self.obs_idx = [torch.as_tensor(i) for i in agents.obs_idx]
        self.means = nn.ModuleList(
            mlp(len(o), len(a), out_std=0.01)
            for o, a in zip(agents.obs_idx, agents.act_idx, strict=True)
        )
        self.log_stds = nn.ParameterList(
            nn.Parameter(torch.zeros(len(a))) for a in agents.act_idx
        )
        order = np.concatenate(agents.act_idx)
        self.to_actuator = torch.as_tensor(np.argsort(order))
        owner = np.empty(agents.action_dim, dtype=np.int64)
        for k, idx in enumerate(agents.act_idx):
            owner[idx] = k
        self.agent_of = nn.functional.one_hot(
            torch.as_tensor(owner), agents.k
        ).float()

    def distribution(self, obs: torch.Tensor) -> Normal:
        mean = torch.cat(
            [
                net(obs[:, i])
                for net, i in zip(self.means, self.obs_idx, strict=True)
            ],
            -1,
        )
        log_std = torch.cat(list(self.log_stds)).expand_as(mean)
        return Normal(
            mean[:, self.to_actuator], log_std[:, self.to_actuator].exp()
        )

    def per_agent(self, per_actuator: torch.Tensor) -> torch.Tensor:
        return per_actuator @ self.agent_of


@dataclass
class Rollout:
    obs: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor


def make_env(env_kwargs: dict[str, Any], gamma: float) -> Callable[[], gym.Env]:
    def thunk() -> gym.Env:
        env = gym.make(ENV_ID, **env_kwargs)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(
            env,
            lambda o: np.clip(o, -OBS_CLIP, OBS_CLIP),
            env.observation_space,
        )
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        return gym.wrappers.TransformReward(
            env, lambda r: min(max(float(r), -REWARD_CLIP), REWARD_CLIP)
        )

    return thunk


def obs_normalizer(env: gym.Env) -> gym.wrappers.NormalizeObservation:
    wrapper: Any = env
    while not isinstance(wrapper, gym.wrappers.NormalizeObservation):
        wrapper = wrapper.env
    return wrapper


class MAPPOTrainer:
    def __init__(
        self,
        agents: Agents,
        env_kwargs: dict[str, Any],
        cfg: PPOConfig,
        seed: int,
    ) -> None:
        torch.manual_seed(seed)
        self.agents, self.env_kwargs = agents, env_kwargs
        self.cfg, self.seed = cfg, seed
        self.rng = np.random.default_rng(seed)
        self.envs = gym.vector.SyncVectorEnv(
            [make_env(env_kwargs, cfg.gamma)] * cfg.num_envs,
            autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
        )
        state_dim = gym.spaces.flatdim(self.envs.single_observation_space)
        action_dim = gym.spaces.flatdim(self.envs.single_action_space)
        agents.check(state_dim, action_dim, f"{ENV_ID} {env_kwargs}")
        # critic first: same init RNG order as CleanRL, so K=1 matches it
        self.critic = mlp(state_dim, 1, out_std=1.0)
        self.actor = MultiAgentActor(agents)
        params = [*self.critic.parameters(), *self.actor.parameters()]
        self.optimizer = torch.optim.Adam(
            params, lr=cfg.learning_rate, eps=1e-5
        )
        self.curve: list[tuple[int, float]] = []
        self.global_step = 0

    def train(self, log: LogFn) -> None:
        obs, _ = self.envs.reset(seed=self.seed)
        next_obs = torch.as_tensor(obs, dtype=torch.float32)
        next_done = torch.zeros(self.cfg.num_envs)
        start = time.time()
        for it in range(self.cfg.num_iterations):
            frac = 1.0 - it / self.cfg.num_iterations
            self.optimizer.param_groups[0]["lr"] = frac * self.cfg.learning_rate
            n_episodes = len(self.curve)
            rollout, next_obs, next_done = self.collect(next_obs, next_done)
            advantages, returns = self.gae(rollout, next_obs, next_done)
            metrics = self.update(rollout, advantages, returns)
            metrics["perf/sps"] = self.global_step / (time.time() - start)
            finished = [r for _, r in self.curve[n_episodes:]]
            if finished:
                metrics["train/episodic_return"] = float(np.mean(finished))
            log(metrics, self.global_step)

    def collect(
        self, next_obs: torch.Tensor, next_done: torch.Tensor
    ) -> tuple[Rollout, torch.Tensor, torch.Tensor]:
        n, e = self.cfg.num_steps, self.cfg.num_envs
        ro = Rollout(
            obs=torch.zeros(n, e, next_obs.shape[1]),
            actions=torch.zeros(n, e, self.agents.action_dim),
            logprobs=torch.zeros(n, e, self.agents.k),
            rewards=torch.zeros(n, e),
            dones=torch.zeros(n, e),
            values=torch.zeros(n, e),
        )
        for step in range(n):
            self.global_step += e
            ro.obs[step], ro.dones[step] = next_obs, next_done
            with torch.no_grad():
                dist = self.actor.distribution(next_obs)
                action = dist.sample()
                ro.logprobs[step] = self.actor.per_agent(dist.log_prob(action))
                ro.values[step] = self.critic(next_obs).flatten()
            ro.actions[step] = action
            obs, reward, term, trunc, info = self.envs.step(action.numpy())
            ro.rewards[step] = torch.as_tensor(reward, dtype=torch.float32)
            next_obs = torch.as_tensor(obs, dtype=torch.float32)
            next_done = torch.as_tensor(term | trunc, dtype=torch.float32)
            self.record_episodes(info)
        return ro, next_obs, next_done

    def record_episodes(self, info: dict[str, Any]) -> None:
        final = info.get("final_info")
        if final is None or "_episode" not in final:
            return
        for r in final["episode"]["r"][final["_episode"]]:
            self.curve.append((self.global_step, float(r)))

    def gae(
        self, ro: Rollout, next_obs: torch.Tensor, next_done: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        with torch.no_grad():
            next_value = self.critic(next_obs).flatten()
        advantages = torch.zeros_like(ro.rewards)
        last = torch.zeros(cfg.num_envs)
        for t in reversed(range(cfg.num_steps)):
            if t == cfg.num_steps - 1:
                nonterminal, values_next = 1.0 - next_done, next_value
            else:
                nonterminal, values_next = (
                    1.0 - ro.dones[t + 1],
                    ro.values[t + 1],
                )
            delta = (
                ro.rewards[t]
                + cfg.gamma * values_next * nonterminal
                - ro.values[t]
            )
            last = delta + cfg.gamma * cfg.gae_lambda * nonterminal * last
            advantages[t] = last
        return advantages, advantages + ro.values

    def update(
        self, ro: Rollout, advantages: torch.Tensor, returns: torch.Tensor
    ) -> dict[str, float]:
        batch = {
            "obs": ro.obs.reshape(-1, ro.obs.shape[-1]),
            "actions": ro.actions.reshape(-1, self.agents.action_dim),
            "logprobs": ro.logprobs.reshape(-1, self.agents.k),
            "advantages": advantages.reshape(-1),
            "returns": returns.reshape(-1),
            "values": ro.values.reshape(-1),
        }
        stats: dict[str, list[float]] = {}
        for _ in range(self.cfg.update_epochs):
            perm = self.rng.permutation(self.cfg.batch_size)
            for start in range(0, self.cfg.batch_size, self.cfg.minibatch_size):
                mb = perm[start : start + self.cfg.minibatch_size]
                step = self.minibatch_step({k: v[mb] for k, v in batch.items()})
                for name, value in step.items():
                    stats.setdefault(name, []).append(value)
        return {name: float(np.mean(v)) for name, v in stats.items()}

    def minibatch_step(self, mb: dict[str, torch.Tensor]) -> dict[str, float]:
        cfg = self.cfg
        dist = self.actor.distribution(mb["obs"])
        newlogprob = self.actor.per_agent(dist.log_prob(mb["actions"]))
        entropy = dist.entropy().sum(-1)
        logratio = newlogprob - mb["logprobs"]
        ratio = logratio.exp()
        adv = mb["advantages"]
        adv = ((adv - adv.mean()) / (adv.std() + 1e-8)).unsqueeze(-1)
        clipped = torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
        # per-agent clipped surrogate, summed over agents (MAPPO)
        pg_loss = torch.max(-adv * ratio, -adv * clipped).mean(0).sum()

        newvalue = self.critic(mb["obs"]).flatten()
        v_clipped = mb["values"] + torch.clamp(
            newvalue - mb["values"], -cfg.clip_coef, cfg.clip_coef
        )
        v_loss = (
            0.5
            * torch.max(
                (newvalue - mb["returns"]) ** 2,
                (v_clipped - mb["returns"]) ** 2,
            ).mean()
        )

        loss = pg_loss - cfg.ent_coef * entropy.mean() + cfg.vf_coef * v_loss
        self.optimizer.zero_grad()
        loss.backward()
        params = self.optimizer.param_groups[0]["params"]
        nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
        self.optimizer.step()
        with torch.no_grad():
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfrac = ((ratio - 1).abs() > cfg.clip_coef).float().mean()
        return {
            "losses/policy_loss": pg_loss.item(),
            "losses/value_loss": v_loss.item(),
            "losses/entropy": entropy.mean().item(),
            "losses/approx_kl": approx_kl.item(),
            "losses/clipfrac": clipfrac.item(),
        }

    def evaluate(self) -> dict[str, float]:
        rms = obs_normalizer(self.envs.envs[0]).obs_rms
        env = gym.make(ENV_ID, **self.env_kwargs)
        returns, velocities = [], []
        for ep in range(self.cfg.eval_episodes):
            obs, _ = env.reset(seed=self.seed + EVAL_SEED_OFFSET + ep)
            total, vel, done = 0.0, [], False
            while not done:
                norm = np.clip(
                    (obs - rms.mean) / np.sqrt(rms.var + 1e-8),
                    -OBS_CLIP,
                    OBS_CLIP,
                )
                with torch.no_grad():
                    x = torch.as_tensor(norm, dtype=torch.float32).unsqueeze(0)
                    action = self.actor.distribution(x).mean[0].numpy()
                obs, reward, term, trunc, info = env.step(
                    np.clip(action, -1, 1)
                )
                total += float(reward)
                vel.append(info["x_velocity"])
                done = term or trunc
            returns.append(total)
            velocities.append(float(np.mean(vel)))
        env.close()
        return {
            "eval_return": float(np.mean(returns)),
            "eval_return_std": float(np.std(returns)),
            "eval_x_velocity": float(np.mean(velocities)),
        }

    def close(self) -> None:
        self.envs.close()
