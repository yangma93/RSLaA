# -*- coding: utf-8 -*-
"""MAPPO 多智能体 PPO 训练 (Torch 原生实现).

架构:
  * 每个 LiDAR 一个 Actor (MLP: obs -> (mu, logstd) for Gaussian policy)
  * 一个中心化 Critic (MLP: global_state -> V(s))
  * 多 agent 同时采样动作, PPO clipped objective + value loss + entropy bonus
  * 奖励来自 marl_env.step() 返回的替代指标 (J_EIG/J_cov/...)

输出:
  checkpoints_dir/policy.pt (state_dict, 供 marl_deploy.py 加载)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rslaa.marl_env import MultiLidarEnv, ACTION_SEMANTICS
from rslaa.common import VOXEL, METRIC_FUNCS, DEFAULT_METRIC, LOWER_IS_BETTER
from rslaa.paths import CHECKPOINTS_DIR


# ---------------------------------------------------------------------------
# 网络
# ---------------------------------------------------------------------------
class Actor(nn.Module):
    """单智能体 Actor: obs -> Gaussian policy (mu, logstd)."""

    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.mu_head = nn.Linear(hidden, act_dim)
        self.logstd = nn.Parameter(torch.zeros(act_dim))  # 可学习全局 logstd

    def forward(self, obs):
        h = self.net(obs)
        mu = self.mu_head(h)
        logstd = self.logstd.expand_as(mu)
        return mu, logstd

    def get_action(self, obs, deterministic=False):
        mu, logstd = self.forward(obs)
        if deterministic:
            return mu, None, None
        dist = Normal(mu, logstd.exp())
        a = dist.sample()
        lp = dist.log_prob(a).sum(dim=-1)
        ent = dist.entropy().sum(dim=-1)
        return a, lp, ent


class Critic(nn.Module):
    """中心化 Critic: global_state -> V(s)."""

    def __init__(self, state_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state):
        return self.net(state).squeeze(-1)


class MAPPO(nn.Module):
    def __init__(self, n_agent, obs_dim, act_dim, state_dim,
                 actor_hidden=128, critic_hidden=256, device="cuda"):
        super().__init__()
        self.n_agent = n_agent
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.state_dim = state_dim
        self.actor_hidden = int(actor_hidden)
        self.critic_hidden = int(critic_hidden)
        self.env_meta = {}
        self.device = torch.device(device if torch.cuda.is_available() and "cuda" in device else "cpu")

        # N 个独立 Actor
        self.actors = nn.ModuleList(
            [Actor(obs_dim, act_dim, actor_hidden) for _ in range(n_agent)])
        self.critic = Critic(state_dim, critic_hidden)

        self.to(self.device)

    def _t(self, x):
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def select_actions(self, obs_dict, deterministic=False):
        actions = {}
        logps = {}
        ents = {}
        for i, actor in enumerate(self.actors):
            obs_i = self._t(obs_dict[i]).unsqueeze(0)
            a, lp, ent = actor.get_action(obs_i, deterministic)
            actions[i] = a.squeeze(0).cpu().numpy()
            if lp is not None:
                logps[i] = lp.squeeze(0)
                ents[i] = ent.squeeze(0)
        return actions, logps, ents

    def evaluate(self, obs_dict, state, actions):
        """为 PPO 更新评估 logprob / value / entropy.

        obs_dict: dict[i -> Tensor (B, obs_dim)]
        state: Tensor (B, state_dim)
        actions: dict[i -> Tensor (B, act_dim)]
        返回:
          logp (B, n_agent), value (B,), entropy (B, n_agent)
        """
        B = state.shape[0]
        logps = torch.zeros(B, self.n_agent, device=self.device)
        ents = torch.zeros(B, self.n_agent, device=self.device)
        for i, actor in enumerate(self.actors):
            mu, logstd = actor.forward(obs_dict[i])
            dist = Normal(mu, logstd.exp())
            logps[:, i] = dist.log_prob(actions[i]).sum(dim=-1)
            ents[:, i] = dist.entropy().sum(dim=-1)
        value = self.critic(state)
        return logps, value, ents

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        torch.save({
            "actor_state_dicts": [a.state_dict() for a in self.actors],
            "critic_state_dict": self.critic.state_dict(),
            "n_agent": self.n_agent,
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "state_dim": self.state_dim,
            "actor_hidden": self.actor_hidden,
            "critic_hidden": self.critic_hidden,
            "env_meta": self.env_meta,
        }, path)

    @classmethod
    def load(cls, path, device="cuda"):
        ckpt = torch.load(path, map_location=device)
        model = cls(ckpt["n_agent"], ckpt["obs_dim"], ckpt["act_dim"],
                    ckpt["state_dim"],
                    actor_hidden=ckpt.get("actor_hidden", 128),
                    critic_hidden=ckpt.get("critic_hidden", 256),
                    device=device)
        for i, sd in enumerate(ckpt["actor_state_dicts"]):
            model.actors[i].load_state_dict(sd)
        model.critic.load_state_dict(ckpt["critic_state_dict"])
        model.env_meta = ckpt.get("env_meta", {})
        return model

    @staticmethod
    def load_meta(path):
        """只读取 checkpoints 元信息 (不构建网络), 供部署/续训做一致性检查."""
        ckpt = torch.load(path, map_location="cpu")
        meta = {k: ckpt.get(k) for k in
                ("n_agent", "obs_dim", "act_dim", "state_dim",
                 "actor_hidden", "critic_hidden")}
        meta["env_meta"] = ckpt.get("env_meta", {})
        return meta

    def load_weights(self, path):
        """把已有权重加载到当前网络 (续训/微调用), 维度不一致时给出明确报错."""
        ckpt = torch.load(path, map_location=self.device)
        keys = ("n_agent", "obs_dim", "act_dim", "state_dim")
        mismatch = [f"{k}: ckpt={ckpt.get(k)} vs env={getattr(self, k)}"
                    for k in keys if ckpt.get(k) != getattr(self, k)]
        if mismatch:
            raise ValueError(
                "初始化权重与当前环境不兼容 (观测/状态维度不一致): "
                + "; ".join(mismatch)
                + "。若观测表征已修改, 请用该权重对应的环境配置或重新训练。")
        old_sem = (ckpt.get("env_meta") or {}).get("action_semantics")
        if old_sem != ACTION_SEMANTICS:
            print("[MAPPO] 警告: 初始权重的动作语义与当前环境不同 "
                  f"(权重={old_sem}, 当前={ACTION_SEMANTICS})。"
                  "旧权重按'相对上一窗的增量'训练, 直接接续会改变动作含义, "
                  "建议重新训练。")
        for i, sd in enumerate(ckpt["actor_state_dicts"]):
            if i >= self.n_agent:
                break
            self.actors[i].load_state_dict(sd)
        self.critic.load_state_dict(ckpt["critic_state_dict"])
        self.env_meta = ckpt.get("env_meta", {})
        return ckpt


# ---------------------------------------------------------------------------
# Rollout Buffer
# ---------------------------------------------------------------------------
class RolloutBuffer:
    def __init__(self):
        self.obs = {}       # agent_id -> list[np.ndarray]
        self.actions = {}   # agent_id -> list[np.ndarray]
        self.logps = {}     # agent_id -> list[np.ndarray] (1,)
        self.rewards = []   # list[float]
        self.states = []    # list[np.ndarray]
        self.dones = []     # list[bool]

    @staticmethod
    def _to_np(x):
        """统一转成独立 numpy 数组 (兼容 CUDA Tensor)."""
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return np.array(x, copy=True)

    def store(self, obs_dict, actions_dict, logps_dict, state, reward, done):
        for i, obs in obs_dict.items():
            self.obs.setdefault(i, []).append(self._to_np(obs))
        for i, a in actions_dict.items():
            self.actions.setdefault(i, []).append(self._to_np(a))
        for i, lp in logps_dict.items():
            self.logps.setdefault(i, []).append(self._to_np(lp).reshape(-1))
        self.states.append(self._to_np(state))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))

    def clear(self):
        self.__init__()

    def compute_gae(self, last_value, gamma=0.99, gae_lambda=0.95):
        """计算 GAE 优势. 这里只有全局 reward, 每个 agent 共享."""
        rewards = np.array(self.rewards, np.float64)
        dones = np.array(self.dones, np.float64)
        T = len(rewards)
        # 用 critic 计算每个 state 的 V(s)
        values = []
        for s in self.states:
            v = float(self.critic_eval(s))
            values.append(v)
        values.append(last_value)
        values = np.array(values, np.float64)

        advantages = np.zeros(T, np.float64)
        gae = 0.0
        for t in reversed(range(T)):
            mask = 1.0 - dones[t]
            delta = rewards[t] + gamma * values[t + 1] * mask - values[t]
            gae = delta + gamma * gae_lambda * mask * gae
            advantages[t] = gae
        returns = advantages + values[:T]
        return advantages, returns

    def to_tensors(self, device):
        """全部 -> torch.Tensor."""
        d = device
        obs = {i: torch.as_tensor(np.stack(v), dtype=torch.float32, device=d)
               for i, v in self.obs.items()}
        act = {i: torch.as_tensor(np.stack(v), dtype=torch.float32, device=d)
               for i, v in self.actions.items()}
        lp = {i: torch.as_tensor(np.stack(v).reshape(-1), dtype=torch.float32, device=d)
              for i, v in self.logps.items()}
        states = torch.as_tensor(np.stack(self.states), dtype=torch.float32, device=d)
        return obs, act, lp, states

    # 需要在绑定 model 后才能 compute_gae
    def bind_model(self, model):
        self._model = model

    def critic_eval(self, state):
        s = torch.as_tensor(state, dtype=torch.float32,
                            device=self._model.device).unsqueeze(0)
        with torch.no_grad():
            return float(self._model.critic(s).cpu().item())


# ---------------------------------------------------------------------------
# 训练曲线可视化
# ---------------------------------------------------------------------------
class TrainingPlotter:
    """训练曲线实时可视化 (字体 Arial, 字号 16).

    每个 interval 轮刷新一次窗口, 并保存 training_curves.png;
    matplotlib 不可用时自动关闭可视化, 不影响训练.
    """

    def __init__(self, save_dir=None, metric=DEFAULT_METRIC, enabled=True,
                 show=True, interval=5):
        self.enabled = bool(enabled)
        self.show = bool(show)
        self.interval = max(int(interval), 1)
        self.save_path = (os.path.join(save_dir, "training_curves.png")
                          if save_dir else None)
        self.plt = None
        if not self.enabled:
            return
        try:
            import matplotlib
            if not self.show:
                matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib import font_manager

            has_arial = any(f.name == "Arial"
                            for f in font_manager.fontManager.ttflist)
            plt.rcParams.update({
                "font.family": "Arial" if has_arial else "DejaVu Sans",
                "font.size": 16,
                "axes.titlesize": 16,
                "axes.labelsize": 16,
                "legend.fontsize": 16,
                "xtick.labelsize": 16,
                "ytick.labelsize": 16,
            })
            if not has_arial:
                print("[Plot] 未找到 Arial 字体, 回退为 DejaVu Sans")

            self.plt = plt
            self.fig, axs = plt.subplots(2, 2, figsize=(14, 9))
            self.ax_rew, self.ax_actor, self.ax_critic, self.ax_ent = axs.ravel()

            self.ln_rew, = self.ax_rew.plot([], [], lw=1.2, color="#7f9cf5",
                                            label="episode reward")
            self.ln_avg, = self.ax_rew.plot([], [], lw=2.2, color="#e53e3e",
                                            label="moving average")
            self.ax_rew.set_title("Episode Reward")
            self.ax_rew.set_xlabel("Episode")
            self.ax_rew.set_ylabel("Reward")
            self.ax_rew.legend(loc="best")

            self.ln_actor, = self.ax_actor.plot([], [], lw=1.6, color="#2b6cb0")
            self.ax_actor.set_title("Actor Loss")
            self.ax_actor.set_xlabel("Episode")
            self.ax_actor.set_ylabel("Loss")

            self.ln_critic, = self.ax_critic.plot([], [], lw=1.6,
                                                  color="#2f855a")
            self.ax_critic.set_title("Critic Loss")
            self.ax_critic.set_xlabel("Episode")
            self.ax_critic.set_ylabel("Loss")

            self.ln_ent, = self.ax_ent.plot([], [], lw=1.6, color="#b7791f")
            self.ax_ent.set_title("Policy Entropy")
            self.ax_ent.set_xlabel("Episode")
            self.ax_ent.set_ylabel("Entropy")

            self.fig.suptitle(f"MAPPO Training — {metric}", fontsize=16)
            self.fig.tight_layout(rect=(0, 0, 1, 0.96))
            if self.show:
                plt.ion()
                plt.show()
        except Exception as exc:       # pragma: no cover - 环境相关
            print(f"[Plot] 可视化不可用 ({exc}), 已关闭")
            self.enabled = False
            self.plt = None

    def update(self, ep, rewards, actor_losses, critic_losses, entropies):
        if not self.enabled:
            return
        if ep % self.interval != 0:
            return
        try:
            x = np.arange(1, len(rewards) + 1)
            self.ln_rew.set_data(x, rewards)
            win = min(10, len(rewards))
            if win > 0:
                avg = np.convolve(np.asarray(rewards, np.float64),
                                  np.ones(win) / win, mode="valid")
                self.ln_avg.set_data(x[win - 1:], avg)
            self.ln_actor.set_data(x, actor_losses)
            self.ln_critic.set_data(x, critic_losses)
            self.ln_ent.set_data(x, entropies)
            xmax = max(len(rewards), 2)
            for ax in (self.ax_rew, self.ax_actor, self.ax_critic, self.ax_ent):
                ax.set_xlim(0.5, xmax + 0.5)
                ax.relim()
                ax.autoscale_view()
            self.fig.canvas.draw_idle()
            if self.show:
                self.plt.pause(0.05)
            if self.save_path:
                self.fig.savefig(self.save_path, dpi=150, bbox_inches="tight")
        except Exception as exc:       # pragma: no cover - 环境相关
            print(f"[Plot] 刷新失败 ({exc}), 已关闭可视化")
            self.enabled = False

    def close(self):
        if not self.enabled:
            return
        try:
            if self.save_path:
                self.fig.savefig(self.save_path, dpi=150, bbox_inches="tight")
            self.plt.close(self.fig)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 训练循环
# ---------------------------------------------------------------------------
def train(env, n_episodes=200, max_steps=None,
          lr_actor=3e-4, lr_critic=1e-3,
          gamma=0.99, gae_lambda=0.95,
          clip_eps=0.2, ent_coef=0.01, vf_coef=0.5,
          ppo_epochs=4, batch_size=64,
          actor_hidden=128, critic_hidden=256,
          device="cuda", save_dir=None,
          log_interval=5,
          init_policy=None,
          plot=True, plot_show=True, plot_interval=None,
          verbose=True):
    """MAPPO 训练主循环.

    每 episode 收集 rollout (max_steps 步), 然后 PPO 多 epoch 更新.
    init_policy 可指定已有权重 (在新场景/新一轮次上接续训练).
    """
    n_agent = env.n_lidar
    obs_dim = env.obs_dim_i
    act_dim = env.action_dim_i
    state_dim = env.state_global_dim
    max_steps = max_steps or env.max_steps

    if verbose:
        print(f"[MAPPO] n_agent={n_agent}, obs_dim={obs_dim}, "
              f"act_dim={act_dim}, state_dim={state_dim}")
        print(f"[MAPPO] episodes={n_episodes}, max_steps={max_steps}, "
              f"device={device}")
        _dir = "越小越好 (min)" if LOWER_IS_BETTER.get(env.metric, False) \
            else "越大越好 (max)"
        print(f"[MAPPO] 奖励指标: {env.metric} [{_dir}] "
              f"(reward 已统一为越大越好, 训练按 ep_reward 最大化)")
        print(f"[MAPPO] 位姿约束: 初始位姿 ±{env.pos_margin} m / "
              f"±{np.rad2deg(env.ang_rad):.1f}°, 全窗锚定 (不累积漂移)")

    model = MAPPO(n_agent, obs_dim, act_dim, state_dim,
                  actor_hidden=actor_hidden, critic_hidden=critic_hidden,
                  device=device)
    model.env_meta = env.meta() if hasattr(env, "meta") else {}
    if init_policy:
        model.load_weights(init_policy)
        if verbose:
            print(f"[MAPPO] 已加载初始权重: {init_policy} (接续训练)")
    buffer = RolloutBuffer()
    buffer.bind_model(model)

    actor_optim = torch.optim.Adam(model.actors.parameters(), lr=lr_actor)
    critic_optim = torch.optim.Adam(model.critic.parameters(), lr=lr_critic)

    save_dir = save_dir or str(CHECKPOINTS_DIR)
    os.makedirs(save_dir, exist_ok=True)

    plotter = TrainingPlotter(save_dir=save_dir, metric=env.metric,
                              enabled=plot, show=plot_show,
                              interval=plot_interval or log_interval)

    best_reward = -np.inf
    reward_history = []
    actor_history = []
    critic_history = []
    entropy_history = []
    len_history = []
    init_pose_history = []

    for ep in range(1, n_episodes + 1):
        t_ep = time.time()
        obs = env.reset()
        init_pose_history.append(
            np.asarray(env._initial_poses, np.float64).tolist())
        ep_reward = 0.0
        ep_len = 0
        buffer.clear()

        for _ in range(max_steps):
            state = env.get_global_state()
            actions, logps, ents = model.select_actions(obs)
            next_obs, reward, done, info = env.step(actions)
            buffer.store(obs, actions, logps, state, reward, done)
            ep_reward += reward
            ep_len += 1
            obs = next_obs
            if done:
                break

        # --- GAE ---
        with torch.no_grad():
            last_state = torch.as_tensor(
                env.get_global_state(), dtype=torch.float32,
                device=model.device).unsqueeze(0)
            last_value = float(model.critic(last_state).cpu().item())
        advantages, returns = buffer.compute_gae(last_value, gamma, gae_lambda)

        # --- 转 tensor ---
        obs_t, act_t, lp_t, states_t = buffer.to_tensors(model.device)
        adv_t = torch.as_tensor(advantages, dtype=torch.float32,
                                device=model.device)
        ret_t = torch.as_tensor(returns, dtype=torch.float32,
                                device=model.device)
        # 每个 agent 共享同一个 advantage
        adv_expanded = adv_t.unsqueeze(1).expand(-1, n_agent)

        T = len(rewards := buffer.rewards)

        # --- PPO 多 epoch 更新 ---
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        indices = np.arange(T)
        for _ in range(ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, T, batch_size):
                idx = indices[start:start + batch_size]
                b_obs = {i: obs_t[i][idx] for i in range(n_agent)}
                b_act = {i: act_t[i][idx] for i in range(n_agent)}
                b_lp_old = {i: lp_t[i][idx] for i in range(n_agent)}
                b_states = states_t[idx]
                b_adv = adv_expanded[idx]
                b_ret = ret_t[idx]

                # evaluate
                b_lp_new, b_vals, b_ent = model.evaluate(b_obs, b_states, b_act)

                # Actor loss (clipped PPO)
                ratios = torch.zeros(len(idx), n_agent, device=model.device)
                for i in range(n_agent):
                    ratios[:, i] = (b_lp_new[:, i] - b_lp_old[i]).exp()
                surr1 = ratios * b_adv
                surr2 = torch.clamp(ratios, 1 - clip_eps, 1 + clip_eps) * b_adv
                actor_loss = -torch.min(surr1, surr2).mean() \
                    - ent_coef * b_ent.mean()

                # Critic loss
                critic_loss = 0.5 * (b_vals - b_ret) ** 2
                critic_loss = critic_loss.mean()

                actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(model.actors.parameters(), 0.5)
                actor_optim.step()

                critic_optim.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(model.critic.parameters(), 0.5)
                critic_optim.step()

                total_actor_loss += float(actor_loss.item())
                total_critic_loss += float(critic_loss.item())
                total_entropy += float(b_ent.mean().item())
                n_updates += 1

        avg_actor = total_actor_loss / max(n_updates, 1)
        avg_critic = total_critic_loss / max(n_updates, 1)
        avg_ent = total_entropy / max(n_updates, 1)

        reward_history.append(ep_reward)
        actor_history.append(avg_actor)
        critic_history.append(avg_critic)
        entropy_history.append(avg_ent)
        len_history.append(ep_len)
        plotter.update(ep, reward_history, actor_history,
                       critic_history, entropy_history)
        if ep_reward > best_reward:
            best_reward = ep_reward
            model.save(os.path.join(save_dir, "policy_best.pt"))

        if verbose and (ep % log_interval == 0 or ep == 1):
            elapsed = time.time() - t_ep
            avg_w = np.mean(reward_history[-log_interval:])
            print(f"[EP {ep:4d}] reward={ep_reward:+.4f} (avg_last={avg_w:+.4f}), "
                  f"len={ep_len}, actor={avg_actor:+.3f}, critic={avg_critic:.3f}, "
                  f"ent={avg_ent:.3f}, t={elapsed:.1f}s")

    # 最终保存
    plotter.close()
    model.save(os.path.join(save_dir, "policy_final.pt"))

    # 训练记录
    with open(os.path.join(save_dir, "train_history.json"), "w") as f:
        json.dump({
            "n_episodes": n_episodes,
            "max_steps": max_steps,
            "rewards": [float(r) for r in reward_history],
            "actor_loss": [float(v) for v in actor_history],
            "critic_loss": [float(v) for v in critic_history],
            "entropy": [float(v) for v in entropy_history],
            "ep_len": [int(v) for v in len_history],
            "init_poses": init_pose_history,
            "env_meta": model.env_meta,
            "config": {
                "lr_actor": lr_actor, "lr_critic": lr_critic,
                "gamma": gamma, "gae_lambda": gae_lambda,
                "clip_eps": clip_eps, "ent_coef": ent_coef,
                "vf_coef": vf_coef, "ppo_epochs": ppo_epochs,
                "actor_hidden": actor_hidden, "critic_hidden": critic_hidden,
                "init_policy": init_policy,
            }
        }, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f"\n[MAPPO] 训练完成! best_reward={best_reward:.4f}, "
              f"policy 保存于 {save_dir}")

    return model, reward_history


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traj", required=True)
    ap.add_argument("--env", default=None)
    ap.add_argument("--models", required=True)
    ap.add_argument("--presets", default="vlp16,vlp16",
                    help="LiDAR 型号列表 (逗号分隔, 自动按 n-lidar 循环)")
    ap.add_argument("--initial-poses", default=None,
                    help="可选: 初始位姿 .json (含 poses 字段或纯数组) "
                         "或 .txt/.csv (逗号分隔, 每行 [x,y,z,roll,pitch,yaw])")
    ap.add_argument("--pose-json", default=None,
                    help="可选: 直接用 lidar_pose.json (与 --initial-poses 互斥)")
    ap.add_argument("--n-lidar", type=int, default=2)
    ap.add_argument("--window-min", type=int, default=5, choices=[5, 15, 20, 30])
    ap.add_argument("--pos-margin", type=float, default=0.5,
                    help="相对初始位姿的位置盒半宽 (m), 全窗锚定")
    ap.add_argument("--ang-margin-deg", type=float, default=15.0,
                    help="相对初始位姿的姿态盒半宽 (°), 全窗锚定")
    ap.add_argument("--metric", default=DEFAULT_METRIC, choices=list(METRIC_FUNCS.keys()),
                    help="训练奖励指标 (默认 J_mv, 与感知指标正相关, 越大越好; "
                         "J_cov/J_dyn/J_traj 越小越好; J_EIG 越大越好)")
    ap.add_argument("--voxel", type=float, default=VOXEL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-episodes", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=0,
                    help="每 episode 最大步数 (0=自动=窗口数)")
    ap.add_argument("--actor-hidden", type=int, default=128)
    ap.add_argument("--critic-hidden", type=int, default=256)
    ap.add_argument("--lr-actor", type=float, default=3e-4)
    ap.add_argument("--lr-critic", type=float, default=1e-3)
    ap.add_argument("--ppo-epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--save-dir", default=None)
    ap.add_argument("--log-interval", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frame-step", type=int, default=1)
    ap.add_argument("--init-policy", default=None,
                    help="已有权重 (.pt): 在新场景/新一轮次上接续训练")
    ap.add_argument("--randomize-init-poses", dest="randomize_init_poses",
                    action="store_true", default=True,
                    help="每轮重采样初始位姿 (默认开启; 首轮用指定位姿)")
    ap.add_argument("--no-randomize-init-poses", dest="randomize_init_poses",
                    action="store_false",
                    help="关闭每轮重采样, 固定使用指定/自动初始位姿")
    ap.add_argument("--init-z-min", type=float, default=5.0,
                    help="随机初始位姿离地高度下限 (m)")
    ap.add_argument("--init-z-max", type=float, default=8.0,
                    help="随机初始位姿离地高度上限 (m)")
    ap.add_argument("--no-plot", dest="plot", action="store_false",
                    default=True, help="关闭训练曲线可视化")
    ap.add_argument("--no-plot-show", dest="plot_show", action="store_false",
                    default=True, help="只保存曲线图, 不弹窗显示")
    ap.add_argument("--plot-interval", type=int, default=0,
                    help="曲线刷新间隔 (0=与 log-interval 相同)")
    return ap.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()

    # 初始位姿处理
    initial_poses = None
    if args.pose_json:
        from rslaa.common import load_lidar_pose_json
        pj = load_lidar_pose_json(args.pose_json)
        initial_poses = pj["poses"]
        args.n_lidar = pj["n_lidar"]
        args.presets = ",".join(pj["presets"])
    elif args.initial_poses:
        # 支持 .json (含 "poses" 字段或纯 [[x,y,z,roll,pitch,yaw], ...])
        # 与 .txt/.csv (np.loadtxt 逗号分隔, 每行一个位姿)
        if str(args.initial_poses).lower().endswith(".json"):
            with open(args.initial_poses, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                initial_poses = np.asarray(obj["poses"], np.float64)
                if obj.get("n_lidar"):
                    args.n_lidar = int(obj["n_lidar"])
                if obj.get("presets"):
                    args.presets = ",".join(obj["presets"])
            else:
                initial_poses = np.asarray(obj, np.float64)
        else:
            initial_poses = np.loadtxt(args.initial_poses, delimiter=",")
        initial_poses = np.atleast_2d(initial_poses).astype(np.float64)
        if initial_poses.shape[1] != 6:
            raise SystemExit("--initial-poses 每个位姿需为 6 维 [x,y,z,roll,pitch,yaw]")
        if initial_poses.shape[0] != args.n_lidar:
            raise SystemExit(
                f"--initial-poses 提供 {initial_poses.shape[0]} 个位姿, "
                f"但 n_lidar={args.n_lidar} (请同时调整 --n-lidar 或 JSON 中的 n_lidar)")

    presets = (args.presets.split(",") * args.n_lidar)[:args.n_lidar]

    env = MultiLidarEnv(
        traj_path=args.traj, env_path=args.env, models_dir=args.models,
        presets=presets, initial_poses=initial_poses, n_lidar=args.n_lidar,
        pos_margin=args.pos_margin, ang_margin_deg=args.ang_margin_deg,
        window_minutes=args.window_min,
        metric=args.metric, voxel=args.voxel, device=args.device,
        max_steps=args.max_steps if args.max_steps > 0 else None,
        seed=args.seed, frame_step=args.frame_step,
        randomize_init_poses=args.randomize_init_poses,
        init_z_range=(args.init_z_min, args.init_z_max),
    )

    train(env, n_episodes=args.n_episodes,
          max_steps=args.max_steps if args.max_steps > 0 else None,
          lr_actor=args.lr_actor, lr_critic=args.lr_critic,
          gamma=args.gamma, gae_lambda=args.gae_lambda,
          clip_eps=args.clip_eps, ent_coef=args.ent_coef,
          vf_coef=0.5, ppo_epochs=args.ppo_epochs,
          batch_size=args.batch_size,
          actor_hidden=args.actor_hidden, critic_hidden=args.critic_hidden,
          device=args.device,
          save_dir=args.save_dir, log_interval=args.log_interval,
          init_policy=args.init_policy,
          plot=args.plot, plot_show=args.plot_show,
          plot_interval=args.plot_interval or None)
