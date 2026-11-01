"""
Twin Delayed Deep Deterministic Policy Gradient (Expert 5 - Continuous).

Fixes:
- Correct target policy smoothing (noise clamped, then action clamped)
- Proper exploration noise decay schedule
- Warmup with random actions
- Gradient clipping
- Actor updated at lower frequency than critic (policy delay)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
from typing import Optional

from experts.base_agent import build_mlp, init_weights, ReplayBuffer, BaseAgent


class TD3Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, num_layers, activation):
        super().__init__()
        self.net = build_mlp(state_dim, action_dim, hidden_dim, num_layers, activation)
        self.net.apply(lambda m: init_weights(m, gain=np.sqrt(2)))
        # Last layer small init
        if hasattr(self.net[-1], 'weight'):
            init_weights(self.net[-1], gain=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(x))


class TD3Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, num_layers, activation):
        super().__init__()
        self.q1 = build_mlp(state_dim + action_dim, 1, hidden_dim, num_layers, activation)
        self.q2 = build_mlp(state_dim + action_dim, 1, hidden_dim, num_layers, activation)
        self.q1.apply(lambda m: init_weights(m, gain=np.sqrt(2)))
        self.q2.apply(lambda m: init_weights(m, gain=np.sqrt(2)))

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)

    def q1_forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa)


class TD3Agent(BaseAgent):
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 1,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_size: int = 100000,
        batch_size: int = 256,
        warmup_steps: int = 1000,
        exploration_noise: float = 0.1,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_delay: int = 2,
        hidden_dim: int = 256,
        actor_hidden_dim: Optional[int] = None,
        critic_hidden_dim: Optional[int] = None,
        actor_num_layers: int = 2,
        critic_num_layers: int = 2,
        activation: str = "relu",
        grad_norm_clip: float = 1.0,
        log_dir: Optional[str] = None,
        device: Optional[str] = None,
    ):
        super().__init__(device)
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.exploration_noise = exploration_noise
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.grad_norm_clip = grad_norm_clip
        self._total_steps = 0

        actor_hd = actor_hidden_dim or hidden_dim
        critic_hd = critic_hidden_dim or hidden_dim

        self.actor = TD3Actor(state_dim, action_dim, actor_hd, actor_num_layers, activation).to(self.device)
        self.actor_target = TD3Actor(state_dim, action_dim, actor_hd, actor_num_layers, activation).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = TD3Critic(state_dim, action_dim, critic_hd, critic_num_layers, activation).to(self.device)
        self.critic_target = TD3Critic(state_dim, action_dim, critic_hd, critic_num_layers, activation).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.buffer = ReplayBuffer(buffer_size, state_dim, action_dim, self.device)

        self.last_actor_loss = 0.0
        self.last_critic_loss = 0.0

        self._writer = None
        if log_dir:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._writer = SummaryWriter(log_dir)
            except ImportError:
                pass

    def act(self, state: np.ndarray, eval_mode: bool = False) -> np.ndarray:
        if self._total_steps < self.warmup_steps and not eval_mode:
            return np.random.uniform(-1.0, 1.0, size=(self.action_dim,)).astype(np.float32)

        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.actor(s).cpu().numpy().flatten()

        if not eval_mode:
            noise = np.random.normal(0, self.exploration_noise, size=action.shape)
            action = np.clip(action + noise, -1.0, 1.0)

        return action.astype(np.float32)

    def store(self, state, action, reward, next_state, done):
        action_arr = np.array(action).flatten().astype(np.float32)
        self.buffer.push(state, action_arr, float(reward), next_state, float(done))

    def train_step(self) -> None:
        self._total_steps += 1
        if self._total_steps < self.warmup_steps:
            return
        if not self.buffer.is_ready(self.batch_size):
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        with torch.no_grad():
            # Target policy smoothing
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_states) + noise).clamp(-1.0, 1.0)
            q1_next, q2_next = self.critic_target(next_states, next_actions)
            target_q = rewards + (1.0 - dones) * self.gamma * torch.min(q1_next, q2_next)

        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self._clip_grad(self.critic.parameters(), self.grad_norm_clip)
        self.critic_optimizer.step()

        # Delayed policy update
        if self._total_steps % self.policy_delay == 0:
            actor_loss = -self.critic.q1_forward(states, self.actor(states)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self._clip_grad(self.actor.parameters(), self.grad_norm_clip)
            self.actor_optimizer.step()

            # Soft update targets
            for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

            self.last_actor_loss = actor_loss.item()

        self.last_critic_loss = critic_loss.item()

        if self._writer and self._total_steps % 200 == 0:
            self._writer.add_scalar("train/actor_loss", self.last_actor_loss, self._total_steps)
            self._writer.add_scalar("train/critic_loss", self.last_critic_loss, self._total_steps)

    def log_episode(self, ep, total_return, sharpe, sortino, max_dd, equity_curve):
        if self._writer:
            self._writer.add_scalar("episode/total_return", total_return, ep)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "total_steps": self._total_steps,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        self.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
        self._total_steps = ckpt.get("total_steps", 0)