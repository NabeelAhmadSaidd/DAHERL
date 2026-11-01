"""
Soft Actor-Critic (Expert 2 - Continuous actions).

Fixes:
- Automatic entropy tuning via target entropy = -action_dim
- Twin critics for overestimation bias reduction
- Correct action scaling to [-1, 1]
- Proper log_alpha gradient flow
- Warmup phase with random actions
- Gradient clipping
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Tuple

from experts.base_agent import build_mlp, init_weights, ReplayBuffer, BaseAgent


LOG_STD_MIN = -5
LOG_STD_MAX = 2


class SACActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, num_layers, activation):
        super().__init__()
        self.net = build_mlp(state_dim, hidden_dim, hidden_dim, num_layers - 1, activation)
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        self.net.apply(lambda m: init_weights(m, gain=np.sqrt(2)))
        init_weights(self.mu_head, gain=0.01)
        init_weights(self.log_std_head, gain=0.01)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.net(x)
        mu = self.mu_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, log_std = self.forward(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mu, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        # Enforcing action bounds (tanh squashing correction)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return y_t, log_prob


class SACCriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, num_layers, activation):
        super().__init__()
        # Twin critics
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


class SACAgent(BaseAgent):
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 1,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        buffer_size: int = 100000,
        batch_size: int = 256,
        warmup_steps: int = 1000,
        max_grad_norm: float = 1.0,
        hidden_dim: int = 256,
        num_layers: int = 2,
        activation: str = "relu",
        log_std_min: float = -5,
        log_std_max: float = 2,
        log_dir: Optional[str] = None,
        device: Optional[str] = None,
    ):
        super().__init__(device)
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self._total_steps = 0

        global LOG_STD_MIN, LOG_STD_MAX
        LOG_STD_MIN = log_std_min
        LOG_STD_MAX = log_std_max

        self.actor = SACActorNetwork(state_dim, action_dim, hidden_dim, num_layers, activation).to(self.device)
        self.critic = SACCriticNetwork(state_dim, action_dim, hidden_dim, num_layers, activation).to(self.device)
        self.critic_target = SACCriticNetwork(state_dim, action_dim, hidden_dim, num_layers, activation).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # Automatic entropy tuning
        self.target_entropy = -float(action_dim)
        self.log_alpha = torch.tensor(np.log(alpha), dtype=torch.float32, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

        self.buffer = ReplayBuffer(buffer_size, state_dim, action_dim, self.device)

        self.last_actor_loss = 0.0
        self.last_critic_loss = 0.0
        self.last_alpha = alpha

        self._writer = None
        if log_dir:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._writer = SummaryWriter(log_dir)
            except ImportError:
                pass

    @property
    def alpha(self):
        return self.log_alpha.exp().item()

    def act(self, state: np.ndarray) -> np.ndarray:
        if self._total_steps < self.warmup_steps:
            return np.random.uniform(-1.0, 1.0, size=(self.action_dim,)).astype(np.float32)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, _ = self.actor.sample(s)
        return action.cpu().numpy().flatten().astype(np.float32)

    def store(self, state, action, reward, next_state, done):
        action_arr = np.array(action).flatten().astype(np.float32)
        self.buffer.push(state, action_arr, float(reward), next_state, float(done))

    def train_step(self) -> None:
        self._total_steps += 1
        if not self.buffer.is_ready(self.batch_size):
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_next, q2_next = self.critic_target(next_states, next_actions)
            min_q_next = torch.min(q1_next, q2_next) - self.log_alpha.exp() * next_log_probs
            target_q = rewards + (1.0 - dones) * self.gamma * min_q_next

        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self._clip_grad(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        # Actor update
        new_actions, log_probs = self.actor.sample(states)
        q1_new = self.critic.q1_forward(states, new_actions)
        actor_loss = (self.log_alpha.exp() * log_probs - q1_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self._clip_grad(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        # Alpha update
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Soft update target
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.last_actor_loss = actor_loss.item()
        self.last_critic_loss = critic_loss.item()
        self.last_alpha = self.alpha

        if self._writer and self._total_steps % 100 == 0:
            self._writer.add_scalar("train/actor_loss", self.last_actor_loss, self._total_steps)
            self._writer.add_scalar("train/critic_loss", self.last_critic_loss, self._total_steps)
            self._writer.add_scalar("train/alpha", self.last_alpha, self._total_steps)

    def log_episode(self, ep, total_return, sharpe, sortino, max_dd, equity_curve):
        if self._writer:
            self._writer.add_scalar("episode/total_return", total_return, ep)
            self._writer.add_scalar("episode/sharpe", sharpe, ep)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.data,
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "total_steps": self._total_steps,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        self.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
        self.log_alpha.data = ckpt["log_alpha"]
        self.alpha_optimizer.load_state_dict(ckpt["alpha_optimizer"])
        self._total_steps = ckpt.get("total_steps", 0)