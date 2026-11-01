"""
Double DQN with Dueling Architecture (Expert 3 - Discrete actions).

Fixes:
- Epsilon decay tied to actual timestep count, not act() call count
- Warmup before any training (pure random exploration)
- Proper target network sync
- Gradient clipping
- Dueling network with correct advantage/value decomposition
- Prioritized experience replay option (uniform for stability)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, List

from experts.base_agent import init_weights, DiscreteReplayBuffer, BaseAgent


class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, num_layers, activation):
        super().__init__()
        act_map = {
            "relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU,
            "leaky_relu": nn.LeakyReLU, "silu": nn.SiLU,
        }
        act_cls = act_map.get(activation.lower(), nn.ReLU)

        # Shared feature extractor
        trunk: List[nn.Module] = []
        in_dim = state_dim
        for _ in range(num_layers):
            trunk.append(nn.Linear(in_dim, hidden_dim))
            trunk.append(act_cls())
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*trunk)

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            act_cls(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            act_cls(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

        self.trunk.apply(lambda m: init_weights(m, gain=np.sqrt(2)))
        self.value_stream.apply(lambda m: init_weights(m, gain=1.0))
        self.advantage_stream.apply(lambda m: init_weights(m, gain=0.01))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.trunk(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        # Dueling: Q = V + (A - mean(A))
        q = values + (advantages - advantages.mean(dim=-1, keepdim=True))
        return q


class DDQNAgent(BaseAgent):
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 3,
        lr: float = 1e-4,
        gamma: float = 0.99,
        batch_size: int = 256,
        buffer_size: int = 100000,
        target_update_freq: int = 1000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 50000,
        dueling: bool = True,
        warmup_steps: int = 1000,
        max_grad_norm: float = 10.0,
        hidden_dim: int = 256,
        num_layers: int = 2,
        activation: str = "relu",
        log_dir: Optional[str] = None,
        device: Optional[str] = None,
    ):
        super().__init__(device)
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self._total_steps = 0

        self.q_network = DuelingQNetwork(state_dim, action_dim, hidden_dim, num_layers, activation).to(self.device)
        self.target_network = DuelingQNetwork(state_dim, action_dim, hidden_dim, num_layers, activation).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.buffer = DiscreteReplayBuffer(buffer_size, state_dim, self.device)

        self.last_loss = 0.0

        self._writer = None
        if log_dir:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._writer = SummaryWriter(log_dir)
            except ImportError:
                pass

    def _get_epsilon(self) -> float:
        """
        Compute epsilon based on total steps.

        Fix: v1 decayed epsilon per act() call including warmup steps,
        causing epsilon to hit minimum before meaningful learning started.
        """
        if self._total_steps < self.warmup_steps:
            return 1.0
        steps_since_warmup = self._total_steps - self.warmup_steps
        eps = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(
            -steps_since_warmup / (self.epsilon_decay + 1e-8)
        )
        return float(eps)

    def act(self, state: np.ndarray) -> int:
        epsilon = self._get_epsilon()
        if np.random.random() < epsilon:
            return int(np.random.randint(0, self.action_dim))
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(s)
            return int(q_values.argmax(dim=-1).item())

    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, int(action), float(reward), next_state, float(done))

    def train_step(self) -> None:
        self._total_steps += 1
        if self._total_steps < self.warmup_steps:
            return
        if not self.buffer.is_ready(self.batch_size):
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        with torch.no_grad():
            # DDQN: use online network to select action, target network to evaluate
            next_actions = self.q_network(next_states).argmax(dim=-1, keepdim=True)
            next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze(-1)
            target_q = rewards.squeeze(-1) + (1.0 - dones.squeeze(-1)) * self.gamma * next_q_values

        current_q = self.q_network(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        loss = F.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self._clip_grad(self.q_network.parameters(), self.max_grad_norm)
        self.optimizer.step()

        if self._total_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        self.last_loss = loss.item()

        if self._writer and self._total_steps % 200 == 0:
            self._writer.add_scalar("train/q_loss", self.last_loss, self._total_steps)
            self._writer.add_scalar("train/epsilon", self._get_epsilon(), self._total_steps)

    def log_episode(self, ep, total_return, sharpe, sortino, max_dd, equity_curve):
        if self._writer:
            self._writer.add_scalar("episode/total_return", total_return, ep)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self._total_steps,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(ckpt["q_network"])
        self.target_network.load_state_dict(ckpt["target_network"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self._total_steps = ckpt.get("total_steps", 0)