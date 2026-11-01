"""
Truncated Quantile Critics (Expert 4 - Continuous).

Fixes:
- Correct quantile regression loss (huber quantile loss)
- Proper top-quantile truncation
- Auto entropy tuning
- Twin critic architecture
- Warmup
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, List, Tuple

from experts.base_agent import build_mlp, init_weights, ReplayBuffer, BaseAgent

LOG_STD_MIN = -5
LOG_STD_MAX = 2


class TQCActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, num_layers, activation):
        super().__init__()
        self.net = build_mlp(state_dim, hidden_dim, hidden_dim, max(1, num_layers - 1), activation)
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.net.apply(lambda m: init_weights(m, gain=np.sqrt(2)))
        init_weights(self.mu, gain=0.01)
        init_weights(self.log_std, gain=0.01)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        f = self.net(x)
        mu = self.mu(f)
        log_std = torch.clamp(self.log_std(f), LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, log_std = self.forward(x)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        x_t = dist.rsample()
        y_t = torch.tanh(x_t)
        log_prob = dist.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return y_t, log_prob


class TQCCritic(nn.Module):
    """Single critic with multiple quantile outputs."""

    def __init__(self, state_dim, action_dim, hidden_dim, num_layers, activation, n_quantiles):
        super().__init__()
        self.n_quantiles = n_quantiles
        self.net = build_mlp(state_dim + action_dim, n_quantiles, hidden_dim, num_layers, activation)
        self.net.apply(lambda m: init_weights(m, gain=np.sqrt(2)))

    def forward(self, state, action) -> torch.Tensor:
        sa = torch.cat([state, action], dim=-1)
        return self.net(sa)  # [batch, n_quantiles]


class TQCAgent(BaseAgent):
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 1,
        lr: float = 3e-4,
        lr_actor: Optional[float] = None,
        lr_critic: Optional[float] = None,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_size: int = 100000,
        batch_size: int = 256,
        warmup_steps: int = 1000,
        hidden_dim: int = 256,
        actor_hidden_dim: Optional[int] = None,
        critic_hidden_dim: Optional[int] = None,
        actor_num_layers: int = 2,
        critic_num_layers: int = 2,
        activation: str = "relu",
        log_std_min: float = -5,
        log_std_max: float = 2,
        n_critics: int = 2,
        n_quantiles: int = 25,
        top_quantiles_to_drop: int = 2,
        auto_entropy_tuning: bool = True,
        alpha: float = 0.2,
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
        self.n_critics = n_critics
        self.n_quantiles = n_quantiles
        self.top_quantiles_to_drop = top_quantiles_to_drop
        self.grad_norm_clip = grad_norm_clip
        self._total_steps = 0

        global LOG_STD_MIN, LOG_STD_MAX
        LOG_STD_MIN = log_std_min
        LOG_STD_MAX = log_std_max

        actor_hd = actor_hidden_dim or hidden_dim
        critic_hd = critic_hidden_dim or hidden_dim
        lr_a = lr_actor or lr
        lr_c = lr_critic or lr

        self.actor = TQCActor(state_dim, action_dim, actor_hd, actor_num_layers, activation).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_a)

        # Multiple independent critics
        self.critics = nn.ModuleList([
            TQCCritic(state_dim, action_dim, critic_hd, critic_num_layers, activation, n_quantiles)
            for _ in range(n_critics)
        ]).to(self.device)
        self.critics_target = nn.ModuleList([
            TQCCritic(state_dim, action_dim, critic_hd, critic_num_layers, activation, n_quantiles)
            for _ in range(n_critics)
        ]).to(self.device)
        for c, ct in zip(self.critics, self.critics_target):
            ct.load_state_dict(c.state_dict())

        self.critic_optimizer = optim.Adam(self.critics.parameters(), lr=lr_c)

        # Alpha (entropy)
        self.auto_entropy_tuning = auto_entropy_tuning
        self.target_entropy = -float(action_dim)
        self.log_alpha = torch.tensor(np.log(alpha), dtype=torch.float32, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_a)

        # Quantile targets
        quantiles_total = n_critics * n_quantiles
        self.quantiles_to_keep = quantiles_total - top_quantiles_to_drop

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
        if self._total_steps < self.warmup_steps:
            return
        if not self.buffer.is_ready(self.batch_size):
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        batch_size = states.shape[0]

        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            # All critic targets: [n_critics, batch, n_quantiles]
            next_quantiles = torch.stack([ct(next_states, next_actions) for ct in self.critics_target], dim=0)
            # Flatten to [batch, n_critics * n_quantiles]
            next_quantiles = next_quantiles.permute(1, 0, 2).reshape(batch_size, -1)
            # Sort and drop top quantiles (pessimism)
            next_quantiles, _ = torch.sort(next_quantiles, dim=-1)
            next_quantiles = next_quantiles[:, :self.quantiles_to_keep]
            # Target: [batch, quantiles_to_keep]
            target_quantiles = rewards + (1.0 - dones) * self.gamma * (
                next_quantiles - self.log_alpha.exp() * next_log_probs
            )

        # Critic loss: quantile Huber loss
        # Current quantiles: [n_critics, batch, n_quantiles]
        current_quantiles = torch.stack([c(states, actions) for c in self.critics], dim=0)

        # Build tau for quantile regression
        taus = (torch.arange(self.n_quantiles, device=self.device).float() + 0.5) / self.n_quantiles

        critic_loss_total = torch.tensor(0.0, device=self.device)
        for i in range(self.n_critics):
            cq = current_quantiles[i]  # [batch, n_quantiles]
            # pairwise: [batch, n_quantiles, quantiles_to_keep]
            diff = target_quantiles.unsqueeze(1) - cq.unsqueeze(2)
            huber = F.huber_loss(cq.unsqueeze(2).expand_as(diff), target_quantiles.unsqueeze(1).expand_as(diff), reduction='none', delta=1.0)
            quantile_loss = torch.abs(taus.view(1, -1, 1) - (diff < 0).float()) * huber
            critic_loss_total = critic_loss_total + quantile_loss.mean()

        critic_loss_total = critic_loss_total / self.n_critics

        self.critic_optimizer.zero_grad()
        critic_loss_total.backward()
        self._clip_grad(self.critics.parameters(), self.grad_norm_clip)
        self.critic_optimizer.step()

        # Actor loss
        new_actions, log_probs = self.actor.sample(states)
        new_quantiles = torch.stack([c(states, new_actions) for c in self.critics], dim=0)
        new_quantiles = new_quantiles.permute(1, 0, 2).reshape(batch_size, -1)
        actor_loss = (self.log_alpha.exp() * log_probs - new_quantiles.mean(dim=-1, keepdim=True)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self._clip_grad(self.actor.parameters(), self.grad_norm_clip)
        self.actor_optimizer.step()

        # Alpha update
        if self.auto_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        # Soft update targets
        for c, ct in zip(self.critics, self.critics_target):
            for p, tp in zip(c.parameters(), ct.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        self.last_actor_loss = actor_loss.item()
        self.last_critic_loss = critic_loss_total.item()

    def log_episode(self, ep, total_return, sharpe, sortino, max_dd, equity_curve):
        if self._writer:
            self._writer.add_scalar("episode/total_return", total_return, ep)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(),
            "critics": [c.state_dict() for c in self.critics],
            "critics_target": [ct.state_dict() for ct in self.critics_target],
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.data,
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "total_steps": self._total_steps,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        for i, c in enumerate(self.critics):
            c.load_state_dict(ckpt["critics"][i])
        for i, ct in enumerate(self.critics_target):
            ct.load_state_dict(ckpt["critics_target"][i])
        self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        self.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
        self.log_alpha.data = ckpt["log_alpha"]
        self.alpha_optimizer.load_state_dict(ckpt["alpha_optimizer"])
        self._total_steps = ckpt.get("total_steps", 0)