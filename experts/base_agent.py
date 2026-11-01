"""
Base agent class with shared utilities.

Fixes:
- Standardized save/load interface
- Gradient norm tracking
- Device management
- Shared replay buffer
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
from collections import deque
import random


def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dim: int = 256,
    num_layers: int = 2,
    activation: str = "relu",
    output_activation: Optional[str] = None,
) -> nn.Sequential:
    """Build an MLP with configurable depth and activation."""
    act_map = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU,
        "silu": nn.SiLU,
    }
    act_cls = act_map.get(activation.lower(), nn.ReLU)

    layers: List[nn.Module] = []
    in_dim = input_dim
    for _ in range(num_layers):
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(act_cls())
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, output_dim))

    if output_activation is not None:
        out_act_cls = act_map.get(output_activation.lower(), None)
        if out_act_cls is not None:
            layers.append(out_act_cls())

    return nn.Sequential(*layers)


def init_weights(module: nn.Module, gain: float = 1.0) -> None:
    """Orthogonal initialization for linear layers."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.zeros_(module.bias)


class ReplayBuffer:
    """
    Fixed-size replay buffer for off-policy algorithms.

    Stores transitions as numpy arrays for memory efficiency.
    Uses uniform sampling.
    """

    def __init__(self, capacity: int, state_dim: int, action_dim: int, device: str = "cpu"):
        self.capacity = capacity
        self.device = device
        self.pos = 0
        self.size = 0

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: float,
    ) -> None:
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = done
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple:
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.states[indices]).to(self.device),
            torch.FloatTensor(self.actions[indices]).to(self.device),
            torch.FloatTensor(self.rewards[indices]).to(self.device),
            torch.FloatTensor(self.next_states[indices]).to(self.device),
            torch.FloatTensor(self.dones[indices]).to(self.device),
        )

    def __len__(self) -> int:
        return self.size

    def is_ready(self, batch_size: int) -> bool:
        return self.size >= batch_size


class DiscreteReplayBuffer:
    """Replay buffer for discrete action spaces (stores scalar actions)."""

    def __init__(self, capacity: int, state_dim: int, device: str = "cpu"):
        self.capacity = capacity
        self.device = device
        self.pos = 0
        self.size = 0

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        self.states[self.pos] = state
        self.actions[self.pos] = int(action)
        self.rewards[self.pos] = float(reward)
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple:
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.states[indices]).to(self.device),
            torch.LongTensor(self.actions[indices]).to(self.device),
            torch.FloatTensor(self.rewards[indices]).to(self.device),
            torch.FloatTensor(self.next_states[indices]).to(self.device),
            torch.FloatTensor(self.dones[indices]).to(self.device),
        )

    def __len__(self) -> int:
        return self.size

    def is_ready(self, batch_size: int) -> bool:
        return self.size >= batch_size


class BaseAgent:
    """Base class for all RL agents."""

    def __init__(self, device: Optional[str] = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    def act(self, state: np.ndarray) -> any:
        raise NotImplementedError

    def store(self, state, action, reward, next_state, done) -> None:
        raise NotImplementedError

    def train_step(self) -> None:
        raise NotImplementedError

    def save(self, path: str) -> None:
        raise NotImplementedError

    def load(self, path: str) -> None:
        raise NotImplementedError

    def _clip_grad(self, parameters, max_norm: float = 1.0) -> float:
        """Clip gradients and return norm."""
        return float(torch.nn.utils.clip_grad_norm_(parameters, max_norm))