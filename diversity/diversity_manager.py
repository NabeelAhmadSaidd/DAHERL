"""
Diversity Manager for DAHERL ensemble.

Implements trajectory‑level behavioral diversity via a rolling Pearson correlation
between the current agent's recent returns and the stored trajectories of previously
trained experts. Penalties are soft and can be scheduled over time using a
DiversityScheduler.
"""

import numpy as np
from collections import deque
from pathlib import Path
from typing import Dict, Optional
import pickle

from .diversity_scheduler import DiversityScheduler


class DiversityManager:
    """
    Manages diversity penalties during ensemble training.

    The penalty scale is determined by a scheduler (if provided) or a fixed scale.
    The scheduler can be used to start with zero penalty and gradually increase it.

    Usage:
        # With scheduler (recommended)
        scheduler = DiversityScheduler(schedule=[(1, 0.0), (10, 0.5), (21, 1.0)])
        manager = DiversityManager(storage_dir="storage", enabled=True, scheduler=scheduler)

        # Without scheduler (fixed penalty scale from the start)
        manager = DiversityManager(storage_dir="storage", enabled=True, penalty_scale=0.02)

        # At each step:
        penalty = manager.compute_penalty(step_return, agent_name)
        reward = original_reward - penalty

        # Reset buffer at episode start:
        manager.reset_buffer()

        # Update epoch for scheduler:
        manager.update_epoch(episode_count)

        # After training completes:
        manager.register_expert(agent_name, trajectory)
    """

    def __init__(
        self,
        storage_dir: str = "storage",
        penalty_scale: float = 0.02,
        similarity_threshold: float = 0.5,
        window_size: int = 30,
        scheduler: Optional[DiversityScheduler] = None,
        enabled: bool = False,
    ):
        """
        Args:
            storage_dir: Directory to save/load expert trajectories.
            penalty_scale: Base penalty scale (multiplied by scheduler's lambda if provided).
            similarity_threshold: Correlation above which penalty is applied.
            window_size: Number of recent returns used for correlation.
            scheduler: Optional DiversityScheduler to adjust penalty over time.
            enabled: Set to True during ensemble fine‑tuning.
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.base_penalty_scale = penalty_scale
        self.similarity_threshold = similarity_threshold
        self.window_size = window_size
        self.scheduler = scheduler
        self.enabled = enabled

        self._current_epoch = 0
        self._expert_trajectories: Dict[str, dict] = {}
        self._current_returns: deque = deque(maxlen=window_size)
        self._load_existing_trajectories()

    def _load_existing_trajectories(self) -> None:
        """Load all previously stored expert trajectories from disk."""
        for pkl_path in self.storage_dir.glob("*.pkl"):
            try:
                with open(pkl_path, "rb") as f:
                    data = pickle.load(f)
                expert_name = pkl_path.stem
                self._expert_trajectories[expert_name] = data
            except Exception:
                pass

    def update_epoch(self, epoch: int) -> None:
        """Update the current episode count and scheduler."""
        self._current_epoch = epoch
        if self.scheduler is not None:
            self.scheduler.update_epoch(epoch)

    def reset_buffer(self) -> None:
        """Clear the rolling return buffer (call at episode start)."""
        self._current_returns.clear()

    def compute_penalty(
        self,
        step_return: float,
        agent_name: str,
    ) -> float:
        """
        Compute diversity penalty for the current step.

        The penalty is based on the correlation between the agent's recent returns
        (over a sliding window) and the corresponding window of each stored trajectory.

        Returns:
            Float penalty in [0, max_penalty].
        """
        if not self.enabled:
            return 0.0
        if len(self._expert_trajectories) == 0:
            return 0.0

        # Update rolling buffer
        self._current_returns.append(float(step_return))
        if len(self._current_returns) < self.window_size:
            return 0.0

        current_win = np.array(self._current_returns, dtype=np.float64)

        max_similarity = 0.0
        for name, traj in self._expert_trajectories.items():
            if name == agent_name:
                continue
            stored_returns = traj.get("returns", np.array([]))
            if len(stored_returns) < self.window_size:
                continue

            stored_win = stored_returns[-self.window_size:].astype(np.float64)

            # Skip constant sequences
            if np.std(current_win) < 1e-8 or np.std(stored_win) < 1e-8:
                continue

            corr = np.corrcoef(current_win, stored_win)[0, 1]
            if np.isnan(corr):
                continue
            similarity = abs(corr)
            if similarity > max_similarity:
                max_similarity = similarity

        excess = max_similarity - self.similarity_threshold
        if excess <= 0.0:
            return 0.0

        # Apply current penalty scale
        if self.scheduler is not None:
            scale = self.base_penalty_scale * self.scheduler.lambda_value
        else:
            scale = self.base_penalty_scale

        penalty = scale * excess
        return float(min(penalty, scale))  # cap at scale

    def register_expert(self, name: str, trajectory: dict) -> None:
        """Save an expert's trajectory after training completes."""
        self._expert_trajectories[name] = trajectory
        save_path = self.storage_dir / f"{name}.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(trajectory, f, protocol=pickle.HIGHEST_PROTOCOL)

    def get_diversity_score(self) -> float:
        """
        Compute pairwise diversity score across all registered experts.

        Returns:
            Float in [0, 1], where 1 = fully diverse (zero average correlation),
            0 = identical behaviour.
        """
        if len(self._expert_trajectories) < 2:
            return 1.0
        names = list(self._expert_trajectories.keys())
        correlations = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                r_i = self._expert_trajectories[names[i]].get("returns", np.array([]))
                r_j = self._expert_trajectories[names[j]].get("returns", np.array([]))
                min_len = min(len(r_i), len(r_j))
                if min_len < 10:
                    continue
                corr = float(np.corrcoef(r_i[-min_len:], r_j[-min_len:])[0, 1])
                if not np.isnan(corr):
                    correlations.append(abs(corr))
        if not correlations:
            return 1.0
        return float(1.0 - np.mean(correlations))