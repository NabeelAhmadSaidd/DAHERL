"""
Allocator‑specific diversity manager.

Encourages the allocator to explore different experts over time by penalising
over‑selection of a single expert. Optionally, regime information can be used
to maintain per‑regime diversity.
"""

import numpy as np
from collections import deque
from typing import Dict, Optional, List


class AllocatorDiversityManager:
    """
    Manages diversity penalties for the allocator's expert selection.

    The manager tracks how often each expert is chosen over a recent window.
    When the selection distribution becomes too skewed, a soft penalty is applied
    to the reward, encouraging more balanced selection.
    """

    def __init__(
        self,
        n_experts: int,
        window_size: int = 100,
        uniform_target: Optional[float] = None,
        threshold: float = 0.45,
        penalty_scale: float = 0.02,
        use_regime: bool = False,
        enabled: bool = True,
    ):
        """
        Args:
            n_experts: Number of available experts.
            window_size: Number of recent steps to consider for selection history.
            uniform_target: Target proportion for each expert. Defaults to 1/n_experts.
            threshold: Maximum allowed proportion before penalty is applied.
            penalty_scale: Base penalty scale; actual penalty = scale * excess.
            use_regime: If True, maintain per‑regime counts and apply penalties based on regime.
            enabled: If False, always returns 0 penalty.
        """
        self.n_experts = n_experts
        self.window_size = window_size
        self.uniform_target = uniform_target or (1.0 / n_experts)
        self.threshold = threshold
        self.penalty_scale = penalty_scale
        self.use_regime = use_regime
        self.enabled = enabled

        # Rolling buffer of selected expert indices
        self._history: deque = deque(maxlen=window_size)
        self._regime_history: Optional[deque] = deque(maxlen=window_size) if use_regime else None
        self._current_regime: int = 0  # default

    def update_regime(self, regime_id: int) -> None:
        """Update the current market regime (if use_regime is True)."""
        if self.use_regime:
            self._current_regime = int(regime_id)

    def compute_penalty(self, action: int) -> float:
        """
        Compute diversity penalty for selecting expert `action`.

        Returns a float in [0, penalty_scale].
        """
        if not self.enabled:
            return 0.0

        # Update history
        self._history.append(action)
        if self._regime_history is not None:
            self._regime_history.append((action, self._current_regime))

        # Need enough history to compute meaningful proportions
        if len(self._history) < min(self.window_size, 20):
            return 0.0

        # Compute proportions over the entire history (or recent window)
        counts = np.bincount(self._history, minlength=self.n_experts)
        total = counts.sum()
        if total == 0:
            return 0.0
        proportions = counts / total

        # Optionally, per‑regime proportions
        if self._regime_history is not None and self._current_regime >= 0:
            # Filter actions for the current regime
            recent = list(self._regime_history)
            if recent:
                regime_actions = [a for a, r in recent if r == self._current_regime]
                if regime_actions:
                    reg_counts = np.bincount(regime_actions, minlength=self.n_experts)
                    reg_total = reg_counts.sum()
                    if reg_total > 0:
                        proportions = reg_counts / reg_total

        # Find maximum proportion
        max_prop = proportions.max()
        # Excess over uniform target? We want uniform, but threshold is absolute.
        # We'll use excess over threshold if max_prop > threshold.
        excess = max_prop - self.threshold
        if excess <= 0.0:
            return 0.0

        # Soft penalty: scale * excess, capped at scale
        penalty = self.penalty_scale * excess
        return float(min(penalty, self.penalty_scale))

    def reset(self) -> None:
        """Clear history (call at start of a new episode or training phase)."""
        self._history.clear()
        if self._regime_history is not None:
            self._regime_history.clear()