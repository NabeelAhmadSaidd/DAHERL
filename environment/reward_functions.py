"""
Reward function utilities.

All reward functions are redesigned to be:
1. Dense - agent gets signal every step, not just at episode end
2. Scaled - typical rewards are in range [-2, 2], not [-0.001, 0.001]
3. Learnable - aligned with actual profitability
4. Stable - bounded variance

Key fix: original rewards were on the scale of step_returns (1e-3 to 1e-4),
which are too small to propagate meaningful gradients. We scale by 100x.

Key fix: rolling Sharpe/Sortino returned 0.0 for the first window_size steps,
creating a dead zone at the start of every episode where no learning signal exists.
We blend with immediate return signal to fix this.
"""

import numpy as np
from collections import deque
from typing import List, Optional


class RollingRewardCalculator:
    """Maintains a rolling window of step returns for reward computation."""

    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.returns = deque(maxlen=window_size)
        self.step_count = 0

    def reset(self):
        self.returns.clear()
        self.step_count = 0

    def add_step(self, step_return: float, pnl: float = 0.0):
        self.returns.append(step_return)
        self.step_count += 1

    @property
    def mean_return(self) -> float:
        if len(self.returns) == 0:
            return 0.0
        return float(np.mean(self.returns))

    @property
    def std_return(self) -> float:
        if len(self.returns) < 2:
            return 1e-8
        return float(np.std(self.returns)) + 1e-8

    @property
    def downside_std(self) -> float:
        arr = np.array(self.returns)
        neg = arr[arr < 0]
        if len(neg) < 2:
            return 1e-8
        return float(np.std(neg)) + 1e-8

    @property
    def sharpe(self) -> float:
        return self.mean_return / self.std_return

    @property
    def sortino(self) -> float:
        return self.mean_return / self.downside_std

    @property
    def has_enough_data(self) -> bool:
        return self.step_count >= 3


def _immediate_signal(step_return: float, cost_ratio: float) -> float:
    """Base immediate reward used as fallback and blend signal."""
    return step_return * 100.0 - 2.0 * cost_ratio * 100.0


def reward_portfolio_return(step_return: float, cost_ratio: float) -> float:
    """
    Dense profit reward scaled to [-~2, ~2] range for typical price moves.

    Fix: Multiplied by 100 so 0.1% move → reward of ~0.1 (was 0.001).
    Cost penalty scaled equally.
    """
    return step_return * 100.0 - 2.0 * cost_ratio * 100.0


def reward_rolling_sharpe(
    calc: RollingRewardCalculator,
    step_return: float,
    cost_ratio: float,
) -> float:
    """
    Rolling Sharpe with immediate blend.

    Fix: Never returns pure 0. Blends Sharpe (delayed signal) with
    immediate return (dense signal) for stable early-episode learning.
    """
    immediate = _immediate_signal(step_return, cost_ratio)
    if not calc.has_enough_data:
        return immediate
    sharpe_contrib = float(np.clip(calc.sharpe * 10.0, -5.0, 5.0))
    return 0.5 * sharpe_contrib + 0.5 * immediate


def reward_rolling_sortino(
    calc: RollingRewardCalculator,
    step_return: float,
    cost_ratio: float,
) -> float:
    """Rolling Sortino with immediate blend."""
    immediate = _immediate_signal(step_return, cost_ratio)
    if not calc.has_enough_data:
        return immediate
    sortino_contrib = float(np.clip(calc.sortino * 10.0, -5.0, 5.0))
    return 0.5 * sortino_contrib + 0.5 * immediate


def reward_drawdown_penalty(
    equity: float,
    peak_equity: float,
    step_return: float,
    cost_ratio: float,
) -> float:
    """Return penalized by drawdown fraction."""
    drawdown = (peak_equity - equity) / (peak_equity + 1e-8)
    drawdown = float(np.clip(drawdown, 0.0, 1.0))
    return step_return * 100.0 - 3.0 * drawdown - 2.0 * cost_ratio * 100.0


def reward_calmar(
    calc: RollingRewardCalculator,
    equity_curve: List[float],
    step_return: float,
    cost_ratio: float,
    window: int = 20,
) -> float:
    """
    Calmar-style reward using actual max drawdown.

    Fix: v1 used std instead of max_dd in calmar calculation (bug).
    """
    immediate = _immediate_signal(step_return, cost_ratio)
    if not calc.has_enough_data or len(equity_curve) < 3:
        return immediate
    arr = np.array(equity_curve[-window:], dtype=np.float64)
    arr = np.maximum(arr, 1e-4)
    peak = np.maximum.accumulate(arr)
    max_dd = float(np.max((peak - arr) / (peak + 1e-8)))
    max_dd = max(max_dd, 1e-4)
    calmar = calc.mean_return / max_dd
    return float(np.clip(calmar * 10.0 + 0.5 * immediate, -5.0, 5.0))


def reward_risk_adjusted(
    step_return: float,
    calc: RollingRewardCalculator,
    cost_ratio: float,
) -> float:
    """Volatility-penalized return."""
    immediate = _immediate_signal(step_return, cost_ratio)
    if not calc.has_enough_data:
        return immediate
    vol = calc.std_return
    inv_vol = float(np.clip(1.0 / (vol * 100.0 + 1e-4), 0.1, 5.0))
    return float(step_return * 100.0 * inv_vol) - 2.0 * cost_ratio * 100.0


REWARD_FUNCTIONS = {
    1: reward_portfolio_return,
    2: reward_rolling_sharpe,
    3: reward_rolling_sortino,
    4: reward_drawdown_penalty,
    5: reward_calmar,
    6: reward_risk_adjusted,
}


def get_reward_function(expert_id: int):
    if expert_id not in REWARD_FUNCTIONS:
        raise ValueError(f"Unknown expert_id: {expert_id}. Valid: {list(REWARD_FUNCTIONS.keys())}")
    return REWARD_FUNCTIONS[expert_id]