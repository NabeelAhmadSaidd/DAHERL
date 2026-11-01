"""
Gymnasium-compliant multi-position trading environment.
Supports Discrete(3) and Box(-1,1) action spaces.

Major fixes vs v1:
- Action space: fixed fractional position sizing (action maps to % of equity, not lot count)
- Portfolio accounting: equity = balance + unrealized enforced as single truth
- Observations: running z-score normalization per feature, stationary state features
- Rewards: stable, learnable, aligned with profitability
- Liquidation: correct balance update sequence
- No more equity-dependent lot sizing that breaks action stationarity

v3 fixes:
- _execute_to_target: cost is returned and deducted in step() only — removed
  internal balance mutation for cost (was double-counted path risk)
- Partial-close realized PnL: correctly scales by fraction of position being closed
- Expert 6 (_reward_risk_adjusted): redesigned to simple dense scaled return
  minus cost penalty. Removes momentum bonus that reinforced losing positions.
- Truncation bootstrap signal: step() passes truncated flag info cleanly
- step() balance/cost accounting consolidated to single location
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import List, Tuple, Optional, Dict, Any
from collections import deque


class RunningNormalizer:
    """Online mean/variance normalization (Welford's algorithm)."""

    def __init__(self, shape: int, clip: float = 10.0, epsilon: float = 1e-8):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 0
        self.clip = clip
        self.epsilon = epsilon

    def update(self, x: np.ndarray) -> None:
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.var += (delta * delta2 - self.var) / self.count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        std = np.sqrt(self.var + self.epsilon)
        normed = (x - self.mean) / std
        return np.clip(normed, -self.clip, self.clip).astype(np.float32)

    def normalize_without_update(self, x: np.ndarray) -> np.ndarray:
        std = np.sqrt(self.var + self.epsilon)
        normed = (x - self.mean) / std
        return np.clip(normed, -self.clip, self.clip).astype(np.float32)


class TradingEnv(gym.Env):
    """
    Multi-position trading environment with corrected accounting and stable RL design.

    Action spaces:
        Discrete(3): 0=Short (target=-max_position_frac), 1=Flat (target=0), 2=Long (target=+max_position_frac)
        Box(-1, 1):  continuous target position as fraction of equity

    Key design principles:
        - Action always means the same thing regardless of account state (stationarity)
        - equity = balance + unrealized_pnl is always the single source of truth
        - Observations are z-score normalized with running statistics
        - Reward is aligned with risk-adjusted portfolio growth
        - Cost accounting: _execute_to_target mutates balance for realized PnL only;
          transaction costs are returned and deducted once in step()
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        expert_id: int,
        asset_type: str = "forex",
        contract_size: Optional[float] = None,
        commission_rate: float = 0.0005,
        spread_rate: float = 0.0001,
        slippage_rate: float = 0.00005,
        leverage: Optional[int] = None,
        maintenance_margin_ratio: float = 0.3,
        initial_balance: float = 10000.0,
        max_position_fraction: float = 0.90,
        window_size: int = 20,
        reward_scale: float = 1.0,
        clip_reward: Tuple[float, float] = (-5.0, 5.0),
        continuous_actions: bool = False,
        render_mode: Optional[str] = None,
        normalize_obs: bool = True,
        reward_lookback: int = 20,
    ):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.expert_id = expert_id
        self.asset_type = asset_type.lower()
        self.commission_rate = float(commission_rate)
        self.spread_rate = float(spread_rate)
        self.slippage_rate = float(slippage_rate)
        self.maintenance_margin_ratio = float(maintenance_margin_ratio)
        self.initial_balance = float(initial_balance)
        self.max_position_fraction = float(max_position_fraction)
        self.window_size = window_size
        self.reward_scale = float(reward_scale)
        self.clip_reward = clip_reward
        self.continuous_actions = continuous_actions
        self.render_mode = render_mode
        self.normalize_obs = normalize_obs
        self.reward_lookback = reward_lookback

        self.contract_size = float(contract_size) if contract_size is not None else (
            100000.0 if self.asset_type == "forex" else 1.0
        )
        self.leverage = int(leverage) if leverage is not None else (
            10 if self.asset_type == "forex" else 3
        )

        # Observation space
        # tech features + [position_frac, unrealized_pnl_frac, drawdown_frac,
        #                   equity_growth_frac, rolling_return, margin_ratio]
        self._n_state_features = 6
        obs_dim = len(feature_cols) + self._n_state_features
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32
        )

        if continuous_actions:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(3)

        # Running normalizer for technical features only
        if self.normalize_obs:
            self._obs_normalizer = RunningNormalizer(shape=len(feature_cols))
        else:
            self._obs_normalizer = None

        # Internal state
        self.current_step: int = 0
        self.balance: float = initial_balance
        self.peak_equity: float = initial_balance
        self.equity_curve: List[float] = []

        # Position tracking: single net position (long positive, short negative)
        # Units are in base currency notional (not lots)
        self.position_notional: float = 0.0
        self.avg_entry_price: float = 0.0
        self.realized_pnl: float = 0.0
        self.total_costs: float = 0.0

        self._prev_equity: float = initial_balance
        self._rolling_returns: deque = deque(maxlen=reward_lookback)

    def seed(self, seed: Optional[int] = None):
        np.random.seed(seed)
        return [seed]

    # ------------------------------------------------------------------
    # Core Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.peak_equity = self.initial_balance
        self._prev_equity = self.initial_balance
        self.equity_curve = [self.initial_balance]

        self.position_notional = 0.0
        self.avg_entry_price = 0.0
        self.realized_pnl = 0.0
        self.total_costs = 0.0
        self._rolling_returns.clear()

        obs = self._get_obs()
        return obs, self._get_info()

    def step(
        self, action: Any
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        terminated = False
        truncated = False

        if self.current_step >= len(self.df):
            truncated = True
            return (
                np.zeros(len(self.feature_cols) + self._n_state_features, dtype=np.float32),
                0.0, terminated, truncated, self._get_info()
            )

        price = self._get_price(self.current_step)

        # --- Compute current equity (source of truth) ---
        unrealized = self._calc_unrealized_pnl(price)
        equity = self.balance + unrealized

        if equity <= self.initial_balance * 0.05:
            # Bankrupt: equity dropped to 5% of starting capital
            cost = self._close_position(price)
            self.total_costs += cost
            self.balance -= cost
            terminated = True
            return (
                self._get_obs(), -5.0,
                terminated, truncated,
                {"bankrupt": True, **self._get_info()}
            )

        # --- Margin call check BEFORE action ---
        if self._is_margin_called(equity, price):
            cost = self._close_position(price)
            self.total_costs += cost
            self.balance -= cost
            terminated = True
            reward = float(np.clip(-3.0 * self.reward_scale, self.clip_reward[0], self.clip_reward[1]))
            return self._get_obs(), reward, terminated, truncated, {"margin_call": True, **self._get_info()}

        # --- Convert action to target position fraction ---
        target_frac = self._action_to_target_frac(action)

        # --- Execute trades to reach target ---
        # _execute_to_target handles realized PnL → balance internally.
        # It returns only the transaction COST (commission + spread + slippage).
        # We deduct the cost from balance here — single authoritative location.
        cost = self._execute_to_target(target_frac, equity, price)
        self.total_costs += cost
        self.balance -= cost

        # --- Recompute equity after trade ---
        unrealized = self._calc_unrealized_pnl(price)
        equity = self.balance + unrealized

        # --- Track equity ---
        self.equity_curve.append(equity)
        if equity > self.peak_equity:
            self.peak_equity = equity

        # --- Step return (for reward) ---
        step_return = (equity - self._prev_equity) / (abs(self._prev_equity) + 1e-8)
        self._prev_equity = equity
        self._rolling_returns.append(step_return)

        # --- Compute reward ---
        reward = self._compute_reward(step_return, cost, equity, price)
        reward = float(np.clip(reward * self.reward_scale, self.clip_reward[0], self.clip_reward[1]))

        # --- Advance ---
        self.current_step += 1
        if self.current_step >= len(self.df):
            truncated = True
            # Close position at end of episode
            if abs(self.position_notional) > 1e-8:
                close_cost = self._close_position(price)
                self.total_costs += close_cost
                self.balance -= close_cost

        info = self._get_info()
        info["step_return"] = step_return
        info["cost"] = cost
        info["equity"] = equity
        info["truncated"] = truncated
        info["terminated"] = terminated

        if terminated or truncated:
            obs = np.zeros(len(self.feature_cols) + self._n_state_features, dtype=np.float32)
        else:
            obs = self._get_obs()

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Position management (simplified: single net position)
    # ------------------------------------------------------------------

    def _action_to_target_frac(self, action: Any) -> float:
        """
        Convert action to target position fraction of current equity.

        Action meaning is CONSTANT regardless of account state.
        Returns value in [-max_position_fraction, +max_position_fraction].
        Positive = long, Negative = short.
        """
        if self.continuous_actions:
            if isinstance(action, np.ndarray):
                raw = float(action.flat[0])
            else:
                raw = float(action)
            raw = float(np.clip(raw, -1.0, 1.0))
            return raw * self.max_position_fraction
        else:
            mapping = {0: -self.max_position_fraction, 1: 0.0, 2: self.max_position_fraction}
            return mapping.get(int(action), 0.0)

    def _execute_to_target(self, target_frac: float, equity: float, price: float) -> float:
        """
        Adjust net position to match target_frac * equity in notional value.

        Returns transaction cost (commission + spread + slippage).

        IMPORTANT: This method mutates self.balance for realized PnL only.
        The caller (step()) is responsible for deducting the returned cost
        from self.balance exactly once.

        target_notional = target_frac * equity
        (position stored as signed USD notional: position_notional = qty_signed * current_price)
        """
        target_notional = target_frac * equity

        # Enforce margin constraint
        max_notional = equity * self.leverage * self.max_position_fraction
        target_notional = float(np.clip(target_notional, -max_notional, max_notional))

        delta_notional = target_notional - self.position_notional

        if abs(delta_notional) < equity * 0.001:
            # Ignore trivially small changes (< 0.1% of equity)
            return 0.0

        traded_notional = abs(delta_notional)
        cost = self._compute_cost(traded_notional, price)

        if abs(self.position_notional) < 1e-8:
            # Opening new position from flat
            self.avg_entry_price = price
            self.position_notional = target_notional

        elif (delta_notional > 0) == (self.position_notional >= 0):
            # Adding to existing position (same side)
            old_notional = abs(self.position_notional)
            add_notional = abs(delta_notional)
            total_notional = old_notional + add_notional
            self.avg_entry_price = (
                (self.avg_entry_price * old_notional + price * add_notional)
                / (total_notional + 1e-8)
            )
            self.position_notional = target_notional

        else:
            # Reducing or reversing position
            if abs(target_notional) < 1e-8:
                # Full close
                realized = self._compute_realized_pnl(price)
                self.balance += realized
                self.realized_pnl += realized
                self.position_notional = 0.0
                self.avg_entry_price = 0.0

            elif (target_notional > 0) != (self.position_notional > 0):
                # Reversal: close existing first, then open new
                realized = self._compute_realized_pnl(price)
                self.balance += realized
                self.realized_pnl += realized
                self.position_notional = target_notional
                self.avg_entry_price = price

            else:
                # Partial close: close |delta_notional| worth of position
                close_frac = abs(delta_notional) / (abs(self.position_notional) + 1e-8)
                close_frac = float(np.clip(close_frac, 0.0, 1.0))
                full_realized = self._compute_realized_pnl(price)
                realized = full_realized * close_frac
                self.balance += realized
                self.realized_pnl += realized
                self.position_notional = target_notional
                # avg_entry_price is unchanged on partial close (correct)

        return cost

    def _compute_realized_pnl(self, exit_price: float) -> float:
        """PnL if entire current position is closed at exit_price."""
        if abs(self.position_notional) < 1e-8 or self.avg_entry_price < 1e-8:
            return 0.0
        # position_notional = qty_signed * avg_entry_price
        # realized = qty_signed * (exit_price - avg_entry_price)
        qty_signed = self.position_notional / (self.avg_entry_price + 1e-8)
        return float(qty_signed * (exit_price - self.avg_entry_price))

    def _calc_unrealized_pnl(self, price: float) -> float:
        """Current mark-to-market PnL."""
        if abs(self.position_notional) < 1e-8 or self.avg_entry_price < 1e-8:
            return 0.0
        qty_signed = self.position_notional / (self.avg_entry_price + 1e-8)
        return float(qty_signed * (price - self.avg_entry_price))

    def _close_position(self, price: float) -> float:
        """
        Close entire position. Adds realized PnL to balance.
        Returns transaction cost (caller must deduct from balance).
        """
        if abs(self.position_notional) < 1e-8:
            return 0.0
        traded_notional = abs(self.position_notional)
        cost = self._compute_cost(traded_notional, price)
        realized = self._compute_realized_pnl(price)
        self.balance += realized
        self.realized_pnl += realized
        self.position_notional = 0.0
        self.avg_entry_price = 0.0
        return cost

    def _compute_cost(self, traded_notional: float, price: float) -> float:
        """Total transaction cost for a trade of given notional value."""
        commission = traded_notional * self.commission_rate
        spread = traded_notional * self.spread_rate
        slippage = traded_notional * self.slippage_rate
        return commission + spread + slippage

    def _is_margin_called(self, equity: float, price: float) -> bool:
        """Check if margin maintenance threshold is breached."""
        if abs(self.position_notional) < 1e-8:
            return False
        used_margin = abs(self.position_notional) / (self.leverage + 1e-8)
        return equity < used_margin * self.maintenance_margin_ratio

    # ------------------------------------------------------------------
    # Reward functions
    # ------------------------------------------------------------------
    def _compute_reward(
        self, step_return: float, cost: float, equity: float, price: float
    ) -> float:
        """Dispatch to per-expert reward. Cost is already reflected in step_return."""
        cost_ratio = cost / (abs(equity) + 1e-8)  # kept only for potential future use

        if self.expert_id == 1:
            return self._reward_portfolio_return(step_return)
        elif self.expert_id == 2:
            return self._reward_rolling_sharpe(step_return)
        elif self.expert_id == 3:
            return self._reward_rolling_sortino(step_return)
        elif self.expert_id == 4:
            return self._reward_drawdown_penalty(step_return, equity)
        elif self.expert_id == 5:
            return self._reward_calmar(step_return)
        elif self.expert_id == 6:
            return self._reward_excess_return(step_return, equity)
        else:
            raise ValueError(f"Unknown expert_id: {self.expert_id}")

    def _reward_portfolio_return(self, step_return: float) -> float:
        """Dense profit‑based reward. Cost already deducted from equity."""
        return step_return * 100.0

    def _reward_rolling_sharpe(self, step_return: float) -> float:
        """Rolling Sharpe with immediate blend. No extra cost penalty."""
        r = self._rolling_returns
        if len(r) < 3:
            return step_return * 100.0
        arr = np.array(r, dtype=np.float64)
        mean_r = float(arr.mean())
        std_r = float(arr.std()) + 1e-8
        sharpe = mean_r / std_r
        immediate = step_return * 100.0
        return float(np.clip(sharpe * 10.0 + 0.5 * immediate, -5.0, 5.0))

    def _reward_rolling_sortino(self, step_return: float) -> float:
        """Rolling Sortino with immediate blend. No extra cost penalty."""
        r = self._rolling_returns
        if len(r) < 3:
            return step_return * 100.0
        arr = np.array(r, dtype=np.float64)
        mean_r = float(arr.mean())
        downside = arr[arr < 0]
        down_std = float(downside.std()) + 1e-8 if len(downside) > 0 else 1e-8
        sortino = mean_r / down_std
        immediate = step_return * 100.0
        return float(np.clip(sortino * 10.0 + 0.5 * immediate, -5.0, 5.0))

    def _reward_drawdown_penalty(self, step_return: float, equity: float) -> float:
        """Return penalised by current drawdown. No extra cost penalty."""
        drawdown = (self.peak_equity - equity) / (self.peak_equity + 1e-8)
        return step_return * 100.0 - 3.0 * drawdown

    def _reward_calmar(self, step_return: float) -> float:
        """Calmar‑style reward using actual max drawdown. No extra cost penalty."""
        if len(self.equity_curve) < 3:
            return step_return * 100.0
        arr = np.array(self.equity_curve[-self.reward_lookback:], dtype=np.float64)
        arr = np.maximum(arr, 1e-4)
        peak = np.maximum.accumulate(arr)
        max_dd = float(np.max((peak - arr) / (peak + 1e-8)))
        max_dd = max(max_dd, 1e-4)
        mean_r = float(np.mean(self._rolling_returns)) if self._rolling_returns else step_return
        calmar = mean_r / max_dd
        immediate = step_return * 100.0
        return float(np.clip(calmar * 10.0 + 0.5 * immediate, -5.0, 5.0))

    def _reward_excess_return(self, step_return: float, equity: float) -> float:
        """
        Dense reward with an EMA baseline, inactivity penalty, and position adjustments.
        """
        # EMA baseline (smoothing factor 0.05)
        if not hasattr(self, "_ema_return"):
            self._ema_return = 0.0
        alpha = 0.05
        baseline = self._ema_return
        excess_return = step_return - baseline

        reward = excess_return * 100.0

        # Update EMA after using it
        self._ema_return = (1.0 - alpha) * self._ema_return + alpha * step_return

        # Position penalties / bonuses
        position_frac = abs(self.position_notional) / (equity + 1e-8)
        in_position = position_frac >= 0.01

        if not in_position:
            reward -= 0.005                     # discourage staying flat
        elif excess_return > 0:
            reward += min(0.01, excess_return * 20.0)   # profit bonus
        elif excess_return < 0:
            reward -= min(0.05, abs(excess_return) * 25.0)  # loss penalty

        if 0.01 <= position_frac < 0.05:
            reward -= 0.002                     # tiny position penalty

        return float(np.clip(reward, -10.0, 10.0))
    # ------------------------------------------------------------------
    # Observation and info
    # ------------------------------------------------------------------

    def _get_price(self, step: int) -> float:
        """Get current close price."""
        row = self.df.iloc[step]
        price = float(row.get("close", row.get("Close", 1.0)))
        return max(price, 1e-8)

    def _get_obs(self) -> np.ndarray:
        """
        Build normalized observation vector.

        Technical features: z-score normalized with running stats (stationary).
        State features: all fractions/ratios (inherently bounded, no raw prices).
        """
        idx = max(0, self.current_step - 1)
        row = self.df.iloc[idx]

        # Technical features (pre-computed in df)
        raw_tech = row[self.feature_cols].values.astype(np.float64)
        raw_tech = np.nan_to_num(raw_tech, nan=0.0, posinf=0.0, neginf=0.0)

        if self.normalize_obs and self._obs_normalizer is not None:
            self._obs_normalizer.update(raw_tech)
            tech_features = self._obs_normalizer.normalize_without_update(raw_tech)
        else:
            tech_features = raw_tech.astype(np.float32)

        # State features — all stationary ratios
        price = self._get_price(idx)
        unrealized = self._calc_unrealized_pnl(price)
        equity = self.balance + unrealized

        # 1. Net position fraction: signed position as fraction of equity
        position_frac = self.position_notional / (equity + 1e-8)
        position_frac = float(np.clip(position_frac, -2.0, 2.0))

        # 2. Unrealized PnL as fraction of initial balance
        unrealized_frac = unrealized / (self.initial_balance + 1e-8)
        unrealized_frac = float(np.clip(unrealized_frac, -1.0, 1.0))

        # 3. Current drawdown
        drawdown = (self.peak_equity - equity) / (self.peak_equity + 1e-8)
        drawdown = float(np.clip(drawdown, 0.0, 1.0))

        # 4. Equity growth fraction vs initial
        equity_growth = (equity - self.initial_balance) / (self.initial_balance + 1e-8)
        equity_growth = float(np.clip(equity_growth, -1.0, 2.0))

        # 5. Rolling mean return (recent momentum signal)
        rolling_return = float(np.mean(self._rolling_returns)) if self._rolling_returns else 0.0
        rolling_return = float(np.clip(rolling_return * 100.0, -5.0, 5.0))

        # 6. Margin usage ratio
        used_margin = abs(self.position_notional) / (self.leverage + 1e-8)
        margin_ratio = used_margin / (equity + 1e-8)
        margin_ratio = float(np.clip(margin_ratio, 0.0, 2.0))

        state_features = np.array([
            position_frac,
            unrealized_frac,
            drawdown,
            equity_growth,
            rolling_return,
            margin_ratio,
        ], dtype=np.float32)

        obs = np.concatenate([tech_features, state_features])
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        return obs.astype(np.float32)

    def _get_info(self) -> Dict[str, Any]:
        price = self._get_price(max(0, self.current_step - 1))
        unrealized = self._calc_unrealized_pnl(price)
        equity = self.balance + unrealized
        drawdown = (self.peak_equity - equity) / (self.peak_equity + 1e-8)
        return {
            "balance": self.balance,
            "equity": equity,
            "peak_equity": self.peak_equity,
            "position_notional": self.position_notional,
            "position_frac": self.position_notional / (equity + 1e-8),
            "unrealized_pnl": unrealized,
            "realized_pnl": self.realized_pnl,
            "drawdown": float(drawdown),
            "total_costs": self.total_costs,
            "step": self.current_step,
            "long_lots": max(0.0, self.position_notional),
            "short_lots": max(0.0, -self.position_notional),
        }

    def render(self) -> None:
        price = self._get_price(max(0, self.current_step - 1))
        unrealized = self._calc_unrealized_pnl(price)
        equity = self.balance + unrealized
        print(
            f"Step {self.current_step:5d} | "
            f"Equity: {equity:10.2f} | "
            f"Position: {self.position_notional:10.2f} | "
            f"Unrealized: {unrealized:8.2f} | "
            f"Realized: {self.realized_pnl:8.2f} | "
            f"Drawdown: {(self.peak_equity - equity) / (self.peak_equity + 1e-8):.2%}"
        )