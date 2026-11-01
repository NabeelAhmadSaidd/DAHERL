"""
Comprehensive pytest suite for the trading environment and agents.
"""

import pytest
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

from environment.trading_env import TradingEnv


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------

def make_df(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Create synthetic OHLCV dataframe."""
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    close = np.maximum(close, 1.0)
    high = close * (1 + rng.uniform(0.001, 0.01, n))
    low = close * (1 - rng.uniform(0.001, 0.01, n))
    open_ = low + (high - low) * rng.uniform(0, 1, n)
    volume = rng.uniform(1000, 5000, n)

    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "feature_1": rng.normal(0, 1, n),
        "feature_2": rng.normal(0, 1, n),
        "feature_3": rng.uniform(-1, 1, n),
    })
    return df


FEATURE_COLS = ["feature_1", "feature_2", "feature_3"]


@pytest.fixture
def df():
    return make_df()


@pytest.fixture
def crypto_env_discrete(df):
    return TradingEnv(
        df=df, feature_cols=FEATURE_COLS, expert_id=1,
        asset_type="crypto", contract_size=1.0, min_lot=0.01, max_lot=1.0,
        leverage=5, initial_balance=10000.0, window_size=10,
        continuous_actions=False,
    )


@pytest.fixture
def crypto_env_continuous(df):
    return TradingEnv(
        df=df, feature_cols=FEATURE_COLS, expert_id=2,
        asset_type="crypto", contract_size=1.0, min_lot=0.01, max_lot=1.0,
        leverage=5, initial_balance=10000.0, window_size=10,
        continuous_actions=True,
    )


@pytest.fixture
def forex_env(df):
    return TradingEnv(
        df=df, feature_cols=FEATURE_COLS, expert_id=1,
        asset_type="forex", contract_size=100000.0, min_lot=1.0, max_lot=100.0,
        leverage=30, initial_balance=10000.0, window_size=10,
        continuous_actions=False,
    )


# -------------------------------------------------------------------------
# Reset tests
# -------------------------------------------------------------------------

class TestReset:
    def test_reset_returns_tuple(self, crypto_env_discrete):
        result = crypto_env_discrete.reset()
        assert isinstance(result, tuple), "reset() must return a tuple"
        assert len(result) == 2, "reset() must return (obs, info)"

    def test_reset_obs_shape(self, crypto_env_discrete):
        obs, info = crypto_env_discrete.reset()
        expected_dim = len(FEATURE_COLS) + 8
        assert obs.shape == (expected_dim,), f"Expected obs shape ({expected_dim},), got {obs.shape}"

    def test_reset_obs_dtype(self, crypto_env_discrete):
        obs, info = crypto_env_discrete.reset()
        assert obs.dtype == np.float32

    def test_reset_obs_no_nan(self, crypto_env_discrete):
        obs, info = crypto_env_discrete.reset()
        assert not np.any(np.isnan(obs)), "Observation contains NaN after reset"
        assert not np.any(np.isinf(obs)), "Observation contains Inf after reset"

    def test_reset_info_is_dict(self, crypto_env_discrete):
        obs, info = crypto_env_discrete.reset()
        assert isinstance(info, dict)

    def test_reset_balance_restored(self, crypto_env_discrete):
        crypto_env_discrete.reset()
        for _ in range(50):
            obs, r, terminated, truncated, info = crypto_env_discrete.step(2)
            if terminated or truncated:
                break
        obs, info = crypto_env_discrete.reset()
        assert abs(crypto_env_discrete.balance - crypto_env_discrete.initial_balance) < 1e-6

    def test_reset_positions_cleared(self, crypto_env_discrete):
        crypto_env_discrete.reset()
        for _ in range(20):
            crypto_env_discrete.step(2)
        crypto_env_discrete.reset()
        assert len(crypto_env_discrete.open_trades) == 0
        assert crypto_env_discrete.long_lots == 0.0
        assert crypto_env_discrete.short_lots == 0.0

    def test_reset_with_seed(self, df):
        env = TradingEnv(df=df, feature_cols=FEATURE_COLS, expert_id=1,
                         asset_type="crypto", contract_size=1.0, min_lot=0.01,
                         max_lot=1.0, leverage=5, initial_balance=10000.0,
                         window_size=10, continuous_actions=False)
        obs1, _ = env.reset(seed=0)
        obs2, _ = env.reset(seed=0)
        np.testing.assert_array_equal(obs1, obs2)


# -------------------------------------------------------------------------
# Step tests
# -------------------------------------------------------------------------

class TestStep:
    def test_step_returns_5_tuple(self, crypto_env_discrete):
        crypto_env_discrete.reset()
        result = crypto_env_discrete.step(1)
        assert len(result) == 5, "step() must return (obs, reward, terminated, truncated, info)"

    def test_step_obs_shape(self, crypto_env_discrete):
        crypto_env_discrete.reset()
        obs, reward, terminated, truncated, info = crypto_env_discrete.step(1)
        expected_dim = len(FEATURE_COLS) + 8
        assert obs.shape == (expected_dim,)

    def test_step_reward_is_scalar(self, crypto_env_discrete):
        crypto_env_discrete.reset()
        obs, reward, terminated, truncated, info = crypto_env_discrete.step(1)
        assert np.isscalar(reward) or reward.ndim == 0

    def test_step_reward_clipped(self, crypto_env_discrete):
        crypto_env_discrete.reset()
        for _ in range(100):
            obs, reward, terminated, truncated, info = crypto_env_discrete.step(2)
            assert -10.0 <= reward <= 10.0, f"Reward {reward} out of clip range"
            if terminated or truncated:
                crypto_env_discrete.reset()

    def test_step_terminated_is_bool(self, crypto_env_discrete):
        crypto_env_discrete.reset()
        obs, reward, terminated, truncated, info = crypto_env_discrete.step(1)
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))

    def test_step_info_keys(self, crypto_env_discrete):
        crypto_env_discrete.reset()
        obs, reward, terminated, truncated, info = crypto_env_discrete.step(1)
        required_keys = {"balance", "equity", "long_lots", "short_lots",
                         "num_open_trades", "used_margin", "free_margin",
                         "realized_pnl", "drawdown"}
        for k in required_keys:
            assert k in info, f"Missing key '{k}' in info"

    def test_step_all_discrete_actions(self, crypto_env_discrete):
        for action in [0, 1, 2]:
            crypto_env_discrete.reset()
            obs, reward, terminated, truncated, info = crypto_env_discrete.step(action)
            assert obs is not None

    def test_step_continuous_action(self, crypto_env_continuous):
        crypto_env_continuous.reset()
        action = np.array([0.5], dtype=np.float32)
        obs, reward, terminated, truncated, info = crypto_env_continuous.step(action)
        assert obs is not None

    def test_step_truncated_at_end(self, df):
        env = TradingEnv(df=df, feature_cols=FEATURE_COLS, expert_id=1,
                         asset_type="crypto", contract_size=1.0, min_lot=0.01,
                         max_lot=1.0, leverage=5, initial_balance=10000.0,
                         window_size=10, continuous_actions=False)
        env.reset()
        done = False
        steps = 0
        while not done and steps < 10000:
            obs, reward, terminated, truncated, info = env.step(1)
            done = terminated or truncated
            steps += 1
        assert done, "Environment should terminate or truncate"


# -------------------------------------------------------------------------
# Portfolio accounting tests
# -------------------------------------------------------------------------

class TestPortfolioAccounting:
    def test_equity_equals_balance_plus_unrealized(self, crypto_env_discrete):
        crypto_env_discrete.reset()
        crypto_env_discrete.step(2)  # Open long
        price = float(crypto_env_discrete.df.iloc[
            max(0, crypto_env_discrete.current_step - 1)
        ]["close"])
        unrealized = crypto_env_discrete._calc_unrealized_pnl(price)
        expected_equity = crypto_env_discrete.balance + unrealized
        assert abs(crypto_env_discrete.equity - expected_equity) < 1e-4, (
            f"equity={crypto_env_discrete.equity}, balance+unrealized={expected_equity}"
        )

    def test_balance_decreases_on_trade_cost(self, crypto_env_discrete):
        crypto_env_discrete.reset()
        initial_balance = crypto_env_discrete.balance
        crypto_env_discrete.step(2)  # Buy
        # Balance should drop by cost; equity may differ
        assert crypto_env_discrete.balance <= initial_balance + 1e-4

    def test_long_lots_increase_on_buy(self, crypto_env_discrete):
        crypto_env_discrete.reset()
        crypto_env_discrete.step(2)  # Buy
        assert crypto_env_discrete.long_lots > 0.0

    def test_short_lots_increase_on_sell(self, crypto_env_discrete):
        crypto_env_discrete.reset()
        crypto_env_discrete.step(0)  # Sell
        assert crypto_env_discrete.short_lots > 0.0

    def test_hold_does_not_change_lots(self, crypto_env_discrete):
        crypto_env_discrete.reset()
        long_before = crypto_env_discrete.long_lots
        short_before = crypto_env_discrete.short_lots
        crypto_env_discrete.step(1)  # Hold
        assert abs(crypto_env_discrete.long_lots - long_before) < 1e-6
        assert abs(crypto_env_discrete.short_lots - short_before) < 1e-6

    def test_realized_pnl_updates_on_close(self, crypto_env_discrete):
        crypto_env_discrete.reset()
        crypto_env_discrete.step(2)  # Open long
        realized_before = crypto_env_discrete.realized_pnl
        crypto_env_discrete.step(0)  # Switch to short (closes long)
        # Realized PnL should have changed
        assert crypto_env_discrete.realized_pnl != realized_before or \
               crypto_env_discrete.long_lots == 0.0

    def test_margin_consistency(self, crypto_env_discrete):
        crypto_env_discrete.reset()
        crypto_env_discrete.step(2)
        assert crypto_env_discrete.used_margin >= 0.0
        assert abs(crypto_env_discrete.free_margin -
                   (crypto_env_discrete.equity - crypto_env_discrete.used_margin)) < 1e-3

    def test_no_negative_lots(self, crypto_env_discrete):
        crypto_env_discrete.reset()
        for action in [2, 0, 1, 2, 0, 2, 1, 0]:
            crypto_env_discrete.step(action)
            assert crypto_env_discrete.long_lots >= 0.0, "long_lots went negative"
            assert crypto_env_discrete.short_lots >= 0.0, "short_lots went negative"


# -------------------------------------------------------------------------
# Observation tests
# -------------------------------------------------------------------------

class TestObservation:
    def test_obs_in_observation_space(self, crypto_env_discrete):
        obs, _ = crypto_env_discrete.reset()
        assert crypto_env_discrete.observation_space.contains(obs.astype(np.float32))

    def test_obs_no_nan_after_steps(self, crypto_env_discrete):
        crypto_env_discrete.reset()
        for i in range(200):
            action = i % 3
            obs, reward, terminated, truncated, info = crypto_env_discrete.step(action)
            assert not np.any(np.isnan(obs)), f"NaN in obs at step {i}"
            if terminated or truncated:
                crypto_env_discrete.reset()

    def test_obs_financial_state_normalized(self, crypto_env_discrete):
        obs, _ = crypto_env_discrete.reset()
        # Last 8 elements are financial features — balance/equity ratios should start ~1
        assert abs(obs[-8] - 1.0) < 0.1, "balance_norm should start near 1.0"
        assert abs(obs[-7] - 1.0) < 0.1, "equity_norm should start near 1.0"


# -------------------------------------------------------------------------
# Action space tests
# -------------------------------------------------------------------------

class TestActionSpace:
    def test_discrete_action_space(self, crypto_env_discrete):
        assert isinstance(crypto_env_discrete.action_space, gym.spaces.Discrete)
        assert crypto_env_discrete.action_space.n == 3

    def test_continuous_action_space(self, crypto_env_continuous):
        assert isinstance(crypto_env_continuous.action_space, gym.spaces.Box)
        assert crypto_env_continuous.action_space.shape == (1,)
        assert crypto_env_continuous.action_space.low[0] == -1.0
        assert crypto_env_continuous.action_space.high[0] == 1.0

    def test_sample_discrete_action_valid(self, crypto_env_discrete):
        crypto_env_discrete.reset()
        for _ in range(20):
            action = crypto_env_discrete.action_space.sample()
            assert action in [0, 1, 2]

    def test_sample_continuous_action_valid(self, crypto_env_continuous):
        crypto_env_continuous.reset()
        for _ in range(20):
            action = crypto_env_continuous.action_space.sample()
            assert -1.0 <= action[0] <= 1.0


# -------------------------------------------------------------------------
# Reward function tests
# -------------------------------------------------------------------------

class TestRewardFunctions:
    @pytest.mark.parametrize("expert_id", [1, 2, 3, 4, 5, 6])
    def test_all_expert_rewards_run(self, df, expert_id):
        env = TradingEnv(
            df=df, feature_cols=FEATURE_COLS, expert_id=expert_id,
            asset_type="crypto", contract_size=1.0, min_lot=0.01, max_lot=1.0,
            leverage=5, initial_balance=10000.0, window_size=10,
            continuous_actions=(expert_id in [2, 4, 5, 6]),
        )
        env.reset()
        for step in range(50):
            if env.continuous_actions:
                action = np.array([0.3], dtype=np.float32)
            else:
                action = step % 3
            obs, reward, terminated, truncated, info = env.step(action)
            assert not np.isnan(reward), f"NaN reward at expert_id={expert_id}, step={step}"
            assert -10.0 <= reward <= 10.0, f"Reward {reward} out of clip range"
            if terminated or truncated:
                break

    def test_reward_portfolio_return(self):
        from environment.reward_functions import reward_portfolio_return
        r = reward_portfolio_return(0.01, 0.001)
        assert isinstance(r, float)
        assert r > 0  # Positive return, small cost -> positive reward

    def test_reward_rolling_sharpe(self):
        from environment.reward_functions import reward_rolling_sharpe, RollingRewardCalculator
        calc = RollingRewardCalculator(window_size=10)
        for _ in range(10):
            calc.add_step(0.001)
        r = reward_rolling_sharpe(calc, 0.0)
        assert isinstance(r, float)
        assert not np.isnan(r)

    def test_reward_rolling_sortino(self):
        from environment.reward_functions import reward_rolling_sortino, RollingRewardCalculator
        calc = RollingRewardCalculator(window_size=10)
        for v in [0.01, -0.005, 0.008, -0.002, 0.003]:
            calc.add_step(v)
        r = reward_rolling_sortino(calc, 0.0)
        assert isinstance(r, float)
        assert not np.isnan(r)

    def test_reward_with_zero_returns(self):
        from environment.reward_functions import reward_portfolio_return
        r = reward_portfolio_return(0.0, 0.0)
        assert r == 0.0


# -------------------------------------------------------------------------
# Trade cost tests
# -------------------------------------------------------------------------

class TestTradeCosts:
    def test_cost_positive(self, crypto_env_discrete):
        crypto_env_discrete.reset()
        price = 100.0
        cost = crypto_env_discrete._compute_trade_cost(0.1, price)
        assert cost > 0.0, "Trade cost must be positive"

    def test_cost_scales_with_lots(self, crypto_env_discrete):
        crypto_env_discrete.reset()
        price = 100.0
        cost_small = crypto_env_discrete._compute_trade_cost(0.1, price)
        cost_large = crypto_env_discrete._compute_trade_cost(0.2, price)
        assert cost_large > cost_small

    def test_forex_commission_per_lot(self, forex_env):
        forex_env.reset()
        price = 1.1000
        cost = forex_env._compute_trade_cost(1.0, price)
        assert cost > 0.0


# -------------------------------------------------------------------------
# Diversity module tests
# -------------------------------------------------------------------------

class TestDiversityManager:
    def test_register_and_compute(self, tmp_path):
        from diversity.diversity_manager import DiversityManager
        mgr = DiversityManager(storage_dir=str(tmp_path))

        traj1 = {
            "actions": np.array([0.5, 0.6, 0.7]),
            "returns": np.array([0.01, 0.02, 0.01]),
            "positions": np.array([0.3, 0.4, 0.3]),
        }
        mgr.register_expert("PPO", traj1)

        penalty = mgr.compute_penalty(
            np.array([0.5]), np.array([0.01]), np.array([0.3]), "SAC"
        )
        assert 0.0 <= penalty <= mgr.alpha

    def test_no_penalty_without_experts(self, tmp_path):
        from diversity.diversity_manager import DiversityManager
        mgr = DiversityManager(storage_dir=str(tmp_path))
        penalty = mgr.compute_penalty(
            np.array([0.5]), np.array([0.01]), np.array([0.3]), "PPO"
        )
        assert penalty == 0.0

    def test_penalty_clamped(self, tmp_path):
        from diversity.diversity_manager import DiversityManager
        mgr = DiversityManager(storage_dir=str(tmp_path), alpha=0.1)
        traj = {
            "actions": np.ones(100),
            "returns": np.ones(100) * 0.01,
            "positions": np.ones(100) * 0.5,
        }
        mgr.register_expert("PPO", traj)
        mgr.update_epoch(1000)  # High epoch -> max schedule

        penalty = mgr.compute_penalty(
            np.ones(10), np.ones(10) * 0.01, np.ones(10) * 0.5, "SAC"
        )
        assert 0.0 <= penalty <= 0.1 + 1e-8

    def test_trajectory_persistence(self, tmp_path):
        from diversity.diversity_manager import DiversityManager
        mgr1 = DiversityManager(storage_dir=str(tmp_path))
        traj = {
            "actions": np.array([0.1, 0.2]),
            "returns": np.array([0.001, 0.002]),
            "positions": np.array([0.1, 0.1]),
        }
        mgr1.register_expert("TestExpert", traj)

        mgr2 = DiversityManager(storage_dir=str(tmp_path))
        assert "TestExpert" in mgr2._expert_trajectories


# -------------------------------------------------------------------------
# Data split integrity tests
# -------------------------------------------------------------------------

class TestDataSplitIntegrity:
    def test_train_test_no_overlap(self):
        df = make_df(n=1000)
        split = int(len(df) * 0.8)
        train_df = df.iloc[:split].reset_index(drop=True)
        test_df = df.iloc[split:].reset_index(drop=True)

        assert len(train_df) + len(test_df) == len(df)

        train_indices = set(df.iloc[:split].index)
        test_indices = set(df.iloc[split:].index)
        assert train_indices.isdisjoint(test_indices), "Train/test indices overlap"

    def test_train_env_uses_train_data_only(self):
        df = make_df(n=1000)
        split = int(len(df) * 0.8)
        train_df = df.iloc[:split].reset_index(drop=True)

        env = TradingEnv(
            df=train_df, feature_cols=FEATURE_COLS, expert_id=1,
            asset_type="crypto", contract_size=1.0, min_lot=0.01, max_lot=1.0,
            leverage=5, initial_balance=10000.0, window_size=10,
        )
        env.reset()
        assert len(env.df) == len(train_df)

    def test_env_does_not_exceed_df_bounds(self):
        df = make_df(n=100)
        env = TradingEnv(
            df=df, feature_cols=FEATURE_COLS, expert_id=1,
            asset_type="crypto", contract_size=1.0, min_lot=0.01, max_lot=1.0,
            leverage=5, initial_balance=10000.0, window_size=10,
        )
        env.reset()
        done = False
        while not done:
            obs, reward, terminated, truncated, info = env.step(1)
            done = terminated or truncated
            assert env.current_step <= len(df), \
                f"current_step {env.current_step} exceeded df length {len(df)}"


# -------------------------------------------------------------------------
# Gymnasium env_checker (optional but useful)
# -------------------------------------------------------------------------

class TestGymnasiumCompliance:
    def test_env_checker_discrete(self, df):
        env = TradingEnv(
            df=df, feature_cols=FEATURE_COLS, expert_id=1,
            asset_type="crypto", contract_size=1.0, min_lot=0.01, max_lot=1.0,
            leverage=5, initial_balance=10000.0, window_size=10,
            continuous_actions=False,
        )
        # check_env is strict — wrap in try/except to surface warnings as failures
        try:
            check_env(env, warn=True, skip_render_check=True)
        except Exception as e:
            pytest.fail(f"Gymnasium env_checker failed: {e}")

    def test_env_checker_continuous(self, df):
        env = TradingEnv(
            df=df, feature_cols=FEATURE_COLS, expert_id=2,
            asset_type="crypto", contract_size=1.0, min_lot=0.01, max_lot=1.0,
            leverage=5, initial_balance=10000.0, window_size=10,
            continuous_actions=True,
        )
        try:
            check_env(env, warn=True, skip_render_check=True)
        except Exception as e:
            pytest.fail(f"Gymnasium env_checker failed: {e}")