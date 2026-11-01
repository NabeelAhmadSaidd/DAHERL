"""
Unified training script for all six DAHERL experts.

Supports --no-diversity flag to train without DiversityManager penalty.
Logs (CSV and TensorBoard) are saved under logs/.../no_diversity/ when the flag is active,
otherwise directly under logs/.../. Checkpoints follow the same pattern under checkpoints/.

USAGE:
    # Fresh training
    python -m experts.train_experts --expert 1 --asset EURUSD --data data/processed/EURUSD/train.csv 

    # Resume training for the configured number of timesteps (e.g., 720k steps) from a checkpoint
    python -m experts.train_experts --expert 2 --asset EURUSD --data data/processed/EURUSD/train.csv \\
        --resume checkpoints/EURUSD/diversity/SAC/SAC_expert_30.pt
"""

import argparse
import csv
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import re
import pickle  # for loading/saving trajectory

from environment.trading_env import TradingEnv
from experts.ppo_expert import PPOAgent
from experts.sac_expert import SACAgent
from experts.ddqn_expert import DDQNAgent
from experts.tqc_expert import TQCAgent
from experts.td3_expert import TD3Agent
from experts.ppg_expert import PPGAgent         
from diversity.diversity_manager import DiversityManager
from diversity.diversity_scheduler import DiversityScheduler

# -------------------------------------------------------------------------
# Config and asset helpers
# -------------------------------------------------------------------------

def load_config(config_path: str = "configs/experts.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def infer_asset_from_data_path(data_path: str) -> str:
    return Path(data_path).parent.name


def detect_asset_type(asset_name: str) -> str:
    asset_upper = asset_name.upper()
    crypto_suffixes = ("USDT", "BUSD", "USDC", "BTC", "ETH", "BNB",
                       "XRP", "ADA", "DOGE", "SOL", "DOT")
    if any(asset_upper.endswith(s) for s in crypto_suffixes):
        return "crypto"
    forex_bases = ("EUR", "GBP", "USD", "JPY", "CHF", "CAD", "AUD", "NZD")
    if (any(asset_upper.startswith(b) for b in forex_bases) and
            any(asset_upper.endswith(b) for b in forex_bases) and
            len(asset_upper) == 6):
        return "forex"
    return "crypto"


def get_env_params(asset_type: str, expert_id: int) -> dict:
    if asset_type == "forex":
        return dict(
            asset_type="forex",
            contract_size=100000.0,
            commission_rate=1.5 / 100000.0,
            spread_rate=0.00005,
            slippage_rate=0.00001,
            leverage=30,
            maintenance_margin_ratio=0.5,
            initial_balance=10000.0,
            window_size=20,
            reward_scale=1.0,
            clip_reward=(-10.0, 10.0),
            max_position_fraction=0.95,
            continuous_actions=False,
        )
    else:  # crypto
        return dict(
            asset_type="crypto",
            contract_size=1.0,
            commission_rate=0.0005,
            spread_rate=0.0002,
            slippage_rate=0.0001,
            leverage=5,
            maintenance_margin_ratio=0.5,
            initial_balance=10000.0,
            window_size=20,
            reward_scale=1.0,
            clip_reward=(-10.0, 10.0),
            max_position_fraction=0.30,
            continuous_actions=False,
        )


# -------------------------------------------------------------------------
# Metrics helpers
# -------------------------------------------------------------------------

def compute_sharpe(returns: list, eps: float = 1e-8) -> float:
    if len(returns) < 2:
        return 0.0
    arr = np.array(returns, dtype=np.float64)
    std = arr.std(ddof=1)
    if std < eps or np.isnan(std):
        return 0.0
    return float(arr.mean() / (std + eps))


def compute_sortino(returns: list, eps: float = 1e-8) -> float:
    if len(returns) < 2:
        return 0.0
    arr = np.array(returns, dtype=np.float64)
    downside = arr[arr < 0]
    if len(downside) < 2:
        return 0.0
    down_std = downside.std(ddof=1)
    if down_std < eps or np.isnan(down_std):
        return 0.0
    return float(arr.mean() / (down_std + eps))


def compute_max_drawdown(equity_curve: list) -> float:
    if len(equity_curve) < 2:
        return 0.0
    arr = np.array([max(e, 1e-4) for e in equity_curve], dtype=np.float64)
    peak = np.maximum.accumulate(arr)
    dd = (peak - arr) / (peak + 1e-8)
    return float(np.max(dd))


def parse_episode_from_checkpoint(path: str) -> int:
    """Extract episode number from checkpoint filename like 'SAC_expert_30.pt'."""
    match = re.search(r'_expert_(\d+)\.pt$', path)
    if match:
        return int(match.group(1))
    return 0


def load_trajectory(storage_dir: str, agent_name: str) -> dict:
    """Load existing trajectory pickle if it exists, else return None."""
    pkl_path = Path(storage_dir) / f"{agent_name}.pkl"
    if pkl_path.exists():
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        return data
    return None


# -------------------------------------------------------------------------
# PPO training loop (resume-friendly, step-based)
# -------------------------------------------------------------------------

def train_ppo(
    cfg: dict,
    df: pd.DataFrame,
    feature_cols: list,
    save_path: str,
    asset_type: str,
    asset_name: str,
    use_diversity: bool = True,
    resume_from: str = None,
    initial_episode: int = 0,
) -> None:
    print(f"\n{'='*70}\nTRAINING PPO (Expert 1) on {asset_type.upper()} — {asset_name}\n{'='*70}\n")

    env_params = get_env_params(asset_type, expert_id=1)
    env_params["continuous_actions"] = False
    env_params["leverage"] = 1
    env = TradingEnv(df=df, feature_cols=feature_cols, expert_id=1, **env_params)

    state_dim = len(feature_cols) + env._n_state_features
    ppo_cfg = cfg["ppo"]

    base_csv = Path("logs/training/experts") / asset_name / "ppo"
    base_tb  = Path("logs/tensorboard/ppo") / asset_name
    if not use_diversity:
        base_csv = base_csv / "no_diversity"
        base_tb  = base_tb  / "no_diversity"
    csv_log_dir = base_csv
    csv_log_dir.mkdir(parents=True, exist_ok=True)
    log_file = csv_log_dir / "episode_log.csv"
    tb_log_dir = str(base_tb)

    timesteps: int = ppo_cfg.get("timesteps", cfg["training"]["timesteps"])
    rollout_steps: int = ppo_cfg["rollout_steps"]

    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=3,
        lr=ppo_cfg["learning_rate"],
        gamma=ppo_cfg["gamma"],
        gae_lambda=ppo_cfg["gae_lambda"],
        clip_eps=ppo_cfg["clip_eps"],
        entropy_coef=ppo_cfg["entropy_coef"],
        entropy_coef_final=ppo_cfg.get("entropy_coef_final", 0.001),
        value_coef=ppo_cfg["value_coef"],
        max_grad_norm=ppo_cfg["max_grad_norm"],
        update_epochs=ppo_cfg["update_epochs"],
        batch_size=ppo_cfg["batch_size"],
        rollout_steps=rollout_steps,
        hidden_dim=ppo_cfg["hidden_dim"],
        num_layers=ppo_cfg["num_layers"],
        activation=ppo_cfg["activation"],
        lr_anneal_steps=ppo_cfg.get("lr_anneal_steps", timesteps),
        entropy_anneal_steps=ppo_cfg.get("entropy_anneal_steps", timesteps),
        clip_value_loss=ppo_cfg.get("clip_value_loss", True),
        target_kl=ppo_cfg.get("target_kl", 0.02),
        log_dir=tb_log_dir,
    )

    storage_dir = f"storage/{asset_name}"
    if not use_diversity:
        storage_dir = f"{storage_dir}/no_diversity"
    else:
        storage_dir = f"{storage_dir}/diversity"

    scheduler = DiversityScheduler() if use_diversity else None
    manager = DiversityManager(
        storage_dir=storage_dir,
        enabled=use_diversity,
        scheduler=scheduler,
        penalty_scale=0.02,
        similarity_threshold=0.5,
        window_size=30,
    )

    # -------- Resume: load model and existing trajectory --------
    if resume_from and Path(resume_from).exists():
        agent.load(resume_from)
        print(f"[PPO] Loaded checkpoint from {resume_from}")

    all_actions = []
    all_step_returns = []
    all_positions = []
    if resume_from:
        old_traj = load_trajectory(storage_dir, "PPO")
        if old_traj is not None:
            all_actions = old_traj['actions'].tolist() if isinstance(old_traj['actions'], np.ndarray) else list(old_traj['actions'])
            all_step_returns = old_traj['returns'].tolist() if isinstance(old_traj['returns'], np.ndarray) else list(old_traj['returns'])
            all_positions = old_traj['positions'].tolist() if isinstance(old_traj['positions'], np.ndarray) else list(old_traj['positions'])
            print(f"[PPO] Loaded existing trajectory with {len(all_actions)} steps")

    episode_count: int = initial_episode
    episode_equity: list = []
    episode_step_returns: list = []
    episode_positions: list = []
    episode_costs: list = []

    state, info = env.reset()
    manager.reset_buffer()
    steps_done: int = 0

    checkpoint_dir = Path(save_path).parent

    # Append to log if resuming, else overwrite
    file_mode = 'a' if resume_from and log_file.exists() else 'w'
    with open(log_file, file_mode, newline="") as log_fh:
        log_writer = csv.writer(log_fh)
        if log_file.stat().st_size == 0 or file_mode == 'w':
            log_writer.writerow([
                "episode", "total_return", "sharpe", "sortino", "max_drawdown",
                "avg_position", "length", "policy_loss", "value_loss", "entropy",
                "approx_kl", "clip_frac",
            ])

        pbar = tqdm(total=timesteps, desc="PPO Training", unit="step", leave=True)

        while steps_done < timesteps:
            agent.reset_buffer()
            agent.reset_episode_accumulators()

            for _ in range(rollout_steps):
                if steps_done >= timesteps:
                    break

                action, log_prob, value = agent.act(state)

                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                step_return = float(info.get("step_return", 0.0))
                position_frac = float(abs(info.get("position_frac", 0.0)))
                equity = float(info.get("equity", env.initial_balance))
                cost = float(info.get("cost", 0.0))

                if use_diversity:
                    penalty = manager.compute_penalty(step_return, "PPO")
                else:
                    penalty = 0.0
                adjusted_reward = float(reward) - float(penalty)

                agent.store(
                    state=state,
                    action=action,
                    reward=adjusted_reward,
                    done=done,
                    log_prob=log_prob,
                    value=value,
                    next_state=next_state,
                )
                agent.store_step_info(info)

                episode_step_returns.append(step_return)
                episode_equity.append(equity)
                episode_positions.append(position_frac)
                episode_costs.append(cost)

                all_actions.append(float(action))
                all_step_returns.append(step_return)
                all_positions.append(position_frac)

                steps_done += 1
                pbar.update(1)

                if done:
                    episode_count += 1
                    manager.update_epoch(episode_count)

                    if len(episode_equity) > 1:
                        start_eq = episode_equity[0]
                        final_eq = episode_equity[-1]
                        total_return = (final_eq - start_eq) / (abs(start_eq) + 1e-8)
                        sharpe = compute_sharpe(episode_step_returns)
                        sortino = compute_sortino(episode_step_returns)
                        max_dd = compute_max_drawdown(episode_equity)
                        avg_pos = float(np.mean(episode_positions))
                        ep_len = len(episode_step_returns)

                        print(
                            f"[PPO] Ep {episode_count:4d} | "
                            f"Return: {total_return * 100:+7.2f}% | "
                            f"Sharpe: {sharpe:+6.3f} | "
                            f"Sortino: {sortino:+6.3f} | "
                            f"MaxDD: {max_dd * 100:5.2f}% | "
                            f"Steps: {steps_done:>8d} | "
                            f"Len: {ep_len:4d}",
                            flush=True,
                        )

                        agent.log_episode(episode_count, total_return, sharpe,
                                          sortino, max_dd, episode_equity)

                        _last_ep_stats = (
                            episode_count, total_return, sharpe, sortino,
                            max_dd, avg_pos, ep_len,
                        )
                    else:
                        _last_ep_stats = None

                    # Periodic checkpoint every 15 episodes
                    if episode_count % 15 == 0:
                        ckpt_path = checkpoint_dir / f"PPO_expert_{episode_count}.pt"
                        agent.save(str(ckpt_path))
                        print(f"[PPO] Checkpoint saved to {ckpt_path}")

                    episode_step_returns.clear()
                    episode_equity.clear()
                    episode_positions.clear()
                    episode_costs.clear()

                    state, _ = env.reset()
                    manager.reset_buffer()
                    agent.reset_episode_accumulators()
                else:
                    state = next_state

            agent.update()

            if episode_count > 0 and _last_ep_stats is not None:
                log_writer.writerow([
                    _last_ep_stats[0],
                    _last_ep_stats[1],
                    _last_ep_stats[2],
                    _last_ep_stats[3],
                    _last_ep_stats[4],
                    _last_ep_stats[5],
                    _last_ep_stats[6],
                    agent.last_policy_loss,
                    agent.last_value_loss,
                    agent.last_entropy,
                    agent.last_approx_kl,
                    agent.last_clip_frac,
                ])

        pbar.close()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    agent.save(save_path)
    print(f"\nPPO saved to {save_path}")

    trajectory = {
        "actions": np.array(all_actions, dtype=np.float32),
        "returns": np.array(all_step_returns, dtype=np.float32),
        "positions": np.array(all_positions, dtype=np.float32),
    }
    manager.register_expert("PPO", trajectory)
    print(f"PPO trajectory saved to {storage_dir}/PPO.pkl\n")


# -------------------------------------------------------------------------
# Off-policy training loop (resume-friendly, step-based)
# -------------------------------------------------------------------------

def train_off_policy(
    agent_name: str,
    agent,
    env: TradingEnv,
    timesteps: int,
    save_path: str,
    storage_dir: str,
    log_dir: Path,
    continuous: bool,
    use_diversity: bool = True,
    resume_from: str = None,
    initial_episode: int = 0,
) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "episode_log.csv"

    scheduler = DiversityScheduler() if use_diversity else None
    manager = DiversityManager(
        storage_dir=storage_dir,
        enabled=use_diversity,
        scheduler=scheduler,
        penalty_scale=0.02,
        similarity_threshold=0.5,
        window_size=30,
    )

    # -------- Resume: load model and existing trajectory --------
    if resume_from and Path(resume_from).exists():
        agent.load(resume_from)
        print(f"[{agent_name}] Loaded checkpoint from {resume_from}")

    all_actions = []
    all_step_returns = []
    all_positions = []
    if resume_from:
        old_traj = load_trajectory(storage_dir, agent_name)
        if old_traj is not None:
            all_actions = old_traj['actions'].tolist() if isinstance(old_traj['actions'], np.ndarray) else list(old_traj['actions'])
            all_step_returns = old_traj['returns'].tolist() if isinstance(old_traj['returns'], np.ndarray) else list(old_traj['returns'])
            all_positions = old_traj['positions'].tolist() if isinstance(old_traj['positions'], np.ndarray) else list(old_traj['positions'])
            print(f"[{agent_name}] Loaded existing trajectory with {len(all_actions)} steps")

    state, info = env.reset()
    manager.reset_buffer()
    episode_count = initial_episode
    episode_returns: list = []
    episode_equity: list = []
    episode_positions: list = []
    episode_costs: list = []

    checkpoint_dir = Path(save_path).parent

    # Append to log if resuming, else overwrite
    file_mode = 'a' if resume_from and log_file.exists() else 'w'
    with open(log_file, file_mode, newline="") as log_fh:
        log_writer = csv.writer(log_fh)
        if log_file.stat().st_size == 0 or file_mode == 'w':
            log_writer.writerow([
                "episode", "total_return", "sharpe", "sortino",
                "max_drawdown", "avg_position", "cost_ratio", "length",
            ])

        for step in tqdm(range(timesteps), desc=f"{agent_name} Training", leave=True):
            # --- Action & environment step ---
            if agent_name == "TD3":
                action = agent.act(state, eval_mode=False)
            else:
                action = agent.act(state)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            step_return = info.get("step_return", 0.0)
            position_frac = float(abs(info.get("position_frac", 0.0)))
            equity = info.get("equity", env.initial_balance)
            cost = info.get("cost", 0.0)

            episode_returns.append(step_return)
            episode_equity.append(equity)
            episode_positions.append(position_frac)
            episode_costs.append(cost)

            if use_diversity:
                penalty = manager.compute_penalty(step_return, agent_name)
            else:
                penalty = 0.0
            adjusted_reward = reward - penalty

            agent.store(state, action, adjusted_reward, next_state, float(terminated))
            agent.train_step()

            all_actions.append(float(action) if not continuous else float(action[0]))
            all_step_returns.append(step_return)
            all_positions.append(position_frac)

            if done:
                episode_count += 1
                manager.update_epoch(episode_count)

                if len(episode_equity) > 1:
                    start_eq = episode_equity[0]
                    final_eq = episode_equity[-1]
                    total_return = (final_eq - start_eq) / (abs(start_eq) + 1e-8)
                    sharpe = compute_sharpe(episode_returns)
                    sortino = compute_sortino(episode_returns)
                    max_dd = compute_max_drawdown(episode_equity)
                    avg_pos = float(np.mean(episode_positions))
                    total_cost = float(np.sum(episode_costs))
                    total_abs_ret = float(np.sum(np.abs(episode_returns))) * abs(start_eq)
                    cost_ratio = total_cost / (total_abs_ret + 1e-8)
                    ep_len = len(episode_returns)

                    print(
                        f"[{agent_name}] Ep {episode_count:4d} | "
                        f"Return: {total_return*100:+7.2f}% | "
                        f"Sharpe: {sharpe:+6.3f} | Sortino: {sortino:+6.3f} | "
                        f"MaxDD: {max_dd*100:5.2f}% | Len: {ep_len:4d}",
                        flush=True,
                    )

                    if hasattr(agent, "log_episode"):
                        agent.log_episode(episode_count, total_return, sharpe,
                                          sortino, max_dd, episode_equity)

                    log_writer.writerow([
                        episode_count, total_return, sharpe, sortino,
                        max_dd, avg_pos, cost_ratio, ep_len,
                    ])

                if episode_count % 15 == 0:
                    ckpt_path = checkpoint_dir / f"{agent_name}_expert_{episode_count}.pt"
                    agent.save(str(ckpt_path))
                    print(f"[{agent_name}] Checkpoint saved to {ckpt_path}")

                state, info = env.reset()
                manager.reset_buffer()
                episode_returns.clear()
                episode_equity.clear()
                episode_positions.clear()
                episode_costs.clear()
            else:
                state = next_state

    # Save final model
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    agent.save(save_path)
    print(f"\n{agent_name} saved to {save_path}")

    trajectory = {
        "actions": np.array(all_actions, dtype=np.float32),
        "returns": np.array(all_step_returns, dtype=np.float32),
        "positions": np.array(all_positions, dtype=np.float32),
    }
    manager.register_expert(agent_name, trajectory)
    print(f"{agent_name} trajectory saved to {storage_dir}/{agent_name}.pkl\n")


# -------------------------------------------------------------------------
# PPG training loop (resume-friendly, step-based)
# -------------------------------------------------------------------------

def train_ppg(
    cfg: dict,
    df: pd.DataFrame,
    feature_cols: list,
    save_path: str,
    asset_type: str,
    asset_name: str,
    use_diversity: bool = True,
    resume_from: str = None,
    initial_episode: int = 0,
) -> None:
    print(f"\n{'='*70}\nTRAINING PPG (Expert 6) on {asset_type.upper()} — {asset_name}\n{'='*70}\n")

    env_params = get_env_params(asset_type, expert_id=6)
    env_params["continuous_actions"] = False
    env_params["leverage"] = 1
    env = TradingEnv(df=df, feature_cols=feature_cols, expert_id=6, **env_params)

    state_dim = len(feature_cols) + env._n_state_features
    ppg_cfg = cfg["ppg"]

    base_csv = Path("logs/training/experts") / asset_name / "ppg"
    base_tb  = Path("logs/tensorboard/ppg") / asset_name
    if not use_diversity:
        base_csv = base_csv / "no_diversity"
        base_tb  = base_tb  / "no_diversity"
    csv_log_dir = base_csv
    csv_log_dir.mkdir(parents=True, exist_ok=True)
    log_file = csv_log_dir / "episode_log.csv"
    tb_log_dir = str(base_tb)

    timesteps: int = ppg_cfg.get("timesteps", cfg["training"]["timesteps"])
    rollout_steps: int = ppg_cfg["rollout_steps"]

    agent = PPGAgent(
        state_dim=state_dim,
        action_dim=3,
        lr=ppg_cfg["lr"],
        gamma=ppg_cfg["gamma"],
        gae_lambda=ppg_cfg["gae_lambda"],
        clip_eps=ppg_cfg["clip_eps"],
        entropy_coef=ppg_cfg["entropy_coef"],
        entropy_coef_final=ppg_cfg.get("entropy_coef_final", 0.001),
        value_coef=ppg_cfg["value_coef"],
        max_grad_norm=ppg_cfg["max_grad_norm"],
        policy_epochs=ppg_cfg["policy_epochs"],
        batch_size=ppg_cfg["batch_size"],
        rollout_steps=rollout_steps,
        auxiliary_interval=ppg_cfg.get("auxiliary_interval", 8),
        aux_epochs=ppg_cfg["aux_epochs"],
        aux_batch_size=ppg_cfg["aux_batch_size"],
        kl_coef=ppg_cfg["kl_coef"],
        hidden_dim=ppg_cfg["hidden_dim"],
        num_layers=ppg_cfg["num_layers"],
        activation=ppg_cfg["activation"],
        lr_anneal_steps=ppg_cfg.get("lr_anneal_steps", timesteps),
        entropy_anneal_steps=ppg_cfg.get("entropy_anneal_steps", timesteps),
        clip_value_loss=ppg_cfg.get("clip_value_loss", True),
        target_kl=ppg_cfg.get("target_kl", 0.02),
        log_dir=tb_log_dir,
    )

    storage_dir = f"storage/{asset_name}"
    if not use_diversity:
        storage_dir = f"{storage_dir}/no_diversity"
    else:
        storage_dir = f"{storage_dir}/diversity"
    Path(storage_dir).mkdir(parents=True, exist_ok=True)
    scheduler = DiversityScheduler() if use_diversity else None
    manager = DiversityManager(
        storage_dir=storage_dir,
        enabled=use_diversity,
        scheduler=scheduler,
        penalty_scale=0.02,
        similarity_threshold=0.5,
        window_size=30,
    )

    # -------- Resume: load model and existing trajectory --------
    if resume_from and Path(resume_from).exists():
        agent.load(resume_from)
        print(f"[PPG] Loaded checkpoint from {resume_from}")

    all_actions = []
    all_step_returns = []
    all_positions = []
    if resume_from:
        old_traj = load_trajectory(storage_dir, "PPG")
        if old_traj is not None:
            all_actions = old_traj['actions'].tolist() if isinstance(old_traj['actions'], np.ndarray) else list(old_traj['actions'])
            all_step_returns = old_traj['returns'].tolist() if isinstance(old_traj['returns'], np.ndarray) else list(old_traj['returns'])
            all_positions = old_traj['positions'].tolist() if isinstance(old_traj['positions'], np.ndarray) else list(old_traj['positions'])
            print(f"[PPG] Loaded existing trajectory with {len(all_actions)} steps")

    episode_count: int = initial_episode
    episode_equity: list = []
    episode_step_returns: list = []
    episode_positions: list = []
    episode_costs: list = []

    state, info = env.reset()
    manager.reset_buffer()
    steps_done: int = 0

    checkpoint_dir = Path(save_path).parent

    file_mode = 'a' if resume_from and log_file.exists() else 'w'
    with open(log_file, file_mode, newline="") as log_fh:
        log_writer = csv.writer(log_fh)
        if log_file.stat().st_size == 0 or file_mode == 'w':
            log_writer.writerow([
                "episode", "total_return", "sharpe", "sortino", "max_drawdown",
                "avg_position", "length", "policy_loss", "value_loss", "entropy",
                "approx_kl", "clip_frac", "aux_loss",
            ])

        pbar = tqdm(total=timesteps, desc="PPG Training", unit="step", leave=True)

        while steps_done < timesteps:
            agent.reset_buffer()
            agent.reset_episode_accumulators()

            for _ in range(rollout_steps):
                if steps_done >= timesteps:
                    break

                action, log_prob, value = agent.act(state)

                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                step_return = float(info.get("step_return", 0.0))
                position_frac = float(abs(info.get("position_frac", 0.0)))
                equity = float(info.get("equity", env.initial_balance))
                cost = float(info.get("cost", 0.0))

                if use_diversity:
                    penalty = manager.compute_penalty(step_return, "PPG")
                else:
                    penalty = 0.0
                adjusted_reward = float(reward) - float(penalty)

                agent.store(
                    state=state,
                    action=action,
                    reward=adjusted_reward,
                    done=done,
                    log_prob=log_prob,
                    value=value,
                    next_state=next_state,
                )
                agent.store_step_info(info)

                episode_step_returns.append(step_return)
                episode_equity.append(equity)
                episode_positions.append(position_frac)
                episode_costs.append(cost)

                all_actions.append(float(action))
                all_step_returns.append(step_return)
                all_positions.append(position_frac)

                steps_done += 1
                pbar.update(1)

                if done:
                    episode_count += 1
                    manager.update_epoch(episode_count)

                    if len(episode_equity) > 1:
                        start_eq = episode_equity[0]
                        final_eq = episode_equity[-1]
                        total_return = (final_eq - start_eq) / (abs(start_eq) + 1e-8)
                        sharpe = compute_sharpe(episode_step_returns)
                        sortino = compute_sortino(episode_step_returns)
                        max_dd = compute_max_drawdown(episode_equity)
                        avg_pos = float(np.mean(episode_positions))
                        ep_len = len(episode_step_returns)

                        print(
                            f"[PPG] Ep {episode_count:4d} | "
                            f"Return: {total_return * 100:+7.2f}% | "
                            f"Sharpe: {sharpe:+6.3f} | "
                            f"Sortino: {sortino:+6.3f} | "
                            f"MaxDD: {max_dd * 100:5.2f}% | "
                            f"Steps: {steps_done:>8d} | "
                            f"Len: {ep_len:4d}",
                            flush=True,
                        )

                        agent.log_episode(episode_count, total_return, sharpe,
                                          sortino, max_dd, episode_equity)

                        _last_ep_stats = (
                            episode_count, total_return, sharpe, sortino,
                            max_dd, avg_pos, ep_len,
                        )
                    else:
                        _last_ep_stats = None

                    if episode_count % 15 == 0:
                        ckpt_path = checkpoint_dir / f"PPG_expert_{episode_count}.pt"
                        agent.save(str(ckpt_path))
                        print(f"[PPG] Checkpoint saved to {ckpt_path}")

                    episode_step_returns.clear()
                    episode_equity.clear()
                    episode_positions.clear()
                    episode_costs.clear()

                    state, _ = env.reset()
                    manager.reset_buffer()
                    agent.reset_episode_accumulators()
                else:
                    state = next_state

            agent.update()

            if episode_count > 0 and _last_ep_stats is not None:
                log_writer.writerow([
                    _last_ep_stats[0],
                    _last_ep_stats[1],
                    _last_ep_stats[2],
                    _last_ep_stats[3],
                    _last_ep_stats[4],
                    _last_ep_stats[5],
                    _last_ep_stats[6],
                    agent.last_policy_loss,
                    agent.last_value_loss,
                    agent.last_entropy,
                    agent.last_approx_kl,
                    agent.last_clip_frac,
                    agent.last_aux_loss,
                ])

        pbar.close()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    agent.save(save_path)
    print(f"\nPPG saved to {save_path}")

    trajectory = {
        "actions": np.array(all_actions, dtype=np.float32),
        "returns": np.array(all_step_returns, dtype=np.float32),
        "positions": np.array(all_positions, dtype=np.float32),
    }
    manager.register_expert("PPG", trajectory)
    print(f"PPG trajectory saved to {storage_dir}/PPG.pkl\n")


# -------------------------------------------------------------------------
# Main entry point
# -------------------------------------------------------------------------

EXPERT_KEY_MAP = {1: "ppo", 2: "sac", 3: "ddqn", 4: "tqc", 5: "td3", 6: "ppg"}
EXPERT_NAME_MAP = {1: "PPO", 2: "SAC", 3: "DDQN", 4: "TQC", 5: "TD3", 6: "PPG"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert", type=int, required=True, choices=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--config", type=str, default="configs/experts.yaml")
    parser.add_argument("--data", type=str, default="data/processed/BTCUSDT/train.csv")
    parser.add_argument("--asset", type=str, default=None)
    parser.add_argument("--no-diversity", action="store_true",
                        help="Train without diversity penalty")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Override base checkpoint directory")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint .pt file to resume training from")
    # Note: no --extra_episodes – we always train for the configured timesteps
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = pd.read_csv(args.data)

    expert_key = EXPERT_KEY_MAP[args.expert]
    expert_name = EXPERT_NAME_MAP[args.expert]

    try:
        feature_cols = cfg[expert_key]["features"]
    except KeyError:
        raise KeyError(f"Missing '{expert_key}.features' in config.")

    print(f"Expert {args.expert} ({expert_name}) | Features: {len(feature_cols)} | {feature_cols}")

    asset = args.asset or infer_asset_from_data_path(args.data)
    asset_type = detect_asset_type(asset)
    print(f"Asset: {asset} ({asset_type})")

    base_save_dir = args.save_dir if args.save_dir else cfg["training"]["save_dir"]
    out_dir = Path(base_save_dir) / asset
    if args.no_diversity:
        out_dir = out_dir / "no_diversity"
    else:
        out_dir = out_dir / "diversity"

    out_dir = out_dir / expert_name
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = str(out_dir / f"{expert_key}_expert.pt")

    storage_dir = f"storage/{asset}"
    if args.no_diversity:
        storage_dir = f"{storage_dir}/no_diversity"
    else:
        storage_dir = f"{storage_dir}/diversity"
    Path(storage_dir).mkdir(parents=True, exist_ok=True)

    timesteps = cfg[expert_key].get("timesteps", cfg["training"]["timesteps"])
    state_dim = len(feature_cols) + 6

    use_diversity = not args.no_diversity

    initial_episode = 0
    if args.resume:
        initial_episode = parse_episode_from_checkpoint(args.resume)
        if initial_episode == 0:
            print("Warning: Could not parse episode number from checkpoint filename. Starting from episode 0.")
        else:
            print(f"Resuming from episode {initial_episode}")

    # --- PPO (expert 1) ---
    if args.expert == 1:
        train_ppo(cfg, df, feature_cols, save_path, asset_type, asset,
                  use_diversity=use_diversity,
                  resume_from=args.resume,
                  initial_episode=initial_episode)
        return

    # --- PPG (expert 6) ---
    if args.expert == 6:
        train_ppg(cfg, df, feature_cols, save_path, asset_type, asset,
                  use_diversity=use_diversity,
                  resume_from=args.resume,
                  initial_episode=initial_episode)
        return

    # --- Off-policy experts (2-5) ---
    continuous = args.expert in (2, 4, 5)
    env_params = get_env_params(asset_type, args.expert)
    env_params["continuous_actions"] = continuous
    env = TradingEnv(df=df, feature_cols=feature_cols, expert_id=args.expert, **env_params)

    log_dir = Path("logs/training/experts") / asset / expert_key.lower()
    tb_dir  = Path("logs/tensorboard") / expert_key.lower() / asset
    if args.no_diversity:
        log_dir = log_dir / "no_diversity"
        tb_dir  = tb_dir  / "no_diversity"
    tb_log_dir = str(tb_dir)

    # Instantiate agents (expert 2-5)
    if args.expert == 2:
        ecfg = cfg["sac"]
        agent = SACAgent(
            state_dim=state_dim, action_dim=1,
            lr=ecfg["learning_rate"], gamma=ecfg["gamma"], tau=ecfg["tau"],
            alpha=ecfg["alpha"], buffer_size=ecfg["buffer_size"],
            batch_size=ecfg["batch_size"], warmup_steps=ecfg["warmup_steps"],
            max_grad_norm=ecfg["max_grad_norm"], hidden_dim=ecfg["hidden_dim"],
            num_layers=ecfg["num_layers"], activation=ecfg["activation"],
            log_std_min=ecfg["log_std_min"], log_std_max=ecfg["log_std_max"],
            log_dir=tb_log_dir,
        )
    elif args.expert == 3:
        ecfg = cfg["ddqn"]
        agent = DDQNAgent(
            state_dim=state_dim, action_dim=3,
            lr=ecfg["learning_rate"], gamma=ecfg["gamma"],
            batch_size=ecfg["batch_size"], buffer_size=ecfg["buffer_size"],
            target_update_freq=ecfg["target_update_freq"],
            epsilon_start=ecfg["epsilon_start"], epsilon_end=ecfg["epsilon_end"],
            epsilon_decay=ecfg["epsilon_decay"], dueling=ecfg["dueling"],
            warmup_steps=ecfg["warmup_steps"], max_grad_norm=ecfg["max_grad_norm"],
            hidden_dim=ecfg.get("hidden_dim", 256),
            num_layers=ecfg.get("num_layers", 2),
            activation=ecfg.get("activation", "relu"),
            log_dir=tb_log_dir,
        )
    elif args.expert == 4:
        ecfg = cfg["tqc"]
        agent = TQCAgent(
            state_dim=state_dim, action_dim=1,
            lr=ecfg["learning_rate"],
            lr_actor=ecfg.get("lr_actor"), lr_critic=ecfg.get("lr_critic"),
            gamma=ecfg["gamma"], tau=ecfg["tau"],
            buffer_size=ecfg["buffer_size"], batch_size=ecfg["batch_size"],
            warmup_steps=ecfg["warmup_steps"],
            hidden_dim=ecfg["hidden_dim"],
            actor_hidden_dim=ecfg.get("actor_hidden_dim"),
            critic_hidden_dim=ecfg.get("critic_hidden_dim"),
            actor_num_layers=ecfg.get("actor_num_layers", 2),
            critic_num_layers=ecfg.get("critic_num_layers", 2),
            activation=ecfg.get("activation", "relu"),
            log_std_min=ecfg.get("log_std_min", -20),
            log_std_max=ecfg.get("log_std_max", 2),
            n_critics=ecfg["n_critics"], n_quantiles=ecfg["n_quantiles"],
            top_quantiles_to_drop=ecfg["top_quantiles_to_drop"],
            auto_entropy_tuning=ecfg["auto_entropy_tuning"],
            alpha=ecfg["alpha"],
            grad_norm_clip=ecfg.get("grad_norm_clip", 1.0),
            log_dir=tb_log_dir,
        )
    elif args.expert == 5:
        ecfg = cfg["td3"]
        agent = TD3Agent(
            state_dim=state_dim, action_dim=1,
            lr_actor=ecfg["lr_actor"], lr_critic=ecfg["lr_critic"],
            gamma=ecfg["gamma"], tau=ecfg["tau"],
            buffer_size=ecfg["buffer_size"], batch_size=ecfg["batch_size"],
            warmup_steps=ecfg["warmup_steps"],
            exploration_noise=ecfg["exploration_noise"],
            policy_noise=ecfg["policy_noise"], noise_clip=ecfg["noise_clip"],
            policy_delay=ecfg["policy_delay"],
            hidden_dim=ecfg["hidden_dim"],
            actor_hidden_dim=ecfg.get("actor_hidden_dim"),
            critic_hidden_dim=ecfg.get("critic_hidden_dim"),
            actor_num_layers=ecfg.get("actor_num_layers", 2),
            critic_num_layers=ecfg.get("critic_num_layers", 2),
            activation=ecfg.get("activation", "relu"),
            grad_norm_clip=ecfg.get("grad_norm_clip", 1.0),
            log_dir=tb_log_dir,
        )

    train_off_policy(
        agent_name=expert_name,
        agent=agent,
        env=env,
        timesteps=timesteps,
        save_path=save_path,
        storage_dir=storage_dir,
        log_dir=log_dir,
        continuous=continuous,
        use_diversity=use_diversity,
        resume_from=args.resume,
        initial_episode=initial_episode,
    )


if __name__ == "__main__":
    main()