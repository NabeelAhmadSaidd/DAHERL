"""
Train the allocator IQN policy that selects among the six experts.

This script is intended to run end-to-end from a clean checkout:
- loads the expert checkpoints
- creates a wrapped allocator environment
- trains a discrete IQN agent over expert indices
- saves checkpoints and can resume automatically

Usage:
    python -m allocator.train_allocator --asset EURUSD
    python -m allocator.train_allocator --asset BTCUSDT --no-diversity
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from allocator.allocator_agent import IQNAllocator
from allocator.allocator_env import AllocatorEnv
from allocator.diversity_manager import AllocatorDiversityManager   # <-- Correct import
from environment.trading_env import TradingEnv
from regime.config import CONFIG as REGIME_CONFIG
from regime.inference import MultiHorizonRegimeClassifier
from experts.ppo_expert import PPOAgent
from experts.sac_expert import SACAgent
from experts.ddqn_expert import DDQNAgent
from experts.tqc_expert import TQCAgent
from experts.td3_expert import TD3Agent
from experts.ppg_expert import PPGAgent
from experts.train_experts import EXPERT_NAME_MAP, detect_asset_type, get_env_params


def load_config(config_path: str | Path = "configs/experts.yaml") -> Dict[str, Any]:
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ----------------------------------------------------------------------
# Expert loading helpers (unchanged)
# ----------------------------------------------------------------------

def _safe_torch_load(path: Path, device: str) -> Any:
    try:
        return torch.load(path, map_location=device)
    except TypeError:
        return torch.load(path, map_location=device, weights_only=False)


def _extract_state_dict(checkpoint: Any, preferred_keys: Iterable[str]) -> Optional[Dict[str, Any]]:
    if not isinstance(checkpoint, dict):
        return None
    for key in preferred_keys:
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    if checkpoint and all(isinstance(k, str) for k in checkpoint.keys()):
        if any(k.endswith(".weight") or k.endswith(".bias") for k in checkpoint.keys()):
            return checkpoint
    return None


def _load_module_weights(module: torch.nn.Module, checkpoint: Any, preferred_keys: Iterable[str]) -> bool:
    state_dict = _extract_state_dict(checkpoint, preferred_keys)
    if state_dict is None:
        return False
    module.load_state_dict(state_dict)
    return True


def _load_sequence_weights(
    modules: Iterable[torch.nn.Module],
    checkpoint_items: Any,
) -> bool:
    if not isinstance(checkpoint_items, (list, tuple)):
        return False
    modules = list(modules)
    if len(modules) != len(checkpoint_items):
        return False
    for module, state_dict in zip(modules, checkpoint_items):
        if not isinstance(state_dict, dict):
            return False
        module.load_state_dict(state_dict)
    return True


def _expert_kwargs(expert_name: str, cfg: Dict[str, Any], state_dim: int, device: str) -> Dict[str, Any]:
    """Returns the kwargs for instantiating a specific expert agent."""
    name = expert_name.lower()

    if name == "ppo":
        ecfg = cfg["ppo"]
        return {
            "state_dim": state_dim,
            "action_dim": 3,
            "lr": ecfg.get("learning_rate", 3e-4),
            "gamma": ecfg.get("gamma", 0.99),
            "gae_lambda": ecfg.get("gae_lambda", 0.95),
            "clip_eps": ecfg.get("clip_eps", 0.2),
            "entropy_coef": ecfg.get("entropy_coef", 0.01),
            "entropy_coef_final": ecfg.get("entropy_coef_final", 0.001),
            "value_coef": ecfg.get("value_coef", 0.5),
            "max_grad_norm": ecfg.get("max_grad_norm", 0.5),
            "update_epochs": ecfg.get("update_epochs", 4),
            "batch_size": ecfg.get("batch_size", 64),
            "rollout_steps": ecfg.get("rollout_steps", 2048),
            "hidden_dim": ecfg.get("hidden_dim", 256),
            "num_layers": ecfg.get("num_layers", 2),
            "activation": ecfg.get("activation", "tanh"),
            "lr_anneal_steps": ecfg.get("lr_anneal_steps", 0),
            "entropy_anneal_steps": ecfg.get("entropy_anneal_steps", 0),
            "clip_value_loss": ecfg.get("clip_value_loss", True),
            "target_kl": ecfg.get("target_kl", 0.02),
            "device": device,
        }

    if name == "sac":
        ecfg = cfg["sac"]
        return {
            "state_dim": state_dim,
            "action_dim": 1,
            "lr": ecfg.get("learning_rate", 3e-4),
            "gamma": ecfg.get("gamma", 0.99),
            "tau": ecfg.get("tau", 0.005),
            "alpha": ecfg.get("alpha", 0.2),
            "buffer_size": ecfg.get("buffer_size", 100000),
            "batch_size": ecfg.get("batch_size", 256),
            "warmup_steps": ecfg.get("warmup_steps", 1000),
            "max_grad_norm": ecfg.get("max_grad_norm", 1.0),
            "hidden_dim": ecfg.get("hidden_dim", 256),
            "num_layers": ecfg.get("num_layers", 2),
            "activation": ecfg.get("activation", "relu"),
            "log_std_min": ecfg.get("log_std_min", -20),
            "log_std_max": ecfg.get("log_std_max", 2),
            "device": device,
        }

    if name == "ddqn":
        ecfg = cfg["ddqn"]
        return {
            "state_dim": state_dim,
            "action_dim": 3,
            "lr": ecfg.get("learning_rate", 1e-4),
            "gamma": ecfg.get("gamma", 0.99),
            "batch_size": ecfg.get("batch_size", 64),
            "buffer_size": ecfg.get("buffer_size", 100000),
            "target_update_freq": ecfg.get("target_update_freq", 1000),
            "epsilon_start": ecfg.get("epsilon_start", 1.0),
            "epsilon_end": ecfg.get("epsilon_end", 0.05),
            "epsilon_decay": ecfg.get("epsilon_decay", 50000),
            "dueling": ecfg.get("dueling", True),
            "warmup_steps": ecfg.get("warmup_steps", 1000),
            "max_grad_norm": ecfg.get("max_grad_norm", 10.0),
            "hidden_dim": ecfg.get("hidden_dim", 256),
            "num_layers": ecfg.get("num_layers", 2),
            "activation": ecfg.get("activation", "relu"),
            "device": device,
        }

    if name == "tqc":
        ecfg = cfg["tqc"]
        return {
            "state_dim": state_dim,
            "action_dim": 1,
            "lr": ecfg.get("learning_rate", 3e-4),
            "lr_actor": ecfg.get("lr_actor", ecfg.get("learning_rate", 3e-4)),
            "lr_critic": ecfg.get("lr_critic", ecfg.get("learning_rate", 3e-4)),
            "gamma": ecfg.get("gamma", 0.99),
            "tau": ecfg.get("tau", 0.005),
            "buffer_size": ecfg.get("buffer_size", 100000),
            "batch_size": ecfg.get("batch_size", 256),
            "warmup_steps": ecfg.get("warmup_steps", 1000),
            "hidden_dim": ecfg.get("hidden_dim", 256),
            "actor_hidden_dim": ecfg.get("actor_hidden_dim", ecfg.get("hidden_dim", 256)),
            "critic_hidden_dim": ecfg.get("critic_hidden_dim", ecfg.get("hidden_dim", 256)),
            "actor_num_layers": ecfg.get("actor_num_layers", 2),
            "critic_num_layers": ecfg.get("critic_num_layers", 2),
            "activation": ecfg.get("activation", "relu"),
            "log_std_min": ecfg.get("log_std_min", -20),
            "log_std_max": ecfg.get("log_std_max", 2),
            "n_critics": ecfg.get("n_critics", 2),
            "n_quantiles": ecfg.get("n_quantiles", 25),
            "top_quantiles_to_drop": ecfg.get("top_quantiles_to_drop", 2),
            "auto_entropy_tuning": ecfg.get("auto_entropy_tuning", True),
            "alpha": ecfg.get("alpha", 0.2),
            "grad_norm_clip": ecfg.get("grad_norm_clip", 1.0),
            "device": device,
        }

    if name == "td3":
        ecfg = cfg["td3"]
        return {
            "state_dim": state_dim,
            "action_dim": 1,
            "lr_actor": ecfg.get("lr_actor", 3e-4),
            "lr_critic": ecfg.get("lr_critic", 3e-4),
            "gamma": ecfg.get("gamma", 0.99),
            "tau": ecfg.get("tau", 0.005),
            "buffer_size": ecfg.get("buffer_size", 100000),
            "batch_size": ecfg.get("batch_size", 256),
            "warmup_steps": ecfg.get("warmup_steps", 1000),
            "exploration_noise": ecfg.get("exploration_noise", 0.1),
            "policy_noise": ecfg.get("policy_noise", 0.2),
            "noise_clip": ecfg.get("noise_clip", 0.5),
            "policy_delay": ecfg.get("policy_delay", 2),
            "hidden_dim": ecfg.get("hidden_dim", 256),
            "actor_hidden_dim": ecfg.get("actor_hidden_dim", ecfg.get("hidden_dim", 256)),
            "critic_hidden_dim": ecfg.get("critic_hidden_dim", ecfg.get("hidden_dim", 256)),
            "actor_num_layers": ecfg.get("actor_num_layers", 2),
            "critic_num_layers": ecfg.get("critic_num_layers", 2),
            "activation": ecfg.get("activation", "relu"),
            "grad_norm_clip": ecfg.get("grad_norm_clip", 1.0),
            "device": device,
        }

    if name == "ppg":
        ecfg = cfg["ppg"]
        return {
            "state_dim": state_dim,
            "action_dim": 3,
            "lr": ecfg.get("learning_rate", 3e-4),
            "gamma": ecfg.get("gamma", 0.99),
            "gae_lambda": ecfg.get("gae_lambda", 0.95),
            "clip_eps": ecfg.get("clip_eps", 0.2),
            "entropy_coef": ecfg.get("entropy_coef", 0.01),
            "entropy_coef_final": ecfg.get("entropy_coef_final", 0.001),
            "value_coef": ecfg.get("value_coef", 0.5),
            "max_grad_norm": ecfg.get("max_grad_norm", 0.5),
            "policy_epochs": ecfg.get("policy_epochs", 10),
            "batch_size": ecfg.get("batch_size", 64),
            "rollout_steps": ecfg.get("rollout_steps", 2048),
            "aux_epochs": ecfg.get("aux_epochs", 6),
            "aux_batch_size": ecfg.get("aux_batch_size", 256),
            "kl_coef": ecfg.get("kl_coef", 0.1),
            "hidden_dim": ecfg.get("hidden_dim", 256),
            "num_layers": ecfg.get("num_layers", 2),
            "activation": ecfg.get("activation", "tanh"),
            "lr_anneal_steps": ecfg.get("lr_anneal_steps", 0),
            "entropy_anneal_steps": ecfg.get("entropy_anneal_steps", 0),
            "clip_value_loss": ecfg.get("clip_value_loss", True),
            "target_kl": ecfg.get("target_kl", 0.02),
            "device": device,
        }

    raise ValueError(f"Unsupported expert '{expert_name}'.")


def load_expert_agent(
    name: str,
    checkpoint_path: str | Path,
    env_dummy: TradingEnv,
    device: str = "cpu",
    expert_cfg: Optional[Dict[str, Any]] = None,
):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Expert checkpoint missing: {checkpoint_path}")

    state_dim = int(env_dummy.observation_space.shape[0])
    expert_cfg = expert_cfg or load_config(PROJECT_ROOT / "configs" / "experts.yaml")
    kwargs = _expert_kwargs(name, expert_cfg, state_dim, device)

    if name == "PPO":
        agent = PPOAgent(**kwargs)
    elif name == "SAC":
        agent = SACAgent(**kwargs)
    elif name == "DDQN":
        agent = DDQNAgent(**kwargs)
    elif name == "TQC":
        agent = TQCAgent(**kwargs)
    elif name == "TD3":
        agent = TD3Agent(**kwargs)
    elif name == "PPG":
        agent = PPGAgent(**kwargs)
    else:
        raise ValueError(f"Unknown expert: {name}")

    loaded = False
    try:
        agent.load(str(checkpoint_path))
        loaded = True
    except Exception:
        checkpoint = _safe_torch_load(checkpoint_path, device)

        if name == "PPO":
            loaded = _load_module_weights(agent.network, checkpoint, ("network", "network_state_dict"))
            if not loaded and isinstance(checkpoint, dict):
                loaded = _load_module_weights(agent.network, checkpoint, tuple(checkpoint.keys()))
        elif name == "SAC":
            if isinstance(checkpoint, dict):
                loaded = (
                    _load_module_weights(agent.actor, checkpoint, ("actor",))
                    and _load_module_weights(agent.critic, checkpoint, ("critic",))
                    and _load_module_weights(agent.critic_target, checkpoint, ("critic_target",))
                )
        elif name == "DDQN":
            if isinstance(checkpoint, dict):
                loaded = (
                    _load_module_weights(agent.q_network, checkpoint, ("q_network",))
                    and _load_module_weights(agent.target_network, checkpoint, ("target_network",))
                )
        elif name == "TQC":
            if isinstance(checkpoint, dict):
                loaded = _load_module_weights(agent.actor, checkpoint, ("actor",))
                loaded = loaded and _load_sequence_weights(agent.critics, checkpoint.get("critics"))
                loaded = loaded and _load_sequence_weights(agent.critics_target, checkpoint.get("critics_target"))
        elif name == "TD3":
            if isinstance(checkpoint, dict):
                loaded = (
                    _load_module_weights(agent.actor, checkpoint, ("actor",))
                    and _load_module_weights(agent.critic, checkpoint, ("critic",))
                    and _load_module_weights(agent.actor_target, checkpoint, ("actor_target",))
                    and _load_module_weights(agent.critic_target, checkpoint, ("critic_target",))
                )
        elif name == "PPG":
            if isinstance(checkpoint, dict):
                loaded = _load_module_weights(agent.network, checkpoint, ("network", "network_state_dict"))

        if isinstance(checkpoint, dict):
            for optimizer_name in ("optimizer", "actor_optimizer", "critic_optimizer", "alpha_optimizer"):
                optimizer_state = checkpoint.get(optimizer_name)
                if optimizer_state is not None and hasattr(agent, optimizer_name):
                    try:
                        getattr(agent, optimizer_name).load_state_dict(optimizer_state)
                    except Exception:
                        pass
            if hasattr(agent, "_total_steps") and "total_steps" in checkpoint:
                try:
                    agent._total_steps = int(checkpoint["total_steps"])
                except Exception:
                    pass
            if hasattr(agent, "_update_count") and "update_count" in checkpoint:
                try:
                    agent._update_count = int(checkpoint["update_count"])
                except Exception:
                    pass
            if hasattr(agent, "entropy_coef") and "entropy_coef" in checkpoint:
                try:
                    agent.entropy_coef = float(checkpoint["entropy_coef"])
                except Exception:
                    pass

    if not loaded:
        raise RuntimeError(f"Failed to load expert checkpoint: {checkpoint_path}")

    return agent


# ----------------------------------------------------------------------
# Metric helpers (unchanged)
# ----------------------------------------------------------------------

def compute_sharpe(returns: list[float], eps: float = 1e-8) -> float:
    if len(returns) < 2:
        return 0.0
    arr = np.asarray(returns, dtype=np.float64)
    std = float(arr.std(ddof=1))
    if std < eps or np.isnan(std):
        return 0.0
    return float(arr.mean() / (std + eps))


def compute_sortino(returns: list[float], eps: float = 1e-8) -> float:
    if len(returns) < 2:
        return 0.0
    arr = np.asarray(returns, dtype=np.float64)
    downside = arr[arr < 0]
    if downside.size < 2:
        return 0.0
    down_std = float(downside.std(ddof=1))
    if down_std < eps or np.isnan(down_std):
        return 0.0
    return float(arr.mean() / (down_std + eps))


def compute_max_drawdown(equity_curve: list[float]) -> float:
    if len(equity_curve) < 2:
        return 0.0
    arr = np.asarray([max(float(v), 1e-4) for v in equity_curve], dtype=np.float64)
    peak = np.maximum.accumulate(arr)
    drawdown = (peak - arr) / (peak + 1e-8)
    return float(np.max(drawdown))


# ----------------------------------------------------------------------
# Environment builder & regime helpers (unchanged)
# ----------------------------------------------------------------------

def build_base_env(df: pd.DataFrame, feature_cols: list[str], asset_type: str) -> TradingEnv:
    env_params = get_env_params(asset_type, expert_id=1)
    env_params["continuous_actions"] = True
    return TradingEnv(df=df, feature_cols=feature_cols, expert_id=1, **env_params)


def _load_regime_model(
    asset: str,
    checkpoint_dir: Path,
    regime_model_path: Optional[str] = None,
) -> Optional[MultiHorizonRegimeClassifier]:
    if regime_model_path is None:
        return None
    path = Path(regime_model_path)
    if path.exists():
        return MultiHorizonRegimeClassifier(path)
    return None


def _regime_reward_bonus(info: Dict[str, Any], expert_signal: float) -> float:
    probs = np.asarray(info.get("allocator_regime_probabilities", np.zeros(3, dtype=np.float32)), dtype=np.float32)
    if probs.shape != (3,):
        return 0.0
    confidence = float(np.clip(info.get("allocator_regime_confidence", 0.0), 0.0, 1.0))
    if confidence <= 0.0:
        return 0.0
    regime_class = int(info.get("allocator_regime_class", 0))
    sideways = float(probs[0])
    trending_up = float(probs[1])
    trending_down = float(probs[2])
    signal = float(np.clip(expert_signal, -1.0, 1.0))
    directional_edge = trending_up - trending_down

    if regime_class == 0:
        class_bonus = sideways * (1.0 - abs(signal))
    elif regime_class == 1:
        class_bonus = trending_up * max(signal, 0.0) - trending_down * max(-signal, 0.0)
    else:
        class_bonus = trending_down * max(-signal, 0.0) - trending_up * max(signal, 0.0)

    activity_penalty = sideways * abs(signal)
    alignment_bonus = directional_edge * signal
    bonus = confidence * (0.6 * alignment_bonus + 0.4 * class_bonus - 0.5 * activity_penalty)
    return float(np.clip(bonus, -0.05, 0.05))


# ----------------------------------------------------------------------
# Main training function – now uses IQN and AllocatorDiversityManager
# ----------------------------------------------------------------------
def train_allocator(
    cfg: Dict[str, Any],
    experts_cfg: Dict[str, Any],
    df: pd.DataFrame,
    feature_cols: list[str],
    asset: str,
    asset_type: str,
    device: str,
    save_path: Path,
    regime_model_path: Optional[str] = None,
    resume: bool = True,
    expert_subfolder: str = "diversity",
    include_regime_features: bool = True,
    diversity_enabled: bool = True,
) -> None:
    print(f"\n{'=' * 70}\nTRAINING ALLOCATOR IQN ON {asset_type.upper()} - {asset}\n{'=' * 70}\n")
    print(f"Using expert subfolder: {expert_subfolder}")

    base_env = build_base_env(df, feature_cols, asset_type)

    expert_root = save_path.parent / expert_subfolder
    expert_names = [EXPERT_NAME_MAP[i] for i in range(1, 7)]
    experts = []
    for name in expert_names:
        ckpt_path = expert_root / name / f"{name.lower()}_expert.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing expert checkpoint: {ckpt_path}")
        expert = load_expert_agent(name, ckpt_path, base_env, device=device, expert_cfg=experts_cfg)
        experts.append(expert)
        print(f"Loaded {name} from {ckpt_path}")

    lookback = int(cfg["allocator"].get("lookback", 20))
    regime_model = _load_regime_model(asset, save_path.parent, regime_model_path=regime_model_path)
    print(f"Regime model: {regime_model.__class__.__name__ if regime_model else 'None'}")

    alloc_env = AllocatorEnv(
        base_env,
        experts,
        lookback=lookback,
        regime_model=regime_model,
        regime_feature_cols=list(REGIME_CONFIG["features"]),
        regime_window=int(REGIME_CONFIG["model"]["window"]),
        include_regime_features=include_regime_features,
    )
    alloc_state_dim = alloc_env.observation_space.shape[0]
    print(f"Allocator observation dim: {alloc_state_dim}")

    iqn_cfg = cfg["allocator"]
    allocator = IQNAllocator(
        state_dim=alloc_state_dim,
        action_dim=6,
        lr=iqn_cfg.get("lr", 3e-4),
        gamma=iqn_cfg.get("gamma", 0.995),
        buffer_size=iqn_cfg.get("buffer_size", 30000),
        batch_size=iqn_cfg.get("batch_size", 128),
        hidden_dim=iqn_cfg.get("hidden_dim", 128),
        n_quantiles=iqn_cfg.get("n_quantiles", 32),
        embedding_dim=iqn_cfg.get("embedding_dim", 64),
        num_layers=iqn_cfg.get("num_layers", 2),
        activation=iqn_cfg.get("activation", "relu"),
        target_update_freq=iqn_cfg.get("target_update_freq", 2000),
        epsilon_start=iqn_cfg.get("epsilon_start", 1.0),
        epsilon_end=iqn_cfg.get("epsilon_end", 0.05),
        epsilon_decay=iqn_cfg.get("epsilon_decay", 400000),
        risk_tau=iqn_cfg.get("risk_tau", 0.5),
        log_dir=str(PROJECT_ROOT / "logs" / "tensorboard" / "allocator" / asset / expert_subfolder),
        device=device,
    )

    if resume and save_path.exists():
        try:
            allocator.load(save_path)
            print(f"Resumed allocator from {save_path} (steps={allocator.total_steps})")
        except Exception as exc:
            print(f"Could not resume allocator from {save_path}: {exc}")

    # Instantiate the correct allocator diversity manager
    manager = AllocatorDiversityManager(
        n_experts=len(experts),
        window_size=100,          # or read from cfg
        threshold=0.45,
        penalty_scale=0.02,
        use_regime=False,         # can be set based on ablation
        enabled=diversity_enabled,
    )

    total_timesteps = int(cfg["allocator"].get("total_timesteps", 100000))
    save_interval = int(cfg["allocator"].get("save_freq", 100000))

    log_dir = PROJECT_ROOT / "logs" / "training" / "allocator" / asset / expert_subfolder
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "episode_log.csv"
    append_mode = resume and log_file.exists() and log_file.stat().st_size > 0

    steps_done = allocator.total_steps
    episode_count = 0
    if append_mode:
        try:
            with open(log_file, "r", encoding="utf-8", newline="") as fh:
                episode_count = max(0, sum(1 for _ in fh) - 1)
        except Exception:
            episode_count = 0

    obs, _ = alloc_env.reset()
    episode_equity = [float(base_env.initial_balance)]
    episode_step_returns: list[float] = []
    episode_positions: list[float] = []
    episode_reward_sum = 0.0
    episode_regime_bonus_sum = 0.0

    all_actions: list[float] = []
    all_step_returns: list[float] = []
    all_positions: list[float] = []

    with open(log_file, "a" if append_mode else "w", newline="", encoding="utf-8") as log_fh:
        writer = csv.writer(log_fh)
        if not append_mode:
            writer.writerow([
                "episode", "total_return", "sharpe", "sortino", "max_drawdown",
                "avg_position", "length", "episode_reward", "regime_bonus",
                "loss", "mean_q", "td_error", "epsilon"
            ])

        pbar = tqdm(total=total_timesteps, initial=steps_done, desc="IQN Allocator Training", unit="step")

        while steps_done < total_timesteps:
            action, _, _ = allocator.sample_action(obs)
            next_obs, reward, terminated, truncated, info = alloc_env.step(action)
            done = bool(terminated or truncated)

            step_return = float(info.get("step_return", 0.0))
            position_frac = float(abs(info.get("position_frac", 0.0)))
            equity = float(info.get("equity", base_env.initial_balance))

            # Compute penalty using the correct manager – only action is needed
            penalty = manager.compute_penalty(action)
            regime_bonus = _regime_reward_bonus(info, float(info.get("allocator_expert_signal", 0.0)))
            adjusted_reward = float(reward) - float(penalty) + float(regime_bonus)

            allocator.store_transition(
                state=obs,
                action=action,
                reward=adjusted_reward,
                done=done,
                next_state=next_obs,
                log_prob=None,
                value=None,
            )

            episode_reward_sum += adjusted_reward
            episode_regime_bonus_sum += float(regime_bonus)
            episode_step_returns.append(step_return)
            episode_equity.append(equity)
            episode_positions.append(position_frac)

            all_actions.append(float(action))
            all_step_returns.append(step_return)
            all_positions.append(position_frac)

            obs = next_obs
            steps_done += 1
            pbar.update(1)

            if allocator.buffer_full():
                metrics = allocator.update()
                if metrics:
                    pbar.set_postfix({
                        "loss": f"{metrics['loss']:.4f}",
                        "mean_q": f"{metrics['mean_q']:.3f}",
                        "td": f"{metrics['td_error']:.3f}",
                        "eps": f"{metrics['epsilon']:.3f}",
                    })

            if done:
                episode_count += 1
                manager.reset()   # reset selection history at episode end

                if len(episode_equity) > 1:
                    total_return = (episode_equity[-1] - episode_equity[0]) / (abs(episode_equity[0]) + 1e-8)
                    sharpe = compute_sharpe(episode_step_returns)
                    sortino = compute_sortino(episode_step_returns)
                    max_dd = compute_max_drawdown(episode_equity)
                    avg_position = float(np.mean(episode_positions)) if episode_positions else 0.0
                    ep_len = len(episode_step_returns)

                    allocator.log_episode(
                        episode_count,
                        total_return,
                        sharpe,
                        sortino,
                        max_dd,
                        episode_equity,
                    )

                    writer.writerow([
                        episode_count,
                        total_return,
                        sharpe,
                        sortino,
                        max_dd,
                        avg_position,
                        ep_len,
                        episode_reward_sum,
                        episode_regime_bonus_sum,
                        allocator.last_loss,
                        allocator.last_mean_q,
                        allocator.last_td_error,
                        allocator.last_epsilon,
                    ])
                    log_fh.flush()

                    print(
                        f"[IQN Allocator] Ep {episode_count:4d} | "
                        f"Return: {total_return * 100:+7.2f}% | "
                        f"Sharpe: {sharpe:+6.3f} | Sortino: {sortino:+6.3f} | "
                        f"MaxDD: {max_dd * 100:5.2f}% | Len: {ep_len:4d} | "
                        f"Loss: {allocator.last_loss:+.4f} | "
                        f"MeanQ: {allocator.last_mean_q:+.4f} | "
                        f"TD: {allocator.last_td_error:+.4f} | "
                        f"Eps: {allocator.last_epsilon:.3f}",
                        flush=True,
                    )

                episode_equity = [float(base_env.initial_balance)]
                episode_step_returns.clear()
                episode_positions.clear()
                episode_reward_sum = 0.0
                episode_regime_bonus_sum = 0.0

                obs, _ = alloc_env.reset()

            if steps_done > 0 and steps_done % save_interval == 0:
                allocator.save(save_path)
                print(f"Saved IQN allocator checkpoint to {save_path}")

        pbar.close()

    allocator.save(save_path)
    print(f"\nIQN Allocator training complete. Saved to {save_path}")

    # No trajectory registration needed for this manager; it's stateless across episodes.
    # But we can optionally save the selection history if needed.


# ----------------------------------------------------------------------
# Command-line entry
# ----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", type=str, required=True, help="Asset name such as BTCUSDT or EURUSD")
    parser.add_argument("--config", type=str, default="configs/allocator.yaml", help="Allocator config path")
    parser.add_argument("--experts-config", type=str, default="configs/experts.yaml", help="Experts config path")
    parser.add_argument("--data", type=str, default=None, help="Optional explicit training data path")
    parser.add_argument("--regime-model", type=str, default=None, help="Optional regime checkpoint path")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--no-resume", action="store_true", help="Disable automatic checkpoint resume")
    parser.add_argument("--no-diversity", action="store_true",
                        help="Disable allocator diversity penalty and use no_diversity expert subfolder")
    args = parser.parse_args()

    alloc_cfg = load_config(args.config)
    experts_cfg = load_config(args.experts_config)

    asset = args.asset.upper()
    asset_type = detect_asset_type(asset)
    data_path = Path(args.data) if args.data else PROJECT_ROOT / "data" / "processed" / asset / "train.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    expert_subfolder = "no_diversity" if args.no_diversity else "diversity"
    checkpoint_dir = PROJECT_ROOT / "checkpoints" / asset / expert_subfolder
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_path = checkpoint_dir / "allocator.pt"

    df = pd.read_csv(data_path)
    feature_cols = list(experts_cfg["ppo"]["features"])

    print(f"Asset: {asset} ({asset_type})")
    print(f"Data: {data_path} -> {df.shape}")
    print(f"Features: {feature_cols}")
    print(f"Expert subfolder: {expert_subfolder}")

    # For ablations, include_regime_features should be set externally (e.g., True for NoDiversity, False for NoRegime)
    # Here we default to True; the ablation runner will override via function call if needed.
    train_allocator(
        cfg=alloc_cfg,
        experts_cfg=experts_cfg,
        df=df,
        feature_cols=feature_cols,
        asset=asset,
        asset_type=asset_type,
        device=args.device,
        save_path=save_path,
        regime_model_path=args.regime_model,
        resume=not args.no_resume,
        expert_subfolder=expert_subfolder,
        include_regime_features=True,           # Can be overridden by caller
        diversity_enabled=not args.no_diversity,
    )


if __name__ == "__main__":
    main()