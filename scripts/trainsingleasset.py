#!/usr/bin/env python3
"""
Trains all six experts sequentially for a single asset.

Usage:
    python -m scripts.trainsingleasset --asset BTCUSDT
    python -m scripts.trainsingleasset --asset EURUSD --config configs/experts.yaml --data_dir data/processed

The script trains in order: PPO → SAC → DDQN → TQC → TD3 → PPG.
Each expert uses the diversity framework: PPO has no penalty, later experts
are penalised for being similar to all previously trained experts (per‑asset storage).
"""

import argparse
import pandas as pd
from pathlib import Path

from experts.train_experts import (
    train_ppo,
    train_ppg,
    train_off_policy,
    load_config,
    detect_asset_type,
    get_env_params,
)
from experts.sac_expert import SACAgent
from experts.ddqn_expert import DDQNAgent
from experts.tqc_expert import TQCAgent
from experts.td3_expert import TD3Agent
from environment.trading_env import TradingEnv


def main():
    parser = argparse.ArgumentParser(description="Train all experts for a single asset")
    parser.add_argument("--asset", type=str, required=True,
                        help="Asset name (e.g., BTCUSDT, EURUSD).")
    parser.add_argument("--config", type=str, default="configs/experts.yaml",
                        help="Path to YAML configuration file.")
    parser.add_argument("--data_dir", type=str, default="data/processed",
                        help="Base directory containing asset subfolders with train.csv.")
    parser.add_argument("--no-diversity", action="store_true",
                        help="Train without diversity penalty (default: with diversity).")
    args = parser.parse_args()

    data_path = Path(args.data_dir) / args.asset / "train.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found at {data_path}")

    cfg = load_config(args.config)
    df = pd.read_csv(data_path)
    asset_type = detect_asset_type(args.asset)

    print(f"\n{'='*80}")
    print(f"Starting full training pipeline for asset: {args.asset} (type: {asset_type})")
    print(f"  • Diversity: {'disabled' if args.no_diversity else 'enabled'}")
    print(f"{'='*80}\n")

    base_save_dir = Path(cfg["training"]["save_dir"])
    use_diversity = not args.no_diversity

    # Build storage directory (shared across experts)
    storage_dir = f"storage/{args.asset}"
    if args.no_diversity:
        storage_dir = f"{storage_dir}/no_diversity"
    else:
        storage_dir = f"{storage_dir}/diversity"
    Path(storage_dir).mkdir(parents=True, exist_ok=True)

    # Expert configuration: (config_key, training function or None for off-policy)
    expert_configs = {
        1: ("ppo", "PPO", train_ppo),
        2: ("sac", "SAC", None),    # off-policy: train_off_policy
        3: ("ddqn", "DDQN", None),
        4: ("tqc", "TQC", None),
        5: ("td3", "TD3", None),
        6: ("ppg", "PPG", train_ppg),
    }

    for expert_id, (key, name, train_func) in expert_configs.items():
        print(f"\n{'*'*80}")
        print(f"Starting training for Expert {expert_id} ({name})")
        print(f"{'*'*80}\n")

        feature_cols = cfg[key]["features"]
        print(f"Using {len(feature_cols)} features: {feature_cols}")

        # Build save path: checkpoints/asset/(diversity|no_diversity)/ExpertName/key_expert.pt
        subfolder = "diversity" if use_diversity else "no_diversity"
        save_dir = base_save_dir / args.asset / subfolder / name
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(save_dir / f"{key}_expert.pt")

        if train_func is not None:
            # PPO or PPG – call directly
            train_func(cfg, df, feature_cols, save_path, asset_type, args.asset, use_diversity=use_diversity)
        else:
            # Off‑policy expert: SAC, DDQN, TQC, TD3
            continuous = expert_id in (2, 4, 5)   # SAC, TQC, TD3 use continuous actions
            env_params = get_env_params(asset_type, expert_id)
            env_params["continuous_actions"] = continuous
            env = TradingEnv(df=df, feature_cols=feature_cols, expert_id=expert_id, **env_params)

            state_dim = len(feature_cols) + 6

            # Build log directory
            log_dir = Path("logs/training/experts") / args.asset / key.lower()
            tb_dir = Path("logs/tensorboard") / key.lower() / args.asset
            if args.no_diversity:
                log_dir = log_dir / "no_diversity"
                tb_dir = tb_dir / "no_diversity"
            log_dir.mkdir(parents=True, exist_ok=True)
            tb_log_dir = str(tb_dir)

            # Instantiate the appropriate agent
            if expert_id == 2:  # SAC
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
            elif expert_id == 3:  # DDQN
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
            elif expert_id == 4:  # TQC
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
            elif expert_id == 5:  # TD3
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

            timesteps = cfg[key].get("timesteps", cfg["training"]["timesteps"])
            train_off_policy(
                agent_name=name,
                agent=agent,
                env=env,
                timesteps=timesteps,
                save_path=save_path,
                storage_dir=storage_dir,
                log_dir=log_dir,
                continuous=continuous,
                use_diversity=use_diversity,
            )

        print(f"\nFinished training for Expert {expert_id} ({name})")
        print(f"Model saved to: {save_path}\n")

    print("\n" + "="*80)
    print(f"All six experts successfully trained for asset: {args.asset}")
    print("="*80)


if __name__ == "__main__":
    main()