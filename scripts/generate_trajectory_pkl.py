"""
Generate a trajectory .pkl file from a trained expert checkpoint.
Usage:
    python -m scripts.generate_trajectory_pkl --checkpoint checkpoints/EURUSD/diversity/TQC/TQC_expert_30.pt --asset EURUSD --data data/processed/EURUSD/train.csv
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from environment.trading_env import TradingEnv
from experts.tqc_expert import TQCAgent
from experts.train_experts import detect_asset_type, get_env_params


def load_config(config_path: str = "configs/experts.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_agent(checkpoint_path: str, state_dim: int, expert_cfg: dict) -> TQCAgent:
    """Instantiate TQCAgent with the saved architecture and load weights."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    agent = TQCAgent(
        state_dim=state_dim,
        action_dim=1,
        lr=expert_cfg["learning_rate"],
        lr_actor=expert_cfg.get("lr_actor", expert_cfg.get("learning_rate")),
        lr_critic=expert_cfg.get("lr_critic", expert_cfg.get("learning_rate")),
        gamma=expert_cfg["gamma"],
        tau=expert_cfg["tau"],
        buffer_size=expert_cfg["buffer_size"],
        batch_size=expert_cfg["batch_size"],
        warmup_steps=expert_cfg["warmup_steps"],
        hidden_dim=expert_cfg["hidden_dim"],
        actor_hidden_dim=expert_cfg.get("actor_hidden_dim"),
        critic_hidden_dim=expert_cfg.get("critic_hidden_dim"),
        actor_num_layers=expert_cfg.get("actor_num_layers", 2),
        critic_num_layers=expert_cfg.get("critic_num_layers", 2),
        activation=expert_cfg.get("activation", "relu"),
        log_std_min=expert_cfg.get("log_std_min", -20),
        log_std_max=expert_cfg.get("log_std_max", 2),
        n_critics=expert_cfg["n_critics"],
        n_quantiles=expert_cfg["n_quantiles"],
        top_quantiles_to_drop=expert_cfg["top_quantiles_to_drop"],
        auto_entropy_tuning=expert_cfg["auto_entropy_tuning"],
        alpha=expert_cfg["alpha"],
        grad_norm_clip=expert_cfg.get("grad_norm_clip", 1.0),
        device="cpu",
    )

    # Load the network weights
    if "actor" in ckpt:
        agent.actor.load_state_dict(ckpt["actor"])
        for critic, critic_state in zip(agent.critics, ckpt["critics"]):
            critic.load_state_dict(critic_state)
    else:
        # Fallback if checkpoint has a different structure
        agent.actor.load_state_dict(ckpt)
        # If critics are not present, you may need to handle differently
        # but typically they are saved together.

    return agent


def run_agent(agent, env):
    """Run agent deterministically and collect trajectory."""
    obs, _ = env.reset()
    done = False
    actions = []
    returns = []
    positions = []

    while not done:
        # Try deterministic first, fallback to eval_mode, then plain act
        try:
            action = agent.act(obs, deterministic=True)
        except TypeError:
            try:
                action = agent.act(obs, eval_mode=True)
            except TypeError:
                action = agent.act(obs)  # hope it's deterministic

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        actions.append(float(action[0]) if isinstance(action, np.ndarray) else float(action))
        returns.append(float(info.get("step_return", 0.0)))
        positions.append(float(info.get("position_frac", 0.0)))

    return {
        "actions": np.array(actions, dtype=np.float32),
        "returns": np.array(returns, dtype=np.float32),
        "positions": np.array(positions, dtype=np.float32),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the .pt checkpoint (e.g., checkpoints/EURUSD/TQC_expert_30.pt)")
    parser.add_argument("--asset", type=str, required=True,
                        help="Asset name (e.g., EURUSD)")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to test/train CSV (e.g., data/processed/EURUSD/train.csv)")
    parser.add_argument("--config", type=str, default="configs/experts.yaml")
    parser.add_argument("--no-diversity", action="store_true",
                        help="Save .pkl in no_diversity subfolder (default: diversity)")
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    tqc_cfg = cfg["tqc"]

    # Load data
    df = pd.read_csv(args.data)
    asset_type = detect_asset_type(args.asset)

    # Determine features
    feature_cols = tqc_cfg["features"]

    # Build environment
    env_params = get_env_params(asset_type, expert_id=4)  # TQC expert id
    env_params["continuous_actions"] = True
    env = TradingEnv(df=df, feature_cols=feature_cols, expert_id=4, **env_params)
    state_dim = env.observation_space.shape[0]

    # Load agent
    agent = load_agent(args.checkpoint, state_dim, tqc_cfg)
    print(f"Loaded agent from {args.checkpoint}")

    # Run agent
    trajectory = run_agent(agent, env)
    print(f"Collected trajectory of length {len(trajectory['actions'])}")

    # Determine storage directory
    subfolder = "no_diversity" if args.no_diversity else "diversity"
    storage_dir = Path("storage") / args.asset / subfolder
    storage_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = storage_dir / "TQC.pkl"

    # Save .pkl
    with open(pkl_path, "wb") as f:
        pickle.dump(trajectory, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Trajectory saved to: {pkl_path}")


if __name__ == "__main__":
    main()