from __future__ import annotations

import json
import os
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

from experts.train_experts import detect_asset_type, get_env_params


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AVAILABLE_ASSETS = ("EURUSD", "BTCUSDT", "ETHUSDT", "USDJPY")
EXPERT_ORDER = ("PPO", "SAC", "DDQN", "TD3", "TQC", "PPG")
EXPERT_CHECKPOINT_NAMES = {
    "PPO": "ppo_expert.pt",
    "SAC": "sac_expert.pt",
    "DDQN": "ddqn_expert.pt",
    "TD3": "td3_expert.pt",
    "TQC": "tqc_expert.pt",
    "PPG": "ppg_expert.pt",
}


def resolve_path(path: str | Path, base: Path = PROJECT_ROOT) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = base / resolved
    return resolved


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(resolve_path(path), "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def deep_merge(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(left)
    for key, value in right.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def seed_everything(seed: int) -> None:
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def make_torch_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def annualized_return(
    initial_value: float,
    final_value: float,
    periods: int,
    periods_per_year: int,
) -> float:
    if periods <= 0 or initial_value <= 0:
        return 0.0
    ratio = max(final_value / initial_value, 1e-12)
    return float((ratio ** (periods_per_year / float(periods))) - 1.0)


def compute_metrics(
    portfolio_values: Sequence[float],
    periods_per_year: int = 8760,
) -> Dict[str, float]:
    values = np.asarray(portfolio_values, dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    if values.size < 2:
        return {
            "total_return_pct": 0.0,
            "annualized_return_pct": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_drawdown_pct": 0.0,
            "win_rate_pct": 0.0,
            "final_portfolio_value": float(values[-1]) if values.size else 0.0,
        }

    step_returns = np.diff(values) / (np.abs(values[:-1]) + 1e-12)
    initial = float(values[0])
    final = float(values[-1])
    total_return = (final - initial) / (abs(initial) + 1e-12)
    ann_return = annualized_return(initial, final, len(step_returns), periods_per_year)

    mean_ret = float(np.mean(step_returns))
    std_ret = float(np.std(step_returns, ddof=1)) if step_returns.size > 1 else 0.0
    sharpe = 0.0 if std_ret < 1e-12 else mean_ret / std_ret * np.sqrt(periods_per_year)

    downside = step_returns[step_returns < 0]
    downside_std = float(np.std(downside, ddof=1)) if downside.size > 1 else 0.0
    sortino = 0.0 if downside_std < 1e-12 else mean_ret / downside_std * np.sqrt(periods_per_year)

    peak = np.maximum.accumulate(np.maximum(values, 1e-12))
    drawdown = (peak - values) / (peak + 1e-12)
    max_drawdown = float(np.max(drawdown)) if drawdown.size else 0.0

    win_rate = float(np.mean(step_returns > 0.0)) if step_returns.size else 0.0

    return {
        "total_return_pct": round(total_return * 100.0, 6),
        "annualized_return_pct": round(ann_return * 100.0, 6),
        "sharpe": round(float(sharpe), 6),
        "sortino": round(float(sortino), 6),
        "max_drawdown_pct": round(max_drawdown * 100.0, 6),
        "win_rate_pct": round(win_rate * 100.0, 6),
        "final_portfolio_value": round(final, 6),
    }


@dataclass(frozen=True)
class AssetArtifacts:
    asset: str
    asset_type: str
    train_path: Path
    test_path: Path
    scaler_path: Path
    regime_metadata_path: Path
    regime_checkpoint_path: Path
    expert_checkpoint_dir: Path
    allocator_checkpoint_path: Path
    processed_dir: Path
    output_dir: Path
    storage_dir: Path

    @property
    def expert_checkpoint_paths(self) -> Dict[str, Path]:
        return {
            name: self.expert_checkpoint_dir / self.asset / filename
            for name, filename in EXPERT_CHECKPOINT_NAMES.items()
        }


def resolve_asset_artifacts(config: Dict[str, Any]) -> AssetArtifacts:
    asset_cfg = config.get("experiment", {})
    asset = str(asset_cfg.get("asset", config.get("asset", "EURUSD"))).upper()
    if asset not in AVAILABLE_ASSETS:
        raise ValueError(
            f"Unsupported asset '{asset}'. Expected one of {list(AVAILABLE_ASSETS)}."
        )

    paths_cfg = config.get("paths", {})
    processed_dir = resolve_path(paths_cfg.get("processed_dir", "data/processed"))
    scaler_dir = resolve_path(paths_cfg.get("scaler_dir", "results/models/scalers"))
    regime_metadata_dir = resolve_path(
        paths_cfg.get("regime_metadata_dir", "results/models/regime")
    )
    regime_checkpoint_path = ( resolve_path(paths_cfg.get("checkpoint_dir", "checkpoints")) / asset / "regime"/ f"{asset}_tcn_multi.pth" )
    checkpoint_dir = resolve_path(paths_cfg.get("checkpoint_dir", "checkpoints"))
    storage_dir = resolve_path(paths_cfg.get("storage_dir", "storage"))
    output_dir = resolve_path(
        asset_cfg.get("output_dir", config.get("output_dir", "results/ablations"))
    )

    train_path = processed_dir / asset / "train.csv"
    test_path = processed_dir / asset / "test.csv"
    scaler_path = scaler_dir / f"{asset}_scaler.pkl"
    regime_metadata_path = regime_metadata_dir / f"{asset}_label_metadata.pkl"
    allocator_checkpoint_path = checkpoint_dir / asset / "allocator.pt"

    return AssetArtifacts(
        asset=asset,
        asset_type=detect_asset_type(asset),
        train_path=train_path,
        test_path=test_path,
        scaler_path=scaler_path,
        regime_metadata_path=regime_metadata_path,
        regime_checkpoint_path=regime_checkpoint_path,
        expert_checkpoint_dir=checkpoint_dir,
        allocator_checkpoint_path=allocator_checkpoint_path,
        processed_dir=processed_dir,
        output_dir=output_dir,
        storage_dir=storage_dir,
    )


def load_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def load_pickle(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing artifact: {path}")
    with open(path, "rb") as fh:
        return pickle.load(fh)


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)

def summarize_portfolio(
    portfolio_values: Sequence[float],
    periods_per_year: int = 8760,
) -> Dict[str, Any]:
    """
    Wrapper used by ablation modules.

    Returns:
        {
            total_return_pct,
            annualized_return_pct,
            sharpe,
            sortino,
            max_drawdown_pct,
            win_rate_pct,
            final_portfolio_value
        }
    """
    return compute_metrics(
        portfolio_values=portfolio_values,
        periods_per_year=periods_per_year,
    )