from __future__ import annotations
from typing import Dict, List

REGIME_LABEL_NAMES: Dict[int, str] = {
    0: "Sideways",
    1: "Trending Up",
    2: "Trending Down",
}
REGIME_NAME_TO_ID: Dict[str, int] = {v: k for k, v in REGIME_LABEL_NAMES.items()}

FEATURE_COLUMNS: List[str] = [
    "returns",
    "ema_20",
    "rsi_14",
    "atr_14",
    "vortex_14",
    "macd",
    "adx",
]

CONFIG = {
    "labels": REGIME_LABEL_NAMES,
    "label_names": REGIME_NAME_TO_ID,
    "features": FEATURE_COLUMNS,
    "model": {
        "window": 30,                 # sequence length (input window)
        "default_horizon": 6,
        "forecast_horizons": [1, 2, 3, 4, 5, 6],
    },
    "labeling": {
        "label_window": 24,
        "trend_score_quantile": 0.05,
        "return_quantile": 0.05,
        "adx_threshold": 0.0,
        "min_trend_score": 0.05,
        "min_abs_return": 0.001,
        "min_horizon": 1,
        "max_horizon": 6,
        "smoothing_ema": 0.2,
    },
    "training": {
        "seed": 42,
        "test_split": 0.20,           # note: dataset already split by DatasetBuilder
        "save_prediction_artifacts": True,
    },
    "tcn": {
        "num_channels": [32, 32, 32, 32],
        "kernel_size": 3,
        "dropout": 0.3,
        "attention_heads": 4,
        "attention_dropout": 0.1,
        "se_reduction": 16,
        "stochastic_depth_prob": 0.1,
        "batch_size": 64,
        "epochs": 100,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "patience": 15,
        "use_class_weights": True,
        "label_smoothing": 0.0,
    },
}

def get_feature_columns() -> List[str]:
    return list(CONFIG["features"])

def get_forecast_horizons() -> List[int]:
    return list(CONFIG["model"]["forecast_horizons"])

def get_label_name(label_id: int) -> str:
    return REGIME_LABEL_NAMES[int(label_id)]