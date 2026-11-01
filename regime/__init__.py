# regime/__init__.py
"""Regime detection package (TCN)."""

from .config import CONFIG, REGIME_LABEL_NAMES, get_feature_columns, get_forecast_horizons, get_label_name
from .dataset import RegimeDataset
from .inference import RegimeClassifier
from .tcn_model import MultiHorizonRegimeTCN

__all__ = [
    "CONFIG",
    "REGIME_LABEL_NAMES",
    "get_feature_columns",
    "get_forecast_horizons",
    "get_label_name",
    "RegimeDataset",
    "RegimeClassifier",
    "MultiHorizonRegimeTCN",
]