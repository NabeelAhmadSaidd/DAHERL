#File: preprocessing/dataset_builder.py
from __future__ import annotations
from pathlib import Path
import pickle
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import yaml
from numpy.lib.stride_tricks import sliding_window_view

from preprocessing.feature_engineering import FeatureEngineer
from regime.config import CONFIG as REGIME_CONFIG, get_feature_columns, get_forecast_horizons


class DatasetBuilder:
    def __init__(self, config_path: str = "configs/data.yaml") -> None:
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.feature_engineer = FeatureEngineer()
        self.feature_columns = get_feature_columns()
        self.horizons = get_forecast_horizons()
        self.default_horizon = int(REGIME_CONFIG["model"]["default_horizon"])

        labeling_cfg = REGIME_CONFIG["labeling"]
        self.label_window = int(labeling_cfg["label_window"])
        self.trend_score_quantile = float(labeling_cfg["trend_score_quantile"])
        self.return_quantile = float(labeling_cfg["return_quantile"])
        self.adx_threshold = float(labeling_cfg["adx_threshold"])
        self.min_trend_score = float(labeling_cfg["min_trend_score"])
        self.min_abs_return = float(labeling_cfg["min_abs_return"])

        output_cfg = self.config.get("output", {})
        self.processed_dir = Path(output_cfg.get("processed_dir", "data/processed"))
        self.scaler_dir = Path("results/models/scalers")
        self.regime_dir = Path("results/models/regime")

        # Will be set dynamically for each asset
        self._available_features: List[str] = []

        self.processed_data: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Splitting and normalization
    # ------------------------------------------------------------------

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_ratio = float(self.config["splits"]["train"])
        test_ratio = float(self.config["splits"]["test"])
        if abs((train_ratio + test_ratio) - 1.0) > 1e-6:
            raise ValueError("Train and test ratios must sum to 1.0")

        n = len(df)
        train_end = int(n * train_ratio)
        if train_end <= 0 or train_end >= n:
            raise ValueError(f"Invalid split size for dataframe with {n} rows")

        train_df = df.iloc[:train_end].copy().reset_index(drop=True)
        test_df = df.iloc[train_end:].copy().reset_index(drop=True)
        return train_df, test_df

    def normalize_features(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        asset_name: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Use only the columns that actually exist (after feature engineering)
        cols = self._available_features
        if not cols:
            raise RuntimeError("No available features. Call process_asset() first.")

        mean = train_df[cols].mean()
        std = train_df[cols].std().replace(0, 1.0)

        train_df.loc[:, cols] = (train_df[cols] - mean) / std
        test_df.loc[:, cols] = (test_df[cols] - mean) / std

        train_df.loc[:, cols] = train_df[cols].clip(-6.0, 6.0)
        test_df.loc[:, cols] = test_df[cols].clip(-6.0, 6.0)

        self.scaler_dir.mkdir(parents=True, exist_ok=True)
        scaler_path = self.scaler_dir / f"{asset_name}_scaler.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(
                {
                    "mean": mean.to_dict(),
                    "std": std.to_dict(),
                    "features": cols,  # save the actual features used
                },
                f,
            )
        return train_df, test_df

    # ------------------------------------------------------------------
    # Regime labeling (all methods below are unchanged – only process_asset
    # and normalize_features were adjusted)
    # ------------------------------------------------------------------

    def _future_window_stats(self, df: pd.DataFrame, horizon: int) -> Dict[str, np.ndarray]:
        close = df["close"].to_numpy(dtype=np.float64)
        adx = df["adx"].to_numpy(dtype=np.float64) if "adx" in df.columns else np.zeros(len(df), dtype=np.float64)
        log_close = np.log(np.clip(close, 1e-12, None))

        if len(df) < self.label_window + horizon:
            return {
                "positions": np.empty((0,), dtype=np.int64),
                "trend_score": np.empty((0,), dtype=np.float64),
                "future_return": np.empty((0,), dtype=np.float64),
                "future_adx": np.empty((0,), dtype=np.float64),
            }

        windows = sliding_window_view(log_close, window_shape=self.label_window)
        starts = np.arange(windows.shape[0], dtype=np.int64)

        centered = windows - windows.mean(axis=1, keepdims=True)
        x = np.arange(self.label_window, dtype=np.float64)
        x = x - x.mean()
        denom = float(np.sum(x ** 2) + 1e-12)
        slope = (centered * x).sum(axis=1) / denom

        diffs = np.diff(windows, axis=1)
        volatility = diffs.std(axis=1, ddof=0) + 1e-12
        trend_score = slope / volatility
        future_return = np.expm1(windows[:, -1] - windows[:, 0])
        future_adx = adx[starts + self.label_window - 1]

        positions = starts - horizon
        valid = positions >= 0
        return {
            "positions": positions[valid],
            "trend_score": trend_score[valid],
            "future_return": future_return[valid],
            "future_adx": future_adx[valid],
        }

    def _fit_thresholds(self, train_df: pd.DataFrame) -> Dict[int, Dict[str, float]]:
        thresholds: Dict[int, Dict[str, float]] = {}
        for horizon in self.horizons:
            stats = self._future_window_stats(train_df, horizon)
            abs_scores = np.abs(stats["trend_score"])
            abs_returns = np.abs(stats["future_return"])

            trend_threshold = self.min_trend_score
            if abs_scores.size > 0:
                trend_threshold = max(
                    self.min_trend_score,
                    float(np.quantile(abs_scores, self.trend_score_quantile)),
                )

            return_threshold = self.min_abs_return
            if abs_returns.size > 0:
                return_threshold = max(
                    self.min_abs_return,
                    float(np.quantile(abs_returns, self.return_quantile)),
                )

            thresholds[horizon] = {
                "trend_score": trend_threshold,
                "abs_return": return_threshold,
                "adx": self.adx_threshold,
            }
        return thresholds

    def _apply_labels(
        self,
        df: pd.DataFrame,
        horizon: int,
        thresholds: Dict[str, float],
    ) -> pd.Series:
        labels = np.full(len(df), -1, dtype=np.int16)
        stats = self._future_window_stats(df, horizon)
        if stats["positions"].size == 0:
            return pd.Series(labels, index=df.index)

        positions = stats["positions"]
        trend_score = stats["trend_score"]
        future_return = stats["future_return"]
        future_adx = stats["future_adx"]

        trend_threshold = float(thresholds["trend_score"])
        return_threshold = float(thresholds["abs_return"])
        adx_threshold = float(thresholds["adx"])

        trend_mask = (
            (np.abs(trend_score) >= trend_threshold)
            & (np.abs(future_return) >= return_threshold)
            & (future_adx >= adx_threshold)
        )

        up_mask = trend_mask & (trend_score > 0.0) & (future_return > 0.0)
        down_mask = trend_mask & (trend_score < 0.0) & (future_return < 0.0)

        labels[positions] = 0
        labels[positions[up_mask]] = 1
        labels[positions[down_mask]] = 2
        return pd.Series(labels, index=df.index)

    def _attach_regime_labels(
        self,
        df: pd.DataFrame,
        thresholds_by_horizon: Dict[int, Dict[str, float]],
    ) -> pd.DataFrame:
        df = df.copy()
        for horizon in self.horizons:
            df[f"regime_h{horizon}"] = self._apply_labels(df, horizon, thresholds_by_horizon[horizon]).astype(np.int16)
        df["regime"] = df[f"regime_h{self.default_horizon}"].astype(np.int16)
        return df

    def _summarize_labels(self, df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
        summary: Dict[str, Dict[str, int]] = {}
        for horizon in self.horizons:
            col = f"regime_h{horizon}"
            counts = df.loc[df[col] >= 0, col].value_counts().to_dict()
            summary[col] = {str(int(k)): int(v) for k, v in counts.items()}
        return summary

    # ------------------------------------------------------------------
    # Asset processing 
    # ------------------------------------------------------------------

    def process_asset(self, asset_name: str, csv_path: str) -> None:
        print(f"Processing {asset_name}")

        # 1. Load raw data
        df = pd.read_csv(csv_path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp").reset_index(drop=True)
        else:
            df = df.sort_index().reset_index(drop=True)

        # --- ADJUSTED DATE FILTER -------------------------------------------
        # Forex: 2023-01-01 to 2025-12-31
        # Crypto: 2024-01-01 to 2025-12-31
        end_date = pd.Timestamp("2025-12-31")
        if "USDT" in asset_name.upper():   # e.g. BTCUSDT, ETHUSDT
            start_date = pd.Timestamp("2024-01-01")
        else:                              # e.g. EURUSD, USDJPY
            start_date = pd.Timestamp("2023-01-01")

        mask = (df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)
        df = df.loc[mask].reset_index(drop=True)
        if len(df) == 0:
            print(f"Warning: No data found between {start_date.date()} and "
                f"{end_date.date()} for {asset_name}")
        # --------------------------------------------------------------------

        # 2. Feature engineering
        df = self.feature_engineer.transform(df)

        # 3. Available features
        self._available_features = [c for c in self.feature_columns if c in df.columns]
        missing = set(self.feature_columns) - set(self._available_features)
        if missing:
            print(f"Warning: the following feature columns were not produced: {missing}")

        # 4. Split
        train_df, test_df = self.split_data(df)

        # 5. Regime labels
        thresholds_by_horizon = self._fit_thresholds(train_df)
        train_df = self._attach_regime_labels(train_df, thresholds_by_horizon)
        test_df = self._attach_regime_labels(test_df, thresholds_by_horizon)

        # 6. Normalise
        train_df, test_df = self.normalize_features(train_df, test_df, asset_name)

        # 7. Save to disk
        output_dir = self.processed_dir / asset_name
        output_dir.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(output_dir / "train.csv", index=False)
        test_df.to_csv(output_dir / "test.csv", index=False)

        self.regime_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "asset": asset_name,
            "feature_columns": self._available_features,
            "horizons": self.horizons,
            "default_horizon": self.default_horizon,
            "thresholds": thresholds_by_horizon,
            "label_window": self.label_window,
            "train_label_counts": self._summarize_labels(train_df),
            "test_label_counts": self._summarize_labels(test_df),
        }
        with open(self.regime_dir / f"{asset_name}_label_metadata.pkl", "wb") as fh:
            pickle.dump(metadata, fh)

        self.processed_data[asset_name] = {
            "train_df": train_df,
            "test_df": test_df,
            "metadata": metadata,
        }

        print(
            f"{asset_name}: {len(train_df)} train | {len(test_df)} test | "
            f"horizons={self.horizons}"
        )

    def run(self) -> None:
        for asset_name, asset_cfg in self.config["datasets"].items():
            self.process_asset(asset_name, asset_cfg["path"])

    # ------------------------------------------------------------------
    # Return all logic
    # ------------------------------------------------------------------
    def return_all(self) -> Dict[str, Dict[str, Any]]:
        if not self.processed_data:
            raise RuntimeError("No data processed yet. Call run() first.")
        return self.processed_data


if __name__ == "__main__":
    builder = DatasetBuilder("configs/data.yaml")
    builder.run()
    all_data = builder.return_all()
    for asset, data in all_data.items():
        print(f"{asset}: train shape {data['train_df'].shape}, test shape {data['test_df'].shape}")