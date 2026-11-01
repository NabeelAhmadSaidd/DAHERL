# regime/inference.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .config import CONFIG, REGIME_LABEL_NAMES
from .tcn_model import MultiHorizonRegimeTCN


class RegimeClassifier:
    """
    Stateful inference wrapper for the multi‑horizon TCN.
    Maintains a rolling window of the latest feature vectors.
    Returns predictions for a single, configurable horizon.
    """

    def __init__(
        self,
        model_path: str | Path,
        horizon: Optional[int] = None,
        smooth_alpha: Optional[float] = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.horizon = horizon or CONFIG["model"]["default_horizon"]
        self.smooth_alpha = float(
            smooth_alpha if smooth_alpha is not None else CONFIG["labeling"]["smoothing_ema"]
        )

        tcn_cfg = CONFIG["tcn"]
        self.feature_cols = CONFIG["features"]
        self.window = CONFIG["model"]["window"]
        self.horizons = CONFIG["model"]["forecast_horizons"]
        self.num_horizons = len(self.horizons)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = MultiHorizonRegimeTCN(
            input_size=len(self.feature_cols),
            num_horizons=self.num_horizons,
            num_channels=tcn_cfg["num_channels"],
            kernel_size=tcn_cfg["kernel_size"],
            dropout=tcn_cfg["dropout"],
            attention_heads=tcn_cfg["attention_heads"],
            attention_dropout=tcn_cfg["attention_dropout"],
            se_reduction=tcn_cfg["se_reduction"],
            stochastic_depth_prob=0.0,  # no stochastic depth during inference
            max_seq_len=self.window + 50,
        ).to(self.device)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()

        self._buffer: List[np.ndarray] = []
        self._smoothed_proba: Optional[np.ndarray] = None

    def reset_state(self) -> None:
        self._buffer = []
        self._smoothed_proba = None

    def _smooth(self, proba: np.ndarray) -> np.ndarray:
        if self._smoothed_proba is None:
            self._smoothed_proba = proba.astype(np.float32)
        else:
            alpha = np.clip(self.smooth_alpha, 0.0, 1.0)
            self._smoothed_proba = alpha * proba + (1.0 - alpha) * self._smoothed_proba
        smoothed = np.clip(self._smoothed_proba, 1e-12, 1.0)
        smoothed /= smoothed.sum() + 1e-12
        return smoothed

    def update(self, feature_vector: np.ndarray) -> None:
        """Add a new feature vector (shape (n_features,))."""
        if len(self._buffer) >= self.window:
            self._buffer.pop(0)
        self._buffer.append(np.asarray(feature_vector, dtype=np.float32))

    def predict_now(self) -> Optional[Dict[str, Any]]:
        """Predict using the current window; returns dict for the default horizon."""
        if len(self._buffer) < self.window:
            return None
        window = np.stack(self._buffer[-self.window:], axis=0)  # (window, n_features)
        x = torch.from_numpy(window).unsqueeze(0).to(self.device)  # (1, window, n_features)
        with torch.no_grad():
            logits = self.model(x)[0]  # (num_horizons, 3)
        # Select the requested horizon
        idx = self.horizons.index(self.horizon) if self.horizon in self.horizons else 0
        logits_h = logits[idx]  # (3,)
        proba = torch.softmax(logits_h, dim=-1).cpu().numpy()
        proba = self._smooth(proba) if self.smooth_alpha > 0 else proba

        pred = int(np.argmax(proba))
        confidence = float(np.max(proba))
        entropy = float(-np.sum(proba * np.log(np.clip(proba, 1e-12, 1.0))) / np.log(len(proba)))
        margin = float(np.sort(proba)[-1] - np.sort(proba)[-2]) if len(proba) > 1 else 1.0
        expected_direction = float(proba[1] - proba[2]) if len(proba) >= 3 else 0.0

        return {
            "class_id": pred,
            "class_name": REGIME_LABEL_NAMES[pred],
            "confidence": confidence,
            "entropy": entropy,
            "margin": margin,
            "expected_direction": expected_direction,
            "probabilities": proba.tolist(),
        }


class MultiHorizonRegimeClassifier:
    """
    TCN‑based inference for all forecast horizons simultaneously.
    Takes a full window (not stateful) and returns a dict with arrays for all horizons.
    """

    def __init__(
        self,
        model_path: str | Path,
        feature_cols: Optional[List[str]] = None,
        window: Optional[int] = None,
        horizons: Optional[List[int]] = None,
        smooth_alpha: Optional[float] = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.feature_cols = list(feature_cols or CONFIG["features"])
        self.window = int(window or CONFIG["model"]["window"])
        self.horizons = list(horizons or CONFIG["model"]["forecast_horizons"])
        self.smooth_alpha = float(
            smooth_alpha if smooth_alpha is not None else CONFIG["labeling"]["smoothing_ema"]
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        tcn_cfg = CONFIG["tcn"]
        self.model = MultiHorizonRegimeTCN(
            input_size=len(self.feature_cols),
            num_horizons=len(self.horizons),
            num_channels=tcn_cfg["num_channels"],
            kernel_size=tcn_cfg["kernel_size"],
            dropout=tcn_cfg["dropout"],
            attention_heads=tcn_cfg["attention_heads"],
            attention_dropout=tcn_cfg["attention_dropout"],
            se_reduction=tcn_cfg["se_reduction"],
            stochastic_depth_prob=0.0,  # inference only
            max_seq_len=self.window + 50,
        ).to(self.device)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()

        # Smoothing state for each horizon: shape (num_horizons, 3)
        self._smoothed_proba: Optional[np.ndarray] = None

    def reset_state(self) -> None:
        self._smoothed_proba = None

    def _smooth(self, proba: np.ndarray) -> np.ndarray:
        """Apply exponential smoothing to the full (H, 3) probability matrix."""
        if self._smoothed_proba is None:
            self._smoothed_proba = proba.astype(np.float32)
        else:
            alpha = np.clip(self.smooth_alpha, 0.0, 1.0)
            self._smoothed_proba = alpha * proba + (1.0 - alpha) * self._smoothed_proba
        smoothed = np.clip(self._smoothed_proba, 1e-12, 1.0)
        smoothed /= smoothed.sum(axis=1, keepdims=True) + 1e-12
        return smoothed

    def predict_all(self, window: np.ndarray, smooth: bool = True) -> Dict[str, Any]:
        """
        Predict regimes for all horizons from a single window.

        Args:
            window: (window_len, n_features) numpy array.
            smooth: whether to apply EMA smoothing.

        Returns:
            dict with keys:
                - "probabilities": (num_horizons, 3) numpy array
                - "class_ids": list of ints
                - "class_names": list of str
                - "confidences": list of floats
                - "entropies": list of floats
                - "margins": list of floats
                - "expected_directions": list of floats
        """
        if window.shape != (self.window, len(self.feature_cols)):
            raise ValueError(
                f"Window shape {window.shape} does not match expected ({self.window}, {len(self.feature_cols)})"
            )

        x = torch.from_numpy(window.astype(np.float32)).unsqueeze(0).to(self.device)  # (1, W, F)
        with torch.no_grad():
            logits = self.model(x)[0]  # (num_horizons, 3)
        proba = torch.softmax(logits, dim=-1).cpu().numpy()  # (H, 3)

        if smooth and self.smooth_alpha > 0.0:
            proba = self._smooth(proba)

        results = {
            "probabilities": proba,
            "class_ids": [],
            "class_names": [],
            "confidences": [],
            "entropies": [],
            "margins": [],
            "expected_directions": [],
        }
        for h in range(proba.shape[0]):
            p = proba[h]
            class_id = int(np.argmax(p))
            confidence = float(np.max(p))
            entropy = float(-np.sum(p * np.log(np.clip(p, 1e-12, 1.0))) / np.log(3))
            sorted_p = np.sort(p)
            margin = float(sorted_p[-1] - sorted_p[-2]) if len(p) > 1 else 1.0
            expected_direction = float(p[1] - p[2])  # Up - Down
            results["class_ids"].append(class_id)
            results["class_names"].append(REGIME_LABEL_NAMES[class_id])
            results["confidences"].append(confidence)
            results["entropies"].append(entropy)
            results["margins"].append(margin)
            results["expected_directions"].append(expected_direction)

        return results

    def predict_horizon(self, window: np.ndarray, horizon: int, smooth: bool = True) -> Dict[str, Any]:
        """Predict for a single horizon."""
        all_results = self.predict_all(window, smooth=smooth)
        try:
            idx = self.horizons.index(horizon)
        except ValueError:
            raise ValueError(f"Horizon {horizon} not in {self.horizons}")
        return {
            "class_id": all_results["class_ids"][idx],
            "class_name": all_results["class_names"][idx],
            "confidence": all_results["confidences"][idx],
            "entropy": all_results["entropies"][idx],
            "margin": all_results["margins"][idx],
            "expected_direction": all_results["expected_directions"][idx],
            "probabilities": all_results["probabilities"][idx],
        }