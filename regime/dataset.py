from __future__ import annotations
from typing import List, Sequence, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class RegimeDataset(Dataset):
    """
    Windowed dataset that yields (feature_sequence, target_vector).
    x: (window, len(feature_cols)) float32 tensor
    y: (len(target_cols),) int64 tensor, -1 for invalid entries.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: Sequence[str],
        target_cols: Sequence[str],
        window: int = 128,
    ) -> None:
        super().__init__()
        self._df = df.reset_index(drop=True).copy()
        self.feature_cols = list(feature_cols)
        self.target_cols = list(target_cols)
        self.window = int(window)

        required = self.feature_cols + self.target_cols
        missing = [c for c in required if c not in self._df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        features = (
            self._df[self.feature_cols]
            .replace([np.inf, -np.inf], np.nan)
            .to_numpy(dtype=np.float32)
        )
        self._features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        n_rows = len(self._df)
        n_targets = len(self.target_cols)
        self._targets = np.full((n_rows, n_targets), -1, dtype=np.int64)
        for i, col in enumerate(self.target_cols):
            raw = self._df[col].to_numpy()
            valid = ~pd.isna(raw)
            self._targets[valid, i] = raw[valid].astype(np.int64)

        self._valid_ends: List[int] = []
        for end in range(self.window - 1, n_rows):
            if np.any(self._targets[end] >= 0):
                self._valid_ends.append(end)

    def __len__(self) -> int:
        return len(self._valid_ends)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        end = self._valid_ends[idx]
        start = end - self.window + 1
        x = self._features[start : end + 1]
        y = self._targets[end]
        return torch.from_numpy(x), torch.from_numpy(y)

    @property
    def indices(self) -> np.ndarray:
        return np.asarray(self._valid_ends, dtype=np.int64)

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(self) == 0:
            return (
                np.empty((0, self.window, len(self.feature_cols)), dtype=np.float32),
                np.empty((0, len(self.target_cols)), dtype=np.int64),
                np.empty((0,), dtype=np.int64),
            )
        X = np.stack([self._features[e - self.window + 1 : e + 1] for e in self._valid_ends])
        y = np.stack([self._targets[e] for e in self._valid_ends])
        idx = self.indices
        return X.astype(np.float32), y.astype(np.int64), idx