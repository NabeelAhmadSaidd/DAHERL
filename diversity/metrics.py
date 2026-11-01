"""Diversity metrics for RL trading agents.

Implements three core diversity metrics: action correlation, return correlation,
and trade overlap. All functions are vectorized and handle NaN/constant inputs.
"""

import numpy as np
from typing import Tuple, Optional

def action_correlation(actions_a: np.ndarray, actions_b: np.ndarray) -> float:
    """Absolute Pearson correlation between two action sequences."""
    a = np.asarray(actions_a, dtype=np.float64).flatten()
    b = np.asarray(actions_b, dtype=np.float64).flatten()
    min_len = min(len(a), len(b))
    if min_len < 2:
        return 0.0
    a = a[:min_len]
    b = b[:min_len]
    mask = ~(np.isnan(a) | np.isnan(b))
    if np.sum(mask) < 2:
        return 0.0
    a = a[mask]
    b = b[mask]
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    corr = np.corrcoef(a, b)[0, 1]
    return float(abs(corr))

def return_correlation(returns_a: np.ndarray, returns_b: np.ndarray) -> float:
    """Absolute Pearson correlation between return sequences."""
    return action_correlation(returns_a, returns_b)

def trade_overlap(actions_a: np.ndarray, actions_b: np.ndarray,
                  threshold: float = 0.01) -> float:
    """Percentage of steps where both agents execute the same non‑hold trade."""
    a = np.asarray(actions_a).flatten()
    b = np.asarray(actions_b).flatten()
    min_len = min(len(a), len(b))
    if min_len == 0:
        return 0.0
    a = a[:min_len]
    b = b[:min_len]

    def to_signal(x: np.ndarray, thresh: float) -> np.ndarray:
        x = x.astype(np.float64)
        sig = np.zeros_like(x, dtype=np.int8)
        sig[x > thresh] = 1
        sig[x < -thresh] = -1
        return sig

    sig_a = to_signal(a, threshold)
    sig_b = to_signal(b, threshold)
    same_nonzero = (sig_a == sig_b) & (sig_a != 0)
    total = len(sig_a)
    if total == 0:
        return 0.0
    return float(np.sum(same_nonzero) / total)

def similarity_score(action_corr: float, return_corr: float, trade_ov: float,
                     weights: Tuple[float, float, float] = (0.4, 0.4, 0.2)) -> float:
    """Combine three diversity metrics into a single similarity score."""
    w1, w2, w3 = weights
    if abs(w1 + w2 + w3 - 1.0) > 1e-6:
        raise ValueError("Weights must sum to 1.0")
    return float(w1 * action_corr + w2 * return_corr + w3 * trade_ov)