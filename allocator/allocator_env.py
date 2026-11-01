"""
Allocator environment for DAHERL expert selection.
Corrected version – fully respects regime_mode and include_regime_features.
"""

from __future__ import annotations

import inspect
from collections import deque
from typing import Any, Dict, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from regime.config import get_forecast_horizons
from regime.inference import MultiHorizonRegimeClassifier, RegimeClassifier


class AllocatorEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        base_env: gym.Env,
        experts: Sequence[Any],
        lookback: int = 20,
        regime_model: Optional[RegimeClassifier | MultiHorizonRegimeClassifier] = None,
        regime_feature_cols: Optional[Sequence[str]] = None,
        regime_window: Optional[int] = None,
        include_regime_features: Optional[bool] = None,
        regime_mode: str = "enabled",
        use_regime: Optional[bool] = None,   # kept for backward compatibility
    ):
        super().__init__()
        if len(experts) == 0:
            raise ValueError("AllocatorEnv requires at least one expert.")

        self.base_env = base_env
        self.experts = list(experts)
        self.n_experts = len(self.experts)
        self.lookback = int(lookback)
        self.regime_model = regime_model
        self.regime_feature_cols = list(
            regime_feature_cols or getattr(self.base_env, "feature_cols", [])
        )

        # Resolve include_regime_features with backward compatibility for use_regime
        if include_regime_features is None:
            if use_regime is not None:
                include_regime_features = bool(use_regime)
            else:
                include_regime_features = True
        self.include_regime_features = bool(include_regime_features)

        self.regime_mode = str(regime_mode or "enabled").lower()
        if self.regime_mode not in {"enabled", "masked", "disabled"}:
            raise ValueError(
                f"Unsupported regime_mode '{regime_mode}'. Expected enabled, masked, or disabled."
            )
        # When regime is disabled, regime features must be off
        if self.regime_mode == "disabled":
            self.include_regime_features = False

        # Window size for regime feature extraction
        self.regime_window = int(
            regime_window
            or getattr(self.regime_model, "window", 0)
            or max(self.lookback, len(self.regime_feature_cols) or self.lookback)
        )

        # Determine forecast horizons
        if self.regime_model is not None and hasattr(self.regime_model, "horizons"):
            self._horizons = list(self.regime_model.horizons)
        else:
            self._horizons = get_forecast_horizons()
        self.num_horizons = len(self._horizons)

        # Base environment observation shape
        base_shape = getattr(self.base_env.observation_space, "shape", None)
        if base_shape is None:
            raise ValueError("base_env must expose a Box observation_space with a shape.")
        self.base_obs_dim = int(np.prod(base_shape))

        # Regime meta features dimension (8 features per horizon)
        self.regime_meta_dim = 8 * self.num_horizons if self.include_regime_features else 0
        self.meta_dim = self.n_experts * 2 + self.regime_meta_dim
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(self.base_obs_dim + self.meta_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.n_experts)

        # Expert history for meta features
        self._returns = [deque(maxlen=self.lookback) for _ in range(self.n_experts)]
        self._cached_action_signals = np.zeros((self.n_experts,), dtype=np.float32)
        self._latest_regime_summary: Dict[str, Any] = self._empty_regime_summary()
        self._last_base_obs: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = self.base_env.reset(seed=seed, options=options)
        self._clear_history()
        base_obs = self._coerce_base_obs(obs)
        self._last_base_obs = base_obs
        self._refresh_cached_actions(base_obs)
        return self._build_observation(base_obs), dict(info)

    def step(self, action: Any):
        expert_idx = self._coerce_expert_index(action)
        current_obs = self._last_base_obs
        if current_obs is None:
            current_obs = self._coerce_base_obs(self.base_env._get_obs())

        expert = self.experts[expert_idx]
        raw_expert_action = self._call_expert_act(expert, current_obs)
        expert_signal = self._expert_action_to_signal(raw_expert_action)
        base_action = self._signal_to_base_action(expert_signal)

        next_obs, reward, terminated, truncated, info = self.base_env.step(base_action)
        next_base_obs = self._coerce_base_obs(next_obs)
        step_return = float(info.get("step_return", reward))

        self._returns[expert_idx].append(step_return)
        self._last_base_obs = next_base_obs
        self._refresh_cached_actions(next_base_obs)

        allocator_obs = self._build_observation(next_base_obs)
        single_summary = self._single_horizon_summary()
        info = dict(info)
        info.update(
            {
                "allocator_expert_index": expert_idx,
                "allocator_expert_signal": float(expert_signal),
                "allocator_base_action": base_action,
                "allocator_cached_actions": self._cached_action_signals.copy(),
                "allocator_regime_class": single_summary["class_id"],
                "allocator_regime_class_name": single_summary["class_name"],
                "allocator_regime_confidence": single_summary["confidence"],
                "allocator_regime_entropy": single_summary["entropy"],
                "allocator_regime_margin": single_summary["margin"],
                "allocator_regime_expected_direction": single_summary["expected_direction"],
                "allocator_regime_probabilities": single_summary["probabilities"].copy(),
                "allocator_regime_all": self._latest_regime_summary.copy(),
            }
        )

        return allocator_obs, reward, terminated, truncated, info

    def render(self):
        if hasattr(self.base_env, "render"):
            return self.base_env.render()
        return None

    def close(self):
        if hasattr(self.base_env, "close"):
            return self.base_env.close()
        return None

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _clear_history(self) -> None:
        for q in self._returns:
            q.clear()
        self._cached_action_signals.fill(0.0)
        self._latest_regime_summary = self._empty_regime_summary()
        if (
            self.regime_model is not None
            and self.regime_mode == "enabled"
            and hasattr(self.regime_model, "reset_state")
        ):
            self.regime_model.reset_state()

    def _coerce_base_obs(self, obs: Any) -> np.ndarray:
        base_obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        if base_obs.shape[0] != self.base_obs_dim:
            raise ValueError(
                f"Base observation shape mismatch: expected {self.base_obs_dim}, "
                f"got {base_obs.shape[0]}"
            )
        return base_obs

    def _build_observation(self, base_obs: np.ndarray) -> np.ndarray:
        meta = self._build_meta_features()
        obs = np.concatenate([base_obs, meta], axis=0).astype(np.float32)
        if obs.shape[0] != self.observation_space.shape[0]:
            raise ValueError(
                f"Allocator observation shape mismatch: expected {self.observation_space.shape[0]}, "
                f"got {obs.shape[0]}"
            )
        return obs

    def _build_meta_features(self) -> np.ndarray:
        meta: list[float] = []
        for idx in range(self.n_experts):
            recent_return = float(np.mean(self._returns[idx])) if self._returns[idx] else 0.0
            recent_return = float(np.clip(recent_return * 100.0, -10.0, 10.0))
            signal = float(np.clip(self._cached_action_signals[idx], -1.0, 1.0))
            meta.append(recent_return)
            meta.append(signal)
        if self.include_regime_features:
            meta.extend(self._regime_meta_features())
        return np.asarray(meta, dtype=np.float32)

    def _refresh_cached_actions(self, base_obs: np.ndarray) -> None:
        # Update regime summary only when regime features are requested
        if self.include_regime_features:
            self._latest_regime_summary = self._compute_regime_summary()
        else:
            self._latest_regime_summary = self._empty_regime_summary()

        for idx, expert in enumerate(self.experts):
            raw_action = self._call_expert_act(expert, base_obs)
            self._cached_action_signals[idx] = self._expert_action_to_signal(raw_action)

    def _empty_regime_summary(self) -> Dict[str, Any]:
        return {
            "horizons": self._horizons,
            "probabilities": np.zeros((self.num_horizons, 3), dtype=np.float32),
            "class_ids": [0] * self.num_horizons,
            "class_names": ["Sideways"] * self.num_horizons,
            "confidences": [0.0] * self.num_horizons,
            "entropies": [0.0] * self.num_horizons,
            "margins": [0.0] * self.num_horizons,
            "expected_directions": [0.0] * self.num_horizons,
        }

    def _extract_regime_window(self) -> np.ndarray:
        if self.regime_model is None or self.regime_mode != "enabled":
            return np.zeros((self.regime_window, len(self.regime_feature_cols)), dtype=np.float32)

        df = getattr(self.base_env, "df", None)
        current_step = int(getattr(self.base_env, "current_step", 0))
        if df is None or not self.regime_feature_cols:
            return np.zeros((self.regime_window, len(self.regime_feature_cols)), dtype=np.float32)

        end = max(1, current_step)
        start = max(0, end - self.regime_window)
        window_df = df.iloc[start:end]
        if window_df.empty:
            return np.zeros((self.regime_window, len(self.regime_feature_cols)), dtype=np.float32)

        window = window_df[self.regime_feature_cols].to_numpy(dtype=np.float32)
        if window.shape[0] < self.regime_window:
            pad_rows = self.regime_window - window.shape[0]
            pad = np.repeat(window[:1], pad_rows, axis=0)
            window = np.concatenate([pad, window], axis=0)
        return window[-self.regime_window :].astype(np.float32)

    def _compute_regime_summary(self) -> Dict[str, Any]:
        # Only compute when regime is fully enabled and model exists
        if self.regime_model is None or self.regime_mode != "enabled":
            return self._empty_regime_summary()

        window = self._extract_regime_window()

        # Multi-horizon model with predict_all
        if hasattr(self.regime_model, "predict_all"):
            try:
                results = self.regime_model.predict_all(window, smooth=True)
                return {
                    "horizons": self._horizons,
                    "probabilities": np.asarray(results["probabilities"], dtype=np.float32),
                    "class_ids": results["class_ids"],
                    "class_names": results["class_names"],
                    "confidences": results["confidences"],
                    "entropies": results["entropies"],
                    "margins": results["margins"],
                    "expected_directions": results["expected_directions"],
                }
            except Exception:
                return self._empty_regime_summary()

        # Single-horizon model with predict_summary
        try:
            if hasattr(self.regime_model, "predict_summary"):
                summary = self.regime_model.predict_summary(window, smooth=True)
            else:
                if hasattr(self.regime_model, "reset_state"):
                    self.regime_model.reset_state()
                raise AttributeError(
                    "Single-horizon model must implement predict_summary or be multi-horizon."
                )
            probs = np.asarray(summary["probabilities"], dtype=np.float32).reshape(1, 3)
            # If num_horizons == 0, repeat will produce (0,3) → handled gracefully later
            return {
                "horizons": self._horizons,
                "probabilities": np.repeat(probs, self.num_horizons, axis=0) if self.num_horizons > 0 else np.zeros((0, 3), dtype=np.float32),
                "class_ids": [summary["class_id"]] * self.num_horizons,
                "class_names": [summary["class_name"]] * self.num_horizons,
                "confidences": [summary["confidence"]] * self.num_horizons,
                "entropies": [summary["entropy"]] * self.num_horizons,
                "margins": [summary["margin"]] * self.num_horizons,
                "expected_directions": [summary["expected_direction"]] * self.num_horizons,
            }
        except Exception:
            return self._empty_regime_summary()

    def _single_horizon_summary(self) -> Dict[str, Any]:
        summary = self._latest_regime_summary
        idx = 0
        probs = np.asarray(
            summary.get("probabilities", np.zeros((self.num_horizons, 3), dtype=np.float32)),
            dtype=np.float32,
        )
        if probs.shape[0] != self.num_horizons:
            p = np.zeros((3,), dtype=np.float32)
        else:
            p = probs[idx]
        class_id = int(summary.get("class_ids", [0])[idx]) if idx < len(summary.get("class_ids", [])) else 0
        class_name = (
            str(summary.get("class_names", ["Sideways"])[idx])
            if idx < len(summary.get("class_names", []))
            else "Sideways"
        )
        confidence = (
            float(summary.get("confidences", [0.0])[idx])
            if idx < len(summary.get("confidences", []))
            else 0.0
        )
        entropy = (
            float(summary.get("entropies", [0.0])[idx])
            if idx < len(summary.get("entropies", []))
            else 0.0
        )
        margin = (
            float(summary.get("margins", [0.0])[idx])
            if idx < len(summary.get("margins", []))
            else 0.0
        )
        expected_direction = (
            float(summary.get("expected_directions", [0.0])[idx])
            if idx < len(summary.get("expected_directions", []))
            else 0.0
        )
        return {
            "class_id": class_id,
            "class_name": class_name,
            "confidence": confidence,
            "entropy": entropy,
            "margin": margin,
            "expected_direction": expected_direction,
            "probabilities": p,
        }

    def _regime_meta_features(self) -> np.ndarray:
        if not self.include_regime_features:
            return np.zeros((0,), dtype=np.float32)

        summary = self._latest_regime_summary
        probs = np.asarray(
            summary.get("probabilities", np.zeros((self.num_horizons, 3), dtype=np.float32)),
            dtype=np.float32,
        )
        # If shape mismatch (e.g., model changed), fallback to zeros
        if probs.shape[0] != self.num_horizons:
            probs = np.zeros((self.num_horizons, 3), dtype=np.float32)

        features = []
        for h in range(self.num_horizons):
            p = probs[h]
            class_id = int(summary.get("class_ids", [0])[h]) if h < len(summary.get("class_ids", [])) else 0
            confidence = (
                float(summary.get("confidences", [0.0])[h])
                if h < len(summary.get("confidences", []))
                else 0.0
            )
            entropy = float(summary.get("entropies", [0.0])[h]) if h < len(summary.get("entropies", [])) else 0.0
            margin = float(summary.get("margins", [0.0])[h]) if h < len(summary.get("margins", [])) else 0.0
            expected_dir = (
                float(summary.get("expected_directions", [0.0])[h])
                if h < len(summary.get("expected_directions", []))
                else 0.0
            )

            class_scaled = float(np.clip(class_id - 1.0, -1.0, 1.0))
            if confidence <= 0.0:
                class_scaled = 0.0
            confidence = float(np.clip(confidence, 0.0, 1.0))
            entropy = float(np.clip(entropy, 0.0, 1.0))
            margin = float(np.clip(margin, 0.0, 1.0))
            expected_dir = float(np.clip(expected_dir, -1.0, 1.0))

            features.extend(
                [
                    class_scaled,
                    confidence,
                    entropy,
                    margin,
                    float(p[0]),
                    float(p[1]),
                    float(p[2]),
                    expected_dir,
                ]
            )
        return np.asarray(features, dtype=np.float32)

    # ------------------------------------------------------------------
    # Expert handling
    # ------------------------------------------------------------------

    def _coerce_expert_index(self, action: Any) -> int:
        idx = int(np.asarray(action).reshape(-1)[0])
        if idx < 0 or idx >= self.n_experts:
            raise ValueError(f"Allocator action {idx} is out of range for {self.n_experts} experts.")
        return idx

    def _call_expert_act(self, expert: Any, obs: np.ndarray) -> Any:
        act_fn = getattr(expert, "act", None)
        if act_fn is None:
            raise AttributeError(f"Expert {type(expert).__name__} does not implement act().")

        try:
            signature = inspect.signature(act_fn)
        except (TypeError, ValueError):
            signature = None

        if signature is not None and "eval_mode" in signature.parameters:
            try:
                result = act_fn(obs, eval_mode=True)
            except TypeError:
                result = act_fn(obs)
        else:
            result = act_fn(obs)

        if isinstance(result, tuple):
            result = result[0]
        return result

    def _expert_action_to_signal(self, action: Any) -> float:
        if isinstance(action, tuple):
            action = action[0]

        if isinstance(action, np.ndarray):
            flat = action.reshape(-1)
            if flat.size == 0:
                raise ValueError("Expert action array is empty.")
            action = flat[0]

        if isinstance(action, (np.integer, int)):
            mapping = {0: -1.0, 1: 0.0, 2: 1.0}
            discrete = int(action)
            if discrete in mapping:
                return mapping[discrete]
            return float(np.clip(discrete, -1, 1))

        return float(np.clip(float(action), -1.0, 1.0))

    def _signal_to_base_action(self, signal: float) -> Any:
        continuous = bool(getattr(self.base_env, "continuous_actions", False))
        action_space = getattr(self.base_env, "action_space", None)
        if continuous or isinstance(action_space, spaces.Box):
            return np.asarray([signal], dtype=np.float32)

        if signal <= -0.33:
            return 0
        if signal >= 0.33:
            return 2
        return 1