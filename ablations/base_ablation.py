"""
Shared ablation scaffold for the DAHERL research pipeline.

Corrected version – diversity, regime, checkpoint compatibility, and
variant‑specific allocator checkpoints are fully functional.
Also supports expert‑diversity flag for loading checkpoints from a
"no_diversity" sub‑directory.

Now uses **IQN allocator** instead of PPO.

MODIFICATION: validation and test sets are removed; the training set is used
for all phases (training, validation, evaluation).
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

from ablations.runtime import (
    AssetArtifacts,
    EXPERT_CHECKPOINT_NAMES,
    compute_metrics,
    dump_json,
    load_dataframe,
    load_pickle,
    resolve_asset_artifacts,
    resolve_path,
    seed_everything,
)
from allocator.allocator_agent import IQNAllocator          # ← IQN agent
from allocator.allocator_env import AllocatorEnv
from allocator.train_allocator import load_expert_agent
from allocator.train_allocator import train_allocator as train_allocator_pipeline
from experts.train_experts import get_env_params
from regime.inference import MultiHorizonRegimeClassifier


@dataclass
class AblationRunContext:
    asset: str
    seed: int
    variant_name: str
    output_dir: Path
    artifacts: AssetArtifacts
    config: Dict[str, Any]
    experts_config: Dict[str, Any]
    allocator_config: Dict[str, Any]
    data_config: Dict[str, Any]
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    validation_df: pd.DataFrame
    feature_cols: List[str]
    env_params: Dict[str, Any]
    scaler_metadata: Dict[str, Any]
    regime_metadata: Dict[str, Any]


class BaseAblation(ABC):
    """Base class that owns the actual DAHERL objects used by ablations."""

    variant_key: str = "base"
    display_name: str = "Base"
    use_allocator: bool = True
    use_random_allocator: bool = False
    use_best_expert: bool = False
    diversity_enabled: bool = True           # allocator diversity
    regime_mode: str = "enabled"             # "enabled" | "masked" | "disabled"
    expert_diversity_enabled: bool = True    # whether expert checkpoints were trained with diversity

    def __init__(self, config: Dict[str, Any], seed: int = 42):
        # 1. Store config immediately – this ensures 'config' attribute always exists
        self.config = dict(config)

        # 2. Safely extract the seed, handling tuple/list if the YAML is malformed
        seed_value = seed
        # If seed is the default (42) and not explicitly passed, use config value
        if seed_value == 42:
            seed_value = self.config.get("experiment", {}).get("seed", 42)
        # Convert list/tuple to a scalar integer
        if isinstance(seed_value, (list, tuple)):
            seed_value = seed_value[0] if seed_value else 42
        self.seed = int(seed_value)
        seed_everything(self.seed)

        # 3. Continue with the rest of the initialization
        self.artifacts = resolve_asset_artifacts(self.config)
        self.asset = self.artifacts.asset
        self.asset_type = self.artifacts.asset_type
        self.output_dir = self._resolve_output_dir()

        self.data_config = self._load_subconfig("data_config", "configs/data.yaml")
        self.experts_config = self._load_subconfig("experts_config", "configs/experts.yaml")
        self.allocator_config = self._load_subconfig(
            "allocator_config", "configs/allocator.yaml"
        )

        self.run_context: Optional[AblationRunContext] = None
        self.train_env = None
        self.test_env = None
        self.validation_env = None
        self.allocator_env = None
        self.regime_model = None
        self.experts: List[Any] = []
        self.allocator: Optional[IQNAllocator] = None
        self.best_expert_name: Optional[str] = None
        self.best_expert: Optional[Any] = None
        self._random_selector = None
        self._allocator_checkpoint_loaded = False
        self.is_built = False
    # ------------------------------------------------------------------
    # Variant‑specific allocator checkpoint path
    # ------------------------------------------------------------------

    @property
    def allocator_checkpoint_path(self) -> Path:
        """Return a path unique to this variant, e.g. checkpoints/EURUSD/allocator_full_daherl.pt."""
        return self.artifacts.allocator_checkpoint_path.parent / f"allocator_{self.variant_key}.pt"

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def _load_subconfig(self, key: str, default_path: str) -> Dict[str, Any]:
        path = self.config.get("paths", {}).get(key, default_path)
        with open(resolve_path(path), "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}

    def _resolve_output_dir(self) -> Path:
        root = self.config.get("experiment", {}).get(
            "output_dir",
            self.config.get("output_dir", "results/ablations"),
        )
        return resolve_path(root) / self.display_name

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> None:
        if self.is_built:
            return

        # ---- MODIFICATION: Use training data for all splits ----
        train_df = load_dataframe(self.artifacts.train_path)
        # Use the same data for test and validation (no separate splits)
        test_df = train_df.copy()
        validation_df = train_df.copy()
        # ---------------------------------------------------------

        scaler_metadata = load_pickle(self.artifacts.scaler_path)
        regime_metadata = load_pickle(self.artifacts.regime_metadata_path)
        feature_cols = list(
            self.experts_config.get("ppo", {}).get(
                "features",
                scaler_metadata.get("features")
                or regime_metadata.get("feature_columns")
                or [],
            )
        )
        if not feature_cols:
            raise RuntimeError(
                "Could not resolve feature columns from experts config, scaler metadata, or regime metadata."
            )

        env_params = self._build_env_params()
        self.run_context = AblationRunContext(
            asset=self.asset,
            seed=self.seed,
            variant_name=self.display_name,
            output_dir=self.output_dir,
            artifacts=self.artifacts,
            config=self.config,
            experts_config=self.experts_config,
            allocator_config=self.allocator_config,
            data_config=self.data_config,
            train_df=train_df,
            test_df=test_df,
            validation_df=validation_df,
            feature_cols=feature_cols,
            env_params=env_params,
            scaler_metadata=scaler_metadata,
            regime_metadata=regime_metadata,
        )

        # All environments now use the training data
        self.train_env = self._make_base_env(train_df, continuous_actions=True)
        self.test_env = self._make_base_env(train_df, continuous_actions=True)
        self.validation_env = self._make_base_env(train_df, continuous_actions=True)

        if self.use_allocator:
            self._build_regime_model()
            self._build_expert_pool()
            self._build_allocator()
        elif self.use_best_expert:
            self._build_expert_pool()
        elif self.use_random_allocator:
            self._build_regime_model()
            self._build_expert_pool()

        self.is_built = True
        self._save_manifest(stage="built")

    def _build_env_params(self) -> Dict[str, Any]:
        asset_type = self.asset_type
        params = get_env_params(asset_type, expert_id=1)
        evaluation_cfg = self.config.get("evaluation", {})
        env_cfg = evaluation_cfg.get("env", {})

        if "transaction_costs" in env_cfg:
            tx_cfg = env_cfg["transaction_costs"]
            params["commission_rate"] = tx_cfg.get("commission_rate", params["commission_rate"])
            params["spread_rate"] = tx_cfg.get("spread_rate", params["spread_rate"])
            params["slippage_rate"] = tx_cfg.get("slippage_rate", params["slippage_rate"])

        for key in (
            "contract_size",
            "commission_rate",
            "spread_rate",
            "slippage_rate",
            "leverage",
            "maintenance_margin_ratio",
            "initial_balance",
            "max_position_fraction",
            "window_size",
            "reward_scale",
            "clip_reward",
            "continuous_actions",
            "normalize_obs",
            "reward_lookback",
        ):
            if key in env_cfg:
                params[key] = env_cfg[key]

        if "continuous_actions" not in params:
            params["continuous_actions"] = False
        params["continuous_actions"] = bool(params["continuous_actions"])
        return params

    def _build_validation_split(self, train_df: pd.DataFrame) -> pd.DataFrame:
        # This method is no longer used but kept for compatibility; it returns the full train set.
        return train_df.copy()

    def _make_base_env(self, df: pd.DataFrame, continuous_actions: bool):
        params = dict(self.run_context.env_params if self.run_context else self._build_env_params())
        params["continuous_actions"] = bool(continuous_actions)
        return self._instantiate_env(df, params)

    def _make_single_expert_env(self, df: pd.DataFrame, expert_name: str):
        params = dict(self.run_context.env_params if self.run_context else self._build_env_params())
        params["continuous_actions"] = self._expert_is_continuous(expert_name)
        expert_id = self._expert_name_to_id(expert_name)
        return self._instantiate_env(df, params, expert_id=expert_id)

    def _instantiate_env(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        expert_id: int = 1,
    ):
        from environment.trading_env import TradingEnv

        feature_cols = self.run_context.feature_cols if self.run_context else list(
            self.experts_config.get("ppo", {}).get("features", [])
        )
        return TradingEnv(
            df=df,
            feature_cols=feature_cols,
            expert_id=expert_id,
            **params,
        )

    @staticmethod
    def _expert_name_to_id(expert_name: str) -> int:
        mapping = {
            "PPO": 1,
            "SAC": 2,
            "DDQN": 3,
            "TD3": 4,
            "TQC": 5,
            "PPG": 6,
        }
        return mapping[expert_name.upper()]

    @staticmethod
    def _expert_is_continuous(expert_name: str) -> bool:
        return expert_name.upper() in {"SAC", "TD3", "TQC"}

    # ------------------------------------------------------------------
    # Regime model
    # ------------------------------------------------------------------

    def _build_regime_model(self) -> None:
        if self.regime_mode == "disabled":
            self.regime_model = None
            return

        if not self.artifacts.regime_checkpoint_path.exists():
            self.regime_model = None
            return

        regime_cfg = self.config.get("regime", {})
        self.regime_model = MultiHorizonRegimeClassifier(
            self.artifacts.regime_checkpoint_path,
            feature_cols=list(
                regime_cfg.get(
                    "feature_cols",
                    self.run_context.regime_metadata.get("feature_columns", [])
                    if self.run_context
                    else [],
                )
            )
            or None,
            window=int(regime_cfg.get("window", regime_cfg.get("model_window", 30))),
            horizons=list(
                regime_cfg.get(
                    "horizons",
                    self.run_context.regime_metadata.get("horizons", [])
                    if self.run_context
                    else [],
                )
            )
            or None,
        )

    # ------------------------------------------------------------------
    # Expert pool
    # ------------------------------------------------------------------

    def _build_expert_pool(self) -> None:
        if self.run_context is None:
            raise RuntimeError("Build context has not been initialized.")

        base_env = self._make_base_env(self.run_context.train_df, continuous_actions=True)

        expert_root = self.artifacts.expert_checkpoint_dir / self.asset
        if self.expert_diversity_enabled:
            expert_root = expert_root / "diversity"  
        else:
            expert_root = expert_root / "no_diversity" 

        experts: List[Any] = []
        missing: List[str] = []
        for expert_name in ("PPO", "SAC", "DDQN", "TD3", "TQC", "PPG"):
            ckpt = expert_root / expert_name / EXPERT_CHECKPOINT_NAMES[expert_name]   # ← subfolder included
            if not ckpt.exists():
                missing.append(f"{expert_name}: {ckpt}")
                continue
            expert = load_expert_agent(
                expert_name,
                ckpt,
                env_dummy=base_env,
                device=self._device_name(),
                expert_cfg=self.experts_config,
            )
            expert.eval() if hasattr(expert, "eval") else None
            experts.append(expert)

        if missing:
            raise FileNotFoundError(
                "Missing expert checkpoints for the selected asset:\n" + "\n".join(missing)
            )

        self.experts = experts

    # ------------------------------------------------------------------
    # Allocator environment & policy (IQN instead of PPO)
    # ------------------------------------------------------------------

    def _build_allocator(self) -> None:
        if self.run_context is None:
            raise RuntimeError("Build context has not been initialized.")

        lookback = int(self.config.get("allocator", {}).get("lookback", 20))
        regime_model = self.regime_model
        if self.regime_mode == "masked":
            regime_model = None

        self.allocator_env = AllocatorEnv(
            base_env=self.train_env,
            experts=self.experts,
            lookback=lookback,
            regime_model=regime_model,
            regime_feature_cols=self.run_context.regime_metadata.get("feature_columns", []),
            regime_window=int(self.run_context.regime_metadata.get("label_window", 30)),
            include_regime_features = (self.regime_mode == "enabled"),
            regime_mode=self.regime_mode,
        )

        alloc_cfg = self.allocator_config.get("allocator", {})
        obs_dim = int(self.allocator_env.observation_space.shape[0])

        ckpt_path = self.allocator_checkpoint_path   # variant‑specific path

        # ------------------------------------------------------------------
        # Checkpoint compatibility – now handles IQN save format
        # ------------------------------------------------------------------
        load_checkpoint = False
        if ckpt_path.exists():
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu")
                saved_dim = None
                if isinstance(ckpt, dict):
                    # IQN stores hyperparameters in a 'hyperparams' dict
                    hyper = ckpt.get("hyperparams", {})
                    if isinstance(hyper, dict) and "state_dim" in hyper:
                        saved_dim = int(hyper["state_dim"])
                    else:
                        # fallback for older formats
                        for key in ("state_dim", "input_dim"):
                            if key in ckpt:
                                saved_dim = int(ckpt[key])
                                break
                        if saved_dim is None:
                            state = ckpt.get("network") or ckpt.get("network_state_dict") or ckpt.get("q_network")
                            if isinstance(state, dict):
                                first_key = next(iter(state.keys()), "")
                                if first_key.endswith(".weight"):
                                    saved_dim = int(state[first_key].shape[1])
                if saved_dim is not None and saved_dim != obs_dim:
                    print(
                        f"[WARNING] Allocator checkpoint dimension {saved_dim} "
                        f"differs from current env dim {obs_dim}. Retraining from scratch."
                    )
                    ckpt_path.unlink()
                else:
                    load_checkpoint = True
            except Exception:
                load_checkpoint = True  # if loading fails, still try

        # Build IQN allocator with IQN‑specific parameters
        self.allocator = IQNAllocator(
            state_dim=obs_dim,
            action_dim=self.allocator_env.action_space.n,
            lr=alloc_cfg.get("lr", 3e-4),
            gamma=alloc_cfg.get("gamma", 0.995),
            buffer_size=alloc_cfg.get("buffer_size", 30000),
            batch_size=alloc_cfg.get("batch_size", 128),
            hidden_dim=alloc_cfg.get("hidden_dim", 128),
            n_quantiles=alloc_cfg.get("n_quantiles", 32),
            embedding_dim=alloc_cfg.get("embedding_dim", 64),
            num_layers=alloc_cfg.get("num_layers", 2),
            activation=alloc_cfg.get("activation", "relu"),
            target_update_freq=alloc_cfg.get("target_update_freq", 2000),
            epsilon_start=alloc_cfg.get("epsilon_start", 1.0),
            epsilon_end=alloc_cfg.get("epsilon_end", 0.05),
            epsilon_decay=alloc_cfg.get("epsilon_decay", 400000),
            risk_tau=alloc_cfg.get("risk_tau", 0.5),
            log_dir=None,          # no TensorBoard logging from here
            device=self._device_name(),
        )

        if load_checkpoint:
            try:
                self.allocator.load(ckpt_path)
                self._allocator_checkpoint_loaded = True
                print(f"Loaded IQN allocator checkpoint from {ckpt_path}")
            except Exception as e:
                print(f"Failed to load IQN allocator checkpoint: {e}. Will train from scratch.")
                self._allocator_checkpoint_loaded = False
        else:
            self._allocator_checkpoint_loaded = False

    def _make_allocator_env(self, base_env):
        """Create a fresh AllocatorEnv for evaluation, respecting regime/diversity settings."""
        if self.run_context is None:
            raise RuntimeError("Build context has not been initialized.")
        lookback = int(self.config.get("allocator", {}).get("lookback", 20))
        regime_model = self.regime_model
        if self.regime_mode == "masked":
            regime_model = None
        return AllocatorEnv(
            base_env=base_env,
            experts=self.experts,
            lookback=lookback,
            regime_model=regime_model,
            regime_feature_cols=self.run_context.regime_metadata.get("feature_columns", []),
            regime_window=int(self.run_context.regime_metadata.get("label_window", 30)),
            include_regime_features = (self.regime_mode == "enabled"),
            regime_mode=self.regime_mode,
        )

    def _device_name(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _save_manifest(self, stage: str) -> None:
        manifest = {
            "asset": self.asset,
            "asset_type": self.asset_type,
            "variant": self.display_name,
            "seed": self.seed,
            "stage": stage,
            "use_allocator": self.use_allocator,
            "use_random_allocator": self.use_random_allocator,
            "use_best_expert": self.use_best_expert,
            "diversity_enabled": self.diversity_enabled,
            "regime_mode": self.regime_mode,
            "expert_diversity_enabled": self.expert_diversity_enabled,
            "artifacts": {
                "train": str(self.artifacts.train_path),
                "test": str(self.artifacts.train_path),      # now same as train
                "validation": str(self.artifacts.train_path), # now same as train
                "scaler": str(self.artifacts.scaler_path),
                "regime_metadata": str(self.artifacts.regime_metadata_path),
                "regime_checkpoint": str(self.artifacts.regime_checkpoint_path),
                "allocator_checkpoint": str(self.allocator_checkpoint_path),
            },
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        dump_json(self.output_dir / "manifest.json", manifest)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self) -> None:
        if not self.is_built:
            self.build()
        if self.use_allocator and not self._allocator_checkpoint_loaded:
            self._train_allocator_if_needed()
        self._save_manifest(stage="trained")

    def _train_allocator_if_needed(self) -> None:
        if self.run_context is None:
            raise RuntimeError("Build context has not been initialized.")
        should_train = bool(
            self.config.get("execution", {}).get("train_allocator_if_missing", True)
        )
        if not should_train:
            raise FileNotFoundError(
                f"Allocator checkpoint missing for {self.asset}: {self.allocator_checkpoint_path}"
            )

        alloc_cfg = copy.deepcopy(self.allocator_config)
        alloc_cfg["diversity_enabled"] = self.diversity_enabled
        alloc_cfg["regime_mode"] = self.regime_mode

        experts_cfg = self.experts_config
        device = self._device_name()

        regime_model_path = str(self.artifacts.regime_checkpoint_path) if self.regime_mode == "enabled" else None

        expert_subfolder = "diversity" if self.expert_diversity_enabled else "no_diversity"

        train_allocator_pipeline(
            cfg=alloc_cfg,
            experts_cfg=experts_cfg,
            df=self.run_context.train_df,
            feature_cols=self.run_context.feature_cols,
            asset=self.asset,
            asset_type=self.asset_type,
            device=device,
            save_path=self.allocator_checkpoint_path,
            regime_model_path=regime_model_path,
            resume=False,
            expert_subfolder=expert_subfolder,
            include_regime_features=(self.regime_mode == "enabled"),   # NEW
            diversity_enabled=self.diversity_enabled,                  # NEW
        )

        self._allocator_checkpoint_loaded = False
        self._build_allocator()
    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def run_evaluation(self) -> Dict[str, Any]:
        if not self.is_built:
            self.build()
        metrics, portfolio_values, details = self.evaluate()
        metrics = dict(metrics)
        metrics.update(
            {
                "asset": self.asset,
                "variant": self.display_name,
                "seed": self.seed,
                "use_allocator": self.use_allocator,
                "use_random_allocator": self.use_random_allocator,
                "use_best_expert": self.use_best_expert,
                "diversity_enabled": self.diversity_enabled,
                "regime_mode": self.regime_mode,
                "expert_diversity_enabled": self.expert_diversity_enabled,
                "num_points": len(portfolio_values),
            }
        )
        self._persist_results(metrics, portfolio_values, details)
        return metrics

    @abstractmethod
    def evaluate(self) -> Tuple[Dict[str, Any], Sequence[float], Dict[str, Any]]:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared rollouts
    # ------------------------------------------------------------------

    def _rollout_allocator(self, env, allocator: IQNAllocator) -> Tuple[List[float], List[Dict[str, Any]]]:
        obs, _ = env.reset(seed=self.seed)
        values = [float(env.base_env.initial_balance)]
        rows: List[Dict[str, Any]] = []
        done = False
        step = 0
        while not done:
            action = allocator.act(obs, deterministic=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            portfolio_value = float(info.get("equity", info.get("portfolio_value", values[-1])))
            values.append(portfolio_value)
            rows.append(
                {
                    "step": step,
                    "selected_expert": int(action),
                    "reward": float(reward),
                    "portfolio_value": portfolio_value,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
            )
            obs = next_obs
            step += 1
        return values, rows

    def _rollout_random_allocator(self, env) -> Tuple[List[float], List[Dict[str, Any]]]:
        import random as _random

        rng = _random.Random(self.seed)
        obs, _ = env.reset(seed=self.seed)
        values = [float(env.base_env.initial_balance)]
        rows: List[Dict[str, Any]] = []
        done = False
        step = 0
        while not done:
            action = rng.randrange(len(self.experts))
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            portfolio_value = float(info.get("equity", info.get("portfolio_value", values[-1])))
            values.append(portfolio_value)
            rows.append(
                {
                    "step": step,
                    "selected_expert": int(action),
                    "reward": float(reward),
                    "portfolio_value": portfolio_value,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
            )
            obs = next_obs
            step += 1
        return values, rows

    def _rollout_single_expert(
        self,
        expert: Any,
        env,
    ) -> Tuple[List[float], List[Dict[str, Any]]]:
        obs, _ = env.reset(seed=self.seed)
        values = [float(env.initial_balance)]
        rows: List[Dict[str, Any]] = []
        done = False
        step = 0
        while not done:
            if hasattr(expert, "act"):
                try:
                    action = expert.act(obs, eval_mode=True)
                except TypeError:
                    action = expert.act(obs)
            else:
                raise AttributeError(f"Expert {type(expert).__name__} does not implement act().")

            # ---------------------------------------------------------------
            # FIX: some continuous agents return a tuple (action, info)
            if isinstance(action, tuple):
                action = action[0]
            # ---------------------------------------------------------------

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            portfolio_value = float(info.get("equity", info.get("portfolio_value", values[-1])))
            values.append(portfolio_value)
            rows.append(
                {
                    "step": step,
                    "action": action if isinstance(action, (int, float, str)) else np.asarray(action).tolist(),
                    "reward": float(reward),
                    "portfolio_value": portfolio_value,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
            )
            obs = next_obs
            step += 1
        return values, rows

    def _persist_results(
        self,
        metrics: Dict[str, Any],
        portfolio_values: Sequence[float],
        details: Dict[str, Any],
    ) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        dump_json(self.output_dir / "metrics.json", metrics)
        curve = pd.DataFrame(
            {
                "step": np.arange(len(portfolio_values), dtype=np.int64),
                "portfolio_value": np.asarray(portfolio_values, dtype=np.float64),
            }
        )
        curve.to_csv(self.output_dir / "portfolio_curve.csv", index=False)
        if details.get("selection_rows"):
            pd.DataFrame(details["selection_rows"]).to_csv(
                self.output_dir / "selection_log.csv", index=False
            )

    def _evaluate_all_experts(self) -> List[Dict[str, Any]]:
        if self.run_context is None:
            raise RuntimeError("Build context has not been initialized.")

        summary: List[Dict[str, Any]] = []
        for expert_name, expert in zip(
            ("PPO", "SAC", "DDQN", "TD3", "TQC", "PPG"),
            self.experts,
        ):
            # Use the training data for expert evaluation (no separate validation set)
            env = self._make_single_expert_env(self.run_context.train_df, expert_name)
            values, _ = self._rollout_single_expert(expert, env)
            metrics = compute_metrics(
                values,
                periods_per_year=self._periods_per_year(),
            )
            summary.append(
                {
                    "expert_name": expert_name,
                    **metrics,
                }
            )
        return summary

    def _periods_per_year(self) -> int:
        return int(self.config.get("evaluation", {}).get("annualization_periods", 8760))

    def _select_best_expert(self) -> Tuple[str, Any, List[Dict[str, Any]]]:
        summaries = self._evaluate_all_experts()
        if not summaries:
            raise RuntimeError("No experts available for selection.")
        best = max(summaries, key=lambda row: row.get("sharpe", float("-inf")))
        best_name = str(best["expert_name"])
        best_agent = self.experts[self._expert_name_to_id(best_name) - 1]
        self.best_expert_name = best_name
        self.best_expert = best_agent
        return best_name, best_agent, summaries


def summarize_portfolio(portfolio_values: Sequence[float], periods_per_year: int = 8760) -> Dict[str, Any]:
    return compute_metrics(portfolio_values, periods_per_year=periods_per_year)