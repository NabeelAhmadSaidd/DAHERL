"""
Run DAHERL ablations on a single selected asset.

Usage Command:
    python -m experiments.run_ablation --asset EURUSD

    # Only the new expert‑diversity ablations
    python -m experiments.run_ablation --asset EURUSD --variants no_expert_diversity no_diversity_no_expert_diversity

    # Only the allocator‑diversity ablation
    python -m experiments.run_ablation --asset EURUSD --variants no_diversity

    # Only the regime ablation
    python -m experiments.run_ablation --asset EURUSD --variants no_regime

    # Baselines only (random + best expert)
    python -m experiments.run_ablation --asset EURUSD --variants random_allocator best_expert

    # Full system + expert‑diversity ablation
    python -m experiments.run_ablation --asset EURUSD --variants full_daherl no_expert_diversity
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ablations.base_ablation import BaseAblation
from ablations.full_daherl import FullDAHERL
from ablations.no_diversity import NoDiversity
from ablations.no_regime import NoRegime
from ablations.random_allocator import RandomAllocator
from ablations.best_expert import BestExpert
from ablations.no_expert_diversity import NoExpertDiversity
from ablations.no_diversity_no_expert_diversity import NoDiversityNoExpertDiversity
from ablations.runtime import dump_json, load_yaml, resolve_path


VARIANT_REGISTRY = {
    "full_daherl": FullDAHERL,
    "no_diversity": NoDiversity,
    "no_regime": NoRegime,
    "random_allocator": RandomAllocator,
    "best_expert": BestExpert,
    "no_expert_diversity": NoExpertDiversity,
    "no_diversity_no_expert_diversity": NoDiversityNoExpertDiversity,
}

ALL_VARIANTS = list(VARIANT_REGISTRY.keys())


def load_config(config_path: str | Path) -> Dict[str, Any]:
    return load_yaml(config_path)


def _apply_overrides(config: Dict[str, Any], asset: str | None, seed: int | None, output_dir: str | None) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(config))
    cfg.setdefault("experiment", {})
    if asset:
        cfg["experiment"]["asset"] = asset.upper()
    if seed is not None:
        cfg["experiment"]["seed"] = int(seed)
    if output_dir:
        cfg["experiment"]["output_dir"] = output_dir
    return cfg


def run_variant(
    variant_key: str,
    config: Dict[str, Any],
    seed: int,
    eval_only: bool,
) -> Dict[str, Any]:
    variant_cls = VARIANT_REGISTRY[variant_key]
    variant: BaseAblation = variant_cls(config=config, seed=seed)

    t_start = time.time()
    variant.build()
    if not eval_only:
        variant.train()
    metrics = variant.run_evaluation()
    metrics["wall_time_s"] = round(time.time() - t_start, 3)
    metrics["variant_key"] = variant_key
    metrics["output_dir"] = str(variant.output_dir)
    return metrics


def write_summary(results: List[Dict[str, Any]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "comparison.csv"
    df = pd.DataFrame(results)
    if not df.empty:
        sort_col = "sharpe" if "sharpe" in df.columns else None
        if sort_col:
            df = df.sort_values(sort_col, ascending=False, kind="stable")
    df.to_csv(summary_path, index=False)
    dump_json(output_dir / "results.json", {"results": results})
    return summary_path


def print_table(results: List[Dict[str, Any]]) -> None:
    if not results:
        print("No results to display.")
        return

    headers = [
        "Variant",
        "Total Return %",
        "Annualized %",
        "Sharpe",
        "Sortino",
        "Max DD %",
        "Win Rate %",
        "Final PV",
    ]
    widths = [22, 14, 14, 10, 10, 12, 12, 14]

    def row(cols: List[Any]) -> str:
        return "  ".join(str(c).ljust(w) for c, w in zip(cols, widths))

    print("\n" + "=" * 110)
    print("DAHERL ABLATION SUMMARY")
    print("=" * 110)
    print(row(headers))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))

    for result in sorted(results, key=lambda item: item.get("sharpe", float("-inf")), reverse=True):
        print(
            row(
                [
                    result.get("variant", result.get("variant_key", "?")),
                    f"{result.get('total_return_pct', 0.0):.3f}",
                    f"{result.get('annualized_return_pct', 0.0):.3f}",
                    f"{result.get('sharpe', 0.0):.3f}",
                    f"{result.get('sortino', 0.0):.3f}",
                    f"{result.get('max_drawdown_pct', 0.0):.3f}",
                    f"{result.get('win_rate_pct', 0.0):.3f}",
                    f"{result.get('final_portfolio_value', 0.0):.2f}",
                ]
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="DAHERL Ablation Runner")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ablation.yaml",
        help="Path to the unified ablation YAML.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=ALL_VARIANTS,
        default=ALL_VARIANTS,
        help="Which variants to run.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override the seed in the config.")
    parser.add_argument("--asset", type=str, default=None, help="Override the asset in the config.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override the output directory in the config.",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip training and only evaluate with existing checkpoints.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    config = _apply_overrides(config, asset=args.asset, seed=args.seed, output_dir=args.output_dir)
    seed = int(config.get("experiment", {}).get("seed", 42))
    output_dir = resolve_path(config.get("experiment", {}).get("output_dir", "results/ablations"))
    output_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    failures: List[str] = []

    for variant_key in args.variants:
        print(f"\nRunning {variant_key} ...")
        try:
            metrics = run_variant(
                variant_key=variant_key,
                config=config,
                seed=seed,
                eval_only=args.eval_only,
            )
            results.append(metrics)
        except Exception:
            # Print the full traceback to stderr so it's visible immediately
            traceback.print_exc()
            failures.append(variant_key)

    if results:
        summary_path = write_summary(results, output_dir)
        print_table(results)
        print(f"\nComparison written to: {summary_path}")

    if failures:
        print(f"\nFailed variants: {', '.join(failures)}")

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())