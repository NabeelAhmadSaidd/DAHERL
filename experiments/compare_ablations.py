"""
Build and print the ablation comparison table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ablations.runtime import dump_json, resolve_path


def _load_variant_metrics(root: Path) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for path in sorted(root.glob("*/metrics.json")):
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        payload.setdefault("variant", path.parent.name)
        payload.setdefault("output_dir", str(path.parent))
        results.append(payload)
    return results


def build_comparison_csv(root: Path) -> Path:
    results = _load_variant_metrics(root)
    if not results:
        raise FileNotFoundError(f"No per-variant metrics found in {root}")

    df = pd.DataFrame(results)
    if "sharpe" in df.columns:
        df = df.sort_values("sharpe", ascending=False, kind="stable")
    output_path = root / "comparison.csv"
    df.to_csv(output_path, index=False)
    return output_path


def print_table(df: pd.DataFrame) -> None:
    cols = [
        "variant",
        "total_return_pct",
        "annualized_return_pct",
        "sharpe",
        "sortino",
        "max_drawdown_pct",
        "win_rate_pct",
        "final_portfolio_value",
    ]
    available = [c for c in cols if c in df.columns]
    if not available:
        print("No comparable metrics found.")
        return

    widths = [22, 15, 18, 10, 10, 14, 12, 16]

    def row(values: List[Any]) -> str:
        return "  ".join(str(v).ljust(w) for v, w in zip(values, widths))

    print("\n" + "=" * 110)
    print("DAHERL ABLATION COMPARISON")
    print("=" * 110)
    print(row([
        "Variant",
        "Total Return %",
        "Annualized %",
        "Sharpe",
        "Sortino",
        "Max DD %",
        "Win Rate %",
        "Final PV",
    ]))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))

    for _, record in df.iterrows():
        print(
            row(
                [
                    record.get("variant", "?"),
                    f"{float(record.get('total_return_pct', 0.0)):.3f}",
                    f"{float(record.get('annualized_return_pct', 0.0)):.3f}",
                    f"{float(record.get('sharpe', 0.0)):.3f}",
                    f"{float(record.get('sortino', 0.0)):.3f}",
                    f"{float(record.get('max_drawdown_pct', 0.0)):.3f}",
                    f"{float(record.get('win_rate_pct', 0.0)):.3f}",
                    f"{float(record.get('final_portfolio_value', 0.0)):.2f}",
                ]
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare DAHERL ablations.")
    parser.add_argument(
        "--root",
        type=str,
        default="results/ablations",
        help="Root directory containing per-variant subdirectories.",
    )
    args = parser.parse_args()

    root = resolve_path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    comparison_path = build_comparison_csv(root)
    df = pd.read_csv(comparison_path)
    if "sharpe" in df.columns:
        df = df.sort_values("sharpe", ascending=False, kind="stable")

    print_table(df)
    print(f"\ncomparison.csv written to: {comparison_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())