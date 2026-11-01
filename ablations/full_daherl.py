"""Full DAHERL ablation."""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

from ablations.base_ablation import BaseAblation
from ablations.runtime import summarize_portfolio


class FullDAHERL(BaseAblation):
    variant_key = "full_daherl"
    display_name = "Full_DAHERL"
    use_allocator = True
    use_random_allocator = False
    use_best_expert = False
    diversity_enabled = True
    regime_mode = "enabled"

    def evaluate(self) -> Tuple[Dict[str, Any], Sequence[float], Dict[str, Any]]:
        if self.allocator is None:
            raise RuntimeError("Allocator is not available. Call train() before evaluation.")

        env = self._make_allocator_env(self.test_env)
        portfolio_values, selection_rows = self._rollout_allocator(env, self.allocator)
        metrics = summarize_portfolio(portfolio_values, periods_per_year=self._periods_per_year())
        return metrics, portfolio_values, {"selection_rows": selection_rows}

