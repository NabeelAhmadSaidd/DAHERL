"""Random Allocator ablation."""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

from ablations.base_ablation import BaseAblation
from ablations.runtime import summarize_portfolio


class RandomAllocator(BaseAblation):
    variant_key = "random_allocator"
    display_name = "Random_Allocator"
    use_allocator = False
    use_random_allocator = True
    use_best_expert = False
    diversity_enabled = True
    regime_mode = "enabled"

    def evaluate(self) -> Tuple[Dict[str, Any], Sequence[float], Dict[str, Any]]:
        env = self._make_allocator_env(self.test_env)
        portfolio_values, selection_rows = self._rollout_random_allocator(env)
        metrics = summarize_portfolio(portfolio_values, periods_per_year=self._periods_per_year())
        return metrics, portfolio_values, {"selection_rows": selection_rows}

