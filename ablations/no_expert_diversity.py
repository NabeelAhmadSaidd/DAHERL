"""NoExpertDiversity ablation – experts trained without diversity, allocator diversity remains ON."""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

from ablations.base_ablation import BaseAblation
from ablations.runtime import summarize_portfolio


class NoExpertDiversity(BaseAblation):
    variant_key = "no_expert_diversity"
    display_name = "NoExpertDiversity"
    use_allocator = True
    diversity_enabled = True          # allocator diversity ON
    regime_mode = "enabled"
    expert_diversity_enabled = False  # load non-diverse checkpoints

    def evaluate(self) -> Tuple[Dict[str, Any], Sequence[float], Dict[str, Any]]:
        if self.allocator is None:
            raise RuntimeError("Allocator is not available. Call train() before evaluation.")

        env = self._make_allocator_env(self.test_env)
        portfolio_values, selection_rows = self._rollout_allocator(env, self.allocator)
        metrics = summarize_portfolio(portfolio_values, periods_per_year=self._periods_per_year())
        return metrics, portfolio_values, {"selection_rows": selection_rows}