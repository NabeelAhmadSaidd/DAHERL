"""Best Expert ablation – corrected to handle continuous‑action experts and DataFrame usage."""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import numpy as np

from ablations.base_ablation import BaseAblation
from ablations.runtime import compute_metrics, summarize_portfolio


class BestExpert(BaseAblation):
    variant_key = "best_expert"
    display_name = "Best_Expert"
    use_allocator = False
    use_random_allocator = False
    use_best_expert = True
    diversity_enabled = False
    regime_mode = "disabled"

    def train(self) -> None:
        if not self.is_built:
            self.build()
        best_name, best_agent, validation_scores = self._select_best_expert()
        self.best_expert_name = best_name
        self.best_expert = best_agent
        self._save_manifest(stage="trained")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        from ablations.runtime import dump_json

        dump_json(
            self.output_dir / "validation_scores.json",
            {
                "selected_expert": best_name,
                "scores": validation_scores,
            },
        )

    def evaluate(self) -> Tuple[Dict[str, Any], Sequence[float], Dict[str, Any]]:
        if self.best_expert is None or self.best_expert_name is None:
            best_name, best_agent, validation_scores = self._select_best_expert()
            self.best_expert_name = best_name
            self.best_expert = best_agent
            self.output_dir.mkdir(parents=True, exist_ok=True)
            from ablations.runtime import dump_json
            dump_json(
                self.output_dir / "validation_scores.json",
                {"selected_expert": best_name, "scores": validation_scores},
            )

        # FIXED: pass the test DataFrame, not the environment
        env = self._make_single_expert_env(self.run_context.test_df, self.best_expert_name)
        portfolio_values, rollout_rows = self._rollout_single_expert(self.best_expert, env)
        metrics = summarize_portfolio(
            portfolio_values, periods_per_year=self._periods_per_year()
        )
        return metrics, portfolio_values, {"selection_rows": rollout_rows}