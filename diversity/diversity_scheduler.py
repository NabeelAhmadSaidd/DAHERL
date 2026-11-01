"""
Diversity scheduler – adjusts the penalty coefficient over training epochs.
"""

from typing import Optional, List, Tuple


class DiversityScheduler:
    """
    Schedule the diversity penalty coefficient lambda over epochs.

    The scheduler returns a multiplier (lambda_value) that scales the base penalty.
    """

    def __init__(
        self,
        initial_lambda: float = 0.0,
        schedule: Optional[List[Tuple[int, float]]] = None,
    ):
        """
        Args:
            initial_lambda: Starting multiplier (default 0.0).
            schedule: List of (epoch_start, lambda_value) tuples.
                      Example: [(1, 0.0), (15, 0.5), (30, 1.0)]
                      Value applies from epoch_start onward.
                      If None, a default schedule is used:
                      [(1, 0.0), (15, 0.5), (30, 1.0)]
        """
        if schedule is None:
            schedule = [(1, 0.0), (15, 0.5), (30, 1.0)]
        self.schedule = sorted(schedule, key=lambda x: x[0])
        self.current_epoch = 0
        self._lambda = initial_lambda

    def update_epoch(self, epoch: int) -> None:
        """Update lambda based on current epoch."""
        self.current_epoch = epoch
        # Find the highest schedule entry with epoch_start <= current_epoch
        for epoch_start, lambda_val in reversed(self.schedule):
            if epoch >= epoch_start:
                self._lambda = lambda_val
                return

    @property
    def lambda_value(self) -> float:
        return self._lambda