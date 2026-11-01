"""
DAHERL ablation variants.

Available variants:
  - FullDAHERL                     : Complete system (regime + diversity + allocator).
  - NoDiversity                    : Allocator diversity penalty disabled.
  - NoRegime                       : Regime signal removed from allocator observation.
  - RandomAllocator                : Uniform random expert selection at each step.
  - BestExpert                     : Best single expert selected by validation Sharpe.
  - NoExpertDiversity              : Experts trained without diversity; allocator diversity ON.
  - NoDiversityNoExpertDiversity   : Both expert and allocator diversity disabled.
"""

from ablations.full_daherl import FullDAHERL
from ablations.no_diversity import NoDiversity
from ablations.no_regime import NoRegime
from ablations.random_allocator import RandomAllocator
from ablations.best_expert import BestExpert
from ablations.no_expert_diversity import NoExpertDiversity
from ablations.no_diversity_no_expert_diversity import NoDiversityNoExpertDiversity

__all__ = [
    "FullDAHERL",
    "NoDiversity",
    "NoRegime",
    "RandomAllocator",
    "BestExpert",
    "NoExpertDiversity",
    "NoDiversityNoExpertDiversity",
]