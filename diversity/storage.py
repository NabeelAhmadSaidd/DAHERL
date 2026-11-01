"""Persistent storage for expert trajectories – optional utility."""

import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np

class ExpertTrajectoryStorage:
    """Save and load expert trajectories to/from disk."""

    def __init__(self, storage_dir: str = "storage"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save(self, expert_name: str, trajectory: Dict[str, np.ndarray]) -> None:
        filepath = self.storage_dir / f"{expert_name.lower()}.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(trajectory, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, expert_name: str) -> Optional[Dict[str, np.ndarray]]:
        filepath = self.storage_dir / f"{expert_name.lower()}.pkl"
        if not filepath.exists():
            return None
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def list_available(self) -> List[str]:
        return [p.stem for p in self.storage_dir.glob("*.pkl")]

    def delete(self, expert_name: str) -> bool:
        filepath = self.storage_dir / f"{expert_name.lower()}.pkl"
        if filepath.exists():
            filepath.unlink()
            return True
        return False