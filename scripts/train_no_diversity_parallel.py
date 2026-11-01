#!/usr/bin/env python3
"""
Trains all six experts in parallel (3 at a time) WITHOUT diversity penalty.

This script maximises GPU utilisation by overlapping CPU-bound data fetching
and environment stepping with GPU-bound gradient computations across multiple
experts running concurrently on the same device.

Usage:
    python -m scripts.train_no_diversity_parallel --asset BTCUSDT
    python -m scripts.train_no_diversity_parallel --asset EURUSD --parallel 4
    python -m scripts.train_no_diversity_parallel --asset ETHUSDT --parallel 2  # if GPU memory is limited
"""

import argparse
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

# Expert IDs (as defined in experts/train_experts.py)
EXPERT_IDS = [1, 2, 3, 4, 5, 6]
EXPERT_NAMES = {1: "PPO", 2: "SAC", 3: "DDQN", 4: "TQC", 5: "TD3", 6: "PPG"} 


# ----------------------------------------------------------------------------
# Worker function (runs in a separate subprocess)
# ----------------------------------------------------------------------------

def run_expert(
    expert_id: int,
    asset: str,
    config_path: str,
    data_path: Path,
    no_diversity: bool = True,
) -> tuple[int, int]:
    """
    Launch a single expert training process via subprocess.

    Returns:
        (expert_id, return_code)
    """
    cmd = [
        sys.executable,
        "-m",
        "experts.train_experts",
        "--expert",
        str(expert_id),
        "--asset",
        asset,
        "--config",
        config_path,
        "--data",
        str(data_path),
    ]

    if no_diversity:
        cmd.append("--no-diversity")

    print(f"[PID:{os.getpid()}] Starting Expert {expert_id} ({EXPERT_NAMES[expert_id]})")
    print(f"[CMD] {' '.join(cmd)}\n")

    # Run with real‑time output streaming (so you see TensorBoard logs live)
    process = subprocess.Popen(
        cmd,
        stdout=sys.stdout,
        stderr=sys.stderr,
        text=True,
    )
    process.wait()

    if process.returncode != 0:
        print(
            f"\n[PID:{os.getpid()}] ❌ Expert {expert_id} failed with code {process.returncode}\n"
        )
    else:
        print(
            f"\n[PID:{os.getpid()}] ✅ Expert {expert_id} completed successfully.\n"
        )

    return expert_id, process.returncode


# ----------------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train all experts in parallel (no diversity penalty)"
    )
    parser.add_argument(
        "--asset",
        type=str,
        required=True,
        help="Asset name (e.g., BTCUSDT, EURUSD).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experts.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed",
        help="Base directory containing asset subfolders with train.csv.",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=3,
        choices=range(1, 7),
        help="Number of experts to train simultaneously (default: 3). "
             "Lower if you hit GPU OOM; higher if you have spare capacity.",
    )
    args = parser.parse_args()

    # Locate training data
    data_path = Path(args.data_dir) / args.asset / "train.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found at {data_path}")

    print(f"\n{'=' * 80}")
    print(f"🚀 Parallel (No‑Diversity) Training for asset: {args.asset}")
    print(f"   • {len(EXPERT_IDS)} experts will be trained")
    print(f"   • {args.parallel} experts running concurrently on the same GPU")
    print(f"   • Data: {data_path}")
    print(f"{'=' * 80}\n")

    # Submit all experts to the process pool
    with ProcessPoolExecutor(max_workers=args.parallel) as executor:
        futures = []
        for eid in EXPERT_IDS:
            future = executor.submit(
                run_expert,
                expert_id=eid,
                asset=args.asset,
                config_path=args.config,
                data_path=data_path,
                no_diversity=True,  # <-- This is the key difference
            )
            futures.append(future)

        # Track progress
        completed = 0
        total = len(futures)

        for future in as_completed(futures):
            eid, code = future.result()
            completed += 1
            status = "✅ SUCCESS" if code == 0 else f"❌ FAILED (code {code})"
            print(f"\n{'*' * 40}")
            print(f"Expert {eid} ({EXPERT_NAMES[eid]}) finished: {status}")
            print(f"Progress: {completed}/{total}")
            print(f"{'*' * 40}\n")

    print(f"\n{'=' * 80}")
    print(f"✅ All {total} experts have finished training for asset: {args.asset}")
    print(f"   Checkpoints saved in: checkpoints/{args.asset}/")
    print(f"   (No diversity penalty was applied – all experts trained independently)")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()