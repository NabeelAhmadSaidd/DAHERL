# DAHERL

**DAHERL (Diversity-Aware Hierarchical Ensemble Reinforcement Learning)** is a hierarchical ensemble reinforcement learning framework for robust financial trading in non-stationary markets. The framework combines multiple heterogeneous RL experts with an IQN-based allocator that dynamically selects the most suitable expert while encouraging behavioral diversity.

## Features

- Hierarchical ensemble reinforcement learning
- IQN-based adaptive expert allocator
- Six RL experts: PPO, PPG, SAC, TD3, TQC, and DDQN
- Diversity-aware expert selection
- Regime-aware trading using a TCN model
- Cryptocurrency and Forex support
- Modular and reproducible training pipeline
- Comprehensive ablation studies

## Repository Structure

```
experts/          # RL expert implementations
allocator/        # IQN allocator
diversity/        # Diversity regularization
regime/           # Regime detection model
environment/      # Trading environment
preprocessing/    # Data preprocessing
ablations/        # Ablation experiments
configs/          # YAML configuration files
scripts/          # Training scripts
DAHERL-Paper/     # Paper and figures
```

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the experts:

```bash
python experts/train_experts.py
```

Train the allocator:

```bash
python allocator/train_allocator.py
```

Run ablation experiments:

```bash
python experiments/run_ablation.py
```

## Datasets

Experiments are conducted on four assets:

- BTCUSDT
- ETHUSDT
- EURUSD
- USDJPY