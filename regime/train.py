"""TCN multi‑horizon regime training script."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

# Allow running both as a module and as a script
sys.path.append(str(Path(__file__).resolve().parents[1]))

from regime.config import CONFIG, REGIME_LABEL_NAMES, get_feature_columns, get_forecast_horizons
from regime.dataset import RegimeDataset
from regime.tcn_model import MultiHorizonRegimeTCN


# ---------------------------------------------------------------------------
# Helper: compute class weights
# ---------------------------------------------------------------------------

def _compute_class_weights(
    dataset: RegimeDataset, device: torch.device
) -> Optional[torch.Tensor]:
    """
    Return inverse‑frequency class weights aggregated across all horizons.
    Returns `None` if class weights are disabled in the config.
    """
    if not CONFIG["tcn"].get("use_class_weights", False):
        return None

    # Collect all targets into a single 2D array
    y_list = [dataset[i][1].numpy() for i in range(len(dataset))]
    y_all = np.stack(y_list, axis=0)  # shape: (N_samples, N_horizons)

    class_counts = np.zeros(3, dtype=np.float64)
    for h in range(y_all.shape[1]):          # iterate over horizons
        valid = y_all[:, h] >= 0             # ignore -1 (invalid)
        counts = np.bincount(y_all[valid, h].astype(int), minlength=3)
        class_counts += counts

    class_counts = np.maximum(class_counts, 1)   # avoid division by zero
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * 3.0      # normalise so mean = 1
    return torch.tensor(weights, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _evaluate_per_horizon(
    model: nn.Module,
    loader: DataLoader,
    regime_cols: List[str],
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    """Return per‑horizon accuracy and confusion matrix."""
    model.eval()
    accuracies = {col: [] for col in regime_cols}
    confusions = {col: np.zeros((3, 3), dtype=int) for col in regime_cols}

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)                # (B, num_horizons, 3)
            preds = logits.argmax(dim=-1)    # (B, num_horizons)

            for i, col in enumerate(regime_cols):
                mask = y[:, i] != -1         # ignore invalid samples
                if mask.sum() == 0:
                    continue
                correct = (preds[mask, i] == y[mask, i]).sum().item()
                accuracies[col].append(correct / mask.sum().item())

                y_true = y[mask, i].cpu().numpy()
                y_pred = preds[mask, i].cpu().numpy()
                for t, p in zip(y_true, y_pred):
                    confusions[col][t, p] += 1

    results = {}
    for col in regime_cols:
        results[col] = {
            "accuracy": np.mean(accuracies[col]) if accuracies[col] else 0.0,
            "confusion_matrix": confusions[col].tolist(),
        }
    return results


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", type=str, required=True)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    cfg = CONFIG
    feature_cols = get_feature_columns()
    horizons = get_forecast_horizons()
    regime_cols = [f"regime_h{h}" for h in horizons]
    window = int(cfg["model"]["window"])
    tcn_cfg = cfg["tcn"]

    device = torch.device(args.device)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    if args.data:
        data_path = Path(args.data)
    else:
        data_path = Path(f"data/processed/{args.asset}/train.csv")

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    if data_path.name == "test.csv":
        test_df = pd.read_csv(data_path)
        train_df = pd.read_csv(data_path.with_name("train.csv"))
    else:
        train_df = pd.read_csv(data_path)
        test_df = pd.read_csv(data_path.with_name("test.csv"))

    # Sort by timestamp if present
    for df in (train_df, test_df):
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df.sort_values("timestamp", inplace=True)
            df.reset_index(drop=True, inplace=True)

    print(f"Loaded {args.asset}: train={train_df.shape}, test={test_df.shape}")
    print(f"Features: {len(feature_cols)}, horizons: {horizons}")

    # ------------------------------------------------------------------
    # 2. Build datasets & loaders
    # ------------------------------------------------------------------
    train_dataset = RegimeDataset(train_df, feature_cols, regime_cols, window=window)
    test_dataset = RegimeDataset(test_df, feature_cols, regime_cols, window=window)

    train_loader = DataLoader(
        train_dataset,
        batch_size=tcn_cfg["batch_size"],
        shuffle=True,
        drop_last=True,         # avoid uneven batches that can break LayerNorm
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=tcn_cfg["batch_size"],
        shuffle=False,
        drop_last=False,
    )

    # ------------------------------------------------------------------
    # 3. Model & loss
    # ------------------------------------------------------------------
    model = MultiHorizonRegimeTCN(
        input_size=len(feature_cols),
        num_horizons=len(horizons),
        num_classes=3,
        num_channels=tcn_cfg["num_channels"],
        kernel_size=tcn_cfg["kernel_size"],
        dropout=tcn_cfg["dropout"],
        attention_heads=tcn_cfg["attention_heads"],
        attention_dropout=tcn_cfg["attention_dropout"],
        se_reduction=tcn_cfg["se_reduction"],
        stochastic_depth_prob=tcn_cfg["stochastic_depth_prob"],
        max_seq_len=window + 50,
    ).to(device)

    class_weights = _compute_class_weights(train_dataset, device)

    # Label smoothing requires PyTorch ≥ 1.10 – fall back gracefully
    label_smoothing = tcn_cfg.get("label_smoothing", 0.0)
    try:
        criterion = nn.CrossEntropyLoss(
            ignore_index=-1,
            weight=class_weights,
            label_smoothing=label_smoothing,
        )
    except TypeError:
        print("Warning: label_smoothing not supported in this PyTorch version, ignoring.")
        criterion = nn.CrossEntropyLoss(
            ignore_index=-1,
            weight=class_weights,
        )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=tcn_cfg["learning_rate"],
        weight_decay=tcn_cfg["weight_decay"],
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

    # ------------------------------------------------------------------
    # 4. Training loop
    # ------------------------------------------------------------------
    best_loss = float("inf")
    patience_counter = 0
    checkpoint_dir = Path("checkpoints/regime")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{args.asset}_tcn_multi.pth"

    for epoch in range(1, tcn_cfg["epochs"] + 1):
        # ----- train -----
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            logits = model(x)                     # (B, num_horizons, 3)
            loss = 0.0
            for h in range(len(horizons)):
                loss += criterion(logits[:, h, :], y[:, h])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        # ----- validate -----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                for h in range(len(horizons)):
                    val_loss += criterion(logits[:, h, :], y[:, h]).item()

        scheduler.step(val_loss)

        # ----- checkpoint -----
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1

        print(
            f"Epoch {epoch:3d} | "
            f"Train Loss {train_loss / len(train_loader):.4f} | "
            f"Val Loss {val_loss / len(test_loader):.4f} | "
            f"LR {scheduler.get_last_lr()[0]:.2e}"
        )

        if patience_counter >= tcn_cfg["patience"]:
            print("Early stopping triggered.")
            break

    # ------------------------------------------------------------------
    # 5. Final evaluation
    # ------------------------------------------------------------------
    print("\nLoading best model for evaluation ...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    per_horizon = _evaluate_per_horizon(model, test_loader, regime_cols, device)

    print("\nTest accuracy per horizon:")
    for col, metrics in per_horizon.items():
        print(f"  {col}: {metrics['accuracy'] * 100:.2f}%")
        cm = np.array(metrics["confusion_matrix"])
        # Use readable class names
        cm_df = pd.DataFrame(
            cm,
            index=[REGIME_LABEL_NAMES[i] for i in range(3)],
            columns=[REGIME_LABEL_NAMES[i] for i in range(3)],
        )
        print(cm_df.to_string())
        print()

    # Save summary CSV
    summary = []
    for col, metrics in per_horizon.items():
        summary.append(
            {
                "horizon": col,
                "accuracy": metrics["accuracy"],
                "confusion_matrix": metrics["confusion_matrix"],
            }
        )
    summary_df = pd.DataFrame(summary)
    results_dir = Path("results/regime")
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / f"{args.asset}_tcn_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()