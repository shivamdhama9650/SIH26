"""
train_mlp.py
=============
Stage: DAY 2, PART 2 — training loop for the point-based MLP.

MSE loss, Adam optimizer, configurable batch size/epochs/lr, early
stopping on validation loss, saves the best checkpoint. Seeds are set
for reproducibility.
"""

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.mlp_dataset import PointDataset
from src.mlp_model import PointMLP


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_mlp(X_train, Y_train, X_val, Y_val, channels: list,
              n_output_depths: int, config: dict, out_dir: Path):
    """
    config keys (all have sensible defaults if missing):
        seed, hidden_dim, batch_size, epochs, lr, patience
    Returns: (best_model, history) where history = {"train_loss": [...],
    "val_loss": [...], "best_epoch": int, "best_val_loss": float}
    """
    seed = config.get("seed", 42)
    hidden_dim = config.get("hidden_dim", 64)
    batch_size = config.get("batch_size", 256)
    epochs = config.get("epochs", 50)
    lr = config.get("lr", 1e-3)
    patience = config.get("patience", 8)

    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_mlp] Using device: {device}")

    train_loader = DataLoader(PointDataset(X_train, Y_train),
                               batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(PointDataset(X_val, Y_val),
                             batch_size=batch_size, shuffle=False)

    model = PointMLP(n_input_channels=len(channels),
                      n_output_depths=n_output_depths,
                      hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    print(f"[train_mlp] Model: {len(channels)} inputs -> {hidden_dim} -> "
          f"{hidden_dim} -> {n_output_depths} outputs. "
          f"{sum(p.numel() for p in model.parameters())} parameters.")
    print(f"[train_mlp] Training on {len(train_loader.dataset)} samples, "
          f"validating on {len(val_loader.dataset)} samples. "
          f"batch_size={batch_size}, epochs={epochs}, lr={lr}, "
          f"early_stop_patience={patience}")

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_since_improvement = 0

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "mlp_best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_train_seen = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
            n_train_seen += xb.size(0)
        train_loss = running_loss / max(n_train_seen, 1)

        model.eval()
        running_val_loss = 0.0
        n_val_seen = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                running_val_loss += loss.item() * xb.size(0)
                n_val_seen += xb.size(0)
        val_loss = running_val_loss / max(n_val_seen, 1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"[train_mlp] epoch {epoch:3d}/{epochs} | "
              f"train_loss={train_loss:.5f} | val_loss={val_loss:.5f}")

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_since_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "n_input_channels": len(channels),
                "n_output_depths": n_output_depths,
                "hidden_dim": hidden_dim,
                "channels": channels,
                "epoch": epoch,
                "val_loss": val_loss,
                "config": config,
            }, checkpoint_path)
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= patience:
                print(f"[train_mlp] Early stopping at epoch {epoch} "
                      f"(no val improvement for {patience} epochs). "
                      f"Best epoch was {best_epoch} with val_loss="
                      f"{best_val_loss:.5f}.")
                break

    print(f"[train_mlp] Training complete. Best val_loss={best_val_loss:.5f} "
          f"at epoch {best_epoch}. Checkpoint saved -> {checkpoint_path}")

    # reload best checkpoint before returning (not just the last epoch's weights)
    best_state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    best_model = PointMLP(n_input_channels=len(channels),
                           n_output_depths=n_output_depths,
                           hidden_dim=hidden_dim).to(device)
    best_model.load_state_dict(best_state["model_state_dict"])
    best_model.eval()

    history["best_epoch"] = best_epoch
    history["best_val_loss"] = best_val_loss
    return best_model, history, device
