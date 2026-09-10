"""
train_ablation.py
====================
Stage: STAGE 5, PART C — trains the ablation's "Center MLP" (Model A).

Deliberately thin: src.train_mlp.train_mlp() is already a fully generic
training loop (MSE loss, Adam, early stopping on validation loss,
checkpointed by best val loss) that only needs X/Y arrays + a channel
list + config - it does not care whether X came from a 1x1 point
sample or a 3x3 patch's center pixel. So this module REUSES it as-is,
just pointed at a distinct output directory
(models/ablation/mlp_best.pt) so it never overwrites Day 2's own
models/mlp_best.pt.
"""

from pathlib import Path

from src.train_mlp import train_mlp


def train_center_mlp(X_train_center, Y_train, X_val_center, Y_val, channels: list,
                      n_output_depths: int, config: dict, out_dir: Path):
    """
    X_*_center: (n_samples, n_channels) - the CNN's own center-pixel
        values (see src.ablation.extract_center_pixel), NOT gathered
        independently, so the exact same samples/normalization the CNN
        used are guaranteed here too.
    config: same shape as CNN_CONFIG/DAY2_CONFIG (seed, hidden_dim,
        batch_size, epochs, lr, patience) - pass the SAME epochs/
        patience/batch_size/lr the CNN used, for a fair comparison.

    Returns (model, history) - same as train_mlp(). Checkpoint saved to
    out_dir/mlp_best.pt (out_dir should be e.g. models/ablation, NOT
    models/, to avoid colliding with Day 2's checkpoint).
    """
    out_dir = Path(out_dir)
    print(f"[train_ablation] Training Center-MLP (Model A, 1x1 context) on "
          f"{X_train_center.shape[0]} samples (same samples the CNN used, "
          f"restricted to the center pixel) -> checkpoint will be saved to "
          f"{out_dir / 'mlp_best.pt'}.")
    return train_mlp(X_train_center, Y_train, X_val_center, Y_val, channels=channels,
                      n_output_depths=n_output_depths, config=config, out_dir=out_dir)
