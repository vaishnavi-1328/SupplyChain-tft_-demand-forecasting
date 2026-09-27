"""
Temporal Attention Pattern Analysis (paper Section 6.2).

Aggregates attention weights across the test set and visualizes
which historical time steps the model attends to when forecasting.
Expected finding: peaks at lag-7 (weekly seasonality) and lag-1.
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from training.config import TFTConfig


def collect_attention(model, loader, device: torch.device, cfg: TFTConfig) -> np.ndarray:
    """
    Returns aggregated attention weights: (horizon, lookback+horizon)
    Averaged over all batches and samples.
    """
    model.eval()
    all_weights = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(
                batch["static_cat"],
                batch["static_cont"],
                batch["known_past"],
                batch["known_future"],
                batch["observed_past"],
            )
            # attn_weights: (B, horizon, lookback+horizon)
            all_weights.append(out["attn_weights"].cpu().numpy())

    stacked = np.concatenate(all_weights, axis=0)  # (N, horizon, T)
    return stacked.mean(axis=0)                     # (horizon, T)


def plot_attention_heatmap(avg_attn: np.ndarray, cfg: TFTConfig, output_dir: str) -> None:
    """
    Heatmap: rows = forecast horizon step, cols = lookback time steps.
    Brighter = more attention.
    """
    os.makedirs(output_dir, exist_ok=True)

    T_total   = avg_attn.shape[1]
    lookback  = cfg.lookback
    horizon   = cfg.horizon

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Full attention heatmap
    ax = axes[0]
    im = ax.imshow(avg_attn, aspect="auto", cmap="viridis", origin="upper")
    ax.set_xlabel("Time position (past ← | → future)")
    ax.set_ylabel("Forecast horizon (days ahead)")
    ax.set_title("Average Temporal Attention Weights")
    # Mark boundary between past and future
    ax.axvline(x=lookback - 0.5, color="red", linestyle="--", linewidth=1.5, label="now")
    ax.legend(fontsize=8)
    plt.colorbar(im, ax=ax)

    # Average attention over past positions only (collapsed over horizon)
    ax2 = axes[1]
    past_attn = avg_attn[:, :lookback].mean(axis=0)  # (lookback,)
    lags = np.arange(lookback, 0, -1)  # lag-1 to lag-90

    ax2.plot(lags, past_attn, color="steelblue", linewidth=1.5)
    ax2.set_xlabel("Lag (days)")
    ax2.set_ylabel("Mean Attention Weight")
    ax2.set_title("Attention by Lag (Past Input)")
    ax2.invert_xaxis()  # lag-1 on right, lag-90 on left

    # Annotate lag-7 and lag-14 peaks
    for lag_mark in [7, 14, 21]:
        if lag_mark <= lookback:
            idx = lookback - lag_mark
            ax2.axvline(x=lag_mark, color="red", linestyle="--", alpha=0.5,
                        label=f"lag-{lag_mark}")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, "temporal_attention.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
