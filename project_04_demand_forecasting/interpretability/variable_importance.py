"""
Variable Importance Analysis (paper Section 6.1).

Extracts softmax weights from Variable Selection Networks and aggregates
across the test set to show which input variables the model relies on most.
"""
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from data.dataset import STATIC_CAT_COLS, STATIC_CONT_COLS, KNOWN_COLS, OBSERVED_COLS
from training.config import TFTConfig


STATIC_VAR_NAMES = STATIC_CAT_COLS + STATIC_CONT_COLS
PAST_VAR_NAMES   = KNOWN_COLS + OBSERVED_COLS
FUTURE_VAR_NAMES = KNOWN_COLS


def collect_weights(model, loader, device: torch.device) -> dict[str, np.ndarray]:
    """Collect VSN weights across entire loader."""
    model.eval()
    all_static, all_past, all_future = [], [], []

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
            all_static.append(out["static_weights"].cpu().numpy())         # (B, n_static)
            # past_vsn_weights: (B, lookback, n_vars) → average over time
            all_past.append(out["past_vsn_weights"].mean(dim=1).cpu().numpy())
            all_future.append(out["fut_vsn_weights"].mean(dim=1).cpu().numpy())

    return {
        "static":  np.concatenate(all_static,  axis=0),
        "past":    np.concatenate(all_past,    axis=0),
        "future":  np.concatenate(all_future,  axis=0),
    }


def plot_variable_importance(weights: dict[str, np.ndarray], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    specs = [
        ("static",  STATIC_VAR_NAMES, "Static Covariate Importance"),
        ("past",    PAST_VAR_NAMES,   "Past Input Variable Importance"),
        ("future",  FUTURE_VAR_NAMES, "Known Future Variable Importance"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (key, var_names, title) in zip(axes, specs):
        w = weights[key]   # (N, n_vars)
        means   = w.mean(axis=0)
        p10     = np.percentile(w, 10, axis=0)
        p90     = np.percentile(w, 90, axis=0)

        # Trim var names for readability
        labels = [v.replace("_enc", "").replace("_norm", "") for v in var_names]

        # Sort by mean descending
        order = np.argsort(means)[::-1]
        means_s = means[order]
        p10_s   = p10[order]
        p90_s   = p90[order]
        labels_s = [labels[i] for i in order]

        y = np.arange(len(labels_s))
        ax.barh(y, means_s, xerr=[means_s - p10_s, p90_s - means_s],
                color="steelblue", ecolor="gray", capsize=3, alpha=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(labels_s, fontsize=9)
        ax.set_xlabel("Mean Selection Weight")
        ax.set_title(title)
        ax.invert_yaxis()

    plt.tight_layout()
    path = os.path.join(output_dir, "variable_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # Also save as CSV
    for key, var_names, _ in specs:
        w = weights[key]
        labels = [v.replace("_enc", "").replace("_norm", "") for v in var_names]
        df = pd.DataFrame({
            "variable": labels,
            "mean_weight": w.mean(axis=0),
            "p10": np.percentile(w, 10, axis=0),
            "p90": np.percentile(w, 90, axis=0),
        }).sort_values("mean_weight", ascending=False)
        df.to_csv(os.path.join(output_dir, f"importance_{key}.csv"), index=False)
