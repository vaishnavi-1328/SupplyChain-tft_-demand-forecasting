"""
Regime / Anomaly Detection using attention patterns (paper Section 6.3).

For each time step t, compute how much the attention pattern deviates
from the average pattern using Bhattacharyya coefficient as similarity.
High deviation → possible regime shift (e.g., holiday surge, disruption).
"""
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from training.config import TFTConfig


def bhattacharyya_distance(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> float:
    """
    Bhattacharyya distance between two distributions p and q.
    D_B = -ln( Σ sqrt(p_i * q_i) )
    Higher = more different.
    """
    bc = np.sum(np.sqrt(np.clip(p, eps, None) * np.clip(q, eps, None)))
    return -np.log(np.clip(bc, eps, None))


def detect_regimes(
    model,
    loader,
    device: torch.device,
    cfg: TFTConfig,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Collect per-sample attention patterns and compute deviation scores.
    Returns a DataFrame with columns: [date, deviation_score].
    """
    model.eval()
    all_attn = []

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
            # attn_weights: (B, horizon, T) → collapse horizon by mean
            attn = out["attn_weights"].cpu().numpy().mean(axis=1)   # (B, T)
            all_attn.append(attn)

    attn_all = np.concatenate(all_attn, axis=0)   # (N, T)

    # Normalize rows to valid distributions
    attn_all = attn_all / (attn_all.sum(axis=1, keepdims=True) + 1e-8)

    # Average (reference) attention pattern
    avg_pattern = attn_all.mean(axis=0)

    # Compute deviation score per sample
    scores = np.array([
        bhattacharyya_distance(avg_pattern, attn_all[i])
        for i in range(len(attn_all))
    ])

    n = min(len(scores), len(dates))
    df = pd.DataFrame({"date": dates[:n], "deviation_score": scores[:n]})
    return df


def plot_regime_detection(df: pd.DataFrame, output_dir: str, sales_df: pd.DataFrame = None) -> None:
    os.makedirs(output_dir, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(14, 5))

    ax1.plot(df["date"], df["deviation_score"], color="darkorange", linewidth=1.2, label="Deviation score")

    # Mark top-10% anomalies
    threshold = np.percentile(df["deviation_score"], 90)
    anomalies = df[df["deviation_score"] >= threshold]
    ax1.scatter(anomalies["date"], anomalies["deviation_score"],
                color="red", s=20, zorder=5, label=f"Top 10% anomalies")
    ax1.axhline(y=threshold, color="red", linestyle="--", alpha=0.4)

    ax1.set_xlabel("Date")
    ax1.set_ylabel("Bhattacharyya Distance from avg. pattern", color="darkorange")
    ax1.set_title("Regime Detection — Attention Pattern Deviation Over Time")
    ax1.legend(loc="upper left")

    if sales_df is not None and "sale_date" in sales_df.columns:
        ax2 = ax1.twinx()
        daily = sales_df.groupby("sale_date")["total_units"].sum().reset_index()
        ax2.plot(daily["sale_date"], daily["total_units"],
                 color="steelblue", alpha=0.3, linewidth=0.8, label="Total units sold")
        ax2.set_ylabel("Units sold (all stores)", color="steelblue")
        ax2.legend(loc="upper right")

    plt.tight_layout()
    path = os.path.join(output_dir, "regime_detection.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
