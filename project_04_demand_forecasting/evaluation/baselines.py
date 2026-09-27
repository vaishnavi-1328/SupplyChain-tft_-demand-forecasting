"""
Baseline forecasters for benchmark comparison.
All return (B, horizon) point forecasts.

Baselines:
  - SeasonalNaive:   forecast = same day last week (lag-7)
  - MovingAverage:   forecast = mean of last N days
"""
import torch
import numpy as np
from training.config import TFTConfig
from training.losses import q_risk


class SeasonalNaive:
    """Repeat the value from 7 days ago (same day of week)."""
    def predict(self, observed_past: torch.Tensor, horizon: int) -> torch.Tensor:
        # observed_past: (B, lookback, n_observed)  — log_units_sold is col 0
        past = observed_past[..., 0]  # (B, lookback)
        B, L = past.shape
        preds = []
        for h in range(horizon):
            lag = h + 7   # target = today + h, so look back 7 more
            idx = L - 1 - lag
            idx = max(idx, 0)
            preds.append(past[:, idx])
        return torch.stack(preds, dim=1)  # (B, horizon)


class MovingAverage:
    """Mean of the last `window` days."""
    def __init__(self, window: int = 28):
        self.window = window

    def predict(self, observed_past: torch.Tensor, horizon: int) -> torch.Tensor:
        past = observed_past[..., 0]  # (B, lookback)
        avg  = past[:, -self.window:].mean(dim=1, keepdim=True)  # (B, 1)
        return avg.expand(-1, horizon)                            # (B, horizon)


def benchmark_baselines(loader, cfg: TFTConfig) -> dict:
    """Evaluate all baselines. Returns dict of {name: {p50: ...}} metrics."""
    sn = SeasonalNaive()
    ma = MovingAverage(window=28)

    results = {}
    for name, baseline in [("seasonal_naive", sn), ("moving_avg_28d", ma)]:
        all_preds, all_targets = [], []
        for batch in loader:
            obs  = batch["observed_past"]
            tgt  = batch["target"]
            pred = baseline.predict(obs, cfg.horizon)   # (B, horizon)
            # Wrap as 3-column tensor (same quantile → P10=P50=P90) for q_risk reuse
            pred3 = pred.unsqueeze(-1).expand(-1, -1, len(cfg.quantiles))
            all_preds.append(pred3)
            all_targets.append(tgt)

        preds   = torch.cat(all_preds,   dim=0)
        targets = torch.cat(all_targets, dim=0)
        results[name] = q_risk(preds, targets, cfg.quantiles)

    return results
