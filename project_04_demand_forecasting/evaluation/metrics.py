"""
Evaluation metrics for TFT forecasts.
"""
import numpy as np
import torch
from training.losses import q_risk
from training.config import TFTConfig


def evaluate_model(model, loader, cfg: TFTConfig, device: torch.device) -> dict:
    """Run full evaluation pass. Returns q-Risk per quantile."""
    model.eval()
    all_preds, all_targets = [], []

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
            all_preds.append(out["quantile_preds"].cpu())
            all_targets.append(batch["target"].cpu())

    preds   = torch.cat(all_preds,   dim=0)  # (N, horizon, Q)
    targets = torch.cat(all_targets, dim=0)  # (N, horizon)

    risks = q_risk(preds, targets, cfg.quantiles)

    # Coverage: what fraction of actuals fall within P10–P90 interval?
    p10 = preds[..., 0]
    p90 = preds[..., -1]
    coverage = ((targets >= p10) & (targets <= p90)).float().mean().item()
    risks["coverage_p10_p90"] = coverage

    return risks
