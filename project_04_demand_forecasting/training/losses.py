"""
Quantile loss for probabilistic forecasting.

QL(y, ŷ, q) = q * max(y - ŷ, 0) + (1 - q) * max(ŷ - y, 0)

Normalized q-Risk (for reporting, not for training):
  q-Risk = 2 * Σ QL / Σ |y|
(paper Eq. 22 — normalized so metrics are comparable across datasets)
"""
import torch
import torch.nn as nn


class QuantileLoss(nn.Module):
    def __init__(self, quantiles: list[float]):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        preds:   (B, horizon, num_quantiles)
        targets: (B, horizon)
        Returns: scalar loss
        """
        targets = targets.unsqueeze(-1)  # (B, horizon, 1)
        q = torch.tensor(self.quantiles, dtype=preds.dtype, device=preds.device)

        errors = targets - preds  # (B, horizon, Q)
        loss = torch.max(q * errors, (q - 1) * errors)  # (B, horizon, Q)
        return loss.mean()


def q_risk(preds: torch.Tensor, targets: torch.Tensor, quantiles: list[float]) -> dict[str, float]:
    """
    Compute normalized q-Risk per quantile.
    Returns dict: {f"p{int(q*100)}": value, ...}
    """
    targets_exp = targets.unsqueeze(-1)
    q = torch.tensor(quantiles, dtype=preds.dtype, device=preds.device)

    errors = targets_exp - preds
    ql = torch.max(q * errors, (q - 1) * errors)   # (B, horizon, Q)

    denom = targets.abs().sum().item()
    if denom == 0:
        return {f"p{int(qi * 100)}": float("nan") for qi in quantiles}

    results = {}
    for i, qi in enumerate(quantiles):
        numerator = 2 * ql[..., i].sum().item()
        results[f"p{int(qi * 100)}"] = numerator / denom
    return results
