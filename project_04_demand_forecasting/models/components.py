"""
Core building blocks for TFT (paper Section 4.1–4.3).

GatedResidualNetwork (GRN)
VariableSelectionNetwork (VSN)
StaticCovariateEncoder
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GatedLinearUnit(nn.Module):
    """GLU(x) = σ(W2·x + b2) ⊙ (W1·x + b1)  — element-wise gating."""
    def __init__(self, input_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.gate = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(torch.sigmoid(self.gate(x)) * self.fc(x))


class GatedResidualNetwork(nn.Module):
    """
    GRN(a, c) = LayerNorm(a + GLU(η₁))
    η₁ = W₁·η₂ + b₁
    η₂ = ELU(W₂·a + b₂ + W_c·c)   (c is optional context)
    (paper Eq. 3)
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
        context_dim: Optional[int] = None,
    ):
        super().__init__()
        self.input_dim  = input_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.context_proj = nn.Linear(context_dim, hidden_dim, bias=False) if context_dim else None

        self.glu   = GatedLinearUnit(hidden_dim, output_dim, dropout)
        self.norm  = nn.LayerNorm(output_dim)

        # Skip connection projection when dims differ
        self.skip  = nn.Linear(input_dim, output_dim, bias=False) if input_dim != output_dim else nn.Identity()

    def forward(self, a: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        # η₂
        eta2 = self.fc1(a)
        if c is not None and self.context_proj is not None:
            eta2 = eta2 + self.context_proj(c)
        eta2 = F.elu(eta2)
        # η₁
        eta1 = self.fc2(eta2)
        # GRN output
        return self.norm(self.skip(a) + self.glu(eta1))


class VariableSelectionNetwork(nn.Module):
    """
    VSN: per-variable GRN + softmax weights → weighted sum.
    Returns (embedding, weights) for interpretability.
    (paper Section 4.2)
    """
    def __init__(
        self,
        num_vars: int,
        input_dim: int,       # per-variable embedding size (= hidden_dim)
        hidden_dim: int,
        dropout: float,
        context_dim: Optional[int] = None,
    ):
        super().__init__()
        self.num_vars = num_vars
        self.hidden_dim = hidden_dim

        # One GRN per variable
        self.var_grns = nn.ModuleList([
            GatedResidualNetwork(input_dim, hidden_dim, hidden_dim, dropout)
            for _ in range(num_vars)
        ])

        # Flatten all variables → softmax selection weights
        self.softmax_grn = GatedResidualNetwork(
            input_dim=num_vars * input_dim,
            hidden_dim=hidden_dim,
            output_dim=num_vars,
            dropout=dropout,
            context_dim=context_dim,
        )

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (..., num_vars, input_dim)  — already projected to input_dim
        Returns:
          combined:  (..., hidden_dim)
          weights:   (..., num_vars)   softmax weights (for interpretability)
        """
        # Per-variable processing
        var_outputs = torch.stack(
            [self.var_grns[i](x[..., i, :]) for i in range(self.num_vars)],
            dim=-2,
        )  # (..., num_vars, hidden_dim)

        # Flatten for selection GRN
        flat = x.flatten(start_dim=-2)  # (..., num_vars * input_dim)
        weights = torch.softmax(self.softmax_grn(flat, context), dim=-1)  # (..., num_vars)

        # Weighted sum
        combined = (var_outputs * weights.unsqueeze(-1)).sum(dim=-2)  # (..., hidden_dim)
        return combined, weights


class StaticCovariateEncoder(nn.Module):
    """
    Encodes static (time-invariant) features into 4 context vectors:
      cs — variable selection context
      ce — static enrichment context
      cc — LSTM initial cell state
      ch — LSTM initial hidden state
    (paper Section 4.3)
    """
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.cs = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.ce = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.cc = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.ch = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)

    def forward(self, static_embedding: torch.Tensor) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        return (
            self.cs(static_embedding),
            self.ce(static_embedding),
            self.cc(static_embedding),
            self.ch(static_embedding),
        )
