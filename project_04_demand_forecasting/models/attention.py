"""
Interpretable Multi-Head Attention (paper Section 4.5.3).

Key difference from standard multi-head attention:
- Shared value weights W_V across all heads
- Additive aggregation: H̃ = (1/H) Σ Attention(Q W_Q^h, K W_K^h, V W_V)
- Allows attention patterns to be averaged/interpreted across heads

Returns both the output and the attention weights (needed for interpretability).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, d_model: int, dropout: float):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.num_heads = num_heads
        self.d_model   = d_model
        self.d_k       = d_model // num_heads

        # Separate Q/K projections per head
        self.W_q = nn.ModuleList([nn.Linear(d_model, self.d_k, bias=False) for _ in range(num_heads)])
        self.W_k = nn.ModuleList([nn.Linear(d_model, self.d_k, bias=False) for _ in range(num_heads)])

        # Shared value projection (interpretability-enabling design choice)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # Output projection
        self.W_out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,   # (B, T_q, d_model)
        key:   torch.Tensor,   # (B, T_k, d_model)
        value: torch.Tensor,   # (B, T_k, d_model)
        mask:  torch.Tensor | None = None,  # (B, T_q, T_k) or (T_q, T_k), True = mask out
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          output:   (B, T_q, d_model)
          attn_weights: (B, T_q, T_k)   averaged across heads
        """
        B, T_q, _ = query.shape
        T_k = key.shape[1]

        # Shared values: (B, T_k, d_model)
        V = self.W_v(value)

        head_outputs = []
        head_weights = []

        for h in range(self.num_heads):
            Q_h = self.W_q[h](query)   # (B, T_q, d_k)
            K_h = self.W_k[h](key)     # (B, T_k, d_k)

            # Scaled dot-product attention
            scores = torch.bmm(Q_h, K_h.transpose(1, 2)) / math.sqrt(self.d_k)  # (B, T_q, T_k)

            if mask is not None:
                scores = scores.masked_fill(mask, float("-inf"))

            weights_h = torch.softmax(scores, dim=-1)                            # (B, T_q, T_k)
            weights_h = self.dropout(weights_h)

            # Attend over shared values (full d_model, not d_k)
            # Split V into head-sized chunks for this head
            v_chunk = V[:, :, h * self.d_k : (h + 1) * self.d_k]                # (B, T_k, d_k)
            out_h = torch.bmm(weights_h, v_chunk)                                # (B, T_q, d_k)

            head_outputs.append(out_h)
            head_weights.append(weights_h)

        # Additive aggregation: average across heads (paper Eq. 10)
        H_tilde = torch.stack(head_outputs, dim=0).mean(dim=0)                   # (B, T_q, d_k * ... wait)
        # Concatenate head outputs, then average attention weights
        H_concat = torch.cat(head_outputs, dim=-1)                               # (B, T_q, d_model)
        attn_weights = torch.stack(head_weights, dim=0).mean(dim=0)              # (B, T_q, T_k)

        output = self.W_out(H_concat)                                            # (B, T_q, d_model)
        return output, attn_weights


def causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Upper-triangular mask: position i cannot attend to j > i."""
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    return mask
