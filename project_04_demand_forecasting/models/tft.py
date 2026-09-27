"""
Temporal Fusion Transformer — full architecture.
Follows paper Section 4 exactly.

Forward pass returns:
  quantile_preds:  (B, horizon, num_quantiles)
  attn_weights:    (B, horizon, lookback+horizon)   temporal attention
  vsn_weights:     dict of variable selection weights (for interpretability)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from models.components import (
    GatedResidualNetwork,
    GatedLinearUnit,
    VariableSelectionNetwork,
    StaticCovariateEncoder,
)
from models.attention import InterpretableMultiHeadAttention, causal_mask
from training.config import TFTConfig
from data.dataset import (
    STATIC_CAT_COLS, STATIC_CONT_COLS, KNOWN_COLS, OBSERVED_COLS
)


class TemporalFusionTransformer(nn.Module):
    def __init__(self, cfg: TFTConfig, cat_cardinalities: list[int]):
        """
        cat_cardinalities: number of unique values per static categorical variable,
                           in the same order as STATIC_CAT_COLS.
        """
        super().__init__()
        self.cfg      = cfg
        H             = cfg.hidden_dim
        dropout       = cfg.dropout
        num_quantiles = len(cfg.quantiles)

        n_static_cat  = len(STATIC_CAT_COLS)
        n_static_cont = len(STATIC_CONT_COLS)
        n_known       = len(KNOWN_COLS)
        n_observed    = len(OBSERVED_COLS)

        # ── 1. Input embeddings ──────────────────────────────────────────────────
        # Categorical static → embedding (each var gets its own embedding table)
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(card + 1, H)   # +1 for unknown/padding
            for card in cat_cardinalities
        ])
        # Continuous static → linear projection
        self.static_cont_proj = nn.Linear(n_static_cont, H) if n_static_cont > 0 else None

        # Known temporal (past + future) → per-variable linear projections
        self.known_projs = nn.ModuleList([nn.Linear(1, H) for _ in range(n_known)])

        # Observed past → per-variable linear projections
        self.obs_projs = nn.ModuleList([nn.Linear(1, H) for _ in range(n_observed)])

        # ── 2. Variable Selection Networks ────────────────────────────────────────
        n_static_total = n_static_cat + (1 if n_static_cont > 0 else 0)
        self.static_vsn = VariableSelectionNetwork(
            n_static_total, H, H, dropout
        )

        self.past_vsn = VariableSelectionNetwork(
            n_known + n_observed, H, H, dropout, context_dim=H
        )

        self.future_vsn = VariableSelectionNetwork(
            n_known, H, H, dropout, context_dim=H
        )

        # ── 3. Static Covariate Encoder ──────────────────────────────────────────
        self.static_encoder = StaticCovariateEncoder(H, dropout)

        # ── 4. LSTM Encoder-Decoder (local processing) ───────────────────────────
        self.lstm_encoder = nn.LSTM(H, H, num_layers=cfg.num_lstm_layers,
                                    batch_first=True, dropout=0.0)
        self.lstm_decoder = nn.LSTM(H, H, num_layers=cfg.num_lstm_layers,
                                    batch_first=True, dropout=0.0)

        # Gated residual skip after LSTM (paper Eq. 8)
        self.post_lstm_gate = nn.Linear(H, 2 * H)  # GLU: split into gate + signal
        self.post_lstm_norm = nn.LayerNorm(H)

        # ── 5. Static Enrichment ─────────────────────────────────────────────────
        self.static_enrichment = GatedResidualNetwork(H, H, H, dropout, context_dim=H)

        # ── 6. Temporal Self-Attention ────────────────────────────────────────────
        self.attn = InterpretableMultiHeadAttention(cfg.num_heads, H, dropout)
        self.post_attn_gate = nn.Linear(H, 2 * H)
        self.post_attn_norm = nn.LayerNorm(H)

        # ── 7. Position-wise Feed-forward ─────────────────────────────────────────
        self.ff_grn  = GatedResidualNetwork(H, H, H, dropout)
        self.post_ff_gate = nn.Linear(H, 2 * H)
        self.post_ff_norm = nn.LayerNorm(H)

        # ── 8. Quantile output layer ───────────────────────────────────────────────
        self.quantile_heads = nn.ModuleList([
            nn.Linear(H, 1) for _ in range(num_quantiles)
        ])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _glu_skip(self, x: torch.Tensor, skip: torch.Tensor,
                  gate_layer: nn.Linear, norm: nn.LayerNorm) -> torch.Tensor:
        """Apply GLU gating + residual + LayerNorm."""
        gated = gate_layer(x)
        signal, gate = gated.chunk(2, dim=-1)
        return norm(skip + torch.sigmoid(gate) * signal)

    def forward(
        self,
        static_cat:    torch.Tensor,   # (B, n_static_cat)
        static_cont:   torch.Tensor,   # (B, n_static_cont)
        known_past:    torch.Tensor,   # (B, lookback, n_known)
        known_future:  torch.Tensor,   # (B, horizon, n_known)
        observed_past: torch.Tensor,   # (B, lookback, n_observed)
    ) -> dict[str, torch.Tensor]:

        B = static_cat.shape[0]
        H = self.cfg.hidden_dim

        # ── Step 1: Embed inputs ──────────────────────────────────────────────────
        # Static categorical embeddings
        static_cat_embs = [
            self.cat_embeddings[i](static_cat[:, i])   # (B, H)
            for i in range(static_cat.shape[1])
        ]

        static_inputs = static_cat_embs
        if self.static_cont_proj is not None:
            static_inputs.append(self.static_cont_proj(static_cont))

        # Stack: (B, n_static, H)
        static_stack = torch.stack(static_inputs, dim=1)

        # Per-variable projections for temporal inputs
        known_past_emb = torch.stack([
            self.known_projs[i](known_past[..., i:i+1])
            for i in range(known_past.shape[-1])
        ], dim=2)  # (B, lookback, n_known, H)

        known_fut_emb = torch.stack([
            self.known_projs[i](known_future[..., i:i+1])
            for i in range(known_future.shape[-1])
        ], dim=2)  # (B, horizon, n_known, H)

        obs_emb = torch.stack([
            self.obs_projs[i](observed_past[..., i:i+1])
            for i in range(observed_past.shape[-1])
        ], dim=2)  # (B, lookback, n_observed, H)

        # ── Step 2: Static VSN ────────────────────────────────────────────────────
        static_combined, static_weights = self.static_vsn(static_stack)  # (B, H)

        # ── Step 3: Static covariate encoder → 4 context vectors ─────────────────
        cs, ce, cc, ch = self.static_encoder(static_combined)

        # ── Step 4: Temporal VSNs with static context ─────────────────────────────
        # Past: known + observed concatenated along variable dim
        past_stack = torch.cat([known_past_emb, obs_emb], dim=2)  # (B, lookback, n_known+n_obs, H)
        # cs broadcast to (B, lookback, H)
        cs_past = cs.unsqueeze(1).expand(-1, past_stack.shape[1], -1)

        past_combined_list = []
        past_weights_list  = []
        for t in range(past_stack.shape[1]):
            out, w = self.past_vsn(past_stack[:, t, :, :], cs_past[:, t, :])
            past_combined_list.append(out)
            past_weights_list.append(w)
        past_combined = torch.stack(past_combined_list, dim=1)   # (B, lookback, H)
        past_vsn_weights = torch.stack(past_weights_list, dim=1) # (B, lookback, n_vars)

        cs_fut = cs.unsqueeze(1).expand(-1, known_fut_emb.shape[1], -1)
        fut_combined_list = []
        fut_weights_list  = []
        for t in range(known_fut_emb.shape[1]):
            out, w = self.future_vsn(known_fut_emb[:, t, :, :], cs_fut[:, t, :])
            fut_combined_list.append(out)
            fut_weights_list.append(w)
        fut_combined  = torch.stack(fut_combined_list, dim=1)    # (B, horizon, H)
        fut_vsn_weights = torch.stack(fut_weights_list, dim=1)   # (B, horizon, n_vars)

        # ── Step 5: LSTM encoder-decoder ──────────────────────────────────────────
        # Initialize LSTM state with static context (paper Section 4.5.1)
        h0 = ch.unsqueeze(0)  # (1, B, H)
        c0 = cc.unsqueeze(0)  # (1, B, H)

        enc_out, (hn, cn) = self.lstm_encoder(past_combined, (h0, c0))
        dec_out, _        = self.lstm_decoder(fut_combined,  (hn, cn))

        # Gated skip connections (Eq. 8)
        enc_out = self._glu_skip(enc_out, past_combined, self.post_lstm_gate, self.post_lstm_norm)
        dec_out = self._glu_skip(dec_out, fut_combined,  self.post_lstm_gate, self.post_lstm_norm)

        # Full temporal sequence: past encoder output + future decoder output
        temporal = torch.cat([enc_out, dec_out], dim=1)  # (B, lookback+horizon, H)

        # ── Step 6: Static enrichment ──────────────────────────────────────────────
        ce_exp = ce.unsqueeze(1).expand_as(temporal)
        temporal = self.static_enrichment(temporal, ce_exp)

        # ── Step 7: Interpretable self-attention (only over future positions) ─────
        # Decoder mask: future positions can't attend to later positions (causal)
        T = temporal.shape[1]
        mask = causal_mask(T, temporal.device)  # (T, T)

        attn_out, attn_weights = self.attn(temporal, temporal, temporal, mask)

        # Gated skip
        attn_out = self._glu_skip(attn_out, temporal, self.post_attn_gate, self.post_attn_norm)

        # ── Step 8: Position-wise feed-forward ────────────────────────────────────
        ff_out = self.ff_grn(attn_out)
        ff_out = self._glu_skip(ff_out, attn_out, self.post_ff_gate, self.post_ff_norm)

        # ── Step 9: Quantile output (forecast horizon only) ───────────────────────
        forecast = ff_out[:, self.cfg.lookback:, :]  # (B, horizon, H)

        quantile_preds = torch.cat([
            head(forecast) for head in self.quantile_heads
        ], dim=-1)  # (B, horizon, num_quantiles)

        return {
            "quantile_preds":    quantile_preds,
            "attn_weights":      attn_weights[:, self.cfg.lookback:, :],  # future rows
            "static_weights":    static_weights,
            "past_vsn_weights":  past_vsn_weights,
            "fut_vsn_weights":   fut_vsn_weights,
        }
