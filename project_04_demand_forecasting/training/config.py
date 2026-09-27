from dataclasses import dataclass, field
from typing import List
import os


@dataclass
class TFTConfig:
    # Data
    db_url: str = field(default_factory=lambda: os.getenv("SUPPLY_CHAIN_DB_URL", ""))
    schema: str = "supply_chain"
    data_dir: str = "data/parquet"

    # Subset for training (manageable size)
    top_n_stores: int = 50
    top_n_skus: int = 200

    # Time splits
    train_end: str = "2023-09-30"
    val_end: str = "2023-12-31"
    test_end: str = "2024-12-31"

    # Sequence lengths
    lookback: int = 90       # τ = 90 days of past context
    horizon: int = 14        # forecast 14 days ahead

    # Features
    static_cat_vars: List[str] = field(default_factory=lambda: [
        "store_format", "dc_region", "category", "subcategory", "is_perishable"
    ])
    static_cont_vars: List[str] = field(default_factory=lambda: [
        "store_size_sqft_norm", "unit_price_norm"
    ])
    known_future_vars: List[str] = field(default_factory=lambda: [
        "day_of_week", "month", "is_holiday", "is_weekend", "season_enc",
        "promotion_flag", "discount_pct_norm"
    ])
    observed_past_vars: List[str] = field(default_factory=lambda: [
        "log_units_sold", "stockout_flag", "weather_severity_norm"
    ])
    target: str = "log_units_sold"

    # Model architecture (paper defaults)
    hidden_dim: int = 160
    num_heads: int = 4
    dropout: float = 0.1
    num_lstm_layers: int = 1

    # Training
    batch_size: int = 64
    epochs: int = 50
    lr: float = 1e-3
    grad_clip: float = 0.1
    patience: int = 5           # early stopping

    # Loss
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
