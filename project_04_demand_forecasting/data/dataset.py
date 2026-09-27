"""
PyTorch Dataset for TFT.

Each sample is a (lookback + horizon) window for one store×SKU entity.
Returns tensors split into TFT's four input types:
  - static_cat:    (num_static_cat,)              int64
  - static_cont:   (num_static_cont,)             float32
  - known_past:    (lookback, num_known)           float32
  - known_future:  (horizon,  num_known)           float32
  - observed_past: (lookback, num_observed)        float32
  - target:        (horizon,)                      float32   (log units, ground truth)
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Optional

from training.config import TFTConfig


# Column name lists matching config (order matters — matches embedding table indices)
STATIC_CAT_COLS   = ["store_format_enc", "dc_region_enc", "category_enc",
                      "subcategory_enc", "is_perishable_enc"]
STATIC_CONT_COLS  = ["store_size_sqft_norm", "unit_price_norm"]
KNOWN_COLS        = ["day_of_week", "month", "is_holiday_enc", "is_weekend_enc",
                      "season_enc", "had_promotion_enc", "discount_pct_norm"]
OBSERVED_COLS     = ["log_units_sold", "had_stockout_enc", "weather_severity_norm"]
TARGET_COL        = "log_units_sold"


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        lookback: int,
        horizon: int,
        split: str = "train",        # "train" | "val" | "test"
        train_end: str = "2023-09-30",
        val_end: str   = "2023-12-31",
    ):
        self.lookback = lookback
        self.horizon  = horizon
        self.window   = lookback + horizon

        # Filter to split date range
        df = df.sort_values(["store_id", "product_id", "sale_date"]).copy()
        if split == "train":
            df = df[df["sale_date"] <= train_end]
        elif split == "val":
            # val needs lookback days of history from train, so include from lookback days before val start
            val_start = pd.Timestamp(train_end) - pd.Timedelta(days=lookback)
            df = df[(df["sale_date"] >= val_start) & (df["sale_date"] <= val_end)]
        else:  # test
            test_start = pd.Timestamp(val_end) - pd.Timedelta(days=lookback)
            df = df[df["sale_date"] >= test_start]

        self.samples = self._build_samples(df)

    def _build_samples(self, df: pd.DataFrame) -> list[dict]:
        samples = []
        # Group by entity
        for (store_id, product_id), grp in df.groupby(["store_id", "product_id"], sort=False):
            grp = grp.sort_values("sale_date").reset_index(drop=True)
            n = len(grp)
            if n < self.window:
                continue  # skip entities with insufficient history

            # Static features are constant per entity — take first row
            static_cat  = grp[STATIC_CAT_COLS].iloc[0].values.astype(np.int64)
            static_cont = grp[STATIC_CONT_COLS].iloc[0].values.astype(np.float32)

            # Slide window across time
            for start in range(0, n - self.window + 1):
                end = start + self.window
                window_df = grp.iloc[start:end]

                past   = window_df.iloc[:self.lookback]
                future = window_df.iloc[self.lookback:]

                known_past   = past[KNOWN_COLS].values.astype(np.float32)
                known_future = future[KNOWN_COLS].values.astype(np.float32)
                observed     = past[OBSERVED_COLS].values.astype(np.float32)
                target       = future[TARGET_COL].values.astype(np.float32)

                samples.append({
                    "static_cat":    static_cat,
                    "static_cont":   static_cont,
                    "known_past":    known_past,
                    "known_future":  known_future,
                    "observed_past": observed,
                    "target":        target,
                })
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = self.samples[idx]
        return {
            "static_cat":    torch.tensor(s["static_cat"],    dtype=torch.long),
            "static_cont":   torch.tensor(s["static_cont"],   dtype=torch.float32),
            "known_past":    torch.tensor(s["known_past"],     dtype=torch.float32),
            "known_future":  torch.tensor(s["known_future"],  dtype=torch.float32),
            "observed_past": torch.tensor(s["observed_past"], dtype=torch.float32),
            "target":        torch.tensor(s["target"],        dtype=torch.float32),
        }


def load_datasets(cfg: TFTConfig, data_path: str = "data/parquet/demand_processed.parquet"):
    df = pd.read_parquet(data_path, engine="pyarrow")
    df["sale_date"] = pd.to_datetime(df["sale_date"])

    train_ds = TimeSeriesDataset(df, cfg.lookback, cfg.horizon, "train",
                                  cfg.train_end, cfg.val_end)
    val_ds   = TimeSeriesDataset(df, cfg.lookback, cfg.horizon, "val",
                                  cfg.train_end, cfg.val_end)
    test_ds  = TimeSeriesDataset(df, cfg.lookback, cfg.horizon, "test",
                                  cfg.train_end, cfg.val_end)

    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    # Quick shape verification
    cfg = TFTConfig()
    train_ds, val_ds, test_ds = load_datasets(cfg)
    print(f"Train samples: {len(train_ds):,}")
    print(f"Val   samples: {len(val_ds):,}")
    print(f"Test  samples: {len(test_ds):,}")
    sample = train_ds[0]
    for k, v in sample.items():
        print(f"  {k:20s}: {tuple(v.shape)}  dtype={v.dtype}")
