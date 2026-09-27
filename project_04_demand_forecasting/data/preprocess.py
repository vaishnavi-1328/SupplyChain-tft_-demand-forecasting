"""
Feature engineering and normalization for TFT training.
Input:  data/parquet/demand_raw.parquet
Output: data/parquet/demand_processed.parquet
        data/parquet/encoders.pkl   (scaler/encoder state for inference)
"""
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


# Maps categorical values to integer indices
CAT_COLS = ["store_format", "dc_region", "category", "subcategory", "weather_severity"]
SEASON_MAP = {"Winter": 0, "Spring": 1, "Summer": 2, "Fall": 3}


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    df = df.sort_values(["store_id", "product_id", "sale_date"]).reset_index(drop=True)

    encoders: dict = {}

    # ── Target ──────────────────────────────────────────────────────────────────
    # Log-transform: log(x + 1) handles zeros; inverse is exp(x) - 1
    df["log_units_sold"] = np.log1p(df["total_units"].clip(lower=0))

    # ── Categorical → integer encoding ─────────────────────────────────────────
    for col in CAT_COLS:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].fillna("Unknown").astype(str))
        encoders[col] = le

    df["is_perishable_enc"] = df["is_perishable"].astype(int)
    df["is_holiday_enc"] = df["is_holiday"].astype(int)
    df["is_weekend_enc"] = df["is_weekend"].astype(int)
    df["had_stockout_enc"] = df["had_stockout"].astype(int)
    df["had_promotion_enc"] = df["had_promotion"].astype(int)
    df["season_enc"] = df["season"].map(SEASON_MAP).fillna(0).astype(int)
    # day_of_week: 0 (Mon) – 6 (Sun) already an int in calendar table
    # month: 1–12 already an int

    # ── Continuous normalization (z-score, fitted on train set only) ───────────
    cont_cols = {"store_size_sqft": "store_size_sqft_norm",
                 "unit_price":      "unit_price_norm",
                 "discount_pct":    "discount_pct_norm"}

    for src, tgt in cont_cols.items():
        scaler = StandardScaler()
        df[tgt] = scaler.fit_transform(df[[src]].fillna(0))
        encoders[src] = scaler

    # weather severity: map to numeric severity score
    sev_map = {"Normal": 0.0, "Moderate": 0.5, "Severe": 1.0}
    df["weather_severity_norm"] = df["weather_severity"].map(sev_map).fillna(0.0)

    return df, encoders


def process(input_path: str, output_dir: str) -> None:
    print(f"Loading {input_path}...")
    df = pd.read_parquet(input_path)

    print("Building features...")
    df_proc, encoders = build_features(df)

    out_path = os.path.join(output_dir, "demand_processed.parquet")
    enc_path = os.path.join(output_dir, "encoders.pkl")

    df_proc.to_parquet(out_path, index=False, engine="pyarrow")
    with open(enc_path, "wb") as f:
        pickle.dump(encoders, f)

    print(f"Saved processed data: {out_path}")
    print(f"Saved encoders:       {enc_path}")
    print(f"Columns: {list(df_proc.columns)}")


if __name__ == "__main__":
    process("data/parquet/demand_raw.parquet", "data/parquet")
