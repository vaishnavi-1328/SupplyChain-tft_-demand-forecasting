"""
Extract daily store-SKU demand data from Project 1 PostgreSQL warehouse.
Outputs parquet files to data/parquet/.
"""
import os
import sys
import argparse
import pandas as pd
from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from training.config import TFTConfig


EXTRACT_SQL = """
WITH top_stores AS (
    SELECT store_id, SUM(total_units) AS total_units
    FROM supply_chain.v_daily_store_sales
    GROUP BY store_id
    ORDER BY total_units DESC
    LIMIT :top_n_stores
),
top_skus AS (
    SELECT product_id, SUM(total_units) AS total_units
    FROM supply_chain.v_daily_store_sales
    GROUP BY product_id
    ORDER BY total_units DESC
    LIMIT :top_n_skus
),
base AS (
    SELECT
        v.store_id,
        v.product_id,
        v.sale_date,
        v.total_units,
        v.total_revenue,
        v.had_stockout,
        v.had_promotion,
        v.discount_pct,
        v.is_holiday,
        v.is_weekend,
        v.season,
        -- calendar features
        c.day_of_week,
        c.month,
        -- store features
        s.store_format,
        s.store_size_sqft,
        s.dc_id,
        -- product features
        p.category,
        p.subcategory,
        p.is_perishable,
        p.unit_price,
        -- dc region
        d.region AS dc_region,
        -- weather (daily avg per store)
        COALESCE(w.weather_severity, 'Normal') AS weather_severity
    FROM supply_chain.v_daily_store_sales v
    JOIN top_stores ts ON ts.store_id = v.store_id
    JOIN top_skus tk ON tk.product_id = v.product_id
    JOIN supply_chain.calendar c ON c.full_date = v.sale_date
    JOIN supply_chain.stores s ON s.store_id = v.store_id
    JOIN supply_chain.products p ON p.product_id = v.product_id
    JOIN supply_chain.distribution_centers d ON d.dc_id = s.dc_id
    LEFT JOIN (
        SELECT store_id, date_id,
               CASE
                   WHEN weather_severity IN ('Severe','Extreme') THEN 'Severe'
                   WHEN weather_severity = 'Moderate' THEN 'Moderate'
                   ELSE 'Normal'
               END AS weather_severity
        FROM supply_chain.weather
    ) w ON w.store_id = v.store_id
      AND w.date_id = c.date_id
)
SELECT * FROM base
ORDER BY store_id, product_id, sale_date
"""


def extract(cfg: TFTConfig, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    print(f"Connecting to database...")
    engine = create_engine(cfg.db_url)

    print(f"Extracting top {cfg.top_n_stores} stores × top {cfg.top_n_skus} SKUs...")
    with engine.connect() as conn:
        df = pd.read_sql(
            text(EXTRACT_SQL),
            conn,
            params={"top_n_stores": cfg.top_n_stores, "top_n_skus": cfg.top_n_skus},
            parse_dates=["sale_date"],
        )

    print(f"Extracted {len(df):,} rows, {df['store_id'].nunique()} stores, "
          f"{df['product_id'].nunique()} SKUs")

    # Save as single parquet (small enough: ~50 stores * 200 SKUs * 731 days = 7.3M rows max)
    out_path = os.path.join(output_dir, "demand_raw.parquet")
    df.to_parquet(out_path, index=False, engine="pyarrow")
    print(f"Saved to {out_path}")

    # Basic sanity check
    assert df["sale_date"].min().year == 2023, "Expected data starting 2023"
    assert df["total_units"].min() >= 0, "Negative units found"
    print("Sanity checks passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/parquet")
    args = parser.parse_args()

    cfg = TFTConfig()
    if not cfg.db_url:
        raise EnvironmentError("SUPPLY_CHAIN_DB_URL environment variable not set.")
    extract(cfg, args.output_dir)
