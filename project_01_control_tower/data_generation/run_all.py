"""
Orchestrator: generate all tables in dependency order and load into PostgreSQL.

Usage:
    # Set your connection string first:
    export SUPPLY_CHAIN_DB_URL="postgresql://user:pass@host:5432/dbname"

    # Run everything:
    python run_all.py

    # Run with verification queries after load:
    python run_all.py --verify

    # Regenerate only specific tables (comma-separated):
    python run_all.py --tables sales,inventory

    # Dry run (generate data but don't write to DB):
    python run_all.py --dry-run
"""

import argparse
import sys
import time
from datetime import datetime

import pandas as pd
from sqlalchemy import create_engine, text

from config import DB_URL, SCHEMA, BATCH_SIZE

# ── Generator imports ────────────────────────────────────────
from generate_dimensions import (
    generate_distribution_centers, generate_stores, generate_suppliers,
    generate_products, generate_customers, generate_calendar,
    generate_product_supplier,
)
from generate_promotions  import generate_promotions
from generate_weather     import generate_weather
from generate_purchase_orders import generate_purchase_orders
from generate_shipments   import generate_shipments
from generate_sales       import generate_sales
from generate_inventory   import generate_inventory
from generate_returns     import generate_returns
from generate_spoilage    import generate_spoilage


def get_engine():
    if not DB_URL:
        raise ValueError(
            "DB_URL is not set. "
            "Run: export SUPPLY_CHAIN_DB_URL='postgresql://user:pass@host:5432/dbname'"
        )
    return create_engine(DB_URL)


def load_df(engine, df: pd.DataFrame, table: str, if_exists="append", chunksize=BATCH_SIZE):
    """Write a DataFrame to PostgreSQL in batches."""
    start = time.time()
    df.to_sql(
        table,
        engine,
        schema=SCHEMA,
        if_exists=if_exists,
        index=False,
        chunksize=chunksize,
        method="multi",
    )
    elapsed = time.time() - start
    print(f"    Loaded {len(df):>10,} rows → {table:<30}  ({elapsed:.1f}s)")


def load_in_batches(engine, generator, table: str):
    """Load a generator of DataFrames (for large tables like sales)."""
    total = 0
    for batch in generator:
        if len(batch) == 0:
            continue
        batch.to_sql(
            table,
            engine,
            schema=SCHEMA,
            if_exists="append",
            index=False,
            chunksize=BATCH_SIZE,
            method="multi",
        )
        total += len(batch)
        print(f"    Written batch {total:>12,} rows so far → {table}")
    return total


def truncate_table(engine, table: str):
    with engine.connect() as conn:
        conn.execute(text(f"TRUNCATE TABLE {SCHEMA}.{table} CASCADE"))
        conn.commit()


def run_verification(engine):
    print("\n── Running verification queries ────────────────────────")
    checks = [
        ("Stockout with positive sales (expect 0)",
         f"SELECT COUNT(*) FROM {SCHEMA}.sales_transactions WHERE is_stockout_day = TRUE AND quantity_sold > 0"),
        ("Negative inventory (expect 0)",
         f"SELECT COUNT(*) FROM {SCHEMA}.inventory_snapshots WHERE units_on_hand < 0"),
        ("Spoilage on non-perishables (expect 0)",
         f"SELECT COUNT(*) FROM {SCHEMA}.spoilage sp JOIN {SCHEMA}.products p ON p.product_id = sp.product_id WHERE p.is_perishable = FALSE"),
        ("Bad promo dates (expect 0)",
         f"SELECT COUNT(*) FROM {SCHEMA}.promotions WHERE end_date < start_date"),
        ("Stockout rate %",
         f"SELECT ROUND(100.0 * SUM(CASE WHEN is_stockout THEN 1 ELSE 0 END) / COUNT(*), 2) FROM {SCHEMA}.inventory_snapshots WHERE location_type = 'Store'"),
    ]
    with engine.connect() as conn:
        for desc, sql in checks:
            result = conn.execute(text(sql)).scalar()
            print(f"  {desc}: {result}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify",   action="store_true", help="Run verification after load")
    parser.add_argument("--dry-run",  action="store_true", help="Generate data but don't write to DB")
    parser.add_argument("--tables",   default="all",       help="Comma-separated list of tables to run")
    args = parser.parse_args()

    requested = set(args.tables.split(",")) if args.tables != "all" else None
    dry_run   = args.dry_run

    engine = None if dry_run else get_engine()

    def should_run(name: str) -> bool:
        return requested is None or name in requested

    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"  Supply Chain Control Tower — Data Generation")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    # ── Step 1: Dimension tables ────────────────────────────
    print("Step 1: Generating dimension tables...")
    dcs       = generate_distribution_centers();  dcs["dc_id"]       = range(1, len(dcs) + 1)
    stores    = generate_stores(dcs);             stores["store_id"] = range(1, len(stores) + 1)
    suppliers = generate_suppliers();             suppliers["supplier_id"] = range(1, len(suppliers) + 1)
    products  = generate_products();              products["product_id"]   = range(1, len(products) + 1)
    customers = generate_customers();             customers["customer_id"] = range(1, len(customers) + 1)
    calendar  = generate_calendar()

    if not dry_run:
        for tbl, df in [("distribution_centers", dcs), ("stores", stores),
                         ("suppliers", suppliers), ("products", products),
                         ("customers", customers), ("calendar", calendar)]:
            if should_run(tbl):
                truncate_table(engine, tbl)
                load_df(engine, df, tbl, if_exists="append")

    # ── Step 2: Relationship table ──────────────────────────
    print("\nStep 2: Generating product_supplier...")
    prod_sup = generate_product_supplier(products, suppliers)
    if not dry_run and should_run("product_supplier"):
        truncate_table(engine, "product_supplier")
        load_df(engine, prod_sup, "product_supplier", if_exists="append")

    # ── Step 3: Promotions ──────────────────────────────────
    print("\nStep 3: Generating promotions...")
    promotions = generate_promotions(products)
    promotions["promotion_id"] = range(1, len(promotions) + 1)
    if not dry_run and should_run("promotions"):
        truncate_table(engine, "promotions")
        load_df(engine, promotions, "promotions", if_exists="append")

    # ── Step 4: Weather ─────────────────────────────────────
    print("\nStep 4: Generating weather...")
    weather_df = generate_weather(stores)
    if not dry_run and should_run("weather"):
        truncate_table(engine, "weather")
        load_df(engine, weather_df, "weather", if_exists="append")

    # ── Step 5: Purchase orders ─────────────────────────────
    print("\nStep 5: Generating purchase orders...")
    po_df, pol_df = generate_purchase_orders(products, suppliers, prod_sup, dcs)
    if not dry_run:
        if should_run("purchase_orders"):
            truncate_table(engine, "purchase_orders")
            load_df(engine, po_df, "purchase_orders", if_exists="append")
        if should_run("purchase_order_lines"):
            truncate_table(engine, "purchase_order_lines")
            load_df(engine, pol_df, "purchase_order_lines", if_exists="append")

    # ── Step 6: Shipments ────────────────────────────────────
    print("\nStep 6: Generating shipments...")
    ship_df, event_df = generate_shipments(po_df, dcs, stores, weather_df)
    if not dry_run:
        if should_run("shipments"):
            truncate_table(engine, "shipments")
            load_df(engine, ship_df, "shipments", if_exists="append")
        if should_run("shipment_events"):
            truncate_table(engine, "shipment_events")
            load_df(engine, event_df, "shipment_events", if_exists="append")

    # ── Step 7: Sales transactions (large — streamed in batches) ──
    print("\nStep 7: Generating sales transactions (this will take a while)...")
    sales_total = 0
    sales_batches = []  # accumulate for steps 8–10

    if not dry_run and should_run("sales_transactions"):
        truncate_table(engine, "sales_transactions")

    sales_gen = generate_sales(stores, products, calendar, promotions, weather_df, customers)
    for batch in sales_gen:
        sales_batches.append(batch)
        if not dry_run and should_run("sales_transactions"):
            batch.to_sql("sales_transactions", engine, schema=SCHEMA,
                         if_exists="append", index=False, chunksize=BATCH_SIZE, method="multi")
        sales_total += len(batch)
        print(f"    Sales running total: {sales_total:>12,}")

    sales_df = pd.concat(sales_batches, ignore_index=True) if sales_batches else pd.DataFrame()
    print(f"  Total sales rows: {sales_total:,}")

    # ── Step 8: Spoilage (needs inventory concept; computed before inventory for dep order) ──
    print("\nStep 8: Generating spoilage (placeholder — requires inventory snapshots)...")
    # Spoilage uses inventory snapshots but we need a simplified version first.
    # We generate a lightweight spoilage estimate from sales data directly.
    spoilage_df = _generate_spoilage_from_sales(sales_df, products, weather_df)
    if not dry_run and should_run("spoilage"):
        truncate_table(engine, "spoilage")
        load_df(engine, spoilage_df, "spoilage", if_exists="append")

    # ── Step 9: Inventory snapshots ─────────────────────────
    print("\nStep 9: Generating inventory snapshots...")
    inv_total = 0
    if not dry_run and should_run("inventory_snapshots"):
        truncate_table(engine, "inventory_snapshots")

    inv_gen = generate_inventory(stores, products, sales_df, ship_df, spoilage_df)
    for batch in inv_gen:
        if not dry_run and should_run("inventory_snapshots"):
            batch.to_sql("inventory_snapshots", engine, schema=SCHEMA,
                         if_exists="append", index=False, chunksize=BATCH_SIZE, method="multi")
        inv_total += len(batch)
        print(f"    Inventory snapshots running total: {inv_total:>10,}")
    print(f"  Total inventory snapshot rows: {inv_total:,}")

    # ── Step 10: Returns ─────────────────────────────────────
    print("\nStep 10: Generating returns...")
    returns_df = generate_returns(sales_df, products)
    if not dry_run and should_run("returns"):
        truncate_table(engine, "returns")
        load_df(engine, returns_df, "returns", if_exists="append")

    # ── Verification ─────────────────────────────────────────
    if args.verify and not dry_run:
        run_verification(engine)

    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  All done!  Total time: {total_time/60:.1f} minutes")
    print(f"{'='*60}\n")


def _generate_spoilage_from_sales(
    sales: pd.DataFrame,
    products: pd.DataFrame,
    weather: pd.DataFrame,
) -> pd.DataFrame:
    """
    Lightweight spoilage generation using sales data without full inventory simulation.
    Perishable products with low weekly sales get stochastic spoilage events.
    """
    if sales.empty:
        return pd.DataFrame()

    from generate_spoilage import generate_spoilage as gen_spoil
    # Create a minimal inventory snapshot proxy
    import numpy as np
    rng2 = np.random.default_rng(42)

    perishable_pids = set(products[products["is_perishable"] == True]["product_id"].tolist())
    cat_sales = sales[sales["product_id"].isin(perishable_pids)].copy()
    if cat_sales.empty:
        return pd.DataFrame()

    # Build a simplified inventory snapshot for spoilage calculation
    avg_demand = cat_sales.groupby(["store_id", "product_id"])["quantity_sold"].mean()
    shelf_life_map = products.set_index("product_id")["shelf_life_days"].to_dict()
    cost_map       = products.set_index("product_id")["unit_cost"].to_dict()
    wx_sev = weather.set_index(["store_id", "date_id"])["weather_severity"].to_dict()

    records = []
    sp_id = 1
    all_dates = sorted(cat_sales["date_id"].unique())

    # Sample 10% of store-product-perishable pairs to keep it tractable
    pairs = avg_demand.reset_index().sample(frac=0.10, random_state=42)

    for _, row in pairs.iterrows():
        sid, pid = int(row["store_id"]), int(row["product_id"])
        avg_d     = float(row["quantity_sold"])
        sl        = shelf_life_map.get(pid, 7)
        uc        = float(cost_map.get(pid, 1.0))
        inv       = round(avg_d * rng2.integers(sl, sl * 2 + 1))

        for d in all_dates[::sl]:  # check every shelf_life days
            severity = wx_sev.get((sid, d), 1)
            base_rate = 1.0 / max(sl, 1)
            if severity >= 3:
                base_rate *= 1.5
            spoiled = int(rng2.poisson(inv * base_rate))
            if spoiled <= 0:
                continue
            spoiled = min(spoiled, inv)
            reason = "Temperature_Break" if severity >= 3 else (
                "Overstock" if inv > avg_d * sl * 2 else "Expired"
            )
            records.append({
                "spoilage_id":     sp_id,
                "store_id":        sid,
                "product_id":      pid,
                "spoilage_date":   d,
                "units_spoiled":   spoiled,
                "spoilage_reason": reason,
                "cost_of_spoilage": round(spoiled * uc, 2),
            })
            sp_id += 1
            inv = max(0, inv - spoiled)

    df = pd.DataFrame(records)
    print(f"  Spoilage: {len(df):,} rows")
    return df


if __name__ == "__main__":
    main()
