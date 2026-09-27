"""
Generate inventory_snapshots.

Strategy: daily forward simulation using sales + received shipments.
Opens Jan 1 2023 with 14–30 days of supply per store × product.
Records daily end-of-day snapshot for every active store-product pair.

To keep volume tractable, snapshots are written monthly.
"""

from datetime import date, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd

from config import RANDOM_SEED, START_DATE, END_DATE, OPENING_DOS_MIN, OPENING_DOS_MAX

rng = np.random.default_rng(RANDOM_SEED + 6)

FORMAT_VOLUME = {"Superstore": 1.40, "Neighborhood": 0.80, "Express": 0.40, "Warehouse": 1.80}


def generate_inventory(
    stores: pd.DataFrame,
    products: pd.DataFrame,
    sales: pd.DataFrame,           # full sales_transactions (or iterator batches)
    shipments: pd.DataFrame,       # outbound DC→Store shipments
    spoilage: pd.DataFrame | None = None,
) -> pd.DataFrame:

    all_dates = pd.date_range(START_DATE, END_DATE).date.tolist()

    # Build store-product universe from sales
    store_product_pairs = sales[["store_id", "product_id"]].drop_duplicates()

    # Product metadata
    prod_meta = products.set_index("product_id")[["unit_cost", "category"]].to_dict("index")

    # Store metadata
    store_meta = stores.set_index("store_id")[["store_format", "dc_id"]].to_dict("index")

    # Product base rates (avg daily demand — use mean of sales)
    avg_demand = (
        sales.groupby(["store_id", "product_id"])["quantity_sold"]
        .mean()
        .to_dict()
    )

    # Opening inventory per store × product
    inv: dict[tuple, int] = {}
    reorder_pt: dict[tuple, int] = {}
    safety_stk: dict[tuple, int] = {}

    for _, row in store_product_pairs.iterrows():
        sid, pid = int(row["store_id"]), int(row["product_id"])
        avg_d = avg_demand.get((sid, pid), 1.0)
        dos   = rng.integers(OPENING_DOS_MIN, OPENING_DOS_MAX + 1)
        inv[(sid, pid)]       = max(0, round(avg_d * dos))
        reorder_pt[(sid, pid)] = max(1, round(avg_d * 7))   # 7-day reorder point
        safety_stk[(sid, pid)] = max(1, round(avg_d * 3))   # 3-day safety stock

    # Inbound receipts: outbound shipments (DC→Store) received on actual_arrival
    receipts: dict[tuple, dict] = defaultdict(lambda: defaultdict(int))  # (store_id, product_id, date) → qty
    store_shipments = shipments[
        (shipments["dest_type"] == "Store") &
        (shipments["actual_arrival"].notna())
    ]
    for _, shp in store_shipments.iterrows():
        arr_date = shp["actual_arrival"]
        sid      = int(shp["dest_store_id"])
        # Distribute total_units evenly across active products for that store (simplified)
        active_prods = store_product_pairs[store_product_pairs["store_id"] == sid]["product_id"].tolist()
        if not active_prods:
            continue
        units_per_product = max(1, int(shp["total_units"]) // len(active_prods))
        for pid in active_prods[:50]:  # top 50 products per shipment (simplification)
            receipts[(sid, pid)][arr_date] += units_per_product

    # Daily sales lookup: (store_id, product_id, date) → qty_sold
    daily_sales = (
        sales.groupby(["store_id", "product_id", "date_id"])["quantity_sold"]
        .sum()
        .to_dict()
    )

    # Daily spoilage lookup
    daily_spoilage: dict = {}
    if spoilage is not None:
        daily_spoilage = (
            spoilage.groupby(["store_id", "product_id", "spoilage_date"])["units_spoiled"]
            .sum()
            .to_dict()
        )

    snapshot_records = []
    snap_id = 1

    for d in all_dates:
        for _, row in store_product_pairs.iterrows():
            sid, pid = int(row["store_id"]), int(row["product_id"])
            key = (sid, pid)

            # Receive shipments
            received = receipts[key].get(d, 0)
            inv[key] = inv.get(key, 0) + received

            # Sell
            sold = daily_sales.get((sid, pid, d), 0)
            inv[key] = max(0, inv[key] - sold)

            # Spoilage
            spoiled = daily_spoilage.get((sid, pid, d), 0)
            inv[key] = max(0, inv[key] - spoiled)

            units_oh = inv[key]
            avg_d    = avg_demand.get(key, 1.0)
            dos      = round(units_oh / avg_d, 2) if avg_d > 0 else 0.0

            snapshot_records.append({
                "snapshot_id":      snap_id,
                "store_id":         sid,
                "product_id":       pid,
                "dc_id":            None,
                "snapshot_date":    d,
                "location_type":    "Store",
                "units_on_hand":    units_oh,
                "units_on_order":   0,
                "units_in_transit": 0,
                "days_of_supply":   dos,
                "is_stockout":      (units_oh == 0),
                "reorder_point":    reorder_pt.get(key),
                "safety_stock":     safety_stk.get(key),
            })
            snap_id += 1

        # Yield monthly batches to avoid memory pressure
        if d.day == 1 and d != date.fromisoformat(START_DATE):
            df = pd.DataFrame(snapshot_records)
            yield df
            snapshot_records = []

    if snapshot_records:
        yield pd.DataFrame(snapshot_records)

    print("  Inventory snapshot generation complete.")
