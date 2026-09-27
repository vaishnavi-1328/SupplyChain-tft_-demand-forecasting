"""
Generate purchase_orders and purchase_order_lines.

Logic:
- Each DC places POs weekly with its primary suppliers.
- Order quantity driven by a simple (s, Q) replenishment policy.
- Lead time realized as base + noise + disruption spike.
- Seasonal lead time inflation +20% in Nov–Dec.
- ~3% of POs have a disruption (extra 3–14 days late).
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd

from config import RANDOM_SEED, START_DATE, END_DATE, PO_CONFIRM_RATE

rng = np.random.default_rng(RANDOM_SEED + 3)

STATUSES = ["Received", "Received", "Received", "Received", "Cancelled"]  # 80% received, 20% cancelled

CARRIERS = ["FreightPro", "TransLink", "NationalTruck", "SwiftFreight", "RegionalHaul"]


def generate_purchase_orders(
    products: pd.DataFrame,
    suppliers: pd.DataFrame,
    product_supplier: pd.DataFrame,
    dcs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    start = date.fromisoformat(START_DATE)
    end   = date.fromisoformat(END_DATE)

    # Get primary supplier per product
    primary_ps = product_supplier[product_supplier["is_primary"] == True].copy()
    # Drop min_order_qty from product_supplier before merging — products table is authoritative
    primary_ps = primary_ps.drop(columns=["min_order_qty"], errors="ignore")
    primary_ps = primary_ps.merge(suppliers[["supplier_id", "supplier_type", "lead_time_days_avg",
                                              "lead_time_days_std", "disruption_prob", "fill_rate_baseline"]],
                                   on="supplier_id")

    # Products list with costs
    prod_info = products[["product_id", "unit_cost", "category", "min_order_qty", "case_pack_qty"]].copy()
    primary_ps = primary_ps.merge(prod_info, on="product_id")

    po_records   = []
    line_records = []
    po_id   = 1
    line_id = 1

    dc_list = dcs["dc_id"].tolist()

    # Weekly ordering cycle — iterate over each Monday in the date range
    current = start
    while current.weekday() != 0:  # advance to first Monday
        current += timedelta(days=1)

    while current <= end - timedelta(days=30):
        week_date = current
        month = week_date.month
        seasonal_lt_multiplier = 1.20 if month in (11, 12) else 1.0

        for dc_id in dc_list:
            # Each DC orders from a random subset of products this week (~200–400 products)
            n_products = int(rng.integers(200, 401))
            sampled = primary_ps.sample(n=min(n_products, len(primary_ps)),
                                        random_state=int(rng.integers(1e6)))

            # Group by supplier (batch products per supplier into one PO)
            for sup_id, grp in sampled.groupby("supplier_id"):
                sup_row = suppliers[suppliers["supplier_id"] == sup_id].iloc[0]
                s_type  = sup_row["supplier_type"]

                order_date  = week_date + timedelta(days=int(rng.integers(0, 3)))  # Mon–Wed
                base_lt     = float(sup_row["lead_time_days_avg"]) * seasonal_lt_multiplier
                lt_std      = float(sup_row["lead_time_days_std"])
                realized_lt = max(1, rng.normal(base_lt, lt_std))

                # Disruption?
                disrupted   = rng.random() < float(sup_row["disruption_prob"])
                extra_days  = int(rng.integers(3, 15)) if disrupted else 0

                req_delivery    = order_date + timedelta(days=int(round(base_lt)))
                actual_delivery = order_date + timedelta(days=int(round(realized_lt)) + extra_days)

                if actual_delivery > end:
                    actual_delivery = None  # PO still in flight at end of sim
                    status = "Shipped"
                elif rng.random() < 0.02:
                    status = "Cancelled"
                    actual_delivery = None
                else:
                    status = "Received"

                is_late  = (actual_delivery is not None and actual_delivery > req_delivery)
                late_days = max(0, (actual_delivery - req_delivery).days) if is_late and actual_delivery else 0

                # Confirmation rate
                confirm_rate = PO_CONFIRM_RATE[s_type]

                lines = []
                total_units = 0
                total_cost  = 0.0

                for _, prod_row in grp.iterrows():
                    # Order quantity: 2–8 weeks of supply (roughly 50–300 units per product)
                    base_qty   = int(rng.integers(50, 301))
                    ordered_qty = max(int(prod_row["min_order_qty"]),
                                      int(round(base_qty / prod_row["case_pack_qty"])) * int(prod_row["case_pack_qty"]))
                    confirmed_qty = max(0, round(ordered_qty * rng.uniform(confirm_rate - 0.05, 1.0)))
                    received_qty  = confirmed_qty if status == "Received" else (
                        round(confirmed_qty * float(rng.uniform(float(prod_row["fill_rate_baseline"]) - 0.05, 1.0)))
                        if status == "Received" else 0
                    )
                    uc   = float(prod_row["unit_cost"])
                    fill = round(received_qty / ordered_qty, 4) if ordered_qty > 0 else 0.0

                    lines.append({
                        "po_line_id":    line_id,
                        "po_id":         po_id,
                        "product_id":    int(prod_row["product_id"]),
                        "ordered_qty":   ordered_qty,
                        "confirmed_qty": int(confirmed_qty),
                        "received_qty":  int(received_qty),
                        "unit_cost":     round(uc, 2),
                        "line_total":    round(received_qty * uc, 2),
                        "fill_rate":     fill,
                    })
                    total_units += ordered_qty
                    total_cost  += ordered_qty * uc
                    line_id     += 1

                po_records.append({
                    "po_id":               po_id,
                    "po_number":           f"PO-{po_id:07d}",
                    "supplier_id":         int(sup_id),
                    "dc_id":               int(dc_id),
                    "order_date":          order_date,
                    "requested_delivery":  req_delivery,
                    "actual_delivery":     actual_delivery,
                    "status":              status,
                    "total_lines":         len(lines),
                    "total_units":         total_units,
                    "total_cost":          round(total_cost, 2),
                    "is_late":             bool(is_late),
                    "late_days":           late_days,
                })
                line_records.extend(lines)
                po_id += 1

        current += timedelta(weeks=1)

    po_df   = pd.DataFrame(po_records)
    line_df = pd.DataFrame(line_records)
    print(f"  POs: {len(po_df):,}  |  PO Lines: {len(line_df):,}")
    return po_df, line_df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from generate_dimensions import (generate_distribution_centers, generate_stores,
                                     generate_suppliers, generate_products, generate_product_supplier)
    dcs  = generate_distribution_centers()
    dcs["dc_id"] = range(1, len(dcs) + 1)
    sups = generate_suppliers()
    sups["supplier_id"] = range(1, len(sups) + 1)
    prod = generate_products()
    prod["product_id"] = range(1, len(prod) + 1)
    ps   = generate_product_supplier(prod, sups)
    po, pol = generate_purchase_orders(prod, sups, ps, dcs)
    print(po["status"].value_counts())
    print(f"Late rate: {po['is_late'].mean():.2%}")
