"""
Generate sales_transactions — the causal core (~10–20M rows in optimized mode).

Demand model per store × product × date:
  demand = base_rate
         × store_volume_factor
         × category_seasonality(month)
         × day_of_week_factor
         × holiday_factor
         × promotion_factor  (if product is on promo that day)
         × weather_factor    (category-specific response to temperature/precip)
         × post_promo_dip    (−15% for 3 days after promo ends)
         × stockout_cap      (cannot exceed available inventory)

Noise: Poisson-distributed integer demand.

PERFORMANCE NOTE:
  Full 300 stores × 7,500 SKUs × 731 days = 1.6B combinations.
  We generate ACTIVE store-product pairs only (not every store carries every SKU).
  Each store carries 800–1,200 products, giving ~300 × 1,000 × 731 ≈ 220M row-days.
  We skip rows where expected demand rounds to 0, reducing actual rows to ~15–25M.
"""

from datetime import date, datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd

from config import RANDOM_SEED, START_DATE, END_DATE

rng = np.random.default_rng(RANDOM_SEED + 5)

# ── Category seasonality: multiplier by month ────────────────
CATEGORY_SEASONALITY = {
    "Grocery": {1: 0.90, 2: 0.85, 3: 0.95, 4: 1.00, 5: 1.05, 6: 1.10,
                7: 1.15, 8: 1.10, 9: 1.00, 10: 1.05, 11: 1.30, 12: 1.40},
    "FMCG": {1: 1.05, 2: 0.95, 3: 0.98, 4: 1.00, 5: 1.02, 6: 1.00,
             7: 0.98, 8: 1.00, 9: 1.02, 10: 1.05, 11: 1.08, 12: 1.10},
    "Household": {1: 1.10, 2: 1.05, 3: 1.05, 4: 1.00, 5: 0.95, 6: 0.90,
                  7: 0.88, 8: 0.90, 9: 0.95, 10: 1.00, 11: 1.05, 12: 1.10},
    "Health & Beauty": {1: 1.20, 2: 1.10, 3: 1.00, 4: 0.95, 5: 0.90, 6: 0.95,
                        7: 0.98, 8: 1.00, 9: 1.00, 10: 1.05, 11: 1.10, 12: 1.15},
    "General Merchandise": {1: 0.80, 2: 0.80, 3: 0.90, 4: 0.95, 5: 1.00, 6: 1.00,
                             7: 1.00, 8: 1.10, 9: 1.05, 10: 1.05, 11: 1.40, 12: 1.80},
}

# ── Day-of-week factors (1=Mon … 7=Sun) ─────────────────────
DOW_FACTORS = {
    "Grocery":             {1: 0.80, 2: 0.80, 3: 0.85, 4: 0.90, 5: 1.05, 6: 1.30, 7: 1.30},
    "FMCG":                {1: 0.95, 2: 0.95, 3: 1.00, 4: 1.00, 5: 1.05, 6: 1.10, 7: 1.00},
    "Household":           {1: 0.90, 2: 0.90, 3: 0.95, 4: 1.00, 5: 1.05, 6: 1.15, 7: 1.15},
    "Health & Beauty":     {1: 0.95, 2: 0.95, 3: 1.00, 4: 1.00, 5: 1.05, 6: 1.10, 7: 1.05},
    "General Merchandise": {1: 0.85, 2: 0.85, 3: 0.90, 4: 0.95, 5: 1.05, 6: 1.25, 7: 1.30},
}

# ── Holiday demand multipliers ────────────────────────────────
HOLIDAY_MULTIPLIERS = {
    "Thanksgiving":        {"Grocery": 2.5, "FMCG": 1.4, "Household": 1.2, "Health & Beauty": 1.1, "General Merchandise": 1.5},
    "Black Friday":        {"Grocery": 1.2, "FMCG": 1.3, "Household": 1.3, "Health & Beauty": 1.3, "General Merchandise": 3.5},
    "Christmas Day":       {"Grocery": 1.8, "FMCG": 1.2, "Household": 1.2, "Health & Beauty": 1.3, "General Merchandise": 2.0},
    "Christmas Eve":       {"Grocery": 2.2, "FMCG": 1.3, "Household": 1.2, "Health & Beauty": 1.2, "General Merchandise": 2.5},
    "Super Bowl Sunday":   {"Grocery": 1.8, "FMCG": 1.2, "Household": 1.1, "Health & Beauty": 1.0, "General Merchandise": 1.2},
    "Easter":              {"Grocery": 1.5, "FMCG": 1.1, "Household": 1.1, "Health & Beauty": 1.1, "General Merchandise": 1.2},
    "Independence Day":    {"Grocery": 1.6, "FMCG": 1.1, "Household": 1.1, "Health & Beauty": 1.0, "General Merchandise": 1.3},
    "Halloween":           {"Grocery": 1.4, "FMCG": 1.0, "Household": 1.0, "Health & Beauty": 1.0, "General Merchandise": 1.5},
    "default":             {"Grocery": 1.2, "FMCG": 1.1, "Household": 1.1, "Health & Beauty": 1.1, "General Merchandise": 1.2},
}

# ── Subcategory-level weather effects ────────────────────────
# (temp_coeff: demand multiplier per 10°F above/below 65°F baseline)
WEATHER_EFFECTS = {
    # subcategory → (heat_coeff, cold_coeff, precip_coeff)
    # heat_coeff:  +X per 10°F above 75°F
    # cold_coeff:  +X per 10°F below 45°F
    # precip_coeff: +X per inch of precipitation
    "Beverages":        (0.12,  0.00,  0.00),
    "Fresh Produce":    (-0.05, -0.08, -0.05),
    "Frozen Foods":     (0.08,  0.00,  0.00),
    "Breakfast":        (0.00,  0.05,  0.02),
    "Snacks":           (0.05,  0.00,  0.00),
    "Soup":             (0.00,  0.15,  0.10),
    "Canned Goods":     (0.00,  0.05,  0.08),
    "Pain Relief":      (0.00,  0.08,  0.05),
    "Cold & Allergy":   (0.00,  0.12,  0.05),
    "Seasonal":         (0.00,  0.00,  0.00),  # override per season
    "default":          (0.00,  0.00,  0.00),
}

# ── Store volume factors (larger stores sell more) ───────────
FORMAT_VOLUME = {
    "Superstore": 1.40, "Neighborhood": 0.80, "Express": 0.40, "Warehouse": 1.80,
}


def _build_promo_lookup(promotions: pd.DataFrame) -> dict:
    """Returns dict: (product_id, date) → (promotion_id, discount_pct, lift_pct)"""
    lookup = {}
    for _, row in promotions.iterrows():
        d = row["start_date"]
        while d <= row["end_date"]:
            key = (int(row["product_id"]), d)
            if key not in lookup:
                lookup[key] = (int(row["promotion_id"]), float(row["discount_pct"]), float(row["expected_lift_pct"]))
            d += timedelta(days=1)
    return lookup


def _build_promo_end_lookup(promotions: pd.DataFrame) -> dict:
    """Returns dict: product_id → sorted list of end_dates."""
    ends = defaultdict(list)
    for _, row in promotions.iterrows():
        ends[int(row["product_id"])].append(row["end_date"])
    return {k: sorted(v) for k, v in ends.items()}


def _weather_factor(subcat: str, avg_temp: float, precip: float) -> float:
    heat_c, cold_c, precip_c = WEATHER_EFFECTS.get(subcat, WEATHER_EFFECTS["default"])
    factor = 1.0
    if avg_temp > 75:
        factor += heat_c * (avg_temp - 75) / 10
    if avg_temp < 45:
        factor += cold_c * (45 - avg_temp) / 10
    factor += precip_c * precip
    # Severe weather reduces trips overall (−5% per severity point above 1)
    return max(0.1, factor)


def generate_sales(
    stores: pd.DataFrame,
    products: pd.DataFrame,
    calendar: pd.DataFrame,
    promotions: pd.DataFrame,
    weather: pd.DataFrame,
    customers: pd.DataFrame,
) -> pd.DataFrame:

    start = date.fromisoformat(START_DATE)
    end   = date.fromisoformat(END_DATE)
    all_dates = sorted(calendar["date_id"].tolist())

    promo_lookup    = _build_promo_lookup(promotions)
    promo_end_dates = _build_promo_end_lookup(promotions)

    # Calendar lookup
    cal_lookup = calendar.set_index("date_id")[["day_of_week", "is_holiday", "holiday_name",
                                                  "is_weekend", "month", "season"]].to_dict("index")

    # Weather lookup: (store_id, date) → (avg_temp, precipitation, severity)
    wx_lookup = {}
    for _, row in weather.iterrows():
        wx_lookup[(int(row["store_id"]), row["date_id"])] = (
            float(row["avg_temp_f"]), float(row["precipitation_in"]), int(row["weather_severity"])
        )

    # Store metadata
    store_meta = {}
    for _, row in stores.iterrows():
        store_meta[int(row["store_id"])] = {
            "dc_id":   int(row["dc_id"]),
            "format":  row["store_format"],
        }

    # Product metadata
    prod_meta = {}
    for _, row in products.iterrows():
        base_rate = float(rng.uniform(0.5, 8.0))  # avg daily units sold in a baseline store
        prod_meta[int(row["product_id"])] = {
            "category":    row["category"],
            "subcategory": row["subcategory"],
            "unit_cost":   float(row["unit_cost"]),
            "unit_price":  float(row["unit_price"]),
            "is_perish":   bool(row["is_perishable"]),
            "base_rate":   base_rate,
        }

    # Assign each store a subset of products it carries (800–1,200 out of 7,500)
    all_product_ids = list(prod_meta.keys())
    n_all_products = len(all_product_ids)
    store_product_map = {}
    for sid in store_meta:
        n = min(int(rng.integers(800, 1201)), n_all_products)
        store_product_map[sid] = set(rng.choice(all_product_ids, size=n, replace=False).tolist())

    # Customer ID pool for loyalty attribution (~40% of transactions linked)
    customer_ids = customers["customer_id"].tolist()
    sample_size = min(50_000, len(customer_ids))
    customer_sample = rng.choice(customer_ids, size=sample_size, replace=False).tolist()

    # Opening inventory (units) per store × product
    inventory: dict[tuple, int] = {}
    for sid, pids in store_product_map.items():
        for pid in pids:
            avg_daily = prod_meta[pid]["base_rate"] * FORMAT_VOLUME[store_meta[sid]["format"]]
            dos = int(rng.integers(14, 31))
            inventory[(sid, pid)] = max(1, round(avg_daily * dos))

    # Post-promo dip tracker: (store_id, product_id) → days_remaining
    dip_tracker: dict[tuple, int] = {}

    records = []
    tid = 1

    print(f"  Generating sales for {len(all_dates)} dates × {len(store_meta)} stores...")

    for d in all_dates:
        cal = cal_lookup[d]
        month     = cal["month"]
        dow       = cal["day_of_week"]
        is_holiday = cal["is_holiday"]
        holiday_nm = cal["holiday_name"]

        holiday_mults = (HOLIDAY_MULTIPLIERS.get(holiday_nm, HOLIDAY_MULTIPLIERS["default"])
                         if is_holiday else None)

        for sid, pids in store_product_map.items():
            wx = wx_lookup.get((sid, d), (65.0, 0.0, 1))
            avg_temp, precip, severity = wx
            # Severe weather reduces overall store traffic
            trip_factor = 1.0 - (severity - 1) * 0.05

            for pid in pids:
                pm = prod_meta[pid]
                cat    = pm["category"]
                subcat = pm["subcategory"]
                key    = (sid, pid)

                # Inventory check
                inv = inventory.get(key, 0)
                if inv <= 0:
                    inventory[key] = 0
                    # Stockout: record a zero-sale marker? No — we just skip (lost sale)
                    # The is_stockout flag is set on inventory_snapshots instead.
                    continue

                # Build demand multiplier
                sea_factor  = CATEGORY_SEASONALITY[cat][month]
                dow_factor  = DOW_FACTORS[cat][dow]
                vol_factor  = FORMAT_VOLUME[store_meta[sid]["format"]]
                hol_factor  = holiday_mults[cat] if holiday_mults else 1.0
                wx_factor   = _weather_factor(subcat, avg_temp, precip)

                # Promotion
                promo_key  = (pid, d)
                promo_factor = 1.0
                discount_amt = 0.0
                promo_id     = None
                if promo_key in promo_lookup:
                    pid_promo, disc, lift = promo_lookup[promo_key]
                    promo_factor = lift
                    discount_amt = pm["unit_price"] * disc
                    promo_id     = pid_promo

                # Post-promo dip
                dip_factor = 1.0
                if key in dip_tracker and dip_tracker[key] > 0:
                    dip_factor = 0.85
                    dip_tracker[key] -= 1
                    if dip_tracker[key] == 0:
                        del dip_tracker[key]
                elif promo_id is None and pid in promo_end_dates:
                    # Check if promo just ended (within 3 days)
                    for end_dt in promo_end_dates[pid]:
                        if 0 < (d - end_dt).days <= 3:
                            dip_tracker[key] = 3 - (d - end_dt).days
                            dip_factor = 0.85
                            break

                mean_demand = (
                    pm["base_rate"]
                    * vol_factor
                    * sea_factor
                    * dow_factor
                    * hol_factor
                    * promo_factor
                    * wx_factor
                    * dip_factor
                    * trip_factor
                )

                if mean_demand <= 0.05:
                    continue

                # Poisson draw for integer demand
                qty = int(rng.poisson(mean_demand))
                if qty == 0:
                    continue

                # Cap at available inventory
                is_stockout_day = False
                if qty >= inv:
                    qty = inv
                    is_stockout_day = True
                inventory[key] = inv - qty

                # Customer attribution (~40%)
                cust_id = int(rng.choice(customer_sample)) if rng.random() < 0.40 else None

                up = pm["unit_price"]
                uc = pm["unit_cost"]
                net_rev = round(qty * (up - discount_amt), 2)

                records.append({
                    "transaction_id":  tid,
                    "store_id":        sid,
                    "product_id":      pid,
                    "customer_id":     cust_id,
                    "date_id":         d,
                    "transaction_time": datetime.combine(d, datetime.min.time()).replace(
                        hour=int(rng.integers(6, 22)),
                        minute=int(rng.integers(0, 60)),
                    ),
                    "quantity_sold":   qty,
                    "unit_price":      round(up, 2),
                    "unit_cost":       round(uc, 2),
                    "discount_amt":    round(discount_amt, 2),
                    "net_revenue":     net_rev,
                    "promotion_id":    promo_id,
                    "is_stockout_day": is_stockout_day,
                })
                tid += 1

                # Batch flush every 500k rows to avoid memory explosion
                if len(records) >= 500_000:
                    yield pd.DataFrame(records)
                    records = []

    if records:
        yield pd.DataFrame(records)

    print(f"  Sales generation complete. Total transactions: {tid - 1:,}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from generate_dimensions import (generate_distribution_centers, generate_stores,
                                     generate_products, generate_customers, generate_calendar)
    from generate_promotions import generate_promotions
    from generate_weather import generate_weather

    dcs   = generate_distribution_centers()
    dcs["dc_id"] = range(1, len(dcs) + 1)
    stores = generate_stores(dcs)
    stores["store_id"] = range(1, len(stores) + 1)
    prods  = generate_products()
    prods["product_id"] = range(1, len(prods) + 1)
    cust   = generate_customers()
    cust["customer_id"] = range(1, len(cust) + 1)
    cal    = generate_calendar()
    promos = generate_promotions(prods)
    promos["promotion_id"] = range(1, len(promos) + 1)
    wx     = generate_weather(stores)

    total = 0
    for batch in generate_sales(stores, prods, cal, promos, wx, cust):
        total += len(batch)
        print(f"  Batch: {len(batch):,}  Running total: {total:,}")
        if total >= 1_000_000:
            print("  (sample run — stopping at 1M rows)")
            break
