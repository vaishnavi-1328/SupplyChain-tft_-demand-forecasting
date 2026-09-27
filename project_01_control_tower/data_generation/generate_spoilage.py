"""
Generate the spoilage table (perishables only).

Spoilage is triggered when:
1. Inventory sits unsold beyond a product's shelf_life_days (expiry)
2. Post-promotion demand dip leaves excess perishable inventory
3. Severe weather (temperature break) triggers accelerated spoilage
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd

from config import RANDOM_SEED

rng = np.random.default_rng(RANDOM_SEED + 8)

SPOILAGE_REASONS = ["Expired", "Overstock", "Temperature_Break", "Damaged"]
REASON_WEIGHTS   = [0.50,     0.25,        0.15,                 0.10]


def generate_spoilage(
    stores: pd.DataFrame,
    products: pd.DataFrame,
    inventory_snapshots: pd.DataFrame,
    weather: pd.DataFrame,
) -> pd.DataFrame:

    perishable_pids = set(products[products["is_perishable"] == True]["product_id"].tolist())
    if not perishable_pids:
        return pd.DataFrame()

    # Shelf life map
    shelf_life_map = products.set_index("product_id")["shelf_life_days"].to_dict()
    cost_map       = products.set_index("product_id")["unit_cost"].to_dict()

    # Weather severity lookup
    wx_sev = weather.set_index(["store_id", "date_id"])["weather_severity"].to_dict()

    # Filter inventory snapshots to perishables only
    perishable_inv = inventory_snapshots[
        inventory_snapshots["product_id"].isin(perishable_pids) &
        (inventory_snapshots["location_type"] == "Store")
    ].copy()

    if perishable_inv.empty:
        return pd.DataFrame()

    records = []
    sp_id = 1

    # Group by store × product, look at inventory trajectory
    for (sid, pid), grp in perishable_inv.groupby(["store_id", "product_id"]):
        grp = grp.sort_values("snapshot_date")
        shelf_life = shelf_life_map.get(pid, 7)
        uc = float(cost_map.get(pid, 1.0))

        for _, row in grp.iterrows():
            d         = row["snapshot_date"]
            units_oh  = int(row["units_on_hand"])
            severity  = wx_sev.get((int(sid), d), 1)

            # Expiry: units that have been sitting for shelf_life days accumulate spoilage
            # Approximation: on any given day, expected spoilage ≈ (1 / shelf_life) × units_on_hand
            # with higher rates when overstocked (DOS > 2 × shelf_life)
            dos = float(row["days_of_supply"]) if row["days_of_supply"] else 0

            base_spoilage_rate = 1.0 / max(shelf_life, 1)
            if dos > 2 * shelf_life:
                base_spoilage_rate *= 2.0  # overstock accelerates spoilage

            # Weather temperature break
            if severity >= 3:
                base_spoilage_rate *= 1.5

            expected_spoilage = units_oh * base_spoilage_rate
            if expected_spoilage < 0.1:
                continue

            spoiled = int(rng.poisson(expected_spoilage))
            if spoiled <= 0:
                continue
            spoiled = min(spoiled, units_oh)

            if severity >= 3:
                reason = "Temperature_Break"
            elif dos > 2 * shelf_life:
                reason = "Overstock"
            else:
                reason = str(rng.choice(SPOILAGE_REASONS, p=REASON_WEIGHTS))

            records.append({
                "spoilage_id":     sp_id,
                "store_id":        int(sid),
                "product_id":      int(pid),
                "spoilage_date":   d,
                "units_spoiled":   spoiled,
                "spoilage_reason": reason,
                "cost_of_spoilage":round(spoiled * uc, 2),
            })
            sp_id += 1

    df = pd.DataFrame(records)
    print(f"  Spoilage: {len(df):,} rows")
    return df
