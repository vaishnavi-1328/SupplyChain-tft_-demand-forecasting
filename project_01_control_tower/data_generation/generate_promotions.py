"""
Generate the promotions table.
~10% of products on promotion in any given week, clustered around holidays/seasons.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd

from config import RANDOM_SEED

rng = np.random.default_rng(RANDOM_SEED + 1)

PROMO_TYPES   = ["Price_Cut", "BOGO", "Feature", "Display", "Circular"]
PROMO_WEIGHTS = [0.40, 0.15, 0.20, 0.15, 0.10]

# Discount depth by category
DISCOUNT_RANGE = {
    "Grocery":             (0.05, 0.30),
    "FMCG":                (0.10, 0.35),
    "Household":           (0.10, 0.35),
    "Health & Beauty":     (0.10, 0.40),
    "General Merchandise": (0.15, 0.40),
}

# Lift multiplier by promo type and discount depth
BASE_LIFT = {
    "Price_Cut": 1.3, "BOGO": 1.8, "Feature": 1.4, "Display": 1.25, "Circular": 1.5,
}

# Campaign definitions: (name, weeks_before_holiday, duration_days, pct_products_on_promo)
CAMPAIGNS = [
    # Grocery peaks
    ("Thanksgiving_Push",    1, 14, 0.25),
    ("Christmas_Holiday",    3, 21, 0.20),
    ("Summer_BBQ",          0, 28, 0.15),   # runs July 4 window
    ("Back_to_School",      0, 21, 0.12),   # Aug
    ("Super_Bowl",          1,  7, 0.12),
    ("Easter_Spring",       2, 14, 0.12),
    ("Labor_Day_Weekend",   1,  5, 0.10),
    ("Memorial_Day_Weekend",1,  5, 0.10),
    ("Valentine_Day",       2,  7, 0.08),
    # General clearance cycles (every 8 weeks)
    ("Clearance_Q1",        0, 14, 0.08),
    ("Clearance_Q2",        0, 14, 0.08),
    ("Clearance_Q3",        0, 14, 0.08),
    ("Clearance_Q4",        0, 14, 0.08),
]

# Fixed start dates for clearance rounds (not holiday-relative)
CLEARANCE_STARTS = {
    "Clearance_Q1": [date(2023, 2, 6), date(2024, 2, 5)],
    "Clearance_Q2": [date(2023, 5, 8), date(2024, 5, 6)],
    "Clearance_Q3": [date(2023, 8, 7), date(2024, 8, 5)],
    "Clearance_Q4": [date(2023, 11, 6), date(2024, 11, 4)],
}

HOLIDAY_ANCHORS_2023 = {
    "Thanksgiving_Push": date(2023, 11, 23),
    "Christmas_Holiday": date(2023, 12, 25),
    "Summer_BBQ":        date(2023, 7, 4),
    "Back_to_School":    date(2023, 8, 15),
    "Super_Bowl":        date(2023, 2, 12),
    "Easter_Spring":     date(2023, 3, 26) + timedelta(weeks=2),
    "Labor_Day_Weekend": date(2023, 9, 4),
    "Memorial_Day_Weekend": date(2023, 5, 29),
    "Valentine_Day":     date(2023, 2, 14),
}
HOLIDAY_ANCHORS_2024 = {
    "Thanksgiving_Push": date(2024, 11, 28),
    "Christmas_Holiday": date(2024, 12, 25),
    "Summer_BBQ":        date(2024, 7, 4),
    "Back_to_School":    date(2024, 8, 15),
    "Super_Bowl":        date(2024, 2, 11),
    "Easter_Spring":     date(2024, 3, 31) + timedelta(weeks=2),
    "Labor_Day_Weekend": date(2024, 9, 2),
    "Memorial_Day_Weekend": date(2024, 5, 27),
    "Valentine_Day":     date(2024, 2, 14),
}


def _campaign_starts(name: str, weeks_before: int, duration_days: int) -> list:
    """Return list of (start_date, end_date) across 2023–2024 for a campaign."""
    result = []
    for anchors in [HOLIDAY_ANCHORS_2023, HOLIDAY_ANCHORS_2024]:
        if name in CLEARANCE_STARTS:
            for sd in CLEARANCE_STARTS[name]:
                result.append((sd, sd + timedelta(days=duration_days - 1)))
        elif name in anchors:
            anchor = anchors[name]
            sd = anchor - timedelta(weeks=weeks_before)
            result.append((sd, sd + timedelta(days=duration_days - 1)))
    return result


def generate_promotions(products: pd.DataFrame) -> pd.DataFrame:
    records = []
    pid_counter = 1

    for name, weeks_before, duration, pct in CAMPAIGNS:
        windows = _campaign_starts(name, weeks_before, duration)
        if not windows:
            continue

        for start_date, end_date in windows:
            # Select a random subset of products for this campaign
            n_promos = max(1, int(len(products) * pct))
            selected = products.sample(n=n_promos, random_state=int(rng.integers(1e6))).copy()

            for _, prod in selected.iterrows():
                cat = prod["category"]
                disc_min, disc_max = DISCOUNT_RANGE.get(cat, (0.05, 0.30))
                discount_pct = round(float(rng.uniform(disc_min, disc_max)), 4)
                promo_type   = str(rng.choice(PROMO_TYPES, p=PROMO_WEIGHTS))
                base_lift    = BASE_LIFT[promo_type]
                # Lift scales with discount depth
                expected_lift = round(float(base_lift * (1 + discount_pct * 2)), 4)

                records.append({
                    "promotion_id":      pid_counter,
                    "promotion_name":    name,
                    "product_id":        int(prod["product_id"]),
                    "store_id":          None,   # chain-wide promotion
                    "start_date":        start_date,
                    "end_date":          end_date,
                    "discount_pct":      discount_pct,
                    "promotion_type":    promo_type,
                    "expected_lift_pct": expected_lift,
                    "is_active":         True,
                })
                pid_counter += 1

    df = pd.DataFrame(records)
    print(f"  Promotions: {len(df):,} rows across {df['promotion_name'].nunique()} campaigns")
    return df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from generate_dimensions import generate_products
    products = generate_products()
    promos = generate_promotions(products)
    print(promos.groupby("promotion_name")["promotion_id"].count())
