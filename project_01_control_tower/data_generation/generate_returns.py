"""
Generate the returns table.
Return rate: 1–3% of transactions by category.
Return date: 1–14 days after original purchase.
"""

import numpy as np
import pandas as pd
from datetime import timedelta

from config import RANDOM_SEED, END_DATE

rng = np.random.default_rng(RANDOM_SEED + 7)

RETURN_RATE = {
    "Grocery":             0.01,
    "FMCG":                0.02,
    "Household":           0.025,
    "Health & Beauty":     0.02,
    "General Merchandise": 0.03,
}

RETURN_REASONS_BY_CAT = {
    "Grocery":             ["Spoiled", "Defective", "Wrong_Item"],
    "FMCG":                ["Defective", "Changed_Mind", "Wrong_Item", "Damaged"],
    "Household":           ["Defective", "Changed_Mind", "Damaged", "Wrong_Item"],
    "Health & Beauty":     ["Defective", "Changed_Mind", "Wrong_Item"],
    "General Merchandise": ["Changed_Mind", "Defective", "Damaged", "Wrong_Item"],
}

RETURN_REASON_WEIGHTS = {
    "Grocery":             [0.50, 0.35, 0.15],
    "FMCG":                [0.40, 0.30, 0.20, 0.10],
    "Household":           [0.30, 0.35, 0.20, 0.15],
    "Health & Beauty":     [0.40, 0.40, 0.20],
    "General Merchandise": [0.50, 0.25, 0.15, 0.10],
}


def generate_returns(
    sales: pd.DataFrame,
    products: pd.DataFrame,
) -> pd.DataFrame:
    end_date = pd.to_datetime(END_DATE).date()

    # Category lookup
    cat_map = products.set_index("product_id")["category"].to_dict()
    price_map = products.set_index("product_id")["unit_price"].to_dict()

    records = []
    ret_id = 1

    for cat, rate in RETURN_RATE.items():
        cat_sales = sales[sales["product_id"].map(cat_map) == cat]
        if cat_sales.empty:
            continue

        n_returns = max(1, round(len(cat_sales) * rate))
        sampled   = cat_sales.sample(n=min(n_returns, len(cat_sales)),
                                      random_state=int(rng.integers(1e6)))

        reasons = RETURN_REASONS_BY_CAT[cat]
        weights = RETURN_REASON_WEIGHTS[cat]

        for _, row in sampled.iterrows():
            lag_days = int(rng.integers(1, 15))
            ret_date = row["date_id"] + timedelta(days=lag_days)
            if ret_date > end_date:
                ret_date = end_date

            qty_ret = min(int(row["quantity_sold"]), max(1, round(int(rng.integers(1, int(row["quantity_sold"]) + 1)))))
            reason  = str(rng.choice(reasons, p=weights))
            restock = (reason not in ("Spoiled", "Expired"))

            records.append({
                "return_id":         ret_id,
                "transaction_id":    int(row["transaction_id"]),
                "store_id":          int(row["store_id"]),
                "product_id":        int(row["product_id"]),
                "return_date":       ret_date,
                "quantity_returned": qty_ret,
                "return_reason":     reason,
                "refund_amount":     round(qty_ret * float(price_map.get(int(row["product_id"]), row["unit_price"])), 2),
                "restocked":         bool(restock),
            })
            ret_id += 1

    df = pd.DataFrame(records)
    print(f"  Returns: {len(df):,} rows")
    return df
