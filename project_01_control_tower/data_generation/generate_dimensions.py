"""
Generate all dimension tables:
  distribution_centers, stores, suppliers, products, customers, calendar
"""

import math
import random
from datetime import date, timedelta

import numpy as np
import pandas as pd
from faker import Faker

from config import (
    RANDOM_SEED, START_DATE, END_DATE, NUM_STORES, NUM_PRODUCTS, NUM_SUPPLIERS,
    NUM_CUSTOMERS, NUM_NATIONAL_SUPPLIERS, NUM_REGIONAL_SUPPLIERS, NUM_LOCAL_SUPPLIERS,
    CATEGORY_DIST, LEAD_TIME, SUPPLIER_RELIABILITY, DC_HUBS, STATE_REGIONS,
)

fake = Faker("en_US")
Faker.seed(RANDOM_SEED)
rng = np.random.default_rng(RANDOM_SEED)
random.seed(RANDOM_SEED)


# ─────────────────────────────────────────────
# distribution_centers
# ─────────────────────────────────────────────

def generate_distribution_centers() -> pd.DataFrame:
    return pd.DataFrame(DC_HUBS)


# ─────────────────────────────────────────────
# stores  (clustered around DC hubs)
# ─────────────────────────────────────────────

STORE_FORMATS = ["Superstore", "Neighborhood", "Express", "Warehouse"]
STORE_FORMAT_WEIGHTS = [0.40, 0.35, 0.15, 0.10]
FORMAT_SIZE = {
    "Superstore":   (60_000, 120_000),
    "Neighborhood": (20_000,  40_000),
    "Express":       (5_000,  15_000),
    "Warehouse":   (100_000, 200_000),
}
FORMAT_REVENUE = {
    "Superstore": "High", "Neighborhood": "Medium", "Express": "Low", "Warehouse": "High",
}

# US state bounding boxes (approx lat/lon ranges) per region
REGION_STATES = {
    "Northeast":    [("NY", 40.5, 45.0, -79.8, -72.0), ("PA", 39.7, 42.3, -80.5, -74.7),
                     ("MA", 41.5, 42.9, -73.5, -70.0), ("NJ", 38.9, 41.4, -75.6, -73.9),
                     ("CT", 41.0, 42.1, -73.7, -71.8)],
    "Southeast":    [("FL", 24.4, 31.0, -87.6, -80.0), ("GA", 30.4, 35.0, -85.6, -80.8),
                     ("NC", 33.8, 36.6, -84.3, -75.5), ("VA", 36.5, 39.5, -83.7, -75.2),
                     ("TN", 34.9, 36.7, -90.3, -81.6)],
    "Great Lakes":  [("OH", 38.4, 41.9, -84.8, -80.5), ("MI", 41.7, 48.2, -90.4, -82.4),
                     ("IL", 36.9, 42.5, -91.5, -87.0), ("WI", 42.5, 47.3, -92.9, -86.2)],
    "Midwest":      [("MO", 36.0, 40.6, -95.8, -89.1), ("MN", 43.5, 49.4, -97.2, -89.5),
                     ("IA", 40.4, 43.5, -96.6, -90.1), ("KS", 36.9, 40.0, -102.1, -94.6)],
    "South Central":[("TX", 25.8, 36.5, -106.6, -93.5), ("OK", 33.6, 37.0, -103.0, -94.4),
                     ("LA", 28.9, 33.0, -94.0, -89.0)],
    "Mountain":     [("CO", 36.9, 41.0, -109.0, -102.0), ("AZ", 31.3, 37.0, -114.8, -109.0),
                     ("UT", 36.9, 42.0, -114.1, -109.0)],
    "Southwest":    [("CA", 32.5, 42.0, -124.5, -114.1)],
    "Northwest":    [("WA", 45.5, 49.0, -124.8, -116.9), ("OR", 41.9, 46.3, -124.6, -116.5)],
}


def _random_lat_lon_in_region(region: str, rng: np.random.Generator):
    states = REGION_STATES.get(region, list(REGION_STATES.values())[0])
    state_info = states[rng.integers(len(states))]
    _, lat_min, lat_max, lon_min, lon_max = state_info
    lat = rng.uniform(lat_min, lat_max)
    lon = rng.uniform(lon_min, lon_max)
    return state_info[0], round(lat, 6), round(lon, 6)


def generate_stores(dcs: pd.DataFrame) -> pd.DataFrame:
    records = []
    # Distribute 300 stores proportionally across DCs (roughly by capacity)
    total_cap = dcs["capacity_pallets"].sum()
    counts = (dcs["capacity_pallets"] / total_cap * NUM_STORES).round().astype(int).tolist()
    # Adjust to exactly NUM_STORES
    diff = NUM_STORES - sum(counts)
    counts[0] += diff

    store_id = 1
    for _, dc_row in dcs.iterrows():
        n = counts[dcs.index.get_loc(dc_row.name)]
        region = dc_row["region"]
        for _ in range(n):
            fmt = rng.choice(STORE_FORMATS, p=STORE_FORMAT_WEIGHTS)
            size_min, size_max = FORMAT_SIZE[fmt]
            state, lat, lon = _random_lat_lon_in_region(region, rng)
            city = fake.city()
            zip_code = fake.zipcode()
            opened = date(
                rng.integers(2000, 2022).item(),
                rng.integers(1, 13).item(),
                rng.integers(1, 29).item()
            )
            records.append({
                "store_id":            store_id,
                "store_code":          f"ST-{store_id:04d}",
                "store_name":          f"{city} {fmt}",
                "city":                city,
                "state":               state,
                "zip_code":            zip_code,
                "latitude":            lat,
                "longitude":           lon,
                "store_format":        fmt,
                "store_size_sqft":     int(rng.integers(size_min, size_max)),
                "annual_revenue_band": FORMAT_REVENUE[fmt],
                "dc_id":               int(dc_row["dc_id"]),
                "opened_date":         opened,
                "is_active":           True,
            })
            store_id += 1

    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# suppliers
# ─────────────────────────────────────────────

SUPPLIER_STATES = list(STATE_REGIONS.keys())


def _supplier_profile(stype: str, i: int, rng: np.random.Generator) -> dict:
    lt = LEAD_TIME[stype]
    rel = SUPPLIER_RELIABILITY[stype]
    state = SUPPLIER_STATES[rng.integers(len(SUPPLIER_STATES))]
    lat = rng.uniform(25.0, 49.0)
    lon = rng.uniform(-125.0, -67.0)
    return {
        "supplier_name":       fake.company() + (" Corp." if stype == "National" else ""),
        "supplier_type":       stype,
        "city":                fake.city(),
        "state":               state,
        "country":             "USA",
        "latitude":            round(float(lat), 6),
        "longitude":           round(float(lon), 6),
        "lead_time_days_avg":  round(float(rng.uniform(lt["avg_min"], lt["avg_max"])), 2),
        "lead_time_days_std":  round(float(rng.uniform(lt["std_min"], lt["std_max"])), 2),
        "fill_rate_baseline":  round(float(rng.uniform(*rel["fill_rate"])), 4),
        "otif_baseline":       round(float(rng.uniform(*rel["otif"])), 4),
        "disruption_prob":     round(float(rng.uniform(*rel["disruption_prob"])), 4),
        "payment_terms_days":  int(rng.choice([15, 30, 45, 60], p=[0.1, 0.5, 0.3, 0.1])),
        "is_active":           True,
    }


def generate_suppliers() -> pd.DataFrame:
    records = []
    sid = 1
    for stype, count in [("National", NUM_NATIONAL_SUPPLIERS),
                          ("Regional", NUM_REGIONAL_SUPPLIERS),
                          ("Local",    NUM_LOCAL_SUPPLIERS)]:
        for i in range(count):
            rec = _supplier_profile(stype, i, rng)
            rec["supplier_id"] = sid
            records.append(rec)
            sid += 1
    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# products
# ─────────────────────────────────────────────

SUBCATEGORIES = {
    "Grocery": ["Fresh Produce", "Dairy", "Bakery", "Meat & Seafood", "Frozen Foods",
                "Canned Goods", "Breakfast", "Snacks", "Beverages", "Condiments"],
    "FMCG": ["Laundry", "Dish Care", "Paper Products", "Air Fresheners",
              "Baby Care", "Pet Food", "Tobacco", "Batteries & Bulbs", "Oral Care", "Shaving"],
    "Household": ["Cleaning Supplies", "Storage & Organization", "Kitchen Tools",
                   "Candles & Decor", "Trash Bags", "Pest Control", "Automotive", "Hardware"],
    "Health & Beauty": ["Pain Relief", "Cold & Allergy", "Vitamins", "First Aid",
                         "Skin Care", "Hair Care", "Cosmetics", "Feminine Care", "Foot Care", "Sunscreen"],
    "General Merchandise": ["Electronics Accessories", "Seasonal", "Toys & Games",
                              "Clothing Basics", "School Supplies", "Sports & Fitness",
                              "Garden", "Books & Magazines"],
}

BRANDS = {
    "Grocery": ["FreshHarvest", "GreenField", "SunValley", "HomeBaked", "OceanCatch", "NaturePath"],
    "FMCG": ["CleanPro", "FreshScent", "PowerWash", "DailyEase", "BrightHome", "PureTouch"],
    "Household": ["HomeSense", "OrganizeIt", "CleanMate", "SafeHome", "EcoHouse", "TidyPlus"],
    "Health & Beauty": ["VitaHealth", "PureSkin", "HairPro", "WellBeing", "NatureCare", "ClearMed"],
    "General Merchandise": ["EverReady", "TechBasics", "SeasonRight", "LearnFun", "ActiveGear", "GardenGrow"],
}

UOM_BY_CATEGORY = {
    "Grocery": ["each", "lb", "oz", "liter", "kg", "pack"],
    "FMCG": ["each", "pack", "oz", "liter"],
    "Household": ["each", "pack", "oz"],
    "Health & Beauty": ["each", "oz", "pack", "ml"],
    "General Merchandise": ["each", "pack"],
}

# Price ranges (unit_cost, unit_price) by subcategory type
PRICE_RANGES = {
    "Fresh Produce": (0.50, 3.50, 0.99, 6.99),
    "Dairy": (1.20, 5.00, 2.49, 8.99),
    "Bakery": (1.00, 4.00, 1.99, 7.99),
    "Meat & Seafood": (3.00, 15.00, 5.99, 24.99),
    "Frozen Foods": (1.50, 7.00, 2.99, 12.99),
    "default": (0.80, 20.00, 1.49, 34.99),
}

PERISHABLE_SHELF_LIFE = {
    "Fresh Produce": (3, 7),
    "Dairy": (7, 21),
    "Bakery": (3, 10),
    "Meat & Seafood": (3, 7),
    "Frozen Foods": (60, 365),
    "default": (14, 90),
}


def generate_products() -> pd.DataFrame:
    records = []
    cat_counts = {cat: round(NUM_PRODUCTS * frac) for cat, frac in CATEGORY_DIST.items()}
    cat_counts["Grocery"] += NUM_PRODUCTS - sum(cat_counts.values())  # fix rounding

    pid = 1
    intro_start = date(2018, 1, 1)
    intro_end   = date(2022, 12, 31)
    intro_range = (intro_end - intro_start).days

    for cat, count in cat_counts.items():
        subcats = SUBCATEGORIES[cat]
        brands  = BRANDS[cat]
        uoms    = UOM_BY_CATEGORY[cat]
        for i in range(count):
            subcat = subcats[i % len(subcats)]
            is_perishable = (cat == "Grocery") or (cat == "Health & Beauty" and rng.random() < 0.10)
            shelf_life = None
            if is_perishable:
                lo, hi = PERISHABLE_SHELF_LIFE.get(subcat, PERISHABLE_SHELF_LIFE["default"])
                shelf_life = int(rng.integers(lo, hi + 1))

            pr = PRICE_RANGES.get(subcat, PRICE_RANGES["default"])
            unit_cost  = round(float(rng.uniform(pr[0], pr[1])), 2)
            unit_price = round(float(rng.uniform(max(unit_cost * 1.15, pr[2]), pr[3])), 2)

            category_prefix = cat[:3].upper().replace(" ", "")
            sku = f"{category_prefix}-{pid:05d}"

            intro_date = intro_start + timedelta(days=int(rng.integers(0, intro_range)))

            records.append({
                "product_id":       pid,
                "sku":              sku,
                "product_name":     f"{rng.choice(brands)} {subcat} #{i+1}",
                "category":         cat,
                "subcategory":      subcat,
                "brand":            str(rng.choice(brands)),
                "is_perishable":    bool(is_perishable),
                "shelf_life_days":  shelf_life,
                "unit_of_measure":  str(rng.choice(uoms)),
                "unit_cost":        unit_cost,
                "unit_price":       unit_price,
                "weight_kg":        round(float(rng.uniform(0.05, 10.0)), 3),
                "volume_liters":    round(float(rng.uniform(0.05, 5.0)), 3),
                "min_order_qty":    int(rng.choice([1, 6, 12, 24], p=[0.5, 0.2, 0.2, 0.1])),
                "case_pack_qty":    int(rng.choice([1, 6, 12, 24, 48], p=[0.3, 0.2, 0.25, 0.15, 0.1])),
                "abc_class":        None,  # set after sales data exists
                "is_active":        True,
                "introduced_date":  intro_date,
                "discontinued_date":None,
            })
            pid += 1

    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# customers
# ─────────────────────────────────────────────

LOYALTY_TIERS  = ["Gold", "Silver", "Bronze", "None"]
LOYALTY_WEIGHTS = [0.10, 0.20, 0.35, 0.35]


def generate_customers() -> pd.DataFrame:
    records = []
    acq_start = date(2015, 1, 1)
    acq_range = (date(2023, 12, 31) - acq_start).days
    for cid in range(1, NUM_CUSTOMERS + 1):
        state = SUPPLIER_STATES[rng.integers(len(SUPPLIER_STATES))]
        acq_date = acq_start + timedelta(days=int(rng.integers(0, acq_range)))
        records.append({
            "customer_id":      cid,
            "customer_name":    fake.name(),
            "email":            fake.email(),
            "city":             fake.city(),
            "state":            state,
            "zip_code":         fake.zipcode(),
            "loyalty_tier":     str(rng.choice(LOYALTY_TIERS, p=LOYALTY_WEIGHTS)),
            "acquisition_date": acq_date,
            "is_active":        True,
        })
    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# calendar
# ─────────────────────────────────────────────

US_HOLIDAYS_2023_2024 = {
    date(2023,  1,  1): "New Year's Day",
    date(2023,  1, 16): "MLK Day",
    date(2023,  2, 20): "Presidents Day",
    date(2023,  5, 29): "Memorial Day",
    date(2023,  6, 19): "Juneteenth",
    date(2023,  7,  4): "Independence Day",
    date(2023,  9,  4): "Labor Day",
    date(2023, 10,  9): "Columbus Day",
    date(2023, 11, 10): "Veterans Day (observed)",
    date(2023, 11, 23): "Thanksgiving",
    date(2023, 11, 24): "Black Friday",
    date(2023, 12, 24): "Christmas Eve",
    date(2023, 12, 25): "Christmas Day",
    date(2023, 12, 31): "New Year's Eve",
    date(2024,  1,  1): "New Year's Day",
    date(2024,  1, 15): "MLK Day",
    date(2024,  2, 19): "Presidents Day",
    date(2024,  5, 27): "Memorial Day",
    date(2024,  6, 19): "Juneteenth",
    date(2024,  7,  4): "Independence Day",
    date(2024,  9,  2): "Labor Day",
    date(2024, 10, 14): "Columbus Day",
    date(2024, 11, 11): "Veterans Day",
    date(2024, 11, 28): "Thanksgiving",
    date(2024, 11, 29): "Black Friday",
    date(2024, 12, 24): "Christmas Eve",
    date(2024, 12, 25): "Christmas Day",
    date(2024, 12, 31): "New Year's Eve",
    # Retail events
    date(2023,  2, 12): "Super Bowl Sunday",
    date(2024,  2, 11): "Super Bowl Sunday",
    date(2023,  3, 26): "Easter",
    date(2024,  3, 31): "Easter",
    date(2023,  5, 14): "Mother's Day",
    date(2024,  5, 12): "Mother's Day",
    date(2023,  6, 18): "Father's Day",
    date(2024,  6, 16): "Father's Day",
    date(2023, 10, 31): "Halloween",
    date(2024, 10, 31): "Halloween",
    date(2023, 12, 26): "Day After Christmas",
    date(2024, 12, 26): "Day After Christmas",
}

SEASON_MAP = {1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
              6: "Summer", 7: "Summer", 8: "Summer", 9: "Fall", 10: "Fall",
              11: "Fall", 12: "Winter"}


def generate_calendar() -> pd.DataFrame:
    start = date.fromisoformat(START_DATE)
    end   = date.fromisoformat(END_DATE)
    records = []
    d = start
    while d <= end:
        iso_cal = d.isocalendar()
        month   = d.month
        # Retail fiscal week (simplified: 52-week year starting Feb 1)
        fiscal_start = date(d.year, 2, 1)
        if d < fiscal_start:
            fiscal_start = date(d.year - 1, 2, 1)
        fiscal_week = min(52, (d - fiscal_start).days // 7 + 1)
        fiscal_period = (fiscal_week - 1) // 4 + 1  # simplified 4-4-4

        records.append({
            "date_id":        d,
            "year":           d.year,
            "quarter":        (month - 1) // 3 + 1,
            "month":          month,
            "month_name":     d.strftime("%B"),
            "week_of_year":   iso_cal.week,
            "day_of_week":    iso_cal.weekday,  # 1=Mon … 7=Sun
            "day_name":       d.strftime("%A"),
            "is_weekend":     iso_cal.weekday >= 6,
            "is_holiday":     d in US_HOLIDAYS_2023_2024,
            "holiday_name":   US_HOLIDAYS_2023_2024.get(d),
            "is_month_start": d.day == 1,
            "is_month_end":   (d + timedelta(days=1)).month != month,
            "fiscal_week":    fiscal_week,
            "fiscal_period":  fiscal_period,
            "season":         SEASON_MAP[month],
        })
        d += timedelta(days=1)
    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# product_supplier  (relationship table)
# ─────────────────────────────────────────────

def generate_product_supplier(products: pd.DataFrame, suppliers: pd.DataFrame) -> pd.DataFrame:
    national = suppliers[suppliers["supplier_type"] == "National"]["supplier_id"].tolist()
    regional = suppliers[suppliers["supplier_type"] == "Regional"]["supplier_id"].tolist()
    local    = suppliers[suppliers["supplier_type"] == "Local"]["supplier_id"].tolist()

    records = []
    eff_date = date.fromisoformat(START_DATE) - timedelta(days=90)

    for _, prod in products.iterrows():
        cat = prod["category"]
        # National suppliers available to all; regional/local are category-weighted
        pool_national = national
        pool_regional = regional
        pool_local    = local

        # Each product gets 1–3 suppliers
        n_suppliers = int(rng.choice([1, 2, 3], p=[0.50, 0.35, 0.15]))
        # Always include 1 primary
        supplier_pool = (
            rng.choice(pool_national, min(1, len(pool_national)), replace=False).tolist()
            + rng.choice(pool_regional, min(n_suppliers - 1, len(pool_regional)), replace=False).tolist()
        )
        if len(supplier_pool) > n_suppliers:
            supplier_pool = supplier_pool[:n_suppliers]
        if len(supplier_pool) < 1:
            supplier_pool = rng.choice(pool_national, 1, replace=False).tolist()

        primary_sid = supplier_pool[0]
        for i, sid in enumerate(supplier_pool):
            s_row = suppliers[suppliers["supplier_id"] == sid].iloc[0]
            lt = round(float(rng.normal(s_row["lead_time_days_avg"], s_row["lead_time_days_std"] / 2)), 2)
            lt = max(1.0, lt)
            records.append({
                "product_id":         int(prod["product_id"]),
                "supplier_id":        int(sid),
                "is_primary":         (sid == primary_sid),
                "lead_time_days":     lt,
                "unit_cost_override": None,
                "min_order_qty":      int(prod["min_order_qty"]),
                "effective_date":     eff_date,
                "expiry_date":        None,
            })

    return pd.DataFrame(records)


if __name__ == "__main__":
    print("Generating dimensions...")
    dcs       = generate_distribution_centers()
    stores    = generate_stores(dcs)
    suppliers = generate_suppliers()
    products  = generate_products()
    customers = generate_customers()
    calendar  = generate_calendar()
    prod_sup  = generate_product_supplier(products, suppliers)

    for name, df in [("DCs", dcs), ("Stores", stores), ("Suppliers", suppliers),
                     ("Products", products), ("Customers", customers),
                     ("Calendar", calendar), ("ProductSupplier", prod_sup)]:
        print(f"  {name}: {len(df):,} rows")
    print("Done.")
