import os

RANDOM_SEED = 42
START_DATE = "2023-01-01"
END_DATE = "2024-12-31"

NUM_STORES = 300
NUM_DCS = 8
NUM_PRODUCTS = 7500
NUM_SUPPLIERS = 400
NUM_CUSTOMERS = 500_000

# Category distribution (must sum to 1.0)
CATEGORY_DIST = {
    "Grocery":             0.30,
    "FMCG":                0.25,
    "Household":           0.20,
    "Health & Beauty":     0.15,
    "General Merchandise": 0.10,
}

# Supplier type counts
NUM_NATIONAL_SUPPLIERS  = 50
NUM_REGIONAL_SUPPLIERS  = 200
NUM_LOCAL_SUPPLIERS     = 150

# Lead time ranges by supplier type (days)
LEAD_TIME = {
    "National":  {"avg_min": 3,  "avg_max": 7,  "std_min": 0.5, "std_max": 1.5},
    "Regional":  {"avg_min": 5,  "avg_max": 14, "std_min": 1.0, "std_max": 3.0},
    "Local":     {"avg_min": 7,  "avg_max": 21, "std_min": 2.0, "std_max": 5.0},
}

# Supplier reliability by type
SUPPLIER_RELIABILITY = {
    "National": {"fill_rate": (0.95, 0.99), "otif": (0.92, 0.98), "disruption_prob": (0.005, 0.015)},
    "Regional": {"fill_rate": (0.88, 0.96), "otif": (0.82, 0.94), "disruption_prob": (0.015, 0.04)},
    "Local":    {"fill_rate": (0.78, 0.92), "otif": (0.70, 0.88), "disruption_prob": (0.03, 0.08)},
}

# PO confirmation rates by supplier type
PO_CONFIRM_RATE = {
    "National": 0.98,
    "Regional": 0.92,
    "Local":    0.85,
}

# Opening inventory: days of supply on Jan 1 2023
OPENING_DOS_MIN = 14
OPENING_DOS_MAX = 30

# Batch size for DB inserts
BATCH_SIZE = 10_000

# PostgreSQL connection — set DB_URL in environment or .env file
DB_URL = os.getenv("SUPPLY_CHAIN_DB_URL", "")

SCHEMA = "supply_chain"

# 8 DC hub cities (real US logistics hubs)
DC_HUBS = [
    {"dc_code": "DC-NE",  "dc_name": "Northeast DC",    "city": "Philadelphia",  "state": "PA", "latitude": 39.9526, "longitude": -75.1652, "capacity_pallets": 45000, "region": "Northeast"},
    {"dc_code": "DC-SE",  "dc_name": "Southeast DC",    "city": "Atlanta",       "state": "GA", "latitude": 33.7490, "longitude": -84.3880, "capacity_pallets": 50000, "region": "Southeast"},
    {"dc_code": "DC-GL",  "dc_name": "Great Lakes DC",  "city": "Columbus",      "state": "OH", "latitude": 39.9612, "longitude": -82.9988, "capacity_pallets": 42000, "region": "Great Lakes"},
    {"dc_code": "DC-MW",  "dc_name": "Midwest DC",      "city": "Kansas City",   "state": "MO", "latitude": 39.0997, "longitude": -94.5786, "capacity_pallets": 48000, "region": "Midwest"},
    {"dc_code": "DC-TX",  "dc_name": "Texas DC",        "city": "Dallas",        "state": "TX", "latitude": 32.7767, "longitude": -96.7970, "capacity_pallets": 55000, "region": "South Central"},
    {"dc_code": "DC-MT",  "dc_name": "Mountain DC",     "city": "Denver",        "state": "CO", "latitude": 39.7392, "longitude": -104.9903,"capacity_pallets": 35000, "region": "Mountain"},
    {"dc_code": "DC-SW",  "dc_name": "Southwest DC",    "city": "Los Angeles",   "state": "CA", "latitude": 34.0522, "longitude": -118.2437,"capacity_pallets": 60000, "region": "Southwest"},
    {"dc_code": "DC-NW",  "dc_name": "Northwest DC",    "city": "Seattle",       "state": "WA", "latitude": 47.6062, "longitude": -122.3321,"capacity_pallets": 38000, "region": "Northwest"},
]

# US states with regional assignment for store generation
STATE_REGIONS = {
    "ME": "Northeast", "NH": "Northeast", "VT": "Northeast", "MA": "Northeast",
    "RI": "Northeast", "CT": "Northeast", "NY": "Northeast", "NJ": "Northeast",
    "PA": "Northeast", "DE": "Northeast", "MD": "Northeast",
    "VA": "Southeast", "NC": "Southeast", "SC": "Southeast", "GA": "Southeast",
    "FL": "Southeast", "AL": "Southeast", "MS": "Southeast", "TN": "Southeast",
    "KY": "Southeast", "WV": "Southeast",
    "OH": "Great Lakes", "MI": "Great Lakes", "IN": "Great Lakes",
    "IL": "Great Lakes", "WI": "Great Lakes", "MN": "Great Lakes",
    "MO": "Midwest", "IA": "Midwest", "KS": "Midwest", "NE": "Midwest",
    "SD": "Midwest", "ND": "Midwest",
    "TX": "South Central", "OK": "South Central", "AR": "South Central", "LA": "South Central",
    "CO": "Mountain", "UT": "Mountain", "NM": "Mountain", "AZ": "Mountain",
    "ID": "Mountain", "MT": "Mountain", "WY": "Mountain", "NV": "Mountain",
    "CA": "Southwest", "HI": "Southwest",
    "WA": "Northwest", "OR": "Northwest", "AK": "Northwest",
}
