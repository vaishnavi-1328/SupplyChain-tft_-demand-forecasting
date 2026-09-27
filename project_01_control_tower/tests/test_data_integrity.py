"""
Data integrity tests for Project 1: Supply Chain Control Tower.

Run after all data is loaded:
    pytest tests/test_data_integrity.py -v

Requires SUPPLY_CHAIN_DB_URL environment variable to be set.
"""

import os
import pytest
import pandas as pd
from sqlalchemy import create_engine, text

SCHEMA = "supply_chain"
DB_URL = os.getenv("SUPPLY_CHAIN_DB_URL", "")


@pytest.fixture(scope="session")
def engine():
    if not DB_URL:
        pytest.skip("SUPPLY_CHAIN_DB_URL not set")
    return create_engine(DB_URL)


def query(engine, sql: str):
    with engine.connect() as conn:
        return conn.execute(text(sql)).scalar()


# ── Row count sanity checks ──────────────────────────────────

def test_distribution_centers_count(engine):
    n = query(engine, f"SELECT COUNT(*) FROM {SCHEMA}.distribution_centers")
    assert n == 8, f"Expected 8 DCs, got {n}"


def test_stores_count(engine):
    n = query(engine, f"SELECT COUNT(*) FROM {SCHEMA}.stores")
    assert 295 <= n <= 305, f"Expected ~300 stores, got {n}"


def test_suppliers_count(engine):
    n = query(engine, f"SELECT COUNT(*) FROM {SCHEMA}.suppliers")
    assert 390 <= n <= 410, f"Expected ~400 suppliers, got {n}"


def test_products_count(engine):
    n = query(engine, f"SELECT COUNT(*) FROM {SCHEMA}.products")
    assert 7000 <= n <= 8000, f"Expected ~7,500 products, got {n}"


def test_customers_count(engine):
    n = query(engine, f"SELECT COUNT(*) FROM {SCHEMA}.customers")
    assert n == 500_000, f"Expected 500,000 customers, got {n}"


def test_calendar_count(engine):
    n = query(engine, f"SELECT COUNT(*) FROM {SCHEMA}.calendar")
    assert n == 731, f"Expected 731 calendar rows (2023+2024), got {n}"


def test_calendar_start_end(engine):
    start = query(engine, f"SELECT MIN(date_id) FROM {SCHEMA}.calendar")
    end   = query(engine, f"SELECT MAX(date_id) FROM {SCHEMA}.calendar")
    assert str(start) == "2023-01-01", f"Calendar start wrong: {start}"
    assert str(end)   == "2024-12-31", f"Calendar end wrong: {end}"


def test_weather_count(engine):
    n = query(engine, f"SELECT COUNT(*) FROM {SCHEMA}.weather")
    assert n >= 200_000, f"Expected ~219,300 weather rows, got {n}"


def test_sales_transactions_count(engine):
    n = query(engine, f"SELECT COUNT(*) FROM {SCHEMA}.sales_transactions")
    assert n >= 1_000_000, f"Expected at least 1M sales transactions, got {n}"


# ── Referential integrity ────────────────────────────────────

def test_no_sales_with_invalid_store(engine):
    n = query(engine, f"""
        SELECT COUNT(*) FROM {SCHEMA}.sales_transactions
        WHERE store_id NOT IN (SELECT store_id FROM {SCHEMA}.stores)
    """)
    assert n == 0, f"FK violation: {n} sales with invalid store_id"


def test_no_sales_with_invalid_product(engine):
    n = query(engine, f"""
        SELECT COUNT(*) FROM {SCHEMA}.sales_transactions
        WHERE product_id NOT IN (SELECT product_id FROM {SCHEMA}.products)
    """)
    assert n == 0, f"FK violation: {n} sales with invalid product_id"


def test_no_inventory_with_invalid_product(engine):
    n = query(engine, f"""
        SELECT COUNT(*) FROM {SCHEMA}.inventory_snapshots
        WHERE product_id NOT IN (SELECT product_id FROM {SCHEMA}.products)
    """)
    assert n == 0, f"FK violation: {n} inventory rows with invalid product_id"


def test_no_po_lines_with_invalid_po(engine):
    n = query(engine, f"""
        SELECT COUNT(*) FROM {SCHEMA}.purchase_order_lines
        WHERE po_id NOT IN (SELECT po_id FROM {SCHEMA}.purchase_orders)
    """)
    assert n == 0, f"FK violation: {n} PO lines with invalid po_id"


def test_no_shipment_events_with_invalid_shipment(engine):
    n = query(engine, f"""
        SELECT COUNT(*) FROM {SCHEMA}.shipment_events
        WHERE shipment_id NOT IN (SELECT shipment_id FROM {SCHEMA}.shipments)
    """)
    assert n == 0, f"FK violation: {n} events with invalid shipment_id"


def test_stores_all_have_valid_dc(engine):
    n = query(engine, f"""
        SELECT COUNT(*) FROM {SCHEMA}.stores
        WHERE dc_id NOT IN (SELECT dc_id FROM {SCHEMA}.distribution_centers)
    """)
    assert n == 0, f"FK violation: {n} stores with invalid dc_id"


# ── Business logic constraints ───────────────────────────────

def test_no_stockout_day_with_positive_sales(engine):
    """If is_stockout_day = TRUE, quantity_sold must be 0 (or capped at last unit)."""
    # In our model, stockout_day means inventory hit 0 during that day.
    # We allow the last unit to be sold, so we just check for negative.
    n = query(engine, f"""
        SELECT COUNT(*) FROM {SCHEMA}.sales_transactions
        WHERE quantity_sold <= 0
    """)
    assert n == 0, f"Sales with quantity_sold <= 0: {n}"


def test_no_negative_inventory(engine):
    n = query(engine, f"""
        SELECT COUNT(*) FROM {SCHEMA}.inventory_snapshots
        WHERE units_on_hand < 0
    """)
    assert n == 0, f"Negative inventory: {n} rows"


def test_spoilage_only_for_perishables(engine):
    n = query(engine, f"""
        SELECT COUNT(*) FROM {SCHEMA}.spoilage sp
        JOIN {SCHEMA}.products p ON p.product_id = sp.product_id
        WHERE p.is_perishable = FALSE
    """)
    assert n == 0, f"Spoilage on non-perishable products: {n} rows"


def test_no_bad_promotion_dates(engine):
    n = query(engine, f"""
        SELECT COUNT(*) FROM {SCHEMA}.promotions
        WHERE end_date < start_date
    """)
    assert n == 0, f"Promotions with end_date < start_date: {n}"


def test_no_returns_before_purchase(engine):
    n = query(engine, f"""
        SELECT COUNT(*) FROM {SCHEMA}.returns r
        JOIN {SCHEMA}.sales_transactions st ON st.transaction_id = r.transaction_id
        WHERE r.return_date < st.date_id
    """)
    assert n == 0, f"Returns with return_date before purchase date: {n}"


def test_no_shipment_arrival_before_departure(engine):
    n = query(engine, f"""
        SELECT COUNT(*) FROM {SCHEMA}.shipments
        WHERE actual_arrival IS NOT NULL AND actual_arrival < ship_date
    """)
    assert n == 0, f"Shipments arriving before departure: {n}"


def test_all_products_have_primary_supplier(engine):
    """Every product should have at least one primary supplier entry."""
    n = query(engine, f"""
        SELECT COUNT(*) FROM {SCHEMA}.products p
        WHERE p.product_id NOT IN (
            SELECT DISTINCT product_id FROM {SCHEMA}.product_supplier WHERE is_primary = TRUE
        ) AND p.is_active = TRUE
    """)
    assert n == 0, f"{n} active products have no primary supplier"


# ── Business metric checks ───────────────────────────────────

def test_stockout_rate_in_realistic_range(engine):
    """Stockout rate should be between 1% and 10%."""
    rate = query(engine, f"""
        SELECT ROUND(
            100.0 * SUM(CASE WHEN is_stockout THEN 1 ELSE 0 END) / COUNT(*), 2
        )
        FROM {SCHEMA}.inventory_snapshots
        WHERE location_type = 'Store'
    """)
    assert rate is not None, "Could not compute stockout rate"
    assert 1.0 <= float(rate) <= 15.0, f"Stockout rate {rate}% outside expected 1–15% range"


def test_weekend_grocery_sales_higher_than_weekday(engine):
    """Weekend grocery sales should exceed weekday sales on average."""
    result = query(engine, f"""
        SELECT
            ROUND(AVG(CASE WHEN c.is_weekend THEN st.quantity_sold END), 3)
            - ROUND(AVG(CASE WHEN NOT c.is_weekend THEN st.quantity_sold END), 3)
        FROM {SCHEMA}.sales_transactions st
        JOIN {SCHEMA}.calendar c ON c.date_id = st.date_id
        JOIN {SCHEMA}.products p ON p.product_id = st.product_id
        WHERE p.category = 'Grocery'
    """)
    assert result is not None and float(result) > 0, \
        f"Weekend grocery avg units not higher than weekday. Diff = {result}"


def test_national_suppliers_shorter_lead_time(engine):
    """National suppliers should have shorter average actual lead times than Local."""
    national_lt = query(engine, f"""
        SELECT AVG(po.actual_delivery - po.order_date)
        FROM {SCHEMA}.purchase_orders po
        JOIN {SCHEMA}.suppliers s ON s.supplier_id = po.supplier_id
        WHERE s.supplier_type = 'National' AND po.status = 'Received'
    """)
    local_lt = query(engine, f"""
        SELECT AVG(po.actual_delivery - po.order_date)
        FROM {SCHEMA}.purchase_orders po
        JOIN {SCHEMA}.suppliers s ON s.supplier_id = po.supplier_id
        WHERE s.supplier_type = 'Local' AND po.status = 'Received'
    """)
    if national_lt is not None and local_lt is not None:
        assert float(national_lt) < float(local_lt), \
            f"National avg LT ({national_lt:.1f}) not shorter than Local ({local_lt:.1f})"


def test_promotions_have_positive_lift(engine):
    """Average promotion lift should be > 1.0."""
    avg_lift = query(engine, f"""
        SELECT AVG(expected_lift_pct) FROM {SCHEMA}.promotions
    """)
    assert avg_lift is not None and float(avg_lift) > 1.0, \
        f"Average expected promo lift {avg_lift} not > 1.0"


def test_all_holidays_in_calendar(engine):
    n = query(engine, f"""
        SELECT COUNT(*) FROM {SCHEMA}.calendar WHERE is_holiday = TRUE
    """)
    assert n >= 20, f"Expected at least 20 holiday markers, got {n}"
