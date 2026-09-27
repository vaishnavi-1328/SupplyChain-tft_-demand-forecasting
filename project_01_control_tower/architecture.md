# Architecture: Supply Chain Control Tower

## The 7-Layer Data Architecture

This document describes the data flow architecture for the Supply Chain Control Tower. Learn this well — it is a standard supply chain DS interview question.

```
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 1 — Source Systems (Operational / OLTP)                          │
│                                                                         │
│  POS System          → sales transactions (1 row per item per sale)     │
│  WMS (Warehouse Mgmt)→ inventory snapshots, inbound/outbound shipments  │
│  OMS (Order Mgmt)    → purchase orders, purchase order lines            │
│  Supplier Portal     → shipment tracking events                         │
│  Weather API         → daily weather per location                       │
│  Merchandising Sys   → promotions, pricing                              │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │  CDC or batch extract (daily)
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 2 — Ingestion & Raw Landing Zone                                 │
│                                                                         │
│  Real-world tools: Azure Data Factory, Fivetran, Airbyte, Spark         │
│  Pattern: Extract → Land in raw zone (S3/ADLS) → Validate → Load        │
│                                                                         │
│  This project: Python scripts simulate this layer. run_all.py           │
│  orchestrates generation and loads directly into PostgreSQL.            │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 3 — Data Warehouse (what we built in Project 1)                  │
│                                                                         │
│  Database: PostgreSQL (managed via Ghost MCP)                           │
│  Schema: supply_chain                                                   │
│  Pattern: Star Schema                                                   │
│                                                                         │
│  Dimension tables (slow-changing, descriptive):                         │
│    distribution_centers, stores, suppliers, products,                   │
│    customers, calendar                                                  │
│                                                                         │
│  Relationship table:                                                    │
│    product_supplier (products × suppliers with lead time metadata)      │
│                                                                         │
│  Fact tables (append-only, time-stamped events):                        │
│    sales_transactions, inventory_snapshots, purchase_orders,            │
│    purchase_order_lines, shipments, shipment_events,                    │
│    promotions, weather, returns, spoilage                               │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 4 — Feature / Analytical Layer (Project 1 views + Project 2+)   │
│                                                                         │
│  SQL views (pre-aggregated, query-optimized):                           │
│    v_daily_store_sales     — daily aggregated sales per store-product   │
│    v_supplier_performance  — OTIF, fill rate, lead time per supplier    │
│    v_inventory_health      — current stock status with alert flags      │
│    v_promotion_lift        — measured vs. expected lift per promo       │
│    v_stockout_calendar     — stockout episode durations                 │
│                                                                         │
│  Project 2+: feature tables for ML (rolling demand, lead-time stats,   │
│  safety stock calculations, supplier risk scores)                       │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 5 — ML Layer (Projects 3–6)                                      │
│                                                                         │
│  Demand forecasting models (ARIMA, XGBoost, LightGBM, probabilistic)   │
│  Supplier risk classification                                           │
│  Stockout root cause classification                                     │
│  Experiment tracking: MLflow                                            │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 6 — Optimization Layer (Projects 7–10)                           │
│                                                                         │
│  Replenishment decisions (EOQ, safety stock, reorder points)            │
│  Inventory allocation (linear/integer programming via OR-Tools)         │
│  Multi-echelon inventory positioning                                    │
│  DC capacity simulation                                                 │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 7 — Decision Layer (Project 12)                                  │
│                                                                         │
│  API: FastAPI serving recommendations                                   │
│  Dashboard: Streamlit (Supply Chain Decision Intelligence Platform)     │
│  What-if scenario engine                                                │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Why Star Schema?

A **star schema** organizes data as a central fact table surrounded by dimension tables.

**Advantages for supply chain analytics:**

1. **Query performance**: Joining one fact table to pre-built dimension tables is faster than navigating a normalized 3NF schema with many joins.
2. **Consistency**: Conformed dimensions (calendar, products) are shared across all fact tables. One source of truth for product definitions.
3. **Familiarity**: Every major analytics platform (Snowflake, BigQuery, Databricks, Power BI) is optimized for star schemas.
4. **Interpretability**: Easy to explain to business stakeholders — "we have a sales table and a products table, and you join them."

**The tradeoff**: Denormalization means some data duplication. Acceptable in a warehouse where read performance matters more than write efficiency.

---

## Key Design Decisions

### Partitioned sales_transactions table
`sales_transactions` is partitioned by month. With 50–80M rows over 2 years, this means:
- A query for "December 2023 sales" scans only 1 partition, not the full table
- Archiving old data is a simple partition detach
- This mirrors how real retailers manage petabyte-scale transaction tables

### calendar as a dimension (not a function)
Never compute `EXTRACT(DOW FROM date)` inline in queries. Pre-materializing date attributes:
- Eliminates repetitive computation across every query
- Makes `is_holiday`, `fiscal_week`, `season` available for free
- Standard in every production data warehouse

### Shipment events as a log table
`shipment_events` is an append-only event log — not a status column on the shipments table. This mirrors real carrier APIs (FedEx, UPS, project44) which stream status events. It preserves history: you can reconstruct exactly what happened at each step of a delayed shipment.

### inventory_snapshots grain
We snapshot inventory once per day (end-of-day). Real WMS systems may do this every 15 minutes. Daily is sufficient for strategic replenishment decisions; intraday granularity is needed for store replenishment during the day (a more complex simulation).

---

## OLTP vs. OLAP — the key concept for interviews

| | OLTP (Source Systems) | OLAP (Data Warehouse) |
|---|---|---|
| Purpose | Run the business | Analyze the business |
| Write pattern | Frequent small writes | Bulk loads |
| Read pattern | Row-level lookups | Large aggregations |
| Schema | Normalized (3NF) | Denormalized (star) |
| Examples | POS, OMS, WMS | Snowflake, BigQuery, PostgreSQL analytics |

In this project, we simulate OLTP data and load it directly into an OLAP warehouse. In a real company, this is done via ETL pipelines (Azure Data Factory, Fivetran, dbt).

---

## The Data Flow for One Sale

To answer "where does a sale come from, end to end":

```
Supplier ships product
  → Purchase Order created in OMS
  → Shipment departs supplier (shipment_events: Dispatched)
  → Shipment arrives at DC (shipment_events: Delivered)
  → DC ships to store (outbound shipment)
  → Store receives inventory (inventory_snapshots.units_on_hand increases)
  → Customer purchases product (sales_transactions)
  → Inventory decreases (inventory_snapshots.units_on_hand decreases)
  → If inventory hits zero → is_stockout = TRUE
```

You can trace this entire path using joins across our 17 tables.
