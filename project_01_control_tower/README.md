# Project 1 — Supply Chain Control Tower

## Business Context

RetailSim Co. operates 300 stores, 8 distribution centers, and sources 7,500 SKUs from 400 suppliers across the US. Before building ML models or optimization engines, we need a reliable, queryable foundation: a data warehouse that mirrors what a real retail supply chain analytics team would use day-to-day.

This project builds that foundation from scratch.

---

## What We Built

A fully simulated, 17-table star-schema data warehouse spanning 2 years (Jan 2023 – Dec 2024) with:

| Table | Rows (approx) | What it represents |
|---|---|---|
| distribution_centers | 8 | The 8 regional DC hubs |
| stores | 300 | Retail stores across the US |
| suppliers | 400 | National, regional, and local suppliers |
| products | 7,500 | SKUs across 5 categories |
| customers | 500,000 | Loyalty card holders |
| calendar | 731 | One row per date, with holidays and fiscal flags |
| product_supplier | ~9,000 | Which suppliers supply which products |
| promotions | ~8,000 | Promotional campaigns by product and window |
| weather | ~219,300 | Daily weather per store |
| purchase_orders | ~180,000 | POs placed by DCs with suppliers |
| purchase_order_lines | ~720,000 | Individual product lines per PO |
| shipments | ~200,000 | Supplier→DC and DC→Store shipments |
| shipment_events | ~600,000 | Event log per shipment |
| sales_transactions | ~15–25M | Individual product sales |
| inventory_snapshots | ~20M | Daily end-of-day inventory per store-product |
| returns | ~500,000 | Product returns |
| spoilage | ~400,000 | Perishable product waste events |

---

## Architecture

See `architecture.md` for the full 7-layer architecture diagram.

**Short version:**
```
Source Systems → Ingestion → Data Warehouse → Feature Layer → ML → Optimization → Decision Layer
```

We built Layer 3 (the data warehouse) in this project. All subsequent projects build upward from here.

---

## How to Run

### Prerequisites

```bash
pip install sqlalchemy psycopg2-binary pandas numpy faker pytest matplotlib
```

### 1. Provision PostgreSQL

Use Ghost MCP to create a managed PostgreSQL instance, then:

```bash
export SUPPLY_CHAIN_DB_URL="postgresql://user:pass@host:5432/dbname"
```

### 2. Create the schema

```bash
cd project_01_control_tower
psql $SUPPLY_CHAIN_DB_URL -f sql/00_create_schema.sql
psql $SUPPLY_CHAIN_DB_URL -f sql/01_create_dimensions.sql
psql $SUPPLY_CHAIN_DB_URL -f sql/02_create_facts.sql
psql $SUPPLY_CHAIN_DB_URL -f sql/03_create_relationships.sql
psql $SUPPLY_CHAIN_DB_URL -f sql/04_create_indexes.sql
```

### 3. Generate and load data

```bash
cd data_generation
python run_all.py --verify
```

This will take 30–60 minutes (the sales generation step is the bottleneck).

### 4. Create analytical views

```bash
psql $SUPPLY_CHAIN_DB_URL -f sql/05_create_views.sql
```

### 5. Run integrity tests

```bash
pytest tests/test_data_integrity.py -v
```

### 6. Explore the data

Open the notebooks in order:
1. `exploration/01_schema_exploration.ipynb`
2. `exploration/02_sales_eda.ipynb`
3. `exploration/03_inventory_eda.ipynb`
4. `exploration/04_supplier_eda.ipynb`
5. `exploration/05_promotion_eda.ipynb`

---

## Key Design Decisions

| Decision | Why |
|---|---|
| Star schema | Analytics performance; mirrors Snowflake/BigQuery at real retailers |
| `calendar` as a dimension | Pre-computed date attributes; no inline EXTRACT() in queries |
| `is_stockout_day` on sales_transactions | Captures the lost-sale signal directly — needed for Project 2 |
| Partitioned `sales_transactions` | 50M+ rows require partition pruning for acceptable query speed |
| Shipment events as a log | Preserves full shipment history; mirrors real carrier APIs |
| Spoilage as its own table | Event-based; keeps inventory ledger clean |
| Causal demand model | Demand depends on seasonality, weather, promotions — not random noise |

---

## Sample Questions This Data Can Answer

```sql
-- How much did we lose to stockouts last November?
SELECT SUM(p.unit_price * st.quantity_sold) AS potential_lost_revenue
FROM supply_chain.sales_transactions st
JOIN supply_chain.products p ON p.product_id = st.product_id
WHERE st.is_stockout_day = TRUE
  AND st.date_id BETWEEN '2023-11-01' AND '2023-11-30';

-- Which suppliers have the worst OTIF rate?
SELECT supplier_name, supplier_type, otif_rate, total_pos
FROM supply_chain.v_supplier_performance
ORDER BY otif_rate ASC LIMIT 10;

-- What is the current inventory health across all stores?
SELECT inventory_status, COUNT(*) AS store_product_count
FROM supply_chain.v_inventory_health
GROUP BY inventory_status ORDER BY store_product_count DESC;

-- Did the Thanksgiving promotion actually lift sales?
SELECT promotion_name, category, avg_lift_factor
FROM supply_chain.v_promotion_lift
WHERE promotion_name LIKE '%Thanksgiving%';
```

---

## What I Learned

1. **Star schema design**: How to separate what happened (facts) from who/what was involved (dimensions). Why conformed dimensions matter.
2. **Data grain**: Every table design starts with "what does one row represent?" Getting this wrong corrupts every downstream analysis.
3. **Causal simulation**: Random data is useless for ML. Building demand as a function of real drivers (seasonality, weather, promotions) is what makes models learnable.
4. **Supply chain data flow**: How a sale traces back through shipments → purchase orders → suppliers. The complete data lineage.
5. **Partitioning**: Why large fact tables need partitioning and how PostgreSQL range partitioning works.
6. **Analytical views**: The difference between a raw table and a pre-aggregated view for analyst consumption.

---

## Next Project

**Project 2 — "Why Did We Run Out?"**

With this data warehouse in place, we can now build a Stockout Investigation Engine: given a specific SKU × Store × Date that was out of stock, determine the most likely root cause from the 12 potential causes baked into this simulation.
