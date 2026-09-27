-- Analytical views — the Level 1 query layer.
-- Run AFTER all data has been loaded.

SET search_path TO supply_chain;

-- ─────────────────────────────────────────────────────────────
-- v_daily_store_sales
-- Aggregated daily sales per store × product.
-- Avoids scanning raw transactions for most analytical queries.
-- ─────────────────────────────────────────────────────────────
CREATE OR REPLACE VIEW v_daily_store_sales AS
SELECT
    st.date_id,
    st.store_id,
    s.store_name,
    s.state,
    s.store_format,
    st.product_id,
    p.sku,
    p.product_name,
    p.category,
    p.subcategory,
    p.is_perishable,
    SUM(st.quantity_sold)                           AS total_units_sold,
    SUM(st.net_revenue)                             AS total_net_revenue,
    SUM(st.discount_amt * st.quantity_sold)         AS total_discount,
    COUNT(st.transaction_id)                        AS transaction_count,
    BOOL_OR(st.is_stockout_day)                     AS had_stockout,
    BOOL_OR(st.promotion_id IS NOT NULL)            AS was_promoted,
    c.is_holiday,
    c.is_weekend,
    c.season
FROM sales_transactions st
JOIN stores    s ON s.store_id   = st.store_id
JOIN products  p ON p.product_id = st.product_id
JOIN calendar  c ON c.date_id    = st.date_id
GROUP BY
    st.date_id, st.store_id, s.store_name, s.state, s.store_format,
    st.product_id, p.sku, p.product_name, p.category, p.subcategory, p.is_perishable,
    c.is_holiday, c.is_weekend, c.season;


-- ─────────────────────────────────────────────────────────────
-- v_supplier_performance
-- Per-supplier KPIs derived from purchase orders.
-- Used in Project 6: Supplier Risk & Reliability Engine.
-- ─────────────────────────────────────────────────────────────
CREATE OR REPLACE VIEW v_supplier_performance AS
SELECT
    po.supplier_id,
    s.supplier_name,
    s.supplier_type,
    COUNT(po.po_id)                                                         AS total_pos,
    SUM(po.total_units)                                                     AS total_units_ordered,
    AVG(po.actual_delivery - po.order_date)                                 AS avg_actual_lead_days,
    AVG(po.requested_delivery - po.order_date)                              AS avg_requested_lead_days,
    AVG(po.late_days)                                                       AS avg_late_days,
    ROUND(AVG(CASE WHEN po.is_late THEN 1.0 ELSE 0.0 END)::NUMERIC, 4)     AS late_rate,
    ROUND(1 - AVG(CASE WHEN po.is_late THEN 1.0 ELSE 0.0 END)::NUMERIC, 4) AS otif_rate,
    SUM(po.total_cost)                                                      AS total_spend,
    COUNT(CASE WHEN po.status = 'Cancelled' THEN 1 END)                    AS cancelled_pos
FROM purchase_orders po
JOIN suppliers s ON s.supplier_id = po.supplier_id
WHERE po.status IN ('Received', 'Cancelled')
GROUP BY po.supplier_id, s.supplier_name, s.supplier_type;


-- ─────────────────────────────────────────────────────────────
-- v_inventory_health
-- Latest inventory snapshot per store × product with alert flags.
-- ─────────────────────────────────────────────────────────────
CREATE OR REPLACE VIEW v_inventory_health AS
WITH latest AS (
    SELECT
        store_id,
        product_id,
        MAX(snapshot_date) AS latest_date
    FROM inventory_snapshots
    WHERE location_type = 'Store'
    GROUP BY store_id, product_id
)
SELECT
    i.store_id,
    s.store_name,
    s.state,
    i.product_id,
    p.sku,
    p.product_name,
    p.category,
    p.is_perishable,
    i.snapshot_date,
    i.units_on_hand,
    i.units_on_order,
    i.units_in_transit,
    i.units_on_hand + i.units_in_transit                AS inventory_position,
    i.days_of_supply,
    i.reorder_point,
    i.safety_stock,
    i.is_stockout,
    CASE
        WHEN i.is_stockout THEN 'Stockout'
        WHEN i.units_on_hand <= i.safety_stock THEN 'Below Safety Stock'
        WHEN i.units_on_hand <= i.reorder_point THEN 'Reorder Now'
        WHEN i.days_of_supply <= 3 THEN 'Critical'
        ELSE 'Healthy'
    END AS inventory_status
FROM inventory_snapshots i
JOIN latest  l ON l.store_id = i.store_id AND l.product_id = i.product_id AND l.latest_date = i.snapshot_date
JOIN stores  s ON s.store_id   = i.store_id
JOIN products p ON p.product_id = i.product_id
WHERE i.location_type = 'Store';


-- ─────────────────────────────────────────────────────────────
-- v_promotion_lift
-- Compares promoted sales to a 4-week pre-promotion baseline.
-- Used in Project 5: The Promotion Trap.
-- ─────────────────────────────────────────────────────────────
CREATE OR REPLACE VIEW v_promotion_lift AS
WITH promo_sales AS (
    SELECT
        pr.promotion_id,
        pr.promotion_name,
        pr.product_id,
        pr.start_date,
        pr.end_date,
        pr.discount_pct,
        pr.promotion_type,
        SUM(st.quantity_sold) AS promo_units,
        COUNT(DISTINCT st.date_id) AS promo_days
    FROM promotions pr
    JOIN sales_transactions st
        ON  st.product_id    = pr.product_id
        AND st.promotion_id  = pr.promotion_id
        AND st.date_id BETWEEN pr.start_date AND pr.end_date
    GROUP BY pr.promotion_id, pr.promotion_name, pr.product_id, pr.start_date, pr.end_date, pr.discount_pct, pr.promotion_type
),
baseline_sales AS (
    SELECT
        pr.promotion_id,
        pr.product_id,
        SUM(st.quantity_sold) AS baseline_units,
        COUNT(DISTINCT st.date_id) AS baseline_days
    FROM promotions pr
    JOIN sales_transactions st
        ON  st.product_id = pr.product_id
        AND st.date_id BETWEEN pr.start_date - 28 AND pr.start_date - 1
        AND st.promotion_id IS NULL
    GROUP BY pr.promotion_id, pr.product_id
)
SELECT
    ps.promotion_id,
    ps.promotion_name,
    ps.product_id,
    p.product_name,
    p.category,
    ps.start_date,
    ps.end_date,
    ps.discount_pct,
    ps.promotion_type,
    ROUND(ps.promo_units::NUMERIC / NULLIF(ps.promo_days, 0), 2)        AS avg_daily_promo_units,
    ROUND(bs.baseline_units::NUMERIC / NULLIF(bs.baseline_days, 0), 2)  AS avg_daily_baseline_units,
    ROUND(
        (ps.promo_units::NUMERIC / NULLIF(ps.promo_days, 0))
        / NULLIF(bs.baseline_units::NUMERIC / NULLIF(bs.baseline_days, 0), 0)
    , 3) AS lift_factor
FROM promo_sales  ps
LEFT JOIN baseline_sales bs ON bs.promotion_id = ps.promotion_id AND bs.product_id = ps.product_id
JOIN products p ON p.product_id = ps.product_id;


-- ─────────────────────────────────────────────────────────────
-- v_stockout_calendar
-- Stockout episodes per store × product: start date, end date, duration.
-- Used in Project 2: Why Did We Run Out?
-- ─────────────────────────────────────────────────────────────
CREATE OR REPLACE VIEW v_stockout_calendar AS
WITH stockout_days AS (
    SELECT
        store_id,
        product_id,
        snapshot_date,
        is_stockout,
        -- island detection: consecutive stockout days belong to same episode
        snapshot_date
            - (ROW_NUMBER() OVER (PARTITION BY store_id, product_id, is_stockout ORDER BY snapshot_date))::INTEGER
            AS grp
    FROM inventory_snapshots
    WHERE location_type = 'Store' AND is_stockout = TRUE
)
SELECT
    sd.store_id,
    s.store_name,
    sd.product_id,
    p.sku,
    p.product_name,
    p.category,
    MIN(sd.snapshot_date) AS stockout_start,
    MAX(sd.snapshot_date) AS stockout_end,
    COUNT(*)              AS stockout_days
FROM stockout_days sd
JOIN stores   s ON s.store_id   = sd.store_id
JOIN products p ON p.product_id = sd.product_id
GROUP BY sd.store_id, s.store_name, sd.product_id, p.sku, p.product_name, p.category, sd.grp
ORDER BY stockout_days DESC;
