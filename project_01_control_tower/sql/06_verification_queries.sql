-- Verification queries. Run AFTER all data is loaded.
-- Every query should return 0 rows or the expected value stated in the comment.

SET search_path TO supply_chain;

-- ── 1. Row counts (review manually) ─────────────────────────
SELECT 'distribution_centers' AS tbl, COUNT(*) AS n FROM distribution_centers
UNION ALL SELECT 'stores',              COUNT(*) FROM stores
UNION ALL SELECT 'suppliers',           COUNT(*) FROM suppliers
UNION ALL SELECT 'products',            COUNT(*) FROM products
UNION ALL SELECT 'customers',           COUNT(*) FROM customers
UNION ALL SELECT 'calendar',            COUNT(*) FROM calendar
UNION ALL SELECT 'product_supplier',    COUNT(*) FROM product_supplier
UNION ALL SELECT 'promotions',          COUNT(*) FROM promotions
UNION ALL SELECT 'weather',             COUNT(*) FROM weather
UNION ALL SELECT 'purchase_orders',     COUNT(*) FROM purchase_orders
UNION ALL SELECT 'purchase_order_lines',COUNT(*) FROM purchase_order_lines
UNION ALL SELECT 'shipments',           COUNT(*) FROM shipments
UNION ALL SELECT 'shipment_events',     COUNT(*) FROM shipment_events
UNION ALL SELECT 'sales_transactions',  COUNT(*) FROM sales_transactions
UNION ALL SELECT 'inventory_snapshots', COUNT(*) FROM inventory_snapshots
UNION ALL SELECT 'returns',             COUNT(*) FROM returns
UNION ALL SELECT 'spoilage',            COUNT(*) FROM spoilage
ORDER BY tbl;


-- ── 2. FK violations: sales → stores ────────────────────────
-- Expected: 0 rows
SELECT COUNT(*) AS fk_violation_sales_stores
FROM sales_transactions
WHERE store_id NOT IN (SELECT store_id FROM stores);

-- ── 3. FK violations: sales → products ──────────────────────
SELECT COUNT(*) AS fk_violation_sales_products
FROM sales_transactions
WHERE product_id NOT IN (SELECT product_id FROM products);

-- ── 4. Stockout day with positive sales — should never happen
-- Expected: 0 rows
SELECT COUNT(*) AS impossible_stockout_with_sales
FROM sales_transactions
WHERE is_stockout_day = TRUE AND quantity_sold > 0;

-- ── 5. Negative inventory — should never happen ──────────────
SELECT COUNT(*) AS negative_inventory
FROM inventory_snapshots
WHERE units_on_hand < 0;

-- ── 6. Spoilage on non-perishable products ───────────────────
-- Expected: 0 rows
SELECT COUNT(*) AS spoilage_non_perishable
FROM spoilage sp
JOIN products p ON p.product_id = sp.product_id
WHERE p.is_perishable = FALSE;

-- ── 7. Promotion date sanity ─────────────────────────────────
-- Expected: 0 rows
SELECT COUNT(*) AS bad_promo_dates
FROM promotions
WHERE end_date < start_date;

-- ── 8. Returns before purchase ───────────────────────────────
-- Expected: 0 rows
SELECT COUNT(*) AS return_before_purchase
FROM returns r
JOIN sales_transactions st ON st.transaction_id = r.transaction_id
WHERE r.return_date < st.date_id;

-- ── 9. Shipment arrival before departure ─────────────────────
-- Expected: 0 rows
SELECT COUNT(*) AS ship_arrival_before_departure
FROM shipments
WHERE actual_arrival IS NOT NULL AND actual_arrival < ship_date;

-- ── 10. Each product has exactly 1 active primary supplier ───
-- Expected: 0 rows (products with 0 or >1 primary supplier)
SELECT product_id, COUNT(*) AS primary_count
FROM product_supplier
WHERE is_primary = TRUE AND (expiry_date IS NULL OR expiry_date > CURRENT_DATE)
GROUP BY product_id
HAVING COUNT(*) <> 1
LIMIT 20;

-- ── 11. Business logic: weekend grocery sales > weekday ──────
-- Expected: weekend_avg > weekday_avg
SELECT
    c.is_weekend,
    ROUND(AVG(ds.total_units_sold), 2) AS avg_daily_units
FROM v_daily_store_sales ds
JOIN calendar c ON c.date_id = ds.date_id
WHERE ds.category = 'Grocery'
GROUP BY c.is_weekend;

-- ── 12. Stockout rate — should be 2–5% of store-product-days
SELECT
    ROUND(100.0 * SUM(CASE WHEN is_stockout THEN 1 ELSE 0 END) / COUNT(*), 2) AS stockout_rate_pct
FROM inventory_snapshots
WHERE location_type = 'Store';

-- ── 13. National suppliers: shorter avg lead time than local ─
SELECT
    supplier_type,
    ROUND(AVG(actual_delivery - order_date), 1) AS avg_actual_lead_days
FROM purchase_orders po
JOIN suppliers s ON s.supplier_id = po.supplier_id
WHERE po.status = 'Received'
GROUP BY supplier_type
ORDER BY avg_actual_lead_days;

-- ── 14. Promotion lift visible ───────────────────────────────
-- Expected: lift_factor > 1.0 for most rows
SELECT
    promotion_type,
    COUNT(*)                             AS promos,
    ROUND(AVG(lift_factor), 3)           AS avg_lift_factor
FROM v_promotion_lift
WHERE lift_factor IS NOT NULL
GROUP BY promotion_type
ORDER BY avg_lift_factor DESC;

-- ── 15. Nov–Dec Grocery revenue 30–50% above average ────────
SELECT
    EXTRACT(MONTH FROM date_id) AS month,
    ROUND(SUM(total_net_revenue) / COUNT(DISTINCT date_id), 0) AS avg_daily_revenue
FROM v_daily_store_sales
WHERE category = 'Grocery'
GROUP BY month
ORDER BY month;
