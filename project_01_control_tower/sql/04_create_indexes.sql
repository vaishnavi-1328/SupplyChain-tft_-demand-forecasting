-- Performance indexes. Run AFTER 02_create_facts.sql.
-- These reflect the join patterns used in exploration notebooks and future ML feature queries.

SET search_path TO supply_chain;

-- ── sales_transactions (hottest table) ──────────
CREATE INDEX IF NOT EXISTS idx_sales_store_date     ON sales_transactions(store_id, date_id);
CREATE INDEX IF NOT EXISTS idx_sales_product_date   ON sales_transactions(product_id, date_id);
CREATE INDEX IF NOT EXISTS idx_sales_date           ON sales_transactions(date_id);
CREATE INDEX IF NOT EXISTS idx_sales_promotion      ON sales_transactions(promotion_id)
    WHERE promotion_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_sales_stockout       ON sales_transactions(date_id)
    WHERE is_stockout_day = TRUE;

-- ── inventory_snapshots ─────────────────────────
CREATE INDEX IF NOT EXISTS idx_inv_store_product_date
    ON inventory_snapshots(store_id, product_id, snapshot_date);
CREATE INDEX IF NOT EXISTS idx_inv_stockout
    ON inventory_snapshots(snapshot_date)
    WHERE is_stockout = TRUE;
CREATE INDEX IF NOT EXISTS idx_inv_dc
    ON inventory_snapshots(dc_id, product_id, snapshot_date)
    WHERE dc_id IS NOT NULL;

-- ── purchase_orders ─────────────────────────────
CREATE INDEX IF NOT EXISTS idx_po_supplier          ON purchase_orders(supplier_id);
CREATE INDEX IF NOT EXISTS idx_po_dc                ON purchase_orders(dc_id);
CREATE INDEX IF NOT EXISTS idx_po_status            ON purchase_orders(status);
CREATE INDEX IF NOT EXISTS idx_po_order_date        ON purchase_orders(order_date);

-- ── purchase_order_lines ────────────────────────
CREATE INDEX IF NOT EXISTS idx_pol_po               ON purchase_order_lines(po_id);
CREATE INDEX IF NOT EXISTS idx_pol_product          ON purchase_order_lines(product_id);

-- ── shipments ───────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_ship_status          ON shipments(status);
CREATE INDEX IF NOT EXISTS idx_ship_dest_store      ON shipments(dest_store_id)
    WHERE dest_store_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_ship_dest_dc         ON shipments(dest_dc_id)
    WHERE dest_dc_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_ship_po              ON shipments(po_id)
    WHERE po_id IS NOT NULL;

-- ── shipment_events ─────────────────────────────
CREATE INDEX IF NOT EXISTS idx_sevt_shipment        ON shipment_events(shipment_id);
CREATE INDEX IF NOT EXISTS idx_sevt_timestamp       ON shipment_events(event_timestamp);

-- ── weather ─────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_weather_date         ON weather(date_id);

-- ── promotions ──────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_promo_product        ON promotions(product_id);
CREATE INDEX IF NOT EXISTS idx_promo_dates          ON promotions(start_date, end_date);

-- ── returns ─────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_returns_store_date   ON returns(store_id, return_date);
CREATE INDEX IF NOT EXISTS idx_returns_product      ON returns(product_id);

-- ── spoilage ────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_spoilage_store_date  ON spoilage(store_id, spoilage_date);
CREATE INDEX IF NOT EXISTS idx_spoilage_product     ON spoilage(product_id);

-- ── product_supplier ────────────────────────────
CREATE INDEX IF NOT EXISTS idx_ps_supplier          ON product_supplier(supplier_id);
CREATE INDEX IF NOT EXISTS idx_ps_primary           ON product_supplier(product_id)
    WHERE is_primary = TRUE;
