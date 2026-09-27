-- Relationship tables. Run AFTER 01_create_dimensions.sql.

SET search_path TO supply_chain;

-- ─────────────────────────────────────────────
-- product_supplier  (many-to-many with metadata)
-- grain: 1 row per product × supplier × effective_date
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS product_supplier (
    product_id          INTEGER      NOT NULL REFERENCES products(product_id),
    supplier_id         INTEGER      NOT NULL REFERENCES suppliers(supplier_id),
    is_primary          BOOLEAN      NOT NULL DEFAULT FALSE,
    lead_time_days      NUMERIC(5,2) NOT NULL,
    unit_cost_override  NUMERIC(10,2),     -- NULL = use product.unit_cost
    min_order_qty       INTEGER,
    effective_date      DATE         NOT NULL,
    expiry_date         DATE,
    PRIMARY KEY (product_id, supplier_id, effective_date)
);

-- Enforce: each product has at most 1 active primary supplier at any time
-- (validated in Python during generation; partial enforcement here via partial index)
CREATE UNIQUE INDEX IF NOT EXISTS uidx_product_primary_supplier
    ON product_supplier (product_id)
    WHERE is_primary = TRUE AND (expiry_date IS NULL OR expiry_date > '2020-01-01');
