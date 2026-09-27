-- Fact tables. Run AFTER 01_create_dimensions.sql.
-- Creation order respects FK dependencies.

SET search_path TO supply_chain;

-- ─────────────────────────────────────────────
-- 1. promotions
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS promotions (
    promotion_id      SERIAL PRIMARY KEY,
    promotion_name    VARCHAR(200) NOT NULL,
    product_id        INTEGER      NOT NULL REFERENCES products(product_id),
    store_id          INTEGER      REFERENCES stores(store_id),  -- NULL = chain-wide
    start_date        DATE         NOT NULL,
    end_date          DATE         NOT NULL,
    discount_pct      NUMERIC(5,4) NOT NULL,
    promotion_type    VARCHAR(30),            -- Price_Cut, BOGO, Feature, Display, Circular
    expected_lift_pct NUMERIC(6,4),
    is_active         BOOLEAN      NOT NULL DEFAULT TRUE,
    CONSTRAINT chk_promo_dates CHECK (end_date >= start_date)
);

-- ─────────────────────────────────────────────
-- 2. weather
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS weather (
    weather_id       SERIAL PRIMARY KEY,
    store_id         INTEGER      NOT NULL REFERENCES stores(store_id),
    date_id          DATE         NOT NULL REFERENCES calendar(date_id),
    avg_temp_f       NUMERIC(5,1),
    min_temp_f       NUMERIC(5,1),
    max_temp_f       NUMERIC(5,1),
    precipitation_in NUMERIC(6,2),
    snow_in          NUMERIC(6,2),
    condition        VARCHAR(30),             -- Clear, Rain, Snow, Storm, Fog
    weather_severity INTEGER      NOT NULL DEFAULT 1,  -- 1=normal, 2=notable, 3=severe, 4=extreme
    UNIQUE (store_id, date_id)
);

-- ─────────────────────────────────────────────
-- 3. purchase_orders
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS purchase_orders (
    po_id               SERIAL PRIMARY KEY,
    po_number           VARCHAR(30)   UNIQUE NOT NULL,
    supplier_id         INTEGER       NOT NULL REFERENCES suppliers(supplier_id),
    dc_id               INTEGER       NOT NULL REFERENCES distribution_centers(dc_id),
    order_date          DATE          NOT NULL,
    requested_delivery  DATE          NOT NULL,
    actual_delivery     DATE,
    status              VARCHAR(20)   NOT NULL,  -- Pending, Confirmed, Shipped, Received, Cancelled
    total_lines         INTEGER,
    total_units         INTEGER,
    total_cost          NUMERIC(14,2),
    is_late             BOOLEAN,
    late_days           INTEGER
);

-- ─────────────────────────────────────────────
-- 4. purchase_order_lines  (references purchase_orders)
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS purchase_order_lines (
    po_line_id    SERIAL PRIMARY KEY,
    po_id         INTEGER      NOT NULL REFERENCES purchase_orders(po_id),
    product_id    INTEGER      NOT NULL REFERENCES products(product_id),
    ordered_qty   INTEGER      NOT NULL,
    confirmed_qty INTEGER,
    received_qty  INTEGER,
    unit_cost     NUMERIC(10,2) NOT NULL,
    line_total    NUMERIC(12,2),
    fill_rate     NUMERIC(5,4)              -- received_qty / ordered_qty
);

-- ─────────────────────────────────────────────
-- 5. shipments
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS shipments (
    shipment_id        SERIAL PRIMARY KEY,
    shipment_number    VARCHAR(30)   UNIQUE NOT NULL,
    po_id              INTEGER       REFERENCES purchase_orders(po_id),  -- NULL for DC-to-store
    origin_type        VARCHAR(10)   NOT NULL,   -- Supplier or DC
    origin_supplier_id INTEGER       REFERENCES suppliers(supplier_id),
    origin_dc_id       INTEGER       REFERENCES distribution_centers(dc_id),
    dest_type          VARCHAR(10)   NOT NULL,   -- DC or Store
    dest_dc_id         INTEGER       REFERENCES distribution_centers(dc_id),
    dest_store_id      INTEGER       REFERENCES stores(store_id),
    carrier            VARCHAR(80),
    ship_date          DATE          NOT NULL,
    expected_arrival   DATE          NOT NULL,
    actual_arrival     DATE,
    status             VARCHAR(20)   NOT NULL,   -- In_Transit, Delivered, Delayed, Lost
    total_units        INTEGER,
    total_weight_kg    NUMERIC(10,2),
    freight_cost       NUMERIC(10,2),
    CONSTRAINT chk_ship_arrival CHECK (actual_arrival IS NULL OR actual_arrival >= ship_date)
);

-- ─────────────────────────────────────────────
-- 6. shipment_events  (log table, references shipments)
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS shipment_events (
    event_id         SERIAL PRIMARY KEY,
    shipment_id      INTEGER      NOT NULL REFERENCES shipments(shipment_id),
    event_timestamp  TIMESTAMP    NOT NULL,
    event_type       VARCHAR(40)  NOT NULL,
    -- Dispatched | In_Transit | Arrived_DC | Customs_Hold | Delay_Weather
    -- Delay_Mechanical | Delivered | Exception
    location         VARCHAR(100),
    notes            TEXT,
    delay_hours      INTEGER      NOT NULL DEFAULT 0
);

-- ─────────────────────────────────────────────
-- 7. sales_transactions  (the largest table, ~50–80M rows)
-- ─────────────────────────────────────────────
CREATE SEQUENCE IF NOT EXISTS sales_transactions_seq;
CREATE TABLE IF NOT EXISTS sales_transactions (
    transaction_id   BIGINT       NOT NULL DEFAULT nextval('sales_transactions_seq'),
    store_id         INTEGER      NOT NULL REFERENCES stores(store_id),
    product_id       INTEGER      NOT NULL REFERENCES products(product_id),
    customer_id      INTEGER      REFERENCES customers(customer_id),  -- NULL if anonymous
    date_id          DATE         NOT NULL REFERENCES calendar(date_id),
    transaction_time TIMESTAMP    NOT NULL,
    quantity_sold    INTEGER      NOT NULL CHECK (quantity_sold > 0),
    unit_price       NUMERIC(10,2) NOT NULL,
    unit_cost        NUMERIC(10,2) NOT NULL,
    discount_amt     NUMERIC(10,2) NOT NULL DEFAULT 0,
    net_revenue      NUMERIC(12,2),          -- quantity_sold * (unit_price - discount_amt)
    promotion_id     INTEGER      REFERENCES promotions(promotion_id),
    is_stockout_day  BOOLEAN      NOT NULL DEFAULT FALSE,
    PRIMARY KEY (transaction_id, date_id)   -- partition key must be in PK
) PARTITION BY RANGE (date_id);

-- Monthly partitions for 2023 and 2024 (makes large table manageable)
CREATE TABLE IF NOT EXISTS sales_transactions_2023_01 PARTITION OF sales_transactions
    FOR VALUES FROM ('2023-01-01') TO ('2023-02-01');
CREATE TABLE IF NOT EXISTS sales_transactions_2023_02 PARTITION OF sales_transactions
    FOR VALUES FROM ('2023-02-01') TO ('2023-03-01');
CREATE TABLE IF NOT EXISTS sales_transactions_2023_03 PARTITION OF sales_transactions
    FOR VALUES FROM ('2023-03-01') TO ('2023-04-01');
CREATE TABLE IF NOT EXISTS sales_transactions_2023_04 PARTITION OF sales_transactions
    FOR VALUES FROM ('2023-04-01') TO ('2023-05-01');
CREATE TABLE IF NOT EXISTS sales_transactions_2023_05 PARTITION OF sales_transactions
    FOR VALUES FROM ('2023-05-01') TO ('2023-06-01');
CREATE TABLE IF NOT EXISTS sales_transactions_2023_06 PARTITION OF sales_transactions
    FOR VALUES FROM ('2023-06-01') TO ('2023-07-01');
CREATE TABLE IF NOT EXISTS sales_transactions_2023_07 PARTITION OF sales_transactions
    FOR VALUES FROM ('2023-07-01') TO ('2023-08-01');
CREATE TABLE IF NOT EXISTS sales_transactions_2023_08 PARTITION OF sales_transactions
    FOR VALUES FROM ('2023-08-01') TO ('2023-09-01');
CREATE TABLE IF NOT EXISTS sales_transactions_2023_09 PARTITION OF sales_transactions
    FOR VALUES FROM ('2023-09-01') TO ('2023-10-01');
CREATE TABLE IF NOT EXISTS sales_transactions_2023_10 PARTITION OF sales_transactions
    FOR VALUES FROM ('2023-10-01') TO ('2023-11-01');
CREATE TABLE IF NOT EXISTS sales_transactions_2023_11 PARTITION OF sales_transactions
    FOR VALUES FROM ('2023-11-01') TO ('2023-12-01');
CREATE TABLE IF NOT EXISTS sales_transactions_2023_12 PARTITION OF sales_transactions
    FOR VALUES FROM ('2023-12-01') TO ('2024-01-01');
CREATE TABLE IF NOT EXISTS sales_transactions_2024_01 PARTITION OF sales_transactions
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
CREATE TABLE IF NOT EXISTS sales_transactions_2024_02 PARTITION OF sales_transactions
    FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');
CREATE TABLE IF NOT EXISTS sales_transactions_2024_03 PARTITION OF sales_transactions
    FOR VALUES FROM ('2024-03-01') TO ('2024-04-01');
CREATE TABLE IF NOT EXISTS sales_transactions_2024_04 PARTITION OF sales_transactions
    FOR VALUES FROM ('2024-04-01') TO ('2024-05-01');
CREATE TABLE IF NOT EXISTS sales_transactions_2024_05 PARTITION OF sales_transactions
    FOR VALUES FROM ('2024-05-01') TO ('2024-06-01');
CREATE TABLE IF NOT EXISTS sales_transactions_2024_06 PARTITION OF sales_transactions
    FOR VALUES FROM ('2024-06-01') TO ('2024-07-01');
CREATE TABLE IF NOT EXISTS sales_transactions_2024_07 PARTITION OF sales_transactions
    FOR VALUES FROM ('2024-07-01') TO ('2024-08-01');
CREATE TABLE IF NOT EXISTS sales_transactions_2024_08 PARTITION OF sales_transactions
    FOR VALUES FROM ('2024-08-01') TO ('2024-09-01');
CREATE TABLE IF NOT EXISTS sales_transactions_2024_09 PARTITION OF sales_transactions
    FOR VALUES FROM ('2024-09-01') TO ('2024-10-01');
CREATE TABLE IF NOT EXISTS sales_transactions_2024_10 PARTITION OF sales_transactions
    FOR VALUES FROM ('2024-10-01') TO ('2024-11-01');
CREATE TABLE IF NOT EXISTS sales_transactions_2024_11 PARTITION OF sales_transactions
    FOR VALUES FROM ('2024-11-01') TO ('2024-12-01');
CREATE TABLE IF NOT EXISTS sales_transactions_2024_12 PARTITION OF sales_transactions
    FOR VALUES FROM ('2024-12-01') TO ('2025-01-01');

-- ─────────────────────────────────────────────
-- 8. inventory_snapshots
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS inventory_snapshots (
    snapshot_id      BIGSERIAL    PRIMARY KEY,
    store_id         INTEGER      REFERENCES stores(store_id),        -- NULL for DC-level
    product_id       INTEGER      NOT NULL REFERENCES products(product_id),
    dc_id            INTEGER      REFERENCES distribution_centers(dc_id),  -- NULL for store-level
    snapshot_date    DATE         NOT NULL REFERENCES calendar(date_id),
    location_type    VARCHAR(10)  NOT NULL,    -- Store or DC
    units_on_hand    INTEGER      NOT NULL CHECK (units_on_hand >= 0),
    units_on_order   INTEGER      NOT NULL DEFAULT 0,
    units_in_transit INTEGER      NOT NULL DEFAULT 0,
    days_of_supply   NUMERIC(8,2),
    is_stockout      BOOLEAN      NOT NULL DEFAULT FALSE,
    reorder_point    INTEGER,
    safety_stock     INTEGER,
    UNIQUE (store_id, product_id, snapshot_date, location_type)
);

-- ─────────────────────────────────────────────
-- 9. returns
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS returns (
    return_id          SERIAL PRIMARY KEY,
    transaction_id     BIGINT,      -- soft ref to sales_transactions (partitioned table, no FK)
    store_id           INTEGER      NOT NULL REFERENCES stores(store_id),
    product_id         INTEGER      NOT NULL REFERENCES products(product_id),
    return_date        DATE         NOT NULL REFERENCES calendar(date_id),
    quantity_returned  INTEGER      NOT NULL CHECK (quantity_returned > 0),
    return_reason      VARCHAR(50),   -- Defective, Wrong_Item, Spoiled, Changed_Mind, Damaged
    refund_amount      NUMERIC(10,2),
    restocked          BOOLEAN      NOT NULL DEFAULT FALSE
);

-- ─────────────────────────────────────────────
-- 10. spoilage  (perishables only)
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS spoilage (
    spoilage_id      SERIAL PRIMARY KEY,
    store_id         INTEGER      NOT NULL REFERENCES stores(store_id),
    product_id       INTEGER      NOT NULL REFERENCES products(product_id),
    spoilage_date    DATE         NOT NULL REFERENCES calendar(date_id),
    units_spoiled    INTEGER      NOT NULL CHECK (units_spoiled > 0),
    spoilage_reason  VARCHAR(50),   -- Expired, Damaged, Temperature_Break, Overstock
    cost_of_spoilage NUMERIC(10,2)
);
