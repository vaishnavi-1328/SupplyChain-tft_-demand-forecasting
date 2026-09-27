-- Dimension tables. Run AFTER 00_create_schema.sql.
-- Order matters: stores references distribution_centers.

SET search_path TO supply_chain;

-- ─────────────────────────────────────────────
-- 1. distribution_centers
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS distribution_centers (
    dc_id            SERIAL PRIMARY KEY,
    dc_code          VARCHAR(10)  UNIQUE NOT NULL,
    dc_name          VARCHAR(100) NOT NULL,
    city             VARCHAR(80)  NOT NULL,
    state            CHAR(2)      NOT NULL,
    latitude         NUMERIC(9,6) NOT NULL,
    longitude        NUMERIC(9,6) NOT NULL,
    capacity_pallets INTEGER,
    region           VARCHAR(30),
    is_active        BOOLEAN      NOT NULL DEFAULT TRUE
);

-- ─────────────────────────────────────────────
-- 2. stores  (references distribution_centers)
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS stores (
    store_id             SERIAL PRIMARY KEY,
    store_code           VARCHAR(10)  UNIQUE NOT NULL,
    store_name           VARCHAR(100) NOT NULL,
    city                 VARCHAR(80)  NOT NULL,
    state                CHAR(2)      NOT NULL,
    zip_code             VARCHAR(10),
    latitude             NUMERIC(9,6) NOT NULL,
    longitude            NUMERIC(9,6) NOT NULL,
    store_format         VARCHAR(30),            -- Superstore, Neighborhood, Express, Warehouse
    store_size_sqft      INTEGER,
    annual_revenue_band  VARCHAR(10),            -- High, Medium, Low
    dc_id                INTEGER      NOT NULL REFERENCES distribution_centers(dc_id),
    opened_date          DATE,
    is_active            BOOLEAN      NOT NULL DEFAULT TRUE
);

-- ─────────────────────────────────────────────
-- 3. suppliers
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS suppliers (
    supplier_id          SERIAL PRIMARY KEY,
    supplier_name        VARCHAR(150) NOT NULL,
    supplier_type        VARCHAR(20)  NOT NULL,  -- National, Regional, Local
    city                 VARCHAR(80),
    state                CHAR(2),
    country              CHAR(3)      NOT NULL DEFAULT 'USA',
    latitude             NUMERIC(9,6),
    longitude            NUMERIC(9,6),
    lead_time_days_avg   NUMERIC(5,2) NOT NULL,
    lead_time_days_std   NUMERIC(5,2) NOT NULL,
    fill_rate_baseline   NUMERIC(5,4),
    otif_baseline        NUMERIC(5,4),           -- on-time-in-full rate
    disruption_prob      NUMERIC(5,4),           -- probability of disruption per month
    payment_terms_days   INTEGER      NOT NULL DEFAULT 30,
    is_active            BOOLEAN      NOT NULL DEFAULT TRUE
);

-- ─────────────────────────────────────────────
-- 4. products
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS products (
    product_id          SERIAL PRIMARY KEY,
    sku                 VARCHAR(20)   UNIQUE NOT NULL,
    product_name        VARCHAR(200)  NOT NULL,
    category            VARCHAR(50)   NOT NULL,
    subcategory         VARCHAR(80)   NOT NULL,
    brand               VARCHAR(100),
    is_perishable       BOOLEAN       NOT NULL DEFAULT FALSE,
    shelf_life_days     INTEGER,                 -- NULL for non-perishables
    unit_of_measure     VARCHAR(20),             -- each, kg, liter, pack
    unit_cost           NUMERIC(10,2) NOT NULL,
    unit_price          NUMERIC(10,2) NOT NULL,
    weight_kg           NUMERIC(8,3),
    volume_liters       NUMERIC(8,3),
    min_order_qty       INTEGER       NOT NULL DEFAULT 1,
    case_pack_qty       INTEGER       NOT NULL DEFAULT 1,
    abc_class           CHAR(1),                 -- A / B / C (set post-generation)
    is_active           BOOLEAN       NOT NULL DEFAULT TRUE,
    introduced_date     DATE,
    discontinued_date   DATE
);

-- ─────────────────────────────────────────────
-- 5. customers
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS customers (
    customer_id      SERIAL PRIMARY KEY,
    customer_name    VARCHAR(150) NOT NULL,
    email            VARCHAR(200),
    city             VARCHAR(80),
    state            CHAR(2),
    zip_code         VARCHAR(10),
    loyalty_tier     VARCHAR(20),    -- Gold, Silver, Bronze, None
    acquisition_date DATE,
    is_active        BOOLEAN      NOT NULL DEFAULT TRUE
);

-- ─────────────────────────────────────────────
-- 6. calendar  (pre-populated 2023-01-01 → 2024-12-31)
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS calendar (
    date_id         DATE        PRIMARY KEY,
    year            INTEGER     NOT NULL,
    quarter         INTEGER     NOT NULL,
    month           INTEGER     NOT NULL,
    month_name      VARCHAR(12) NOT NULL,
    week_of_year    INTEGER     NOT NULL,
    day_of_week     INTEGER     NOT NULL,    -- 1=Monday … 7=Sunday
    day_name        VARCHAR(12) NOT NULL,
    is_weekend      BOOLEAN     NOT NULL,
    is_holiday      BOOLEAN     NOT NULL DEFAULT FALSE,
    holiday_name    VARCHAR(80),
    is_month_start  BOOLEAN     NOT NULL,
    is_month_end    BOOLEAN     NOT NULL,
    fiscal_week     INTEGER,                -- retail 4-5-4
    fiscal_period   INTEGER,
    season          VARCHAR(10)             -- Spring, Summer, Fall, Winter
);
