-- SafeRoads SN — Schéma PostGIS
-- Exécuté automatiquement au premier démarrage

CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;

-- ── Accidents ──
CREATE TABLE IF NOT EXISTS accidents (
    id              SERIAL PRIMARY KEY,
    source          VARCHAR(50) DEFAULT 'kaggle',   -- kaggle | ansd | police
    datetime        TIMESTAMP NOT NULL,
    year            SMALLINT,
    month           SMALLINT,
    day_of_week     SMALLINT,   -- 0=Lundi
    hour            SMALLINT,
    geom            GEOMETRY(Point, 4326),
    latitude        DOUBLE PRECISION,
    longitude       DOUBLE PRECISION,
    region          VARCHAR(100),
    road_type       VARCHAR(50),
    vehicle_type    VARCHAR(100),
    cause           VARCHAR(100),
    weather         VARCHAR(50),
    gravity         SMALLINT,   -- 1=léger 2=grave 3=mortel
    num_vehicles    SMALLINT,
    num_victims     SMALLINT,
    is_rainy        BOOLEAN DEFAULT FALSE,
    geo_source      VARCHAR(20) DEFAULT 'real',     -- real | simulated
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_accidents_geom     ON accidents USING GIST(geom);
CREATE INDEX IF NOT EXISTS idx_accidents_datetime ON accidents(datetime);
CREATE INDEX IF NOT EXISTS idx_accidents_region   ON accidents(region);

-- ── Segments routiers (depuis OSM) ──
CREATE TABLE IF NOT EXISTS road_segments (
    id              SERIAL PRIMARY KEY,
    osm_id          BIGINT,
    name            VARCHAR(200),
    road_type       VARCHAR(50),
    geom            GEOMETRY(LineString, 4326),
    length_km       DOUBLE PRECISION,
    risk_score      DOUBLE PRECISION DEFAULT 0,
    accident_count  INTEGER DEFAULT 0,
    updated_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_roads_geom ON road_segments USING GIST(geom);

-- ── Hotspots (clusters DBSCAN) ──
CREATE TABLE IF NOT EXISTS hotspots (
    id              SERIAL PRIMARY KEY,
    cluster_id      INTEGER,
    center_lat      DOUBLE PRECISION,
    center_lon      DOUBLE PRECISION,
    geom            GEOMETRY(Point, 4326),
    radius_km       DOUBLE PRECISION,
    accident_count  INTEGER,
    avg_gravity     DOUBLE PRECISION,
    risk_level      VARCHAR(20),  -- faible | moyen | élevé | critique
    region          VARCHAR(100),
    peak_hours      VARCHAR(50),
    updated_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_hotspots_geom ON hotspots USING GIST(geom);

-- ── Météo historique ──
CREATE TABLE IF NOT EXISTS weather_history (
    id              SERIAL PRIMARY KEY,
    datetime        TIMESTAMP NOT NULL,
    latitude        DOUBLE PRECISION,
    longitude       DOUBLE PRECISION,
    city            VARCHAR(100),
    precipitation   DOUBLE PRECISION,   -- mm
    windspeed       DOUBLE PRECISION,   -- km/h
    temperature     DOUBLE PRECISION,   -- °C
    visibility      DOUBLE PRECISION,   -- km (estimé)
    is_rainy        BOOLEAN DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_weather_datetime ON weather_history(datetime);
CREATE INDEX IF NOT EXISTS idx_weather_city     ON weather_history(city);
