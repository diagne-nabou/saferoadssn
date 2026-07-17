"""
SafeRoads SN — Connexion PostgreSQL (psycopg2 + SQLAlchemy)
Sans Docker — connexion directe au serveur PostgreSQL local.

Prérequis :
    sudo apt install postgresql postgresql-contrib   # Ubuntu/Debian
    brew install postgresql                          # macOS
    createdb saferoads
    createuser saferoads_user -P
"""

import os
import logging
from pathlib import Path

import pandas as pd
import psycopg2
import psycopg2.extras
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

log = logging.getLogger(__name__)

# ── Configuration ──
DB_CONFIG = {
    "host":     os.getenv("DB_HOST",     "localhost"),
    "port":     int(os.getenv("DB_PORT", "5432")),
    "dbname":   os.getenv("DB_NAME",     "saferoads"),
    "user":     os.getenv("DB_USER",     "saferoads_user"),
    "password": os.getenv("DB_PASSWORD", "changeme"),
}

DB_URL = (
    f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
    f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
)


def get_engine():
    return create_engine(DB_URL, pool_pre_ping=True)


def get_conn():
    """Retourne une connexion psycopg2 brute."""
    return psycopg2.connect(**DB_CONFIG)


def test_connection() -> tuple[bool, str]:
    """Teste la connexion et retourne (ok, message)."""
    try:
        conn = get_conn()
        cur  = conn.cursor()
        cur.execute("SELECT version()")
        version = cur.fetchone()[0]
        conn.close()
        return True, f"✅ Connecté — {version[:40]}"
    except Exception as e:
        return False, f"❌ Erreur : {e}"


def init_schema():
    """Crée les tables si elles n'existent pas encore."""
    ddl = """
    CREATE TABLE IF NOT EXISTS accidents (
        id              SERIAL PRIMARY KEY,
        source          VARCHAR(50)  DEFAULT 'csv',
        datetime        TIMESTAMP,
        year            SMALLINT,
        month           SMALLINT,
        day_of_week     SMALLINT,
        hour            SMALLINT,
        latitude        DOUBLE PRECISION,
        longitude       DOUBLE PRECISION,
        region          VARCHAR(100),
        road_type       VARCHAR(50),
        vehicle_type    VARCHAR(100),
        cause           VARCHAR(100),
        weather         VARCHAR(50),
        gravity         SMALLINT,
        num_vehicles    SMALLINT,
        num_victims     SMALLINT,
        is_rainy        BOOLEAN DEFAULT FALSE,
        geo_source      VARCHAR(20) DEFAULT 'real',
        created_at      TIMESTAMP DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS hotspots (
        id              SERIAL PRIMARY KEY,
        cluster_id      INTEGER,
        center_lat      DOUBLE PRECISION,
        center_lon      DOUBLE PRECISION,
        accident_count  INTEGER,
        avg_gravity     DOUBLE PRECISION,
        risk_level      VARCHAR(20),
        region          VARCHAR(100),
        peak_hours      VARCHAR(50),
        updated_at      TIMESTAMP DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS predictions_log (
        id              SERIAL PRIMARY KEY,
        latitude        DOUBLE PRECISION,
        longitude       DOUBLE PRECISION,
        hour            SMALLINT,
        region          VARCHAR(100),
        is_rainy        BOOLEAN,
        vehicle_type    VARCHAR(100),
        gravity_label   VARCHAR(20),
        risk_score      DOUBLE PRECISION,
        risk_level      VARCHAR(20),
        created_at      TIMESTAMP DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS idx_accidents_region   ON accidents(region);
    CREATE INDEX IF NOT EXISTS idx_accidents_gravity  ON accidents(gravity);
    CREATE INDEX IF NOT EXISTS idx_accidents_datetime ON accidents(datetime);
    CREATE INDEX IF NOT EXISTS idx_hotspots_risk      ON hotspots(risk_level);
    """
    try:
        conn = get_conn()
        cur  = conn.cursor()
        cur.execute(ddl)
        conn.commit()
        conn.close()
        log.info("Schéma initialisé")
        return True
    except Exception as e:
        log.error(f"Erreur init schéma : {e}")
        return False


def insert_accidents_df(df: pd.DataFrame) -> tuple[int, int]:
    """
    Insère un DataFrame accidents dans PostgreSQL.
    Retourne (n_inserted, n_errors).
    """
    cols = [
        "source", "datetime", "year", "month", "day_of_week", "hour",
        "latitude", "longitude", "region", "road_type", "vehicle_type",
        "cause", "weather", "gravity", "num_vehicles", "num_victims",
        "is_rainy", "geo_source",
    ]

    # Ne garder que les colonnes disponibles
    available = [c for c in cols if c in df.columns]
    df_insert = df[available].copy()

    # Valeurs par défaut pour les colonnes manquantes
    defaults = {
        "source": "csv", "geo_source": "real",
        "num_vehicles": 1, "num_victims": 1, "is_rainy": False,
    }
    for col, val in defaults.items():
        if col not in df_insert.columns:
            df_insert[col] = val

    # Nettoyage types
    for col in ["year", "month", "day_of_week", "hour", "gravity", "num_vehicles", "num_victims"]:
        if col in df_insert.columns:
            df_insert[col] = pd.to_numeric(df_insert[col], errors="coerce").fillna(0).astype(int)

    sql = f"""
        INSERT INTO accidents ({', '.join(df_insert.columns)})
        VALUES ({', '.join(['%s'] * len(df_insert.columns))})
    """

    n_ok = n_err = 0
    conn = get_conn()
    cur  = conn.cursor()

    for row in df_insert.itertuples(index=False):
        try:
            cur.execute(sql, list(row))
            n_ok += 1
        except Exception as e:
            n_err += 1
            conn.rollback()
            log.debug(f"Ligne ignorée : {e}")

    conn.commit()
    conn.close()
    return n_ok, n_err


def insert_hotspots_df(df: pd.DataFrame) -> int:
    """Remplace les hotspots existants par les nouveaux."""
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("TRUNCATE TABLE hotspots RESTART IDENTITY")

    cols = ["cluster_id", "center_lat", "center_lon", "accident_count",
            "avg_gravity", "risk_level", "region", "peak_hours"]
    available = [c for c in cols if c in df.columns]

    sql = f"""
        INSERT INTO hotspots ({', '.join(available)})
        VALUES ({', '.join(['%s'] * len(available))})
    """
    n = 0
    for row in df[available].itertuples(index=False):
        cur.execute(sql, list(row))
        n += 1

    conn.commit()
    conn.close()
    return n


def log_prediction(data: dict):
    """Enregistre une prédiction dans la table predictions_log."""
    sql = """
        INSERT INTO predictions_log
            (latitude, longitude, hour, region, is_rainy,
             vehicle_type, gravity_label, risk_score, risk_level)
        VALUES
            (%(latitude)s, %(longitude)s, %(hour)s, %(region)s, %(is_rainy)s,
             %(vehicle_type)s, %(gravity_label)s, %(risk_score)s, %(risk_level)s)
    """
    try:
        conn = get_conn()
        cur  = conn.cursor()
        cur.execute(sql, data)
        conn.commit()
        conn.close()
    except Exception as e:
        log.warning(f"Log prédiction échoué : {e}")


def query_df(sql: str, params: dict = None) -> pd.DataFrame:
    """Exécute une requête SELECT et retourne un DataFrame."""
    try:
        engine = get_engine()
        return pd.read_sql(sql, engine, params=params)
    except Exception as e:
        log.error(f"Requête échouée : {e}")
        return pd.DataFrame()


def get_stats() -> dict:
    """Calcule les statistiques globales depuis PostgreSQL."""
    queries = {
        "total":    "SELECT COUNT(*) FROM accidents",
        "mortels":  "SELECT COUNT(*) FROM accidents WHERE gravity = 3",
        "graves":   "SELECT COUNT(*) FROM accidents WHERE gravity = 2",
        "legers":   "SELECT COUNT(*) FROM accidents WHERE gravity = 1",
        "hotspots": "SELECT COUNT(*) FROM hotspots WHERE risk_level IN ('critique','élevé')",
        "preds":    "SELECT COUNT(*) FROM predictions_log",
    }
    stats = {}
    conn  = get_conn()
    cur   = conn.cursor()
    for key, sql in queries.items():
        try:
            cur.execute(sql)
            stats[key] = cur.fetchone()[0]
        except Exception:
            stats[key] = 0
    conn.close()
    return stats
