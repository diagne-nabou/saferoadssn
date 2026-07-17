"""
SafeRoads SN — db.py
Connexion SQLAlchemy + PostGIS.
Fournit une session et des helpers d'insertion géospatiale.
"""

import logging
from contextlib import contextmanager

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from geoalchemy2 import Geometry

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import DB_URL

log = logging.getLogger(__name__)


def get_engine():
    """Retourne le moteur SQLAlchemy."""
    return create_engine(DB_URL, pool_pre_ping=True, pool_size=5, max_overflow=10)


@contextmanager
def get_session():
    """Context manager pour une session SQLAlchemy."""
    engine  = get_engine()
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def check_connection() -> bool:
    """Vérifie que la connexion PostGIS est opérationnelle."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT PostGIS_Version()"))
            version = result.scalar()
            log.info(f"PostGIS connecté — version {version}")
            return True
    except Exception as e:
        log.error(f"Connexion PostGIS échouée : {e}")
        return False


def insert_accidents(df: pd.DataFrame) -> int:
    """
    Insère un DataFrame d'accidents dans la table PostgreSQL/PostGIS.
    Retourne le nombre de lignes insérées.
    """
    engine = get_engine()
    rows_inserted = 0

    with engine.connect() as conn:
        for _, row in df.iterrows():
            try:
                conn.execute(text("""
                    INSERT INTO accidents (
                        source, datetime, year, month, day_of_week, hour,
                        geom, latitude, longitude, region, road_type,
                        vehicle_type, cause, weather, gravity,
                        num_vehicles, num_victims
                    ) VALUES (
                        :source, :datetime, :year, :month, :day_of_week, :hour,
                        ST_SetSRID(ST_MakePoint(:longitude, :latitude), 4326),
                        :latitude, :longitude, :region, :road_type,
                        :vehicle_type, :cause, :weather, :gravity,
                        :num_vehicles, :num_victims
                    )
                    ON CONFLICT DO NOTHING
                """), {
                    "source":       row.get("geo_source", "csv"),
                    "datetime":     row.get("datetime"),
                    "year":         int(row.get("year",  2022)),
                    "month":        int(row.get("month", 1)),
                    "day_of_week":  int(row.get("day_of_week", 0)),
                    "hour":         int(row.get("hour", 12)),
                    "latitude":     float(row["latitude"]),
                    "longitude":    float(row["longitude"]),
                    "region":       row.get("region", "Inconnue"),
                    "road_type":    row.get("road_type", "inconnue"),
                    "vehicle_type": row.get("vehicle_type", "Autre"),
                    "cause":        row.get("cause", "Inconnue"),
                    "weather":      row.get("weather", "Inconnu"),
                    "gravity":      int(row.get("gravity", 1)),
                    "num_vehicles": int(row.get("num_vehicles", 1)),
                    "num_victims":  int(row.get("num_victims", 1)),
                })
                rows_inserted += 1
            except Exception as e:
                log.warning(f"Ligne ignorée : {e}")

        conn.commit()

    log.info(f"{rows_inserted:,} accidents insérés dans PostGIS")
    return rows_inserted


def insert_hotspots(df_hotspots: pd.DataFrame) -> int:
    """Insère les hotspots DBSCAN dans PostGIS."""
    engine = get_engine()
    rows_inserted = 0

    # Vider la table avant réinsertion
    with engine.connect() as conn:
        conn.execute(text("TRUNCATE TABLE hotspots RESTART IDENTITY"))
        conn.commit()

    with engine.connect() as conn:
        for _, row in df_hotspots.iterrows():
            try:
                conn.execute(text("""
                    INSERT INTO hotspots (
                        cluster_id, center_lat, center_lon, geom,
                        accident_count, risk_level, region
                    ) VALUES (
                        :cluster_id, :center_lat, :center_lon,
                        ST_SetSRID(ST_MakePoint(:center_lon, :center_lat), 4326),
                        :accident_count, :risk_level, :region
                    )
                """), {
                    "cluster_id":     int(row["cluster_id"]),
                    "center_lat":     float(row["center_lat"]),
                    "center_lon":     float(row["center_lon"]),
                    "accident_count": int(row["accident_count"]),
                    "risk_level":     row["risk_level"],
                    "region":         row.get("region", "Inconnue"),
                })
                rows_inserted += 1
            except Exception as e:
                log.warning(f"Hotspot ignoré : {e}")

        conn.commit()

    log.info(f"{rows_inserted} hotspots insérés dans PostGIS")
    return rows_inserted


def query_accidents_bbox(
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    limit: int = 1000,
) -> pd.DataFrame:
    """
    Requête PostGIS : accidents dans une bounding box.
    """
    engine = get_engine()
    sql = text("""
        SELECT id, datetime, latitude, longitude,
               region, gravity, vehicle_type, cause, weather
        FROM accidents
        WHERE ST_Within(
            geom,
            ST_MakeEnvelope(:lon_min, :lat_min, :lon_max, :lat_max, 4326)
        )
        ORDER BY datetime DESC
        LIMIT :limit
    """)
    with engine.connect() as conn:
        result = conn.execute(sql, {
            "lat_min": lat_min, "lat_max": lat_max,
            "lon_min": lon_min, "lon_max": lon_max,
            "limit": limit,
        })
        return pd.DataFrame(result.fetchall(), columns=result.keys())
