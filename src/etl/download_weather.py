"""
SafeRoads SN — download_weather.py
Télécharge la météo historique (2018-2024) pour les villes principales
du Sénégal via l'API Open-Meteo (gratuite, sans clé).

Variables récupérées :
  - Précipitations (mm/h)
  - Vitesse du vent (km/h)
  - Température (°C)
  - Code météo WMO (soleil, pluie, orage...)

Usage :
    python -m src.etl.download_weather
"""

import sys
import time
import logging
from pathlib import Path
from datetime import date

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import WEATHER_RAW_DIR, SENEGAL_CITIES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [WEATHER] %(message)s")
log = logging.getLogger(__name__)

# ── Paramètres ──
START_DATE   = "2018-01-01"
END_DATE     = date.today().isoformat()
WEATHER_PATH = WEATHER_RAW_DIR / "senegal_weather_history.csv"

# URL Open-Meteo Historical Weather API (gratuite)
API_URL = "https://archive-api.open-meteo.com/v1/archive"

# Variables horaires à récupérer
HOURLY_VARS = [
    "precipitation",           # mm
    "windspeed_10m",           # km/h
    "temperature_2m",          # °C
    "weathercode",             # code WMO
    "visibility",              # mètres
    "relativehumidity_2m",     # %
]

# Mapping codes WMO → catégorie lisible
WMO_TO_WEATHER = {
    0:  "Ensoleillé",
    1:  "Principalement dégagé",
    2:  "Partiellement nuageux",
    3:  "Nuageux",
    45: "Brouillard",
    48: "Brouillard givrant",
    51: "Bruine légère",
    53: "Bruine modérée",
    55: "Bruine forte",
    61: "Pluie légère",
    63: "Pluie modérée",
    65: "Pluie forte",
    80: "Averses légères",
    81: "Averses modérées",
    82: "Averses violentes",
    95: "Orage",
    96: "Orage avec grêle",
    99: "Orage violent",
}


def fetch_city_weather(city: dict, start: str, end: str) -> pd.DataFrame:
    """
    Télécharge les données météo horaires pour une ville donnée.
    """
    params = {
        "latitude":  city["lat"],
        "longitude": city["lon"],
        "start_date": start,
        "end_date":   end,
        "hourly":     ",".join(HOURLY_VARS),
        "timezone":   "Africa/Dakar",
    }

    resp = requests.get(API_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    hourly = data.get("hourly", {})
    if not hourly:
        raise ValueError(f"Pas de données horaires pour {city['name']}")

    df = pd.DataFrame(hourly)
    df["city"]      = city["name"]
    df["latitude"]  = city["lat"]
    df["longitude"] = city["lon"]
    df = df.rename(columns={"time": "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"])

    # Colonnes de renommage
    rename_map = {
        "precipitation":        "precipitation_mm",
        "windspeed_10m":        "windspeed_kmh",
        "temperature_2m":       "temperature_c",
        "weathercode":          "weather_code",
        "visibility":           "visibility_m",
        "relativehumidity_2m":  "humidity_pct",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Catégorie météo lisible
    if "weather_code" in df.columns:
        df["weather_label"] = df["weather_code"].map(WMO_TO_WEATHER).fillna("Inconnu")

    # Feature binaire : pluie (utile pour le modèle ML)
    if "precipitation_mm" in df.columns:
        df["is_rainy"] = df["precipitation_mm"] > 0.1

    # Visibilité en km
    if "visibility_m" in df.columns:
        df["visibility_km"] = (df["visibility_m"] / 1000).round(2)

    return df


def download_weather_history(force: bool = False) -> pd.DataFrame:
    """
    Télécharge la météo pour toutes les villes du Sénégal.
    Si déjà téléchargé, charge depuis le cache sauf si force=True.
    """
    if WEATHER_PATH.exists() and not force:
        log.info(f"Cache trouvé → chargement depuis {WEATHER_PATH}")
        return load_weather_features()

    WEATHER_RAW_DIR.mkdir(parents=True, exist_ok=True)

    log.info(f"Téléchargement météo {START_DATE} → {END_DATE}")
    log.info(f"Villes : {[c['name'] for c in SENEGAL_CITIES]}")

    all_dfs = []
    for i, city in enumerate(SENEGAL_CITIES, 1):
        log.info(f"  [{i}/{len(SENEGAL_CITIES)}] {city['name']}...")
        try:
            df = fetch_city_weather(city, START_DATE, END_DATE)
            all_dfs.append(df)
            log.info(f"    ✓ {len(df):,} enregistrements horaires")
            time.sleep(0.5)  # Respecter le rate limit Open-Meteo
        except Exception as e:
            log.warning(f"    ✗ {city['name']} ignorée : {e}")

    if not all_dfs:
        raise RuntimeError("Aucune donnée météo téléchargée.")

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values(["city", "datetime"]).reset_index(drop=True)

    # Sauvegarder
    combined.to_csv(WEATHER_PATH, index=False)
    log.info(f"✅ Météo sauvegardée : {WEATHER_PATH}")
    log.info(f"   {len(combined):,} enregistrements | {combined['city'].nunique()} villes")

    return combined


def load_weather_features() -> pd.DataFrame:
    """
    Charge la météo historique depuis le cache local.
    """
    if not WEATHER_PATH.exists():
        raise FileNotFoundError(
            f"Météo non trouvée : {WEATHER_PATH}\n"
            f"Lancer : python scripts/setup_data.py --weather-only"
        )
    log.info(f"Chargement météo depuis {WEATHER_PATH}")
    df = pd.read_csv(WEATHER_PATH, parse_dates=["datetime"])
    log.info(f"{len(df):,} enregistrements météo chargés")
    return df


def get_weather_for_datetime(dt: pd.Timestamp, lat: float, lon: float) -> dict:
    """
    Retourne les conditions météo les plus proches pour un datetime et une position.
    Utilisé pour enrichir les accidents lors de la fusion.
    """
    df = load_weather_features()

    # Trouver la ville la plus proche (distance euclidienne approximative)
    import numpy as np
    df_cities = pd.DataFrame(SENEGAL_CITIES)
    df_cities["dist"] = np.sqrt(
        (df_cities["lat"] - lat)**2 + (df_cities["lon"] - lon)**2
    )
    nearest_city = df_cities.loc[df_cities["dist"].idxmin(), "name"]

    # Filtrer sur la ville et l'heure exacte (arrondie à l'heure)
    dt_hour = dt.floor("H")
    mask = (df["city"] == nearest_city) & (df["datetime"] == dt_hour)
    row = df[mask]

    if row.empty:
        return {"weather_label": "Inconnu", "is_rainy": False,
                "windspeed_kmh": 0.0, "precipitation_mm": 0.0}

    row = row.iloc[0]
    return {
        "weather_label":    row.get("weather_label", "Inconnu"),
        "is_rainy":         bool(row.get("is_rainy", False)),
        "windspeed_kmh":    float(row.get("windspeed_kmh", 0)),
        "precipitation_mm": float(row.get("precipitation_mm", 0)),
        "temperature_c":    float(row.get("temperature_c", 25)),
        "visibility_km":    float(row.get("visibility_km", 10)),
        "humidity_pct":     float(row.get("humidity_pct", 60)),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Forcer le re-téléchargement")
    parser.add_argument("--city",  type=str, help="Télécharger une seule ville (test)")
    args = parser.parse_args()

    if args.city:
        city_info = next((c for c in SENEGAL_CITIES if c["name"].lower() == args.city.lower()), None)
        if not city_info:
            print(f"Ville '{args.city}' non trouvée. Options : {[c['name'] for c in SENEGAL_CITIES]}")
        else:
            df = fetch_city_weather(city_info, "2023-01-01", "2023-12-31")
            print(df.head())
            print(f"\n✅ {len(df)} enregistrements pour {args.city}")
    else:
        df = download_weather_history(force=args.force)
        print(f"\n✅ {len(df):,} enregistrements météo disponibles")
        print(df[["datetime", "city", "precipitation_mm", "windspeed_kmh", "weather_label"]].head(10))
