"""
SafeRoads SN -- enrich_data.py
Enrichit accidents_senegal_final.xlsx avec :
  1. Geocodage : lat/lon precise par ville via Nominatim (OpenStreetMap)
  2. Meteo historique : temperature, pluie, vent via Open-Meteo Archive API

Usage :
    python scripts/enrich_data.py
"""

import sys
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [ENRICH] %(message)s")
log = logging.getLogger(__name__)

RAW_XLSX   = Path("data/raw/accidents/accidents_senegal_final.xlsx")
OUTPUT_CSV = Path("data/raw/accidents/accidents.csv")

# ══════════════════════════════════════════════════════
# GEOCODAGE — Coordonnees GPS precises par ville
# ══════════════════════════════════════════════════════

# Coordonnees manuelles fiables pour les villes du Senegal
# (plus precis que Nominatim pour les petites villes)
VILLE_COORDS = {
    # Grandes villes
    "Dakar":        (14.6928, -17.4467),
    "Pikine":       (14.7547, -17.3908),
    "Guediawaye":   (14.7744, -17.3900),
    "Rufisque":     (14.7158, -17.2736),
    "Yoff":         (14.7395, -17.4795),
    # Region Thies
    "Thies":        (14.7910, -16.9260),
    "Mbour":        (14.4158, -16.9664),
    "Tivaouane":    (14.9500, -16.8167),
    "Joal":         (14.1667, -16.8333),
    "Saly":         (14.4333, -17.0167),
    "Pout":         (14.7667, -17.0667),
    # Region Diourbel
    "Diourbel":     (14.6550, -16.2323),
    "Touba":        (14.8500, -15.8833),
    "Mbacke":       (14.7936, -15.9083),
    "Bambey":       (14.7000, -16.4500),
    "Gossas":       (14.4833, -16.0667),
    # Region Saint-Louis
    "Saint-Louis":  (16.0179, -16.4896),
    "Richard-Toll": (16.4667, -15.7000),
    "Podor":        (16.6544, -14.9611),
    "Dagana":       (16.5167, -15.5000),
    "Ndioum":       (16.5167, -14.6500),
    # Region Matam
    "Matam":        (15.6559, -13.2554),
    "Ourossogui":   (15.6000, -13.3167),
    "Kanel":        (15.4917, -13.1750),
    # Region Kaolack
    "Kaolack":      (14.1652, -16.0726),
    "Nioro":        (13.7333, -15.8000),
    "Guinguineo":   (14.2667, -15.9500),
    # Region Fatick
    "Fatick":       (14.3390, -16.4111),
    "Foundiougne":  (14.1333, -16.4667),
    "Tattaguine":   (14.3667, -16.5833),
    # Region Kaffrine
    "Kaffrine":     (14.1058, -15.5508),
    "Koungheul":    (13.9833, -14.8000),
    "Birkelane":    (14.0833, -15.7500),
    # Region Tambacounda
    "Tambacounda":  (13.7707, -13.6673),
    "Bakel":        (14.9000, -12.4500),
    "Goudiry":      (14.1833, -12.7167),
    "Koumpentoum":  (13.9833, -14.5500),
    # Region Kedougou
    "Kedougou":     (12.5605, -12.1747),
    "Saraya":       (12.8333, -11.7500),
    "Salemata":     (12.6333, -12.5833),
    # Region Kolda
    "Kolda":        (12.8983, -14.9412),
    "Velingara":    (13.1500, -14.1167),
    "Medina Gounass": (13.5333, -14.0000),
    # Region Ziguinchor
    "Ziguinchor":   (12.5681, -16.2719),
    "Bignona":      (12.8000, -16.2333),
    "Oussouye":     (12.4833, -16.5500),
    # Region Sedhiou
    "Sedhiou":      (12.7083, -15.5569),
    "Goudomp":      (12.5833, -15.8667),
    "Bounkiling":   (12.8000, -15.7000),
    # Region Louga
    "Louga":        (15.6172, -16.2240),
    "Linguere":     (15.3833, -15.1167),
    "Kebemer":      (15.3667, -16.4500),
    # Autres
    "Autoroute":    (14.7200, -17.2000),  # Autoroute AIBD
}

# Accents -> sans accents pour le matching
ACCENT_MAP = {
    "Thiès": "Thies", "Thies": "Thies",
    "Vélingara": "Velingara",
    "Kédougou": "Kedougou",
    "Sédhiou": "Sedhiou",
    "Linguère": "Linguere",
    "Kébémer": "Kebemer",
    "Mbacké": "Mbacke",
    "Guédiawaye": "Guediawaye",
    "Médina Gounass": "Medina Gounass",
    "Nioro du Rip": "Nioro",
}


def geocode_ville(ville: str, rng) -> tuple:
    """Retourne (lat, lon) pour une ville avec une legere dispersion."""
    if pd.isna(ville) or not isinstance(ville, str):
        return (np.nan, np.nan)

    ville_clean = ville.strip()

    # Essayer le mapping d'accents
    ville_key = ACCENT_MAP.get(ville_clean, ville_clean)

    if ville_key in VILLE_COORDS:
        lat, lon = VILLE_COORDS[ville_key]
        # Ajouter une dispersion de +/- 1-3 km (0.01-0.03 degre)
        # pour que les points ne se superposent pas
        lat += rng.uniform(-0.025, 0.025)
        lon += rng.uniform(-0.025, 0.025)
        return (round(lat, 6), round(lon, 6))

    # Ville inconnue (phrase parasite ou nom non reconnu)
    if len(ville_clean) > 30:
        return (np.nan, np.nan)

    # Tenter Nominatim en dernier recours
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": f"{ville_clean}, Senegal",
            "format": "json",
            "limit": 1,
            "countrycodes": "SN",
        }
        headers = {"User-Agent": "SafeRoads-SN/1.0 (student-project)"}
        resp = requests.get(url, params=params, headers=headers, timeout=5)
        if resp.ok and resp.json():
            data = resp.json()[0]
            lat = float(data["lat"]) + rng.uniform(-0.02, 0.02)
            lon = float(data["lon"]) + rng.uniform(-0.02, 0.02)
            log.info(f"    Nominatim: {ville_clean} -> ({lat:.4f}, {lon:.4f})")
            time.sleep(1.1)  # respecter rate limit
            return (round(lat, 6), round(lon, 6))
    except Exception:
        pass

    return (np.nan, np.nan)


# ══════════════════════════════════════════════════════
# METEO — Open-Meteo Archive API
# ══════════════════════════════════════════════════════

def fetch_meteo(lat: float, lon: float, date_str: str) -> dict:
    """
    Recupere la meteo historique pour un point + date via Open-Meteo.
    Retourne : temperature_c, precipitation_mm, windspeed_kmh, humidity_pct
    """
    defaults = {
        "temperature_c": np.nan,
        "precipitation_mm": np.nan,
        "windspeed_kmh": np.nan,
        "humidity_pct": np.nan,
        "visibility_km": np.nan,
        "weather_code": np.nan,
    }

    if pd.isna(lat) or pd.isna(lon) or pd.isna(date_str):
        return defaults

    try:
        dt = pd.to_datetime(date_str)
        date_only = dt.strftime("%Y-%m-%d")
    except Exception:
        return defaults

    try:
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": round(lat, 4),
            "longitude": round(lon, 4),
            "start_date": date_only,
            "end_date": date_only,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max,relative_humidity_2m_mean",
            "timezone": "Africa/Dakar",
        }
        resp = requests.get(url, params=params, timeout=10)
        if not resp.ok:
            return defaults

        data = resp.json().get("daily", {})
        if not data or not data.get("temperature_2m_max"):
            return defaults

        t_max = data["temperature_2m_max"][0]
        t_min = data["temperature_2m_min"][0]
        temp = round((t_max + t_min) / 2, 1) if t_max is not None and t_min is not None else np.nan
        precip = data.get("precipitation_sum", [None])[0]
        wind = data.get("windspeed_10m_max", [None])[0]
        humidity = data.get("relative_humidity_2m_mean", [None])[0]

        return {
            "temperature_c": temp,
            "precipitation_mm": round(precip, 1) if precip is not None else 0.0,
            "windspeed_kmh": round(wind, 1) if wind is not None else np.nan,
            "humidity_pct": round(humidity, 1) if humidity is not None else np.nan,
            "visibility_km": 10.0,  # pas dispo dans l'API gratuite
            "weather_code": 0,
        }
    except Exception as e:
        return defaults


# ══════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════

def enrich():
    log.info("=" * 55)
    log.info("  SafeRoads SN -- Enrichissement des donnees")
    log.info("=" * 55)

    if not RAW_XLSX.exists():
        raise FileNotFoundError(f"Fichier non trouve : {RAW_XLSX}")

    df = pd.read_excel(RAW_XLSX)
    log.info(f"  {len(df)} accidents charges depuis {RAW_XLSX.name}")
    log.info(f"  Colonnes : {list(df.columns)}")

    # Normaliser les noms de colonnes
    col_map = {
        "Date": "date", "Titre": "titre", "Description": "description",
        "Lien": "lien", "Ville": "ville", "Nb Morts": "nb_morts",
        "Nb Blessés": "nb_blesses", "Nb Blesses": "nb_blesses",
        "Type Véhicule": "type_vehicule", "Type Vehicule": "type_vehicule",
        "Source": "source",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # ── Etape 1 : Geocodage ──
    log.info("\n  [1/2] GEOCODAGE par ville...")
    rng = np.random.default_rng(42)
    lats, lons = [], []
    for _, row in df.iterrows():
        lat, lon = geocode_ville(row.get("ville"), rng)
        lats.append(lat)
        lons.append(lon)

    df["lat"] = lats
    df["lon"] = lons
    n_geo = df["lat"].notna().sum()
    log.info(f"    {n_geo}/{len(df)} lignes geocodees ({len(df)-n_geo} sans coordonnees)")

    # ── Etape 2 : Meteo ──
    log.info("\n  [2/2] METEO historique via Open-Meteo...")

    # Regrouper par (ville, date) pour eviter les appels redondants
    df["_date_str"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    meteo_cache = {}
    meteo_cols = ["temperature_c", "precipitation_mm", "windspeed_kmh", "humidity_pct", "visibility_km"]
    for col in meteo_cols:
        df[col] = np.nan

    # Deduplication : une seule requete par (lat arrondi, date)
    unique_queries = df[["lat", "lon", "_date_str"]].dropna().drop_duplicates()
    unique_queries["_key"] = (
        unique_queries["lat"].round(2).astype(str) + "_" +
        unique_queries["lon"].round(2).astype(str) + "_" +
        unique_queries["_date_str"]
    )
    unique_keys = unique_queries.drop_duplicates(subset="_key")
    log.info(f"    {len(unique_keys)} requetes meteo uniques a faire...")

    for i, (_, row) in enumerate(unique_keys.iterrows()):
        key = row["_key"]
        if key not in meteo_cache:
            meteo_cache[key] = fetch_meteo(row["lat"], row["lon"], row["_date_str"])
            if (i + 1) % 20 == 0:
                log.info(f"    ... {i+1}/{len(unique_keys)} requetes")
            time.sleep(0.15)  # rate limit Open-Meteo

    # Appliquer les resultats
    for idx, row in df.iterrows():
        if pd.notna(row["lat"]) and pd.notna(row["_date_str"]):
            key = f"{round(row['lat'], 2)}_{round(row['lon'], 2)}_{row['_date_str']}"
            meteo = meteo_cache.get(key, {})
            for col in meteo_cols:
                if col in meteo and meteo[col] is not None:
                    df.at[idx, col] = meteo[col]

    n_meteo = df["temperature_c"].notna().sum()
    log.info(f"    {n_meteo}/{len(df)} lignes avec meteo")

    # ── Colonnes derivees ──
    df["is_rainy"] = (df["precipitation_mm"] > 0.5).astype(int)
    df["condition_pluie"] = "sec"
    df.loc[df["precipitation_mm"] > 0.5, "condition_pluie"] = "pluie_legere"
    df.loc[df["precipitation_mm"] > 5.0, "condition_pluie"] = "pluie_moderee"
    df.loc[df["precipitation_mm"] > 15.0, "condition_pluie"] = "pluie_forte"
    df.loc[df["precipitation_mm"] > 30.0, "condition_pluie"] = "pluie_tres_forte"

    # Flag aberrant (villes parasites = phrases longues)
    df["flag_aberrant"] = 0
    if "ville" in df.columns:
        df.loc[df["ville"].str.len() > 30, "flag_aberrant"] = 1

    # Extraire annee, mois, trimestre, jour_sem depuis date
    dt = pd.to_datetime(df["date"], errors="coerce")
    df["annee"] = dt.dt.year
    df["mois"] = dt.dt.month
    df["trimestre"] = dt.dt.quarter
    df["jour_sem"] = dt.dt.day_name()

    # Nettoyage colonnes finales
    df = df.drop(columns=["_date_str"], errors="ignore")

    # ── Sauvegarder ──
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    log.info(f"\n  [OK] Sauvegarde : {OUTPUT_CSV}")
    log.info(f"  {len(df)} lignes | {len(df.columns)} colonnes")
    log.info(f"  Geocodes: {n_geo} | Meteo: {n_meteo}")

    return df


if __name__ == "__main__":
    enrich()
