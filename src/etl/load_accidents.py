"""
SafeRoads SN — load_accidents.py
Charge, valide et normalise le fichier CSV des accidents.
Compatible avec :
  - Dataset Kaggle "Road Traffic Accidents" (Éthiopie)
  - Format SafeRoads natif (voir data/raw/accidents/README_format.md)
  - Tout CSV avec colonnes minimales (datetime, latitude, longitude, gravity)

Usage :
    python -m src.etl.load_accidents
"""

import sys
import logging
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import ACCIDENTS_RAW_PATH, DATA_PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [ACCIDENTS] %(message)s")
log = logging.getLogger(__name__)

ACCIDENTS_CLEAN_PATH = DATA_PROCESSED_DIR / "accidents_clean.csv"

# ── Bbox Sénégal (filtre géographique) ──
SENEGAL_LAT_MIN, SENEGAL_LAT_MAX = 12.3, 16.7
SENEGAL_LON_MIN, SENEGAL_LON_MAX = -17.6, -11.3

# ── Mapping Kaggle → SafeRoads ──
# Dataset : https://kaggle.com/datasets/saurabhshahane/road-traffic-accidents
KAGGLE_COLUMN_MAP = {
    "Time":                    "time_raw",
    "Day_of_week":             "day_of_week_raw",
    "Age_band_of_driver":      "driver_age_band",
    "Sex_of_driver":           "driver_sex",
    "Educational_level":       "driver_education",
    "Vehicle_driver_relation": "driver_relation",
    "Driving_experience":      "driving_experience",
    "Type_of_vehicle":         "vehicle_type_raw",
    "Owner_of_vehicle":        "vehicle_owner",
    "Service_year_of_vehicle": "vehicle_age",
    "Defect_of_vehicle":       "vehicle_defect",
    "Area_accident_occured":   "area_type",
    "Lanes_or_Medians":        "road_lanes",
    "Road_allignment":         "road_alignment",
    "Types_of_Junction":       "junction_type",
    "Road_surface_type":       "road_surface",
    "Road_surface_conditions": "road_condition",
    "Light_conditions":        "light_condition",
    "Weather_conditions":      "weather_raw",
    "Type_of_collision":       "collision_type",
    "Number_of_vehicles_involved": "num_vehicles",
    "Number_of_casualties":    "num_victims",
    "Vehicle_movement":        "vehicle_movement",
    "Casualty_class":          "casualty_class",
    "Sex_of_casualty":         "casualty_sex",
    "Age_band_of_casualty":    "casualty_age_band",
    "Casualty_severity":       "gravity_raw",
    "Cause_of_accident":       "cause_raw",
}

# Mapping gravité Kaggle → 1/2/3
KAGGLE_GRAVITY_MAP = {
    "Slight Injury":   1,
    "Serious Injury":  2,
    "Fatal injury":    3,
}

# Mapping météo Kaggle → labels SafeRoads
KAGGLE_WEATHER_MAP = {
    "Normal":           "Ensoleillé",
    "Raining":          "Pluie légère",
    "Raining and Windy":"Pluie forte",
    "Cloudy":           "Nuageux",
    "Windy":            "Nuageux",
    "Snow":             "Pluie légère",
    "Fog or mist":      "Brouillard",
    "Other":            "Inconnu",
    "Unknown":          "Inconnu",
}

# Mapping type de véhicule Kaggle → SafeRoads
KAGGLE_VEHICLE_MAP = {
    "Automobile":             "Voiture",
    "Lorry (11-40Q)":        "Camion",
    "Lorry (40Q+)":          "Camion",
    "Taxi":                  "Taxi",
    "Public (> 45 seats)":   "Car rapide",
    "Public (12-45 seats)":  "Sept-places",
    "Public (< 12 seats)":   "Sept-places",
    "Motorcycle":            "Moto-Jakarta",
    "Pick up upto 10Q":      "Pickup",
    "Stationwagen":          "Voiture",
    "Ridden horse":          "Charette",
    "Other":                 "Autre",
}

# Régions Sénégal avec leurs coordonnées approximatives
# (utilisé pour simuler la géolocalisation sur le dataset Kaggle)
# Les 14 regions administratives officielles du Senegal
SENEGAL_REGIONS_COORDS = [
    {"region": "Dakar",        "lat": 14.6937, "lon": -17.4441, "weight": 0.18},
    {"region": "Thies",        "lat": 14.7886, "lon": -16.9260, "weight": 0.13},
    {"region": "Kaolack",      "lat": 14.1652, "lon": -16.0726, "weight": 0.07},
    {"region": "Saint-Louis",  "lat": 16.0179, "lon": -16.4896, "weight": 0.05},
    {"region": "Diourbel",     "lat": 14.6550, "lon": -16.2323, "weight": 0.08},
    {"region": "Ziguinchor",   "lat": 12.5681, "lon": -16.2719, "weight": 0.05},
    {"region": "Tambacounda",  "lat": 13.7707, "lon": -13.6673, "weight": 0.06},
    {"region": "Louga",        "lat": 15.6172, "lon": -16.2240, "weight": 0.04},
    {"region": "Kolda",        "lat": 12.8983, "lon": -14.9412, "weight": 0.05},
    {"region": "Matam",        "lat": 15.6559, "lon": -13.2554, "weight": 0.04},
    {"region": "Fatick",       "lat": 14.3390, "lon": -16.4111, "weight": 0.06},
    {"region": "Kaffrine",     "lat": 14.1058, "lon": -15.5508, "weight": 0.05},
    {"region": "Kedougou",     "lat": 12.5605, "lon": -12.1747, "weight": 0.04},
    {"region": "Sedhiou",      "lat": 12.7083, "lon": -15.5569, "weight": 0.05},
    # Mbour = departement de Thies, pas une region
]


def detect_format(df: pd.DataFrame) -> str:
    """
    Detecte automatiquement le format du CSV (kaggle, native, ou senegal_real).
    """
    kaggle_cols = {"Casualty_severity", "Cause_of_accident", "Type_of_vehicle"}
    native_cols = {"datetime", "latitude", "longitude", "gravity"}
    senegal_cols = {"date", "ville", "nb_morts", "nb_blesses"}

    if kaggle_cols.issubset(set(df.columns)):
        return "kaggle"
    elif native_cols.issubset(set(df.columns)):
        return "native"
    elif senegal_cols.issubset(set(df.columns)):
        return "senegal_real"
    else:
        return "unknown"


def _add_senegal_coordinates(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Pour le dataset Kaggle (Éthiopie), assigne des coordonnées sénégalaises
    pondérées selon la densité réelle des accidents par région.
    """
    rng = np.random.default_rng(seed)
    regions = SENEGAL_REGIONS_COORDS
    weights = [r["weight"] for r in regions]

    n = len(df)
    chosen = rng.choice(len(regions), size=n, p=weights)

    lats, lons, region_names = [], [], []
    for idx in chosen:
        r = regions[idx]
        # Dispersion autour du centre de la région (±0.3°)
        lats.append(r["lat"] + rng.uniform(-0.3, 0.3))
        lons.append(r["lon"] + rng.uniform(-0.3, 0.3))
        region_names.append(r["region"])

    df["latitude"]  = lats
    df["longitude"] = lons
    df["region"]    = region_names
    df["geo_source"] = "simulated"  # Indiquer que la géoloc est simulée

    log.warning(
        "Coordonnées GPS non disponibles dans le CSV source → "
        "coordonnées sénégalaises simulées (géoloc approximative par région). "
        "Remplacer par de vraies données GPS pour de meilleures performances ML."
    )
    return df


def normalize_kaggle(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise le format Kaggle vers le format SafeRoads standard.
    """
    df = df_raw.rename(columns=KAGGLE_COLUMN_MAP).copy()

    # Gravité
    if "gravity_raw" in df.columns:
        df["gravity"] = df["gravity_raw"].map(KAGGLE_GRAVITY_MAP).fillna(1).astype(int)

    # Météo
    if "weather_raw" in df.columns:
        df["weather"] = df["weather_raw"].map(KAGGLE_WEATHER_MAP).fillna("Inconnu")

    # Type de véhicule
    if "vehicle_type_raw" in df.columns:
        df["vehicle_type"] = df["vehicle_type_raw"].map(KAGGLE_VEHICLE_MAP).fillna("Autre")

    # Cause
    if "cause_raw" in df.columns:
        df["cause"] = df["cause_raw"].str.strip()

    # Heure depuis la colonne Time (format "17:02:00")
    if "time_raw" in df.columns:
        df["hour"] = pd.to_datetime(df["time_raw"], format="%H:%M:%S", errors="coerce").dt.hour
        df["hour"] = df["hour"].fillna(12).astype(int)

    # Jour de la semaine
    day_map = {"Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,
               "Friday":4,"Saturday":5,"Sunday":6}
    if "day_of_week_raw" in df.columns:
        df["day_of_week"] = df["day_of_week_raw"].map(day_map)

    # Construire un datetime approximatif (année 2022 par défaut)
    df["month"] = np.random.randint(1, 13, size=len(df))
    df["year"]  = 2022
    df["datetime"] = pd.to_datetime({
        "year":  df["year"],
        "month": df["month"],
        "day":   1,
        "hour":  df.get("hour", 12),
    }, errors="coerce")

    # Type de route
    area_to_road = {
        "Office areas":       "urbaine",
        "Residential areas":  "urbaine",
        "Church areas":       "urbaine",
        "Industrial areas":   "urbaine",
        "School areas":       "urbaine",
        "Recreational areas": "urbaine",
        "Outside residential areas": "nationale",
        "Market areas":       "urbaine",
        "Rural vehicle road": "piste",
        "Unknown":            "inconnue",
    }
    if "area_type" in df.columns:
        df["road_type"] = df["area_type"].map(area_to_road).fillna("urbaine")

    # Ajouter coordonnées GPS sénégalaises
    df = _add_senegal_coordinates(df)

    return df


# ── Mappings données réelles Sénégal ──

# Ville -> Region administrative officielle du Senegal (14 regions)
VILLE_TO_REGION = {
    # Region Dakar
    "Dakar": "Dakar", "Pikine": "Dakar", "Guediawaye": "Dakar",
    "Gu\u00e9diawaye": "Dakar", "Rufisque": "Dakar", "Yoff": "Dakar",
    # Region Thies (+ departement Mbour et Tivaouane)
    "Thi\u00e8s": "Thies", "Thies": "Thies", "Thi\ufffd\ufffds": "Thies",
    "Tivaouane": "Thies", "Pout": "Thies",
    "Mbour": "Thies", "Saly": "Thies", "Joal": "Thies",
    # Region Diourbel
    "Diourbel": "Diourbel", "Touba": "Diourbel", "Mback\u00e9": "Diourbel",
    "Mbacke": "Diourbel", "Bambey": "Diourbel", "Gossas": "Diourbel",
    # Region Saint-Louis
    "Saint-Louis": "Saint-Louis", "Richard-Toll": "Saint-Louis",
    "Podor": "Saint-Louis", "Dagana": "Saint-Louis", "Ndioum": "Saint-Louis",
    # Region Matam (region a part entiere depuis 2002)
    "Matam": "Matam", "Ourossogui": "Matam", "Ranérou": "Matam",
    "Kanel": "Matam",
    # Region Kaolack
    "Kaolack": "Kaolack", "Nioro": "Kaolack", "Nioro du Rip": "Kaolack",
    "Guinguineo": "Kaolack",
    # Region Fatick (region a part entiere)
    "Fatick": "Fatick", "Foundiougne": "Fatick", "Gossas": "Fatick",
    "Tattaguine": "Fatick",
    # Region Kaffrine (region a part entiere depuis 2008)
    "Kaffrine": "Kaffrine", "Koungheul": "Kaffrine",
    "Birkelane": "Kaffrine", "Malem-Hodar": "Kaffrine",
    # Region Tambacounda
    "Tambacounda": "Tambacounda", "Bakel": "Tambacounda",
    "Goudiry": "Tambacounda", "Koumpentoum": "Tambacounda",
    # Region Kedougou (region a part entiere depuis 2008)
    "K\u00e9dougou": "Kedougou", "Kedougou": "Kedougou",
    "Saraya": "Kedougou", "Salemata": "Kedougou",
    # Region Kolda
    "Kolda": "Kolda", "V\u00e9lingara": "Kolda", "Velingara": "Kolda",
    "M\u00e9dina Gounass": "Kolda", "Medina Gounass": "Kolda",
    # Region Ziguinchor
    "Ziguinchor": "Ziguinchor", "Bignona": "Ziguinchor", "Oussouye": "Ziguinchor",
    # Region Sedhiou (region a part entiere depuis 2008)
    "S\u00e9dhiou": "Sedhiou", "Sedhiou": "Sedhiou",
    "Goudomp": "Sedhiou", "Bounkiling": "Sedhiou",
    # Region Louga
    "Louga": "Louga", "Lingu\u00e8re": "Louga", "Linguere": "Louga",
    "K\u00e9b\u00e9mer": "Louga", "Kebemer": "Louga",
    # Autoroute -> Dakar (la seule autoroute est l'AIBD pres de Dakar)
    "Autoroute": "Dakar",
    # Inconnue
    "Inconnue": "Dakar",
}

# Type véhicule réel → SafeRoads
SENEGAL_VEHICLE_MAP = {
    "Voiture": "Voiture", "4x4": "Voiture",
    "Camion": "Camion", "Camionnette": "Camion", "Camion-citerne": "Camion",
    "Bus": "Car rapide", "Minicar": "Car rapide", "Car": "Car rapide", "Car rapide": "Car rapide",
    "Moto": "Moto-Jakarta", "Moto-taxi / Jakarta": "Moto-Jakarta", "Deux-roues": "Moto-Jakarta",
    "Taxi": "Taxi",
    "7 places": "Sept-places",
    "Pick-up": "Pickup",
    "Charrette": "Charette",
}

# Road type OSM → SafeRoads
OSM_ROAD_MAP = {
    "motorway": "autoroute", "trunk": "nationale", "primary": "nationale",
    "secondary": "régionale", "tertiary": "départementale",
    "residential": "urbaine", "unclassified": "piste", "track": "piste",
}

# Condition pluie → Weather label SafeRoads
CONDITION_WEATHER_MAP = {
    "sec": "Ensoleillé",
    "pluie_legere": "Pluie légère",
    "pluie_moderee": "Pluie légère",
    "pluie_forte": "Pluie forte",
    "pluie_tres_forte": "Orage",
}

# Jour semaine anglais → int
DAY_EN_MAP = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
              "Friday": 4, "Saturday": 5, "Sunday": 6}


def normalize_senegal_real(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise le format reel Senegal (accidents_senegal_meteo_final.csv)
    vers le format SafeRoads standard.
    """
    rng = np.random.default_rng(42)
    df = df_raw.copy()

    # Filtrer les aberrants si la colonne existe
    if "flag_aberrant" in df.columns:
        n_before = len(df)
        df = df[df["flag_aberrant"] == 0]
        if len(df) < n_before:
            log.info(f"  Filtre flag_aberrant : {n_before} -> {len(df)} lignes")

    # ── Datetime ──
    df["datetime"] = pd.to_datetime(df["date"], errors="coerce")

    # Recuperer les 94 lignes sans date : reconstruire depuis annee + mois
    mask_no_dt = df["datetime"].isna()
    if mask_no_dt.any():
        year_fill = pd.to_numeric(df.loc[mask_no_dt, "annee"], errors="coerce").fillna(2025).astype(int)
        month_fill = pd.to_numeric(df.loc[mask_no_dt, "mois"], errors="coerce").fillna(1).astype(int)
        df.loc[mask_no_dt, "datetime"] = pd.to_datetime(
            year_fill.astype(str) + "-" + month_fill.astype(str) + "-15",
            errors="coerce"
        )
        log.info(f"  {mask_no_dt.sum()} lignes sans date -> reconstruites depuis annee/mois")

    year_raw = pd.to_numeric(df.get("annee", pd.Series(dtype=float)), errors="coerce")
    df["year"] = year_raw.fillna(df["datetime"].dt.year).fillna(2025).astype(int)
    month_raw = pd.to_numeric(df.get("mois", pd.Series(dtype=float)), errors="coerce")
    df["month"] = month_raw.fillna(df["datetime"].dt.month).fillna(1).astype(int)

    # Jour de la semaine
    if "jour_sem" in df.columns:
        df["day_of_week"] = df["jour_sem"].map(DAY_EN_MAP)
        mask_na = df["day_of_week"].isna()
        if mask_na.any():
            df.loc[mask_na, "day_of_week"] = df.loc[mask_na, "datetime"].dt.dayofweek
        df["day_of_week"] = df["day_of_week"].fillna(0).astype(int)

    # Heure : NON DISPONIBLE dans les donnees reelles (pas de simulation)
    df["hour"] = -1  # Marqueur "heure inconnue"

    # ── Gravite ──
    # Derive de nb_morts / nb_blesses
    df["nb_morts"]   = pd.to_numeric(df.get("nb_morts", 0), errors="coerce").fillna(0)
    df["nb_blesses"] = pd.to_numeric(df.get("nb_blesses", 0), errors="coerce").fillna(0)
    df["gravity"] = 1  # leger par defaut
    df.loc[df["nb_blesses"] > 0, "gravity"] = 2  # grave
    df.loc[df["nb_morts"] > 0, "gravity"] = 3    # mortel

    # ── Coordonnees ──
    df["latitude"]  = pd.to_numeric(df.get("lat", np.nan), errors="coerce")
    df["longitude"] = pd.to_numeric(df.get("lon", np.nan), errors="coerce")

    # ── Nettoyage villes parasites ──
    # Certaines lignes ont des phrases entieres au lieu d'un nom de ville
    if "ville" in df.columns:
        # Remplacer les villes trop longues (>30 chars = probablement une phrase)
        df.loc[df["ville"].str.len() > 30, "ville"] = "Inconnue"
        # Normaliser les accents casses (Thi\ufffd\ufffds -> Thies)
        df["ville"] = df["ville"].str.strip()

    # Region depuis ville (14 regions administratives)
    df["region"] = df["ville"].map(VILLE_TO_REGION).fillna("Dakar")
    n_unmapped = (df["ville"].map(VILLE_TO_REGION).isna()).sum()
    if n_unmapped > 0:
        unmapped = df.loc[df["ville"].map(VILLE_TO_REGION).isna(), "ville"].unique()
        log.warning(f"  {n_unmapped} lignes avec ville non mappee: {list(unmapped)[:10]}")

    # Geocoder les lignes sans lat/lon depuis les coordonnees de region
    region_coords = {r["region"]: (r["lat"], r["lon"]) for r in SENEGAL_REGIONS_COORDS}
    mask_no_geo = df["latitude"].isna() | df["longitude"].isna()
    if mask_no_geo.any():
        n_miss = mask_no_geo.sum()
        log.info(f"  {n_miss} lignes sans lat/lon -> geocodage par region")
        for idx in df[mask_no_geo].index:
            reg = df.at[idx, "region"]
            coords = region_coords.get(reg, (14.6937, -17.4441))
            df.at[idx, "latitude"]  = coords[0] + rng.uniform(-0.2, 0.2)
            df.at[idx, "longitude"] = coords[1] + rng.uniform(-0.2, 0.2)

    df["geo_source"] = "real"
    df.loc[mask_no_geo, "geo_source"] = "geocoded"

    # ── Type vehicule ──
    if "type_vehicule" in df.columns:
        df["vehicle_type"] = df["type_vehicule"].map(SENEGAL_VEHICLE_MAP).fillna("Autre")
    else:
        df["vehicle_type"] = "Autre"

    # ── Type de route ──
    if "road_type" in df.columns:
        df["road_type"] = df["road_type"].map(OSM_ROAD_MAP).fillna("urbaine")
    else:
        df["road_type"] = "urbaine"

    # ── Meteo ──
    # Priorite : colonnes deja enrichies > colonnes brutes > defaut
    if "condition_pluie" in df.columns:
        df["weather"] = df["condition_pluie"].map(CONDITION_WEATHER_MAP).fillna("Inconnu")
        df["is_rainy"] = df["condition_pluie"].str.contains("pluie", na=False)
    elif "is_rainy" in df.columns:
        # Deja enrichi par enrich_data.py
        df["is_rainy"] = df["is_rainy"].fillna(0).astype(int)
        df["weather"] = "Inconnu"
        df.loc[df["is_rainy"] == 1, "weather"] = "Pluie"
    else:
        df["weather"] = "Inconnu"
        df["is_rainy"] = False

    # Precipitation : preserver si deja presente (enrichie)
    if "precipitation_mm" in df.columns:
        df["precipitation_mm"] = pd.to_numeric(df["precipitation_mm"], errors="coerce").fillna(0.0)
    elif "precipitation_sum" in df.columns:
        df["precipitation_mm"] = pd.to_numeric(df["precipitation_sum"], errors="coerce").fillna(0.0)
    else:
        df["precipitation_mm"] = 0.0

    # Vent : preserver si deja present
    if "windspeed_kmh" in df.columns:
        df["windspeed_kmh"] = pd.to_numeric(df["windspeed_kmh"], errors="coerce").fillna(15.0)
    elif "wind_speed_10m_max" in df.columns:
        df["windspeed_kmh"] = pd.to_numeric(df["wind_speed_10m_max"], errors="coerce").fillna(15.0)
    else:
        df["windspeed_kmh"] = 15.0

    # Temperature et humidite : preserver si enrichis
    if "temperature_c" in df.columns:
        df["temperature_c"] = pd.to_numeric(df["temperature_c"], errors="coerce")
    if "humidity_pct" in df.columns:
        df["humidity_pct"] = pd.to_numeric(df["humidity_pct"], errors="coerce")

    # Saison
    if "saison" in df.columns:
        df["season"] = df["saison"]
    else:
        df["season"] = df["month"].map(lambda m:
            "hivernage" if m in [6, 7, 8, 9, 10] else "saison_seche"
        )

    # Cause : pas dans le dataset reel, on met Inconnue
    df["cause"] = "Inconnue"

    # Victimes
    df["num_victims"]  = (df["nb_morts"] + df["nb_blesses"]).clip(1, 999).astype(int)
    df["num_vehicles"] = 2

    log.info(f"  Format Senegal reel : {len(df)} accidents normalises")
    log.info(f"  Regions : {df['region'].value_counts().to_dict()}")
    log.info(f"  Gravite : {{1: {(df['gravity']==1).sum()}, 2: {(df['gravity']==2).sum()}, 3: {(df['gravity']==3).sum()}}}")

    return df


def normalize_native(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise un CSV déjà au format SafeRoads (colonnes natives).
    """
    df = df_raw.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["gravity"]  = pd.to_numeric(df["gravity"], errors="coerce").fillna(1).astype(int)
    df["geo_source"] = "real"
    return df


def clean_accidents(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoyage commun après normalisation.
    """
    # Supprimer les lignes sans datetime ou coordonnées
    df = df.dropna(subset=["datetime", "latitude", "longitude"])

    # Filtrer sur le Sénégal (si coordonnées réelles)
    real_geo = df.get("geo_source", "real") == "real"
    if isinstance(real_geo, pd.Series) and real_geo.any():
        mask_real = df.get("geo_source", pd.Series(["real"]*len(df))) == "real"
        df_real = df[mask_real]
        in_senegal = (
            (df_real["latitude"].between(SENEGAL_LAT_MIN, SENEGAL_LAT_MAX)) &
            (df_real["longitude"].between(SENEGAL_LON_MIN, SENEGAL_LON_MAX))
        )
        df_real = df_real[in_senegal]
        df = pd.concat([df_real, df[~mask_real]], ignore_index=True)

    # Colonnes temporelles
    df["year"]        = df["datetime"].dt.year
    df["month"]       = df["datetime"].dt.month
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["hour"]        = df["datetime"].dt.hour
    df["is_weekend"]  = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_night"]    = ((df["hour"] >= 20) | (df["hour"] < 6)).astype(int)  # 20h-5h

    # Saison (Sénégal : saison sèche nov-mai, hivernage jun-oct)
    df["season"] = df["month"].map(lambda m:
        "hivernage" if m in [6, 7, 8, 9, 10] else "saison_seche"
    )

    # Gravité bornée entre 1 et 3
    df["gravity"] = df["gravity"].clip(1, 3)

    # Remplir les valeurs manquantes
    df["weather"]      = df.get("weather",      pd.Series(["Inconnu"]*len(df))).fillna("Inconnu")
    df["vehicle_type"] = df.get("vehicle_type", pd.Series(["Autre"]*len(df))).fillna("Autre")
    df["cause"]        = df.get("cause",        pd.Series(["Inconnue"]*len(df))).fillna("Inconnue")
    df["num_vehicles"] = pd.to_numeric(df.get("num_vehicles", 1), errors="coerce").fillna(1).astype(int)
    df["num_victims"]  = pd.to_numeric(df.get("num_victims",  1), errors="coerce").fillna(1).astype(int)

    return df.reset_index(drop=True)


def load_accidents() -> pd.DataFrame:
    """
    Point d'entrée principal.
    Charge, détecte le format, normalise, nettoie et sauvegarde.
    """
    if not ACCIDENTS_RAW_PATH.exists():
        raise FileNotFoundError(
            f"Fichier accidents non trouvé : {ACCIDENTS_RAW_PATH}\n"
            f"Placer votre CSV dans : data/raw/accidents/accidents.csv\n"
            f"Voir : data/raw/accidents/README_format.md"
        )

    log.info(f"Chargement accidents depuis {ACCIDENTS_RAW_PATH}")
    df_raw = pd.read_csv(ACCIDENTS_RAW_PATH, low_memory=False)
    log.info(f"  {len(df_raw):,} lignes brutes | {len(df_raw.columns)} colonnes")

    fmt = detect_format(df_raw)
    log.info(f"  Format détecté : {fmt}")

    if fmt == "kaggle":
        df = normalize_kaggle(df_raw)
    elif fmt == "senegal_real":
        df = normalize_senegal_real(df_raw)
    elif fmt == "native":
        df = normalize_native(df_raw)
    else:
        log.warning("Format inconnu -- tentative de normalisation native")
        df = normalize_native(df_raw)

    df = clean_accidents(df)

    log.info(f"  {len(df):,} accidents après nettoyage")
    log.info(f"  Gravité : {df['gravity'].value_counts().to_dict()}")

    # Sauvegarder
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(ACCIDENTS_CLEAN_PATH, index=False)
    log.info(f"  Sauvegardé : {ACCIDENTS_CLEAN_PATH}")

    return df


if __name__ == "__main__":
    df = load_accidents()
    print(f"\n[OK] {len(df):,} accidents charges et normalises")
    print(df[["datetime", "latitude", "longitude", "gravity", "weather", "vehicle_type"]].head(10))
