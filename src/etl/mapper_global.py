"""
SafeRoads SN — mapper_global.py
Normalise le dataset Kaggle "Global Road Accidents Dataset" (132k records, 30 features)
vers le format SafeRoads standard, prêt pour le transfer learning.

Dataset source :
    https://www.kaggle.com/datasets/ankushpanday1/global-road-accidents-dataset

Usage :
    python -m src.etl.mapper_global
    python -m src.etl.mapper_global --input data/raw/accidents/accidents.csv --preview
"""

import sys
import logging
import argparse
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import DATA_PROCESSED_DIR, SENEGAL_REGIONS_COORDS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [MAPPER] %(message)s")
log = logging.getLogger(__name__)

OUTPUT_PATH     = DATA_PROCESSED_DIR / "global_mapped.csv"
SENEGAL_REGIONS = SENEGAL_REGIONS_COORDS  # Alias pour compatibilité locale

# ══════════════════════════════════════════════════════
# MAPPINGS — 30 colonnes → format SafeRoads
# ══════════════════════════════════════════════════════

# Gravité
SEVERITY_MAP = {
    # Variants possibles selon la version du dataset
    "Minor":    1, "Slight":   1, "Light":    1, "1": 1,
    "Serious":  2, "Severe":   2, "Major":    2, "2": 2,
    "Fatal":    3, "Critical": 3, "Death":    3, "3": 3,
    # Numériques directs
    1: 1, 2: 2, 3: 3,
}

# Météo
WEATHER_MAP = {
    "Clear":              "Ensoleillé",
    "Sunny":              "Ensoleillé",
    "Fine":               "Ensoleillé",
    "Dry":                "Ensoleillé",
    "Cloudy":             "Nuageux",
    "Overcast":           "Nuageux",
    "Partly Cloudy":      "Nuageux",
    "Rain":               "Pluie légère",
    "Raining":            "Pluie légère",
    "Light Rain":         "Pluie légère",
    "Heavy Rain":         "Pluie forte",
    "Raining and Windy":  "Pluie forte",
    "Flood":              "Pluie forte",
    "Fog":                "Brouillard",
    "Mist":               "Brouillard",
    "Fog or Mist":        "Brouillard",
    "Snow":               "Pluie légère",   # Approx. Sénégal = pas de neige
    "Ice":                "Pluie légère",
    "Windy":              "Nuageux",
    "Sandstorm":          "Brouillard",     # Harmattan au Sénégal
    "Storm":              "Orage",
    "Thunder":            "Orage",
    "Unknown":            "Inconnu",
    "Other":              "Inconnu",
    "NA":                 "Inconnu",
}

# Type de véhicule
VEHICLE_MAP = {
    "Car":                    "Voiture",
    "Automobile":             "Voiture",
    "Sedan":                  "Voiture",
    "SUV":                    "Voiture",
    "Hatchback":              "Voiture",
    "Truck":                  "Camion",
    "Heavy Truck":            "Camion",
    "Lorry":                  "Camion",
    "Bus":                    "Car rapide",
    "Minibus":                "Sept-places",
    "Van":                    "Sept-places",
    "Taxi":                   "Taxi",
    "Motorcycle":             "Moto-Jakarta",
    "Motorbike":              "Moto-Jakarta",
    "Bicycle":                "Moto-Jakarta",
    "Pickup":                 "Pickup",
    "Pickup Truck":           "Pickup",
    "Animal-Drawn Vehicle":   "Charette",
    "Other":                  "Autre",
    "Unknown":                "Autre",
}

# Cause / Facteur contributif
CAUSE_MAP = {
    "Speeding":               "Excès de vitesse",
    "Over Speeding":          "Excès de vitesse",
    "Speed":                  "Excès de vitesse",
    "Fatigue":                "Somnolence/fatigue",
    "Drowsy Driving":         "Somnolence/fatigue",
    "Tired":                  "Somnolence/fatigue",
    "Distracted Driving":     "Téléphone au volant",
    "Phone Use":              "Téléphone au volant",
    "Mobile Phone":           "Téléphone au volant",
    "Drunk Driving":          "Alcool",
    "Alcohol":                "Alcool",
    "DUI":                    "Alcool",
    "Poor Road Condition":    "État dégradé route",
    "Road Defect":            "État dégradé route",
    "Bad Road":               "État dégradé route",
    "Weather":                "Conditions météo",
    "Unknown":                "Inconnue",
    "Other":                  "Inconnue",
}

# Type de route
ROAD_MAP = {
    "Highway":          "autoroute",
    "Motorway":         "autoroute",
    "Expressway":       "autoroute",
    "National Road":    "nationale",
    "Primary Road":     "nationale",
    "Main Road":        "nationale",
    "Regional Road":    "régionale",
    "Secondary Road":   "régionale",
    "Urban Road":       "urbaine",
    "City Street":      "urbaine",
    "Residential":      "urbaine",
    "Rural Road":       "piste",
    "Dirt Road":        "piste",
    "Unknown":          "inconnue",
    "Other":            "inconnue",
}

# Jours de la semaine → entier 0-6
DAY_MAP = {
    "Monday": 0, "Mon": 0,
    "Tuesday": 1, "Tue": 1,
    "Wednesday": 2, "Wed": 2,
    "Thursday": 3, "Thu": 3,
    "Friday": 4, "Fri": 4,
    "Saturday": 5, "Sat": 5,
    "Sunday": 6, "Sun": 6,
}

# Âge conducteur → numérique (centre de la tranche)
AGE_BAND_MAP = {
    "Under 18": 16,
    "18-30":    24,
    "31-50":    40,
    "Over 50":  58,
    "Unknown":  35,
}


# ══════════════════════════════════════════════════════
# DÉTECTION AUTOMATIQUE DES COLONNES
# ══════════════════════════════════════════════════════

def detect_columns(df: pd.DataFrame) -> dict:
    """
    Détecte les noms exacts des colonnes dans le dataset,
    indépendamment de la casse et des espaces.
    """
    cols_lower = {c.lower().strip().replace(" ", "_"): c for c in df.columns}

    candidates = {
        "severity":    ["accident_severity","severity","casualty_severity",
                         "accident_level","injury_severity","fatality_level"],
        "weather":     ["weather_conditions","weather","weather_condition",
                         "climate","meteorological_conditions"],
        "vehicle":     ["type_of_vehicle","vehicle_type","vehicle","vehicle_category"],
        "cause":       ["cause_of_accident","cause","contributing_factor",
                         "accident_cause","primary_cause"],
        "road_type":   ["road_type","type_of_road","road_category","road_class"],
        "hour":        ["time","hour","time_of_day","accident_time","hour_of_day"],
        "day":         ["day_of_week","day","weekday"],
        "month":       ["month","accident_month"],
        "year":        ["year","accident_year"],
        "num_vehicles":["number_of_vehicles_involved","num_vehicles","vehicles_involved"],
        "num_victims": ["number_of_casualties","casualties","victims","num_casualties"],
        "latitude":    ["latitude","lat","gps_lat","accident_lat"],
        "longitude":   ["longitude","lon","lng","gps_lon","accident_lon"],
        "country":     ["country","country_name","nation"],
        "region":      ["region","state","province","area","location"],
        "driver_age":  ["age_band_of_driver","driver_age","age_of_driver"],
        "driver_sex":  ["sex_of_driver","driver_sex","gender"],
        "experience":  ["driving_experience","experience","years_driving"],
        "road_surface":["road_surface_conditions","road_surface","pavement"],
        "light":       ["light_conditions","lighting","light_condition"],
        "junction":    ["types_of_junction","junction_type","junction"],
    }

    found = {}
    for key, names in candidates.items():
        for name in names:
            if name in cols_lower:
                found[key] = cols_lower[name]
                break

    log.info(f"Colonnes détectées : {list(found.keys())}")
    missing = [k for k in ["severity","weather","vehicle"] if k not in found]
    if missing:
        log.warning(f"Colonnes importantes non trouvées : {missing}")

    return found


# ══════════════════════════════════════════════════════
# NORMALISATION PRINCIPALE
# ══════════════════════════════════════════════════════

def map_global_to_saferoads(df_raw: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Transforme le dataset global Kaggle (132k, 30 cols) vers
    le format SafeRoads standard avec toutes les features ML.
    """
    log.info(f"Dataset source : {len(df_raw):,} lignes × {len(df_raw.columns)} colonnes")
    log.info(f"Colonnes : {list(df_raw.columns)}")

    cols = detect_columns(df_raw)
    df   = pd.DataFrame()
    rng  = np.random.default_rng(seed)

    # ── 1. GRAVITÉ ──
    if "severity" in cols:
        raw = df_raw[cols["severity"]].astype(str).str.strip()
        df["gravity"] = raw.map(SEVERITY_MAP)
        # Essayer une conversion numérique si le mapping échoue
        unmapped = df["gravity"].isna()
        if unmapped.any():
            df.loc[unmapped, "gravity"] = pd.to_numeric(
                raw[unmapped], errors="coerce"
            ).clip(1, 3)
        df["gravity"] = df["gravity"].fillna(1).astype(int).clip(1, 3)
    else:
        log.warning("Colonne gravité non trouvée → valeur par défaut 1")
        df["gravity"] = 1

    log.info(f"  Gravité : {df['gravity'].value_counts().sort_index().to_dict()}")

    # ── 2. MÉTÉO ──
    if "weather" in cols:
        df["weather"] = df_raw[cols["weather"]].astype(str).str.strip().map(WEATHER_MAP).fillna("Inconnu")
    else:
        df["weather"] = "Inconnu"

    df["is_rainy"] = df["weather"].isin(["Pluie légère", "Pluie forte", "Orage", "Brouillard"])

    # ── 3. TYPE DE VÉHICULE ──
    if "vehicle" in cols:
        df["vehicle_type"] = df_raw[cols["vehicle"]].astype(str).str.strip().map(VEHICLE_MAP).fillna("Autre")
    else:
        df["vehicle_type"] = "Autre"

    # ── 4. CAUSE ──
    if "cause" in cols:
        df["cause"] = df_raw[cols["cause"]].astype(str).str.strip().map(CAUSE_MAP).fillna("Inconnue")
    else:
        df["cause"] = "Inconnue"

    # ── 5. TYPE DE ROUTE ──
    if "road_type" in cols:
        df["road_type"] = df_raw[cols["road_type"]].astype(str).str.strip().map(ROAD_MAP).fillna("inconnue")
    else:
        df["road_type"] = "inconnue"

    # ── 6. TEMPOREL ──
    # Heure
    if "hour" in cols:
        raw_time = df_raw[cols["hour"]].astype(str)
        # Essayer format HH:MM:SS d'abord
        parsed = pd.to_datetime(raw_time, format="%H:%M:%S", errors="coerce").dt.hour.copy()
        # Sinon format HH:MM
        mask = parsed.isna()
        if mask.any():
            parsed.loc[mask] = pd.to_datetime(raw_time[mask], format="%H:%M", errors="coerce").dt.hour
        # Sinon numérique direct
        mask2 = parsed.isna()
        if mask2.any():
            parsed.loc[mask2] = pd.to_numeric(raw_time[mask2], errors="coerce")
        # Remplir les NaN restants avec des heures aléatoires
        still_na = parsed.isna()
        if still_na.any():
            parsed.loc[still_na] = rng.integers(6, 22, size=still_na.sum())
        df["hour"] = parsed.astype(int).clip(0, 23)
    else:
        df["hour"] = rng.integers(6, 22, size=len(df_raw))

    # Jour de la semaine
    if "day" in cols:
        raw_day = df_raw[cols["day"]].astype(str).str.strip()
        df["day_of_week"] = raw_day.map(DAY_MAP)
        unmapped = df["day_of_week"].isna()
        if unmapped.any():
            df.loc[unmapped, "day_of_week"] = pd.to_numeric(raw_day[unmapped], errors="coerce")
        still_na = df["day_of_week"].isna()
        if still_na.any():
            df.loc[still_na, "day_of_week"] = rng.integers(0, 7, size=still_na.sum())
        df["day_of_week"] = df["day_of_week"].astype(int).clip(0, 6)
    else:
        df["day_of_week"] = rng.integers(0, 7, size=len(df_raw))

    # Mois
    if "month" in cols:
        df["month"] = pd.to_numeric(df_raw[cols["month"]], errors="coerce").fillna(rng.integers(1, 13)).astype(int).clip(1, 12)
    else:
        df["month"] = rng.integers(1, 13, size=len(df_raw))

    # Année
    if "year" in cols:
        df["year"] = pd.to_numeric(df_raw[cols["year"]], errors="coerce").fillna(2022).astype(int)
    else:
        df["year"] = 2022

    # Datetime reconstruit
    df["datetime"] = pd.to_datetime({
        "year": df["year"], "month": df["month"],
        "day": 1, "hour": df["hour"],
    }, errors="coerce")

    # ── 7. NOMBRE DE VÉHICULES / VICTIMES ──
    for key, col in [("num_vehicles", "num_vehicles"), ("num_victims", "num_victims")]:
        if key in cols:
            df[col] = pd.to_numeric(df_raw[cols[key]], errors="coerce").fillna(1).astype(int).clip(1, 20)
        else:
            df[col] = 1

    # ── 8. CARACTÉRISTIQUES CONDUCTEUR (bonus pour le modèle) ──
    if "driver_age" in cols:
        raw_age = df_raw[cols["driver_age"]].astype(str).str.strip()
        df["driver_age"] = raw_age.map(AGE_BAND_MAP)
        unmapped = df["driver_age"].isna()
        df.loc[unmapped, "driver_age"] = pd.to_numeric(raw_age[unmapped], errors="coerce")
        df["driver_age"] = df["driver_age"].fillna(35).astype(float)
    else:
        df["driver_age"] = 35.0

    if "driver_sex" in cols:
        df["driver_sex"] = df_raw[cols["driver_sex"]].astype(str).str.strip().str.lower()
        df["driver_sex_enc"] = df["driver_sex"].map({"male": 0, "female": 1, "m": 0, "f": 1}).fillna(0).astype(int)
    else:
        df["driver_sex_enc"] = 0

    if "experience" in cols:
        raw_exp = df_raw[cols["experience"]].astype(str).str.strip()
        exp_map = {"Below 1yr": 0.5, "1-2yr": 1.5, "2-5yr": 3.5,
                   "5-10yr": 7.5, "Above 10yr": 12, "Unknown": 5}
        df["driving_experience"] = raw_exp.map(exp_map)
        unmapped = df["driving_experience"].isna()
        df.loc[unmapped, "driving_experience"] = pd.to_numeric(raw_exp[unmapped], errors="coerce")
        df["driving_experience"] = df["driving_experience"].fillna(5).astype(float)
    else:
        df["driving_experience"] = 5.0

    # ── 9. SURFACE ET LUMIÈRE (features additionnelles) ──
    if "road_surface" in cols:
        surface_map = {
            "Dry": 0, "Wet": 1, "Ice": 2, "Snow": 2, "Gravel": 1,
            "Muddy": 2, "Sand": 1, "Unknown": 0,
        }
        df["road_surface_enc"] = df_raw[cols["road_surface"]].astype(str).str.strip().map(surface_map).fillna(0).astype(int)
    else:
        df["road_surface_enc"] = 0

    if "light" in cols:
        light_map = {
            "Daylight": 0, "Day": 0, "Day time": 0,
            "Darkness": 1, "Night": 1, "Dark": 1,
            "Dusk/Dawn": 2, "Dusk": 2, "Dawn": 2,
            "Unknown": 0,
        }
        df["light_enc"] = df_raw[cols["light"]].astype(str).str.strip().map(light_map).fillna(0).astype(int)
    else:
        df["light_enc"] = (df["hour"] >= 20) | (df["hour"] < 6)
        df["light_enc"] = df["light_enc"].astype(int)

    # ── 10. COORDONNÉES GPS ──
    # Si le dataset contient des coordonnées réelles → les utiliser
    if "latitude" in cols and "longitude" in cols:
        lat_raw = pd.to_numeric(df_raw[cols["latitude"]], errors="coerce")
        lon_raw = pd.to_numeric(df_raw[cols["longitude"]], errors="coerce")

        # Vérifier si les coordonnées sont valides et dans le monde
        valid_geo = (
            lat_raw.between(-90, 90) & lon_raw.between(-180, 180) &
            lat_raw.notna() & lon_raw.notna()
        )
        pct_valid = valid_geo.mean() * 100
        log.info(f"  Coordonnées GPS valides : {pct_valid:.1f}%")

        if pct_valid > 10:
            df["latitude"]   = lat_raw
            df["longitude"]  = lon_raw
            df["geo_source"] = "real"
            # Remplacer les invalides par des coordonnées simulées Sénégal
            invalid = ~valid_geo
            if invalid.any():
                n_inv = invalid.sum()
                weights = [r["weight"] for r in SENEGAL_REGIONS]
                chosen  = rng.choice(len(SENEGAL_REGIONS), size=n_inv, p=weights)
                df.loc[invalid, "latitude"]  = [SENEGAL_REGIONS[i]["lat"] + rng.uniform(-0.3,0.3) for i in chosen]
                df.loc[invalid, "longitude"] = [SENEGAL_REGIONS[i]["lon"] + rng.uniform(-0.3,0.3) for i in chosen]
                df.loc[invalid, "geo_source"] = "simulated"
        else:
            log.warning("  Coordonnées GPS insuffisantes → simulation Sénégal")
            _assign_senegal_coords(df, rng)
    else:
        log.info("  Pas de colonnes GPS → simulation coordonnées sénégalaises")
        _assign_senegal_coords(df, rng)

    # ── 11. RÉGION ──
    if "region" in cols and "geo_source" in df.columns and (df["geo_source"] == "real").any():
        df["region"] = df_raw[cols["region"]].astype(str).fillna("Inconnue")
    # Sinon région déjà assignée par _assign_senegal_coords

    if "region" not in df.columns:
        df["region"] = "Inconnue"

    # ── 12. PAYS SOURCE (utile pour le transfer learning) ──
    if "country" in cols:
        df["source_country"] = df_raw[cols["country"]].astype(str).str.strip()
    else:
        df["source_country"] = "Global"

    df["source"] = "kaggle_global"

    # ── 13. FEATURES DÉRIVÉES ──
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    df["is_night"]    = ((df["hour"] >= 20) | (df["hour"] < 6)).astype(int)
    df["season"]      = df["month"].map(lambda m: "hivernage" if m in [6,7,8,9,10] else "saison_seche")

    # ── RAPPORT ──
    log.info("=" * 50)
    log.info(f"✅ Mapping terminé : {len(df):,} lignes × {len(df.columns)} colonnes")
    log.info(f"   Gravité    : {df['gravity'].value_counts().sort_index().to_dict()}")
    log.info(f"   Météo      : {df['weather'].value_counts().head(4).to_dict()}")
    log.info(f"   Véhicules  : {df['vehicle_type'].value_counts().head(4).to_dict()}")
    log.info(f"   Geo source : {df['geo_source'].value_counts().to_dict()}")
    log.info("=" * 50)

    return df


def _assign_senegal_coords(df: pd.DataFrame, rng: np.random.Generator):
    """Assigne des coordonnées GPS sénégalaises pondérées par région."""
    weights = [r["weight"] for r in SENEGAL_REGIONS]
    chosen  = rng.choice(len(SENEGAL_REGIONS), size=len(df), p=weights)
    df["latitude"]  = [SENEGAL_REGIONS[i]["lat"] + rng.uniform(-0.3, 0.3) for i in chosen]
    df["longitude"] = [SENEGAL_REGIONS[i]["lon"] + rng.uniform(-0.3, 0.3) for i in chosen]
    df["region"]    = [SENEGAL_REGIONS[i]["region"] for i in chosen]
    df["geo_source"] = "simulated"


def run_mapping(input_path: Path = None, save: bool = True) -> pd.DataFrame:
    """Point d'entrée principal."""
    from src.utils.config import ACCIDENTS_RAW_PATH
    path = input_path or ACCIDENTS_RAW_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Fichier non trouvé : {path}\n"
            "Télécharger depuis : https://kaggle.com/datasets/ankushpanday1/global-road-accidents-dataset"
        )

    log.info(f"Chargement : {path}")
    df_raw = pd.read_csv(path, low_memory=False)
    df     = map_global_to_saferoads(df_raw)

    if save:
        DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUT_PATH, index=False)
        log.info(f"Sauvegardé : {OUTPUT_PATH}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SafeRoads SN — Mapper dataset global")
    parser.add_argument("--input",   type=str, help="Chemin vers le CSV source")
    parser.add_argument("--preview", action="store_true", help="Afficher un aperçu sans sauvegarder")
    args = parser.parse_args()

    input_path = Path(args.input) if args.input else None
    df = run_mapping(input_path, save=not args.preview)

    print(f"\n[OK] {len(df):,} enregistrements mappes")
    print(f"\nAperçu :")
    print(df[["gravity","weather","vehicle_type","cause","road_type",
              "hour","region","geo_source","source_country"]].head(8).to_string())
    print(f"\nColonnes disponibles ({len(df.columns)}) :")
    print(list(df.columns))
