"""
SafeRoads SN — merge_features.py
Fusionne les trois sources de données en un seul dataset enrichi :
  1. Accidents (CSV normalisé)
  2. Réseau routier OSM (type de route, densité locale)
  3. Météo Open-Meteo (conditions au moment de l'accident)

Produit : data/processed/saferoads_dataset.csv — prêt pour le ML

Usage :
    python -m src.etl.merge_features
"""

import sys
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import DATA_PROCESSED_DIR, WEATHER_RAW_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [MERGE] %(message)s")
log = logging.getLogger(__name__)

FINAL_PATH           = DATA_PROCESSED_DIR / "saferoads_dataset.csv"
ACCIDENTS_CLEAN_PATH = DATA_PROCESSED_DIR / "accidents_clean.csv"
WEATHER_PATH         = WEATHER_RAW_DIR   / "senegal_weather_history.csv"

RAD = np.pi / 180  # Conversion degrés → radians pour BallTree haversine


# ══════════════════════════════════════════════════════
# 1. FEATURE GÉOSPATIALE — densité locale d'accidents
# ══════════════════════════════════════════════════════

def add_spatial_density(df: pd.DataFrame, radius_km: float = 5.0) -> pd.DataFrame:
    """
    Pour chaque accident, calcule le nombre d'accidents dans un rayon
    de radius_km km (feature de densité géospatiale).
    Utilise BallTree haversine pour une distance en cercle exact.
    """
    log.info(f"  Calcul densité spatiale (rayon {radius_km} km)...")

    coords_rad = np.deg2rad(df[["latitude", "longitude"]].values)
    tree = BallTree(coords_rad, metric="haversine")

    radius_rad = radius_km / 6371.0  # Rayon Terre en km
    counts = tree.query_radius(coords_rad, r=radius_rad, count_only=True)

    df["nearby_accidents"]    = counts - 1  # Exclure l'accident lui-même
    df["spatial_density"]     = df["nearby_accidents"] / (np.pi * radius_km**2)

    log.info(f"    Densité moyenne : {df['spatial_density'].mean():.4f} acc/km²")
    return df


# ══════════════════════════════════════════════════════
# 2. FEATURE TEMPORELLE — contexte saisonnier
# ══════════════════════════════════════════════════════

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute des features temporelles dérivées.
    """
    log.info("  Ajout features temporelles...")

    # Période de la journée
    def period_of_day(h):
        if 6  <= h < 10: return "matin"
        if 10 <= h < 14: return "milieu_journee"
        if 14 <= h < 18: return "apres_midi"
        if 18 <= h < 22: return "soiree"
        return "nuit"

    df["period_of_day"] = df["hour"].apply(period_of_day)

    # Événements spéciaux Sénégal (semaines à fort trafic)
    # Gamou (mois 12 approx), Tabaski (variable), départs vacances
    df["is_holiday_period"] = (
        ((df["month"] == 12) & (df["day_of_week"].isin([4, 5, 6]))) |
        ((df["month"] == 4)  & (df["day_of_week"].isin([4, 5, 6])))
    ).astype(int)

    # Sin/cos pour la cyclicité heure et mois (utile pour les modèles ML)
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]  / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]  / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7)

    return df


# ══════════════════════════════════════════════════════
# 3. FUSION MÉTÉO
# ══════════════════════════════════════════════════════

def merge_weather(df_acc: pd.DataFrame, df_weather: pd.DataFrame) -> pd.DataFrame:
    """
    Joint les conditions météo à chaque accident via :
    - Ville la plus proche (haversine)
    - Heure la plus proche (floor à l'heure)
    """
    log.info("  Fusion météo...")

    if df_weather is None or df_weather.empty:
        # Preserver les colonnes meteo deja enrichies (via enrich_data.py)
        defaults = {
            "is_rainy": False, "precipitation_mm": 0.0, "windspeed_kmh": 15.0,
            "temperature_c": 28.0, "weather_label": "Inconnu",
            "visibility_km": 10.0, "humidity_pct": 65.0,
        }
        for col, default in defaults.items():
            if col not in df_acc.columns:
                df_acc[col] = default
            else:
                df_acc[col] = df_acc[col].fillna(default)
        n_with_meteo = (df_acc["precipitation_mm"] != 0.0).sum()
        log.info(f"    Météo enrichie préservée : {n_with_meteo} lignes avec précipitations réelles")
        return df_acc

    # Villes disponibles dans la météo
    cities = df_weather[["city", "latitude", "longitude"]].drop_duplicates("city")

    # Pour chaque accident, trouver la ville météo la plus proche
    city_coords = np.deg2rad(cities[["latitude", "longitude"]].values)
    acc_coords  = np.deg2rad(df_acc[["latitude", "longitude"]].values)
    tree = BallTree(city_coords, metric="haversine")
    _, city_idx = tree.query(acc_coords, k=1)
    df_acc["weather_city"] = cities["city"].values[city_idx.flatten()]

    # Arrondir datetime à l'heure pour le join
    df_acc["datetime_hour"] = pd.to_datetime(df_acc["datetime"]).dt.floor("H")
    df_weather["datetime"]  = pd.to_datetime(df_weather["datetime"])
    df_weather_h = df_weather.rename(columns={"city": "weather_city"})

    # Colonnes météo à récupérer
    weather_cols = [c for c in [
        "weather_city", "datetime",
        "precipitation_mm", "windspeed_kmh", "temperature_c",
        "weather_label", "is_rainy", "visibility_km", "humidity_pct",
    ] if c in df_weather_h.columns]

    df_weather_sub = df_weather_h[weather_cols].copy()
    df_weather_sub = df_weather_sub.rename(columns={"datetime": "datetime_hour"})

    df_merged = df_acc.merge(
        df_weather_sub,
        on=["weather_city", "datetime_hour"],
        how="left"
    )

    # Remplir les NaN météo avec des valeurs typiques Sénégal
    defaults = {
        "is_rainy": False, "precipitation_mm": 0.0, "windspeed_kmh": 15.0,
        "temperature_c": 28.0, "weather_label": "Inconnu",
        "visibility_km": 10.0, "humidity_pct": 65.0,
    }
    for col, default in defaults.items():
        if col not in df_merged.columns:
            df_merged[col] = default
        else:
            df_merged[col] = df_merged[col].fillna(default)

    pct_matched = df_merged["is_rainy"].notna().mean() * 100
    log.info(f"    {pct_matched:.1f}% des accidents ont une météo associée")

    return df_merged.drop(columns=["datetime_hour", "weather_city"], errors="ignore")


# ══════════════════════════════════════════════════════
# 4. ENCODAGE CATÉGORIEL
# ══════════════════════════════════════════════════════

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode les variables catégorielles pour le ML.
    Garde les colonnes originales + ajoute les encodées (_enc).
    Utilise des dictionnaires explicites cohérents avec predict.py.
    """
    log.info("  Encodage catégoriel...")

    # Mappings simples
    mappings = {
        "gravity": {1: 0, 2: 1, 3: 2},  # 0-indexed pour sklearn
        "season":  {"saison_seche": 0, "hivernage": 1},
        "period_of_day": {"nuit": 0, "matin": 1, "milieu_journee": 2,
                          "apres_midi": 3, "soiree": 4},
        "is_rainy": {False: 0, True: 1},
    }

    for col, mapping in mappings.items():
        if col in df.columns:
            df[f"{col}_enc"] = df[col].map(mapping).fillna(0).astype(int)

    # Encodages explicites cohérents avec predict.py / streamlit ml.py
    explicit_encodings = {
        "region": {
            "Dakar": 0, "Thies": 1, "Kaolack": 2, "Saint-Louis": 3,
            "Diourbel": 4, "Ziguinchor": 5, "Tambacounda": 6,
            "Louga": 7, "Kolda": 8, "Matam": 9,
            "Fatick": 10, "Kaffrine": 11, "Kedougou": 12, "Sedhiou": 13,
        },
        "vehicle_type": {
            "Voiture": 0, "Camion": 1, "Moto-Jakarta": 2, "Car rapide": 3,
            "Sept-places": 4, "Taxi": 5, "Pickup": 6, "Charette": 7, "Autre": 8,
        },
        "road_type": {
            "autoroute": 0, "nationale": 1, "régionale": 2,
            "départementale": 3, "urbaine": 4, "piste": 5, "inconnue": 6,
        },
        "cause": {
            "Excès de vitesse": 0, "Somnolence/fatigue": 1, "État dégradé route": 2,
            "Téléphone au volant": 3, "Alcool": 4, "Inconnue": 5,
        },
        "weather_label": {
            "Ensoleillé": 0, "Nuageux": 1, "Pluie légère": 2,
            "Pluie forte": 3, "Brouillard": 4, "Orage": 5, "Inconnu": 6,
        },
    }

    for col, enc in explicit_encodings.items():
        if col in df.columns:
            df[f"{col}_enc"] = df[col].map(enc).fillna(0).astype(int)

    return df


# ══════════════════════════════════════════════════════
# 5. POINT D'ENTRÉE PRINCIPAL
# ══════════════════════════════════════════════════════

def merge_all_features() -> pd.DataFrame:
    """
    Fusionne toutes les sources et produit le dataset final ML.
    """
    log.info("=" * 55)
    log.info("  MERGE — Fusion des features SafeRoads SN")
    log.info("=" * 55)

    # ── Charger accidents ──
    if not ACCIDENTS_CLEAN_PATH.exists():
        raise FileNotFoundError(
            f"Accidents nettoyés non trouvés : {ACCIDENTS_CLEAN_PATH}\n"
            "Lancer d'abord : python scripts/run_etl.py"
        )
    log.info("Chargement accidents propres...")
    df = pd.read_csv(ACCIDENTS_CLEAN_PATH, parse_dates=["datetime"])
    log.info(f"  {len(df):,} accidents")

    # ── Charger météo ──
    df_weather = None
    if WEATHER_PATH.exists():
        log.info("Chargement météo...")
        df_weather = pd.read_csv(WEATHER_PATH, parse_dates=["datetime"])
        log.info(f"  {len(df_weather):,} enregistrements météo")
    else:
        log.warning("Météo non disponible — utiliser : python scripts/setup_data.py --weather-only")

    # ── Features spatiales ──
    df = add_spatial_density(df, radius_km=5.0)

    # ── Features temporelles ──
    df = add_temporal_features(df)

    # ── Fusion météo ──
    df = merge_weather(df, df_weather)

    # ── Encodage ──
    df = encode_categoricals(df)

    # ── Colonnes finales ML ──
    ml_features = [
        # Géospatial
        "latitude", "longitude", "spatial_density", "nearby_accidents",
        # Temporel
        "hour", "day_of_week", "month", "year",
        "hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos",
        "is_weekend", "is_night", "is_holiday_period",
        # Météo
        "is_rainy", "precipitation_mm", "windspeed_kmh",
        "temperature_c", "visibility_km", "humidity_pct",
        # Catégorielles encodées
        "vehicle_type_enc", "road_type_enc", "cause_enc",
        "region_enc", "weather_label_enc", "season_enc", "period_of_day_enc",
        # Cibles
        "gravity", "gravity_enc",
    ]

    # Ne garder que les colonnes disponibles
    available = [c for c in ml_features if c in df.columns]
    df_ml = df[available + [c for c in df.columns if c not in available]].copy()

    # ── Sauvegarder ──
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_ml.to_csv(FINAL_PATH, index=False)

    log.info("=" * 55)
    log.info(f"✅ Dataset final : {FINAL_PATH}")
    log.info(f"   {len(df_ml):,} lignes × {len(df_ml.columns)} colonnes")
    log.info(f"   Features ML disponibles : {len(available)}")
    log.info(f"   Gravité : {df_ml['gravity'].value_counts().sort_index().to_dict()}")
    log.info("=" * 55)

    return df_ml


if __name__ == "__main__":
    df = merge_all_features()
    print(f"\n[OK] Dataset final : {len(df):,} lignes x {len(df.columns)} colonnes")
    print("\nAperçu des features ML :")
    print(df[[c for c in [
        "latitude", "longitude", "hour", "is_rainy",
        "spatial_density", "gravity", "weather_label_enc"
    ] if c in df.columns]].head(5))
