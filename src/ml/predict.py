"""
SafeRoads SN — predict.py
Chargement des modèles entraînés et inférence sur de nouvelles observations.
Utilisé par l'API FastAPI.

Usage :
    python -m src.ml.predict --lat 14.69 --lon -17.44 --hour 18 --rainy
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import MODELS_DIR, DATA_PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [PREDICT] %(message)s")
log = logging.getLogger(__name__)

# Catégories disponibles
GRAVITY_LABELS   = {0: "léger", 1: "grave", 2: "mortel"}
RISK_LEVELS      = [(80, "critique"), (60, "élevé"), (40, "moyen"), (0, "faible")]

# Encodages par défaut (reconstruits depuis les valeurs les plus courantes)
REGION_ENC = {
    "Dakar": 0, "Thies": 1, "Kaolack": 2, "Saint-Louis": 3,
    "Diourbel": 4, "Ziguinchor": 5, "Tambacounda": 6,
    "Louga": 7, "Kolda": 8, "Matam": 9,
    "Fatick": 10, "Kaffrine": 11, "Kedougou": 12, "Sedhiou": 13,
}
VEHICLE_ENC = {
    "Voiture": 0, "Camion": 1, "Moto-Jakarta": 2, "Car rapide": 3,
    "Sept-places": 4, "Taxi": 5, "Pickup": 6, "Charette": 7, "Autre": 8,
}
ROAD_ENC = {
    "autoroute": 0, "nationale": 1, "régionale": 2,
    "départementale": 3, "urbaine": 4, "piste": 5, "inconnue": 6,
}
CAUSE_ENC = {
    "Excès de vitesse": 0, "Somnolence/fatigue": 1, "État dégradé route": 2,
    "Téléphone au volant": 3, "Alcool": 4, "Inconnue": 5,
}
WEATHER_ENC = {
    "Ensoleillé": 0, "Nuageux": 1, "Pluie légère": 2,
    "Pluie forte": 3, "Brouillard": 4, "Orage": 5, "Inconnu": 6,
}
SEASON_ENC = {"saison_seche": 0, "hivernage": 1}
PERIOD_ENC = {"nuit": 0, "matin": 1, "milieu_journee": 2, "apres_midi": 3, "soiree": 4}


def _load_model(name: str) -> dict:
    path = MODELS_DIR / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Modèle non trouvé : {path}\n"
            "Lancer : python src/ml/train.py"
        )
    return joblib.load(path)


def _period_of_day(hour: int) -> str:
    if 6  <= hour < 10: return "matin"
    if 10 <= hour < 14: return "milieu_journee"
    if 14 <= hour < 18: return "apres_midi"
    if 18 <= hour < 22: return "soiree"
    return "nuit"


def _build_feature_vector(
    latitude: float,
    longitude: float,
    hour: int,
    day_of_week: int,
    month: int,
    is_rainy: bool,
    precipitation_mm: float,
    windspeed_kmh: float,
    temperature_c: float,
    visibility_km: float,
    humidity_pct: float,
    vehicle_type: str,
    road_type: str,
    region: str,
    cause: str,
    weather_label: str,
    nearby_accidents: int,
    spatial_density: float,
) -> pd.DataFrame:
    """
    Construit le vecteur de features à partir des paramètres bruts.
    """
    period = _period_of_day(hour)
    season = "hivernage" if month in [6, 7, 8, 9, 10] else "saison_seche"
    is_weekend = int(day_of_week >= 5)
    is_night   = int(hour >= 20 or hour < 6)
    is_holiday = int((month == 12 and day_of_week >= 4) or (month == 4 and day_of_week >= 4))

    row = {
        "latitude":           latitude,
        "longitude":          longitude,
        "spatial_density":    spatial_density,
        "nearby_accidents":   nearby_accidents,
        "hour":               hour,
        "day_of_week":        day_of_week,
        "month":              month,
        "hour_sin":           np.sin(2 * np.pi * hour / 24),
        "hour_cos":           np.cos(2 * np.pi * hour / 24),
        "month_sin":          np.sin(2 * np.pi * month / 12),
        "month_cos":          np.cos(2 * np.pi * month / 12),
        "dow_sin":            np.sin(2 * np.pi * day_of_week / 7),
        "dow_cos":            np.cos(2 * np.pi * day_of_week / 7),
        "is_weekend":         is_weekend,
        "is_night":           is_night,
        "is_holiday_period":  is_holiday,
        "is_rainy":           int(is_rainy),
        "precipitation_mm":   precipitation_mm,
        "windspeed_kmh":      windspeed_kmh,
        "temperature_c":      temperature_c,
        "visibility_km":      visibility_km,
        "humidity_pct":       humidity_pct,
        "vehicle_type_enc":   VEHICLE_ENC.get(vehicle_type, 8),
        "road_type_enc":      ROAD_ENC.get(road_type, 6),
        "cause_enc":          CAUSE_ENC.get(cause, 5),
        "region_enc":         REGION_ENC.get(region, 0),
        "weather_label_enc":  WEATHER_ENC.get(weather_label, 6),
        "season_enc":         SEASON_ENC.get(season, 0),
        "period_of_day_enc":  PERIOD_ENC.get(period, 0),
    }
    return pd.DataFrame([row])


def predict(
    latitude: float,
    longitude: float,
    hour: int = None,
    day_of_week: int = None,
    month: int = None,
    is_rainy: bool = False,
    precipitation_mm: float = 0.0,
    windspeed_kmh: float = 15.0,
    temperature_c: float = 28.0,
    visibility_km: float = 10.0,
    humidity_pct: float = 65.0,
    vehicle_type: str = "Voiture",
    road_type: str = "nationale",
    region: str = "Dakar",
    cause: str = "Inconnue",
    weather_label: str = "Ensoleillé",
    nearby_accidents: int = 5,
    spatial_density: float = 0.1,
) -> dict:
    """
    Retourne la prédiction complète pour une observation.
    """
    now = datetime.now()
    hour        = hour        if hour        is not None else now.hour
    day_of_week = day_of_week if day_of_week is not None else now.weekday()
    month       = month       if month       is not None else now.month

    X = _build_feature_vector(
        latitude, longitude, hour, day_of_week, month,
        is_rainy, precipitation_mm, windspeed_kmh,
        temperature_c, visibility_km, humidity_pct,
        vehicle_type, road_type, region, cause, weather_label,
        nearby_accidents, spatial_density,
    )

    # ── Modèle 1 : Gravité ──
    gravity_result = {}
    try:
        m1 = _load_model("gravity_model")
        pipeline = m1["pipeline"]
        feat     = m1["feature_names"]

        # Aligner les colonnes sur celles attendues par le modèle
        X_aligned = X.reindex(columns=feat, fill_value=0)
        proba     = pipeline.predict_proba(X_aligned)[0]
        pred_cls  = int(np.argmax(proba))

        gravity_result = {
            "gravity_class":  pred_cls,
            "gravity_label":  GRAVITY_LABELS[pred_cls],
            "probabilities": {
                "léger":  round(float(proba[0]), 3),
                "grave":  round(float(proba[1]), 3),
                "mortel": round(float(proba[2]), 3),
            },
        }
    except FileNotFoundError:
        gravity_result = {"gravity_label": "modèle_non_entraîné"}

    # ── Modèle 2 : Score de risque ──
    risk_result = {}
    try:
        m2       = _load_model("risk_model")
        pipeline = m2["pipeline"]
        feat     = m2["feature_names"]
        X_aligned = X.reindex(columns=feat, fill_value=0)
        score    = float(pipeline.predict(X_aligned)[0])
        score    = round(np.clip(score, 0, 100), 1)

        risk_level = "faible"
        for threshold, level in RISK_LEVELS:
            if score >= threshold:
                risk_level = level
                break

        risk_result = {
            "risk_score": score,
            "risk_level": risk_level,
        }
    except FileNotFoundError:
        risk_result = {"risk_score": 0, "risk_level": "modèle_non_entraîné"}

    # ── Recommandations ──
    recommendations = _build_recommendations(
        risk_result.get("risk_level", "faible"),
        gravity_result.get("gravity_label", "léger"),
        is_rainy, hour,
    )

    return {
        "input": {
            "latitude": latitude,
            "longitude": longitude,
            "hour": hour,
            "region": region,
            "is_rainy": is_rainy,
            "vehicle_type": vehicle_type,
        },
        **gravity_result,
        **risk_result,
        "recommendations": recommendations,
    }


def _build_recommendations(risk_level: str, gravity: str, is_rainy: bool, hour: int) -> list:
    recs = []
    if risk_level in ("critique", "élevé"):
        recs.append("⚠️ Zone à risque élevé — déploiement police recommandé")
        recs.append("🚧 Vérifier l'état de la chaussée sur ce segment")
    if is_rainy:
        recs.append("🌧️ Pluie détectée — signalisation humidité conseillée")
    if hour >= 20 or hour < 6:
        recs.append("🌙 Circulation nocturne — renforcer l'éclairage")
    if gravity in ("grave", "mortel"):
        recs.append("🚑 Probabilité de victime grave — pré-alerte SAMU")
    if not recs:
        recs.append("✅ Conditions normales")
    return recs


def get_hotspots() -> list:
    """Retourne les hotspots détectés par DBSCAN."""
    path = DATA_PROCESSED_DIR / "hotspots.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    return df.to_dict("records")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lat",   type=float, default=14.6937)
    parser.add_argument("--lon",   type=float, default=-17.4441)
    parser.add_argument("--hour",  type=int,   default=18)
    parser.add_argument("--rainy", action="store_true")
    parser.add_argument("--region", type=str,  default="Dakar")
    args = parser.parse_args()

    result = predict(
        latitude=args.lat,
        longitude=args.lon,
        hour=args.hour,
        is_rainy=args.rainy,
        region=args.region,
    )
    import json
    print(json.dumps(result, indent=2, ensure_ascii=False))
