"""
SafeRoads SN — Helpers ML pour Streamlit
Charge les modèles .pkl et expose une fonction predict() simple.
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import streamlit as st

log = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent.parent / "models"

GRAVITY_LABELS = {0: "léger", 1: "grave", 2: "mortel"}
GRAVITY_COLORS = {0: "#28a745", 1: "#fd7e14", 2: "#dc3545"}
RISK_COLORS    = {"faible": "#28a745", "moyen": "#ffc107", "élevé": "#fd7e14", "critique": "#dc3545"}

ENCODINGS = {
    "region":       {"Dakar":0,"Thies":1,"Kaolack":2,"Saint-Louis":3,
                     "Diourbel":4,"Ziguinchor":5,"Tambacounda":6,"Louga":7,"Kolda":8,
                     "Matam":9,"Fatick":10,"Kaffrine":11,"Kedougou":12,"Sedhiou":13},
    "vehicle_type": {"Voiture":0,"Camion":1,"Moto-Jakarta":2,"Car rapide":3,
                     "Sept-places":4,"Taxi":5,"Pickup":6,"Charette":7,"Autre":8},
    "road_type":    {"autoroute":0,"nationale":1,"régionale":2,"départementale":3,
                     "urbaine":4,"piste":5,"inconnue":6},
    "cause":        {"Excès de vitesse":0,"Somnolence/fatigue":1,"État dégradé route":2,
                     "Téléphone au volant":3,"Alcool":4,"Inconnue":5},
    "weather":      {"Ensoleillé":0,"Nuageux":1,"Pluie légère":2,
                     "Pluie forte":3,"Brouillard":4,"Orage":5,"Inconnu":6},
}


@st.cache_resource(show_spinner="Chargement des modèles ML...")
def load_models() -> dict:
    """Charge les modèles une seule fois (cache Streamlit)."""
    models = {}
    for name in ["gravity_model", "risk_model", "dbscan_model"]:
        path = MODELS_DIR / f"{name}.pkl"
        if path.exists():
            models[name] = joblib.load(path)
            log.info(f"Modèle chargé : {name}")
        else:
            log.warning(f"Modèle absent : {path}")
    return models


def build_features(params: dict) -> pd.DataFrame:
    """Construit le vecteur de features depuis les paramètres du formulaire."""
    h   = params["hour"]
    m   = params["month"]
    dow = params["day_of_week"]

    period = ("nuit" if h < 6 or h >= 22 else
              "matin" if h < 10 else
              "milieu_journee" if h < 14 else
              "apres_midi" if h < 18 else "soiree")

    season = "hivernage" if m in [6,7,8,9,10] else "saison_seche"

    period_enc = {"nuit":0,"matin":1,"milieu_journee":2,"apres_midi":3,"soiree":4}
    season_enc = {"saison_seche":0,"hivernage":1}

    row = {
        "latitude":          params["latitude"],
        "longitude":         params["longitude"],
        "spatial_density":   params.get("spatial_density", 0.1),
        "nearby_accidents":  params.get("nearby_accidents", 5),
        "hour":              h,
        "day_of_week":       dow,
        "month":             m,
        "hour_sin":          np.sin(2 * np.pi * h / 24),
        "hour_cos":          np.cos(2 * np.pi * h / 24),
        "month_sin":         np.sin(2 * np.pi * m / 12),
        "month_cos":         np.cos(2 * np.pi * m / 12),
        "dow_sin":           np.sin(2 * np.pi * dow / 7),
        "dow_cos":           np.cos(2 * np.pi * dow / 7),
        "is_weekend":        int(dow >= 5),
        "is_night":          int(h >= 20 or h < 6),
        "is_holiday_period": int((m == 12 and dow >= 4) or (m == 4 and dow >= 4)),
        "is_rainy":          int(params.get("is_rainy", False)),
        "precipitation_mm":  params.get("precipitation_mm", 0.0),
        "windspeed_kmh":     params.get("windspeed_kmh", 15.0),
        "temperature_c":     params.get("temperature_c", 28.0),
        "visibility_km":     params.get("visibility_km", 10.0),
        "humidity_pct":      params.get("humidity_pct", 65.0),
        "vehicle_type_enc":  ENCODINGS["vehicle_type"].get(params.get("vehicle_type","Voiture"), 8),
        "road_type_enc":     ENCODINGS["road_type"].get(params.get("road_type","nationale"), 6),
        "cause_enc":         ENCODINGS["cause"].get(params.get("cause","Inconnue"), 5),
        "region_enc":        ENCODINGS["region"].get(params.get("region","Dakar"), 0),
        "weather_label_enc": ENCODINGS["weather"].get(params.get("weather","Ensoleillé"), 6),
        "season_enc":        season_enc[season],
        "period_of_day_enc": period_enc[period],
    }
    return pd.DataFrame([row])


def predict(params: dict) -> dict:
    """
    Lance la prédiction à partir des paramètres du formulaire.
    Retourne un dict avec gravity, risk_score, risk_level, recommendations.
    """
    models = load_models()
    X      = build_features(params)

    # ── Gravité ──
    gravity_result = {"gravity_class": 0, "gravity_label": "léger",
                      "probabilities": {"léger": 1.0, "grave": 0.0, "mortel": 0.0}}

    if "gravity_model" in models:
        m        = models["gravity_model"]
        pipeline = m["pipeline"]
        feats    = m["feature_names"]
        X_a      = X.reindex(columns=feats, fill_value=0)
        proba    = pipeline.predict_proba(X_a)[0]
        cls      = int(np.argmax(proba))
        gravity_result = {
            "gravity_class":  cls,
            "gravity_label":  GRAVITY_LABELS[cls],
            "gravity_color":  GRAVITY_COLORS[cls],
            "probabilities":  {
                "léger":  round(float(proba[0]), 3),
                "grave":  round(float(proba[1]), 3),
                "mortel": round(float(proba[2]), 3),
            },
        }

    # ── Score de risque ──
    risk_result = {"risk_score": 35.0, "risk_level": "faible"}

    if "risk_model" in models:
        m        = models["risk_model"]
        pipeline = m["pipeline"]
        feats    = m["feature_names"]
        X_a      = X.reindex(columns=feats, fill_value=0)
        score    = float(np.clip(pipeline.predict(X_a)[0], 0, 100))
        level    = ("critique" if score >= 80 else
                    "élevé"    if score >= 60 else
                    "moyen"    if score >= 40 else "faible")
        risk_result = {
            "risk_score": round(score, 1),
            "risk_level": level,
            "risk_color": RISK_COLORS[level],
        }

    # ── Recommandations ──
    recs = []
    level = risk_result["risk_level"]
    if level in ("critique", "élevé"):
        recs.append("⚠️ Zone à risque élevé — déploiement police recommandé")
    if params.get("is_rainy"):
        recs.append("🌧️ Chaussée mouillée — réduire la vitesse")
    h = params.get("hour", 12)
    if h >= 20 or h < 6:
        recs.append("🌙 Circulation nocturne — vigilance accrue")
    if gravity_result["gravity_label"] in ("grave", "mortel"):
        recs.append("🚑 Risque de victimes graves — pré-alerte SAMU conseillée")
    if params.get("vehicle_type") == "Moto-Jakarta":
        recs.append("🏍️ Moto-Jakarta : port casque obligatoire")
    if not recs:
        recs.append("✅ Conditions normales")

    return {**gravity_result, **risk_result, "recommendations": recs}


def get_hotspots_from_models() -> list:
    """Charge les hotspots depuis le fichier pkl ou csv."""
    models = load_models()
    if "dbscan_model" in models:
        return models["dbscan_model"].get("hotspots", [])

    # Fallback CSV
    csv_path = Path(__file__).parent.parent.parent / "data/processed/hotspots.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path).to_dict("records")

    return []
