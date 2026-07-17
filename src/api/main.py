"""
SafeRoads SN — API FastAPI
Expose les modèles ML via une API REST documentée.

Endpoints :
  GET  /                    → Info API
  GET  /health              → Santé du système
  POST /predict             → Prédiction gravité + score de risque
  GET  /hotspots            → Liste des zones dangereuses (DBSCAN)
  GET  /stats               → Statistiques globales du dataset
  GET  /segments            → Segments routiers à risque élevé
  POST /batch               → Prédictions en lot (max 100)
  GET  /model/info          → Métriques des modèles entraînés

Usage :
    uvicorn src.api.main:app --reload --port 8000
    Documentation : http://localhost:8000/docs
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import MODELS_DIR, DATA_PROCESSED_DIR, API_HOST, API_PORT
from src.ml.predict import predict, get_hotspots

logging.basicConfig(level=logging.INFO, format="%(asctime)s [API] %(message)s")
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════
# INITIALISATION
# ══════════════════════════════════════════════════════

app = FastAPI(
    title="SafeRoads SN API",
    description=(
        "Système intelligent de prédiction des accidents routiers au Sénégal.\n\n"
        "**Gindima Group** · Dakar, Sénégal · contact@gindima-groupe.com\n\n"
        "Sources : OpenStreetMap · Open-Meteo · ANSD"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — autoriser le dashboard frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache en mémoire (évite de relire les CSV à chaque requête)
_cache: dict = {}


def _load_dataset() -> pd.DataFrame:
    if "dataset" not in _cache:
        path = DATA_PROCESSED_DIR / "saferoads_dataset.csv"
        if not path.exists():
            return pd.DataFrame()
        _cache["dataset"] = pd.read_csv(path)
    return _cache["dataset"]


def _load_model_info() -> dict:
    if "model_info" not in _cache:
        info = {}
        for name in ["gravity_model", "risk_model", "dbscan_model"]:
            path = MODELS_DIR / f"{name}_metrics.json"
            if path.exists():
                with open(path) as f:
                    info[name] = json.load(f)
        _cache["model_info"] = info
    return _cache["model_info"]


# ══════════════════════════════════════════════════════
# SCHÉMAS PYDANTIC
# ══════════════════════════════════════════════════════

class PredictRequest(BaseModel):
    latitude:         float = Field(..., ge=12.3, le=16.7,   description="Latitude (Sénégal : 12.3–16.7)")
    longitude:        float = Field(..., ge=-17.6, le=-11.3, description="Longitude (Sénégal : -17.6–-11.3)")
    hour:             Optional[int]   = Field(None, ge=0, le=23,  description="Heure (0-23). Défaut : heure actuelle")
    day_of_week:      Optional[int]   = Field(None, ge=0, le=6,   description="Jour (0=Lundi, 6=Dimanche)")
    month:            Optional[int]   = Field(None, ge=1, le=12,  description="Mois (1-12)")
    is_rainy:         bool            = Field(False,              description="Pluie en cours")
    precipitation_mm: float           = Field(0.0,  ge=0,         description="Précipitations en mm")
    windspeed_kmh:    float           = Field(15.0, ge=0, le=150, description="Vitesse du vent en km/h")
    temperature_c:    float           = Field(28.0, ge=-5, le=55, description="Température en °C")
    visibility_km:    float           = Field(10.0, ge=0, le=50,  description="Visibilité en km")
    humidity_pct:     float           = Field(65.0, ge=0, le=100, description="Humidité relative %")
    vehicle_type:     str             = Field("Voiture",          description="Type de véhicule")
    road_type:        str             = Field("nationale",        description="Type de route")
    region:           str             = Field("Dakar",            description="Région du Sénégal")
    cause:            str             = Field("Inconnue",         description="Cause présumée")
    weather_label:    str             = Field("Ensoleillé",      description="Condition météo (Ensoleillé, Nuageux, Pluie légère, Pluie forte, Brouillard, Orage, Inconnu)")
    nearby_accidents: int             = Field(5,  ge=0,           description="Accidents recensés dans la zone (5km)")
    spatial_density:  float           = Field(0.1, ge=0,          description="Densité spatiale accidents/km²")

    class Config:
        json_schema_extra = {
            "example": {
                "latitude": 14.6937,
                "longitude": -17.4441,
                "hour": 18,
                "is_rainy": False,
                "vehicle_type": "Car rapide",
                "road_type": "nationale",
                "region": "Dakar",
                "nearby_accidents": 12,
                "spatial_density": 0.38,
            }
        }


class BatchPredictRequest(BaseModel):
    observations: List[PredictRequest] = Field(..., max_items=100)


class PredictResponse(BaseModel):
    gravity_class:   Optional[int]
    gravity_label:   str
    probabilities:   Optional[dict]
    risk_score:      float
    risk_level:      str
    recommendations: List[str]
    input:           dict
    processed_at:    str


# ══════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════

@app.get("/", tags=["Général"])
def root():
    """Point d'entrée de l'API — informations générales."""
    return {
        "name":        "SafeRoads SN API",
        "version":     "1.0.0",
        "description": "Prédiction des accidents routiers au Sénégal",
        "author":      "Gindima Group — Dakar, Sénégal",
        "docs":        "/docs",
        "endpoints": {
            "predict":  "POST /predict",
            "hotspots": "GET  /hotspots",
            "stats":    "GET  /stats",
            "segments": "GET  /segments",
            "batch":    "POST /batch",
            "health":   "GET  /health",
            "model":    "GET  /model/info",
        },
    }


@app.get("/health", tags=["Général"])
def health():
    """Vérifie l'état du système et la disponibilité des modèles."""
    status = {
        "status":     "ok",
        "timestamp":  datetime.now().isoformat(),
        "models":     {},
        "data":       {},
    }

    for model in ["gravity_model", "risk_model", "dbscan_model"]:
        status["models"][model] = (MODELS_DIR / f"{model}.pkl").exists()

    dataset_path = DATA_PROCESSED_DIR / "saferoads_dataset.csv"
    hotspot_path = DATA_PROCESSED_DIR / "hotspots.csv"
    status["data"]["dataset_ready"]  = dataset_path.exists()
    status["data"]["hotspots_ready"] = hotspot_path.exists()

    if dataset_path.exists():
        df = _load_dataset()
        status["data"]["n_accidents"] = len(df)

    all_models_ok = all(status["models"].values())
    status["status"] = "ok" if all_models_ok else "degraded"

    return status


@app.post("/predict", response_model=PredictResponse, tags=["Prédiction"])
def predict_accident(req: PredictRequest):
    """
    Prédit la gravité et le score de risque pour une observation.

    Retourne :
    - **gravity_label** : léger / grave / mortel
    - **probabilities** : probabilités par classe
    - **risk_score** : score 0-100
    - **risk_level** : faible / moyen / élevé / critique
    - **recommendations** : actions suggérées
    """
    try:
        result = predict(
            latitude=req.latitude,
            longitude=req.longitude,
            hour=req.hour,
            day_of_week=req.day_of_week,
            month=req.month,
            is_rainy=req.is_rainy,
            precipitation_mm=req.precipitation_mm,
            windspeed_kmh=req.windspeed_kmh,
            temperature_c=req.temperature_c,
            visibility_km=req.visibility_km,
            humidity_pct=req.humidity_pct,
            vehicle_type=req.vehicle_type,
            road_type=req.road_type,
            region=req.region,
            cause=req.cause,
            weather_label=req.weather_label,
            nearby_accidents=req.nearby_accidents,
            spatial_density=req.spatial_density,
        )
        result["processed_at"] = datetime.now().isoformat()
        return result

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Modèle non disponible. Lancer : python src/ml/train.py\n{e}"
        )
    except Exception as e:
        log.error(f"Erreur prédiction : {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch", tags=["Prédiction"])
def batch_predict(req: BatchPredictRequest):
    """
    Prédictions en lot pour plusieurs observations (max 100).
    Utile pour analyser un axe routier entier.
    """
    results = []
    errors  = []

    for i, obs in enumerate(req.observations):
        try:
            result = predict(
                latitude=obs.latitude,
                longitude=obs.longitude,
                hour=obs.hour,
                day_of_week=obs.day_of_week,
                month=obs.month,
                is_rainy=obs.is_rainy,
                precipitation_mm=obs.precipitation_mm,
                windspeed_kmh=obs.windspeed_kmh,
                temperature_c=obs.temperature_c,
                visibility_km=obs.visibility_km,
                humidity_pct=obs.humidity_pct,
                vehicle_type=obs.vehicle_type,
                road_type=obs.road_type,
                region=obs.region,
                cause=obs.cause,
                weather_label=obs.weather_label,
                nearby_accidents=obs.nearby_accidents,
                spatial_density=obs.spatial_density,
            )
            result["index"] = i
            results.append(result)
        except Exception as e:
            errors.append({"index": i, "error": str(e)})

    return {
        "total":     len(req.observations),
        "success":   len(results),
        "failed":    len(errors),
        "results":   results,
        "errors":    errors,
        "processed_at": datetime.now().isoformat(),
    }


@app.get("/hotspots", tags=["Analyse"])
def get_hotspots_endpoint(
    risk_level: Optional[str] = Query(None, description="Filtrer : faible/moyen/élevé/critique"),
    region:     Optional[str] = Query(None, description="Filtrer par région"),
    limit:      int           = Query(50,   ge=1, le=500, description="Nombre max de résultats"),
    min_count:  int           = Query(1,    ge=1,         description="Accidents minimum par hotspot"),
):
    """
    Retourne les zones à forte concentration d'accidents (clusters DBSCAN).
    Compatible avec la carte Folium / Leaflet.
    """
    hotspots = get_hotspots()

    if not hotspots:
        raise HTTPException(
            status_code=404,
            detail="Hotspots non disponibles. Lancer : python src/ml/train.py --model clustering"
        )

    # Filtres
    if risk_level:
        hotspots = [h for h in hotspots if h.get("risk_level") == risk_level]
    if region:
        hotspots = [h for h in hotspots if region.lower() in h.get("region", "").lower()]
    if min_count > 1:
        hotspots = [h for h in hotspots if h.get("accident_count", 0) >= min_count]

    # Trier par nombre d'accidents (desc)
    hotspots = sorted(hotspots, key=lambda x: x.get("accident_count", 0), reverse=True)[:limit]

    return {
        "total":    len(hotspots),
        "hotspots": hotspots,
        "filters":  {"risk_level": risk_level, "region": region, "min_count": min_count},
    }


@app.get("/stats", tags=["Analyse"])
def get_stats(
    region: Optional[str] = Query(None, description="Filtrer par région"),
    year:   Optional[int] = Query(None, description="Filtrer par année"),
):
    """
    Statistiques globales sur les accidents : distribution par heure,
    région, gravité, météo, type de véhicule.
    """
    df = _load_dataset()

    if df.empty:
        raise HTTPException(
            status_code=404,
            detail="Dataset non disponible. Lancer : python scripts/run_etl.py"
        )

    # Filtres
    if region and "region" in df.columns:
        df = df[df["region"].str.contains(region, case=False, na=False)]
    if year and "year" in df.columns:
        df = df[df["year"] == year]

    if df.empty:
        raise HTTPException(status_code=404, detail="Aucun résultat pour ces filtres")

    stats = {
        "total_accidents": len(df),
        "filters":         {"region": region, "year": year},
    }

    # Par heure
    if "hour" in df.columns:
        stats["by_hour"] = df["hour"].value_counts().sort_index().to_dict()

    # Par gravité
    if "gravity" in df.columns:
        g = df["gravity"].value_counts().sort_index()
        stats["by_gravity"] = {
            "1_léger":  int(g.get(1, 0)),
            "2_grave":  int(g.get(2, 0)),
            "3_mortel": int(g.get(3, 0)),
        }

    # Par région
    if "region" in df.columns:
        stats["by_region"] = df["region"].value_counts().head(15).to_dict()

    # Par météo
    if "weather_label" in df.columns:
        stats["by_weather"] = df["weather_label"].value_counts().to_dict()
    elif "weather" in df.columns:
        stats["by_weather"] = df["weather"].value_counts().to_dict()

    # Par type de véhicule
    if "vehicle_type" in df.columns:
        stats["by_vehicle"] = df["vehicle_type"].value_counts().head(10).to_dict()

    # Par jour de la semaine
    if "day_of_week" in df.columns:
        days = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
        dow  = df["day_of_week"].value_counts().sort_index()
        stats["by_day"] = {days[i]: int(dow.get(i, 0)) for i in range(7)}

    # Par saison
    if "season" in df.columns:
        stats["by_season"] = df["season"].value_counts().to_dict()

    # Coordonnées min/max (utile pour centrer la carte)
    stats["bbox"] = {
        "lat_min": round(float(df["latitude"].min()),  4),
        "lat_max": round(float(df["latitude"].max()),  4),
        "lon_min": round(float(df["longitude"].min()), 4),
        "lon_max": round(float(df["longitude"].max()), 4),
    }

    return stats


@app.get("/segments", tags=["Analyse"])
def get_risk_segments(
    min_risk: float = Query(60.0, ge=0, le=100, description="Score de risque minimum"),
    region:   Optional[str] = Query(None, description="Filtrer par région"),
    limit:    int           = Query(20,   ge=1, le=200),
):
    """
    Retourne les segments routiers les plus dangereux.
    Score de risque calculé à partir de la densité d'accidents + gravité.
    """
    df = _load_dataset()

    if df.empty:
        raise HTTPException(status_code=404, detail="Dataset non disponible")

    if "region" in df.columns and region:
        df = df[df["region"].str.contains(region, case=False, na=False)]

    # Agréger par zone géographique (grille 0.05°)
    df["lat_bin"] = (df["latitude"]  / 0.05).round() * 0.05
    df["lon_bin"] = (df["longitude"] / 0.05).round() * 0.05

    grp = df.groupby(["lat_bin", "lon_bin"]).agg(
        accident_count=("latitude",  "count"),
        avg_gravity=(   "gravity",   "mean") if "gravity" in df.columns else ("latitude", "count"),
        region=(        "region",    lambda x: x.mode()[0] if "region" in df.columns and not x.empty else "Inconnue"),
    ).reset_index()

    # Score de risque simplifié par segment
    g_max  = grp["avg_gravity"].max() if "avg_gravity" in grp else 3
    c_max  = grp["accident_count"].max()
    grp["risk_score"] = (
        0.6 * (grp["accident_count"]  / (c_max + 1e-9)) * 100 +
        0.4 * ((grp.get("avg_gravity", 2) - 1) / 2)    * 100
    ).round(1)

    segments = grp[grp["risk_score"] >= min_risk].sort_values(
        "risk_score", ascending=False
    ).head(limit)

    result = []
    for _, row in segments.iterrows():
        result.append({
            "center_lat":     round(float(row["lat_bin"]), 4),
            "center_lon":     round(float(row["lon_bin"]), 4),
            "accident_count": int(row["accident_count"]),
            "avg_gravity":    round(float(row.get("avg_gravity", 2)), 2),
            "risk_score":     float(row["risk_score"]),
            "risk_level":     "critique" if row["risk_score"] >= 80 else
                              "élevé"    if row["risk_score"] >= 60 else "moyen",
            "region":         str(row.get("region", "Inconnue")),
        })

    return {
        "total":    len(result),
        "segments": result,
        "filters":  {"min_risk": min_risk, "region": region},
    }


@app.get("/model/info", tags=["Modèles"])
def model_info():
    """
    Retourne les métriques de performance des modèles entraînés.
    """
    info = _load_model_info()

    if not info:
        raise HTTPException(
            status_code=404,
            detail="Aucun modèle entraîné. Lancer : python src/ml/train.py"
        )

    return {
        "models":     info,
        "models_dir": str(MODELS_DIR),
        "retrieved_at": datetime.now().isoformat(),
    }


@app.get("/forecast", tags=["Analyse"])
def get_forecast(
    region: str  = Query("Dakar", description="Région cible"),
    days:   int  = Query(7,       ge=1, le=30, description="Horizon de prévision en jours"),
):
    """
    Prévision du risque d'accident pour les N prochains jours,
    en tenant compte des patterns historiques (heure, jour, saison).
    """
    from datetime import timedelta

    now = datetime.now()
    forecasts = []

    for d in range(days):
        target_date  = now + timedelta(days=d)
        dow          = target_date.weekday()
        month        = target_date.month
        is_weekend   = dow >= 5
        is_hivernage = month in [6, 7, 8, 9, 10]

        # Score de base selon le pattern historique
        base_risk = 45.0
        if is_weekend:     base_risk += 10
        if is_hivernage:   base_risk += 8
        if dow == 4:        base_risk += 5   # Vendredi = départ week-end

        # Heures de pointe prévues
        peak_hours = ["7h-9h", "17h-20h"]
        if is_weekend:
            peak_hours = ["10h-14h", "18h-22h"]

        forecasts.append({
            "date":        target_date.strftime("%Y-%m-%d"),
            "day_name":    ["Lundi","Mardi","Mercredi","Jeudi","Vendredi","Samedi","Dimanche"][dow],
            "risk_score":  round(min(base_risk, 95), 1),
            "risk_level":  "élevé" if base_risk >= 60 else "moyen" if base_risk >= 40 else "faible",
            "peak_hours":  peak_hours,
            "is_weekend":  is_weekend,
            "notes":       "Hivernage — risque glissance" if is_hivernage else "",
        })

    return {
        "region":    region,
        "horizon":   f"{days} jours",
        "from_date": now.strftime("%Y-%m-%d"),
        "forecasts": forecasts,
    }


# ══════════════════════════════════════════════════════
# LANCEMENT DIRECT
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    log.info(f"Démarrage API SafeRoads SN sur http://{API_HOST}:{API_PORT}")
    uvicorn.run(
        "src.api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
        log_level="info",
    )
