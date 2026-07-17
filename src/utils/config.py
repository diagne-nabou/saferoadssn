"""
SafeRoads SN — Configuration centralisée
Charge les variables depuis .env
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Charger .env depuis la racine du projet
ROOT_DIR = Path(__file__).parent.parent.parent
load_dotenv(ROOT_DIR / ".env")

# ── Chemins ──
DATA_RAW_DIR       = ROOT_DIR / os.getenv("DATA_RAW_DIR", "data/raw")
DATA_PROCESSED_DIR = ROOT_DIR / os.getenv("DATA_PROCESSED_DIR", "data/processed")
MODELS_DIR         = ROOT_DIR / os.getenv("MODELS_DIR", "models")

ACCIDENTS_RAW_PATH = DATA_RAW_DIR / "accidents" / "accidents.csv"
OSM_RAW_DIR        = DATA_RAW_DIR / "osm"
WEATHER_RAW_DIR    = DATA_RAW_DIR / "weather"

FINAL_DATASET_PATH = DATA_PROCESSED_DIR / "saferoads_dataset.csv"

# ── Base de données ──
DB_HOST     = os.getenv("DB_HOST", "localhost")
DB_PORT     = int(os.getenv("DB_PORT", 5432))
DB_NAME     = os.getenv("DB_NAME", "saferoads")
DB_USER     = os.getenv("DB_USER", "saferoads_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "changeme_securise")
DB_URL      = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# ── API ──
API_HOST  = os.getenv("API_HOST", "0.0.0.0")
API_PORT  = int(os.getenv("API_PORT", 8000))
API_DEBUG = os.getenv("API_DEBUG", "false").lower() == "true"

# ── Sénégal bbox ──
SENEGAL_BBOX = {
    "north": float(os.getenv("SENEGAL_BBOX_NORTH", 16.7)),
    "south": float(os.getenv("SENEGAL_BBOX_SOUTH", 12.3)),
    "west":  float(os.getenv("SENEGAL_BBOX_WEST", -17.6)),
    "east":  float(os.getenv("SENEGAL_BBOX_EAST", -11.3)),
}

# ── Régions avec coordonnées et poids (pour simulation GPS) ──
SENEGAL_REGIONS_COORDS = [
    {"region": "Dakar",        "lat": 14.6937, "lon": -17.4441, "weight": 0.25},
    {"region": "Thiès",        "lat": 14.7886, "lon": -16.9260, "weight": 0.15},
    {"region": "Kaolack",      "lat": 14.1652, "lon": -16.0726, "weight": 0.10},
    {"region": "Saint-Louis",  "lat": 16.0179, "lon": -16.4896, "weight": 0.08},
    {"region": "Diourbel",     "lat": 14.6550, "lon": -16.2323, "weight": 0.09},
    {"region": "Ziguinchor",   "lat": 12.5681, "lon": -16.2719, "weight": 0.07},
    {"region": "Tambacounda",  "lat": 13.7707, "lon": -13.6673, "weight": 0.06},
    {"region": "Mbour",        "lat": 14.3850, "lon": -16.9653, "weight": 0.08},
    {"region": "Louga",        "lat": 15.6172, "lon": -16.2240, "weight": 0.06},
    {"region": "Kolda",        "lat": 12.8983, "lon": -14.9412, "weight": 0.06},
]

# ── Villes principales (pour météo) ──
SENEGAL_CITIES = [
    {"name": "Dakar",         "lat": 14.6937, "lon": -17.4441},
    {"name": "Thiès",         "lat": 14.7886, "lon": -16.9260},
    {"name": "Kaolack",       "lat": 14.1652, "lon": -16.0726},
    {"name": "Saint-Louis",   "lat": 16.0179, "lon": -16.4896},
    {"name": "Ziguinchor",    "lat": 12.5681, "lon": -16.2719},
    {"name": "Tambacounda",   "lat": 13.7707, "lon": -13.6673},
    {"name": "Mbour",         "lat": 14.3850, "lon": -16.9653},
    {"name": "Diourbel",      "lat": 14.6550, "lon": -16.2323},
    {"name": "Louga",         "lat": 15.6172, "lon": -16.2240},
    {"name": "Kolda",         "lat": 12.8983, "lon": -14.9412},
]

# ── Créer les dossiers si absents ──
for d in [DATA_RAW_DIR/"accidents", DATA_RAW_DIR/"osm",
          DATA_RAW_DIR/"weather", DATA_PROCESSED_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
