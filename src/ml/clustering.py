"""
SafeRoads SN — clustering.py
Détection des zones à risque (hotspots) via DBSCAN géospatial.
Exporte les résultats en CSV et GeoJSON pour la carte Folium.

Usage :
    python -m src.ml.clustering
    python -m src.ml.clustering --eps 2.0 --min-samples 3
"""

import sys
import json
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import DATA_PROCESSED_DIR, MODELS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [CLUSTER] %(message)s")
log = logging.getLogger(__name__)

HOTSPOTS_CSV     = DATA_PROCESSED_DIR / "hotspots.csv"
HOTSPOTS_GEOJSON = DATA_PROCESSED_DIR / "hotspots.geojson"


def run_dbscan(
    df: pd.DataFrame,
    eps_km: float = 1.0,
    min_samples: int = 5,
) -> pd.DataFrame:
    """
    Exécute DBSCAN géospatial et retourne le dataframe enrichi avec cluster_id.
    """
    coords = df[["latitude", "longitude"]].dropna().values
    coords_rad = np.deg2rad(coords)

    eps_rad = eps_km / 6371.0

    log.info(f"DBSCAN — eps={eps_km}km, min_samples={min_samples}")
    db = DBSCAN(
        eps=eps_rad,
        min_samples=min_samples,
        algorithm="ball_tree",
        metric="haversine",
        n_jobs=-1,
    )
    labels = db.fit_predict(coords_rad)

    df = df.copy()
    df["cluster_id"] = labels

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()
    pct_clustered = (labels != -1).mean() * 100

    log.info(f"  {n_clusters} clusters | {n_noise} points isolés | {pct_clustered:.1f}% clustérisés")
    return df, db, n_clusters


def compute_risk_score(accident_count, avg_gravity, radius_km, total_accidents,
                       proximity_factor=0.0):
    """
    Score de risque composite (0-100) selon la methodologie du projet :
      - Densite (30%) : nb accidents / surface du cluster
      - Gravite (40%) : gravite moyenne normalisee
      - Frequence (20%) : proportion d'accidents dans cette zone vs total
      - Proximite (10%) : facteur base sur la proximite aux axes routiers
    """
    # Densite : accidents par km2 (normalise 0-1)
    area_km2 = max(np.pi * radius_km ** 2, 0.01)
    density  = accident_count / area_km2
    density_norm = min(density / 50.0, 1.0)  # cap a 50 acc/km2

    # Gravite : normalisee entre 1 et 3 → 0-1
    gravity_norm = (avg_gravity - 1.0) / 2.0

    # Frequence : part des accidents totaux
    freq_norm = min(accident_count / max(total_accidents, 1), 1.0)

    # Proximite (0-1, default 0)
    prox_norm = min(proximity_factor, 1.0)

    # Score composite pondere
    score = (
        30 * density_norm +
        40 * gravity_norm +
        20 * freq_norm +
        10 * prox_norm
    )
    return round(min(max(score, 0), 100), 1)


def build_hotspots_table(df_clustered: pd.DataFrame) -> pd.DataFrame:
    """
    Construit le tableau resume des hotspots avec score de risque composite.
    Score = f(densite, gravite, frequence, proximite)
    """
    total_accidents = len(df_clustered[df_clustered["cluster_id"] != -1])
    rows = []

    for cid in sorted(df_clustered["cluster_id"].unique()):
        if cid == -1:
            continue

        cluster = df_clustered[df_clustered["cluster_id"] == cid]
        count   = len(cluster)

        # Centre du cluster (moyenne ponderee par gravite si disponible)
        if "gravity" in cluster.columns:
            weights = cluster["gravity"].values
            center_lat = np.average(cluster["latitude"], weights=weights)
            center_lon = np.average(cluster["longitude"], weights=weights)
            avg_gravity = cluster["gravity"].mean()
        else:
            center_lat  = cluster["latitude"].mean()
            center_lon  = cluster["longitude"].mean()
            avg_gravity = 2.0

        # Rayon approximatif (distance max au centre en km)
        dists = np.sqrt(
            (cluster["latitude"] - center_lat)**2 +
            (cluster["longitude"] - center_lon)**2
        ) * 111  # ~111km par degre
        radius_km = max(dists.max(), 0.1)

        # Score de risque composite
        risk_score = compute_risk_score(
            accident_count=count,
            avg_gravity=avg_gravity,
            radius_km=radius_km,
            total_accidents=total_accidents,
        )

        # Niveau de risque base sur le score composite
        if risk_score >= 70:
            risk_level = "critique"
        elif risk_score >= 50:
            risk_level = "eleve"
        elif risk_score >= 30:
            risk_level = "moyen"
        else:
            risk_level = "faible"

        # Couleur pour la carte
        color_map = {
            "critique": "#FF0000",
            "eleve":    "#FF8C00",
            "moyen":    "#FFD700",
            "faible":   "#00AA00",
        }

        # Region dominante
        region = "Inconnue"
        if "region" in cluster.columns:
            region = cluster["region"].mode().iloc[0] if not cluster["region"].mode().empty else "Inconnue"

        # Heures de pointe dans ce cluster
        peak_hours = ""
        if "hour" in cluster.columns:
            valid_hours = cluster[cluster["hour"] >= 0]["hour"]
            if not valid_hours.empty:
                top_hours = valid_hours.value_counts().head(3).index.tolist()
                peak_hours = ", ".join([f"{h}h" for h in sorted(top_hours)])

        rows.append({
            "cluster_id":     int(cid),
            "center_lat":     round(float(center_lat), 5),
            "center_lon":     round(float(center_lon), 5),
            "accident_count": int(count),
            "avg_gravity":    round(float(avg_gravity), 2),
            "radius_km":      round(float(radius_km), 2),
            "risk_score":     risk_score,
            "risk_level":     risk_level,
            "color":          color_map[risk_level],
            "region":         region,
            "peak_hours":     peak_hours,
        })

    df_hotspots = pd.DataFrame(rows)
    df_hotspots = df_hotspots.sort_values("risk_score", ascending=False).reset_index(drop=True)
    return df_hotspots


def export_geojson(df_hotspots: pd.DataFrame) -> dict:
    """
    Exporte les hotspots en GeoJSON pour affichage sur carte.
    """
    features = []
    for _, row in df_hotspots.iterrows():
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [row["center_lon"], row["center_lat"]],
            },
            "properties": {
                "cluster_id":     int(row["cluster_id"]),
                "accident_count": int(row["accident_count"]),
                "avg_gravity":    float(row["avg_gravity"]),
                "radius_km":      float(row["radius_km"]),
                "risk_level":     row["risk_level"],
                "color":          row["color"],
                "region":         row["region"],
                "peak_hours":     row.get("peak_hours", ""),
            },
        }
        features.append(feature)

    geojson = {
        "type":     "FeatureCollection",
        "features": features,
        "metadata": {
            "total_hotspots": len(features),
            "source":         "SafeRoads SN — DBSCAN géospatial",
        },
    }
    return geojson


def run_clustering(
    eps_km: float = 1.0,
    min_samples: int = 5,
    dataset_path: Path = None,
) -> pd.DataFrame:
    """Point d'entrée principal."""

    path = dataset_path or DATA_PROCESSED_DIR / "saferoads_dataset.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset non trouvé : {path}\n"
            "Lancer : python scripts/run_etl.py"
        )

    log.info(f"Chargement dataset : {path}")
    df = pd.read_csv(path)
    log.info(f"  {len(df):,} accidents")

    # Clustering
    df_clustered, db_model, n_clusters = run_dbscan(df, eps_km, min_samples)

    # Table hotspots
    df_hotspots = build_hotspots_table(df_clustered)

    # Sauvegardes
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_hotspots.to_csv(HOTSPOTS_CSV, index=False)
    log.info(f"  CSV sauvegardé : {HOTSPOTS_CSV}")

    geojson = export_geojson(df_hotspots)
    with open(HOTSPOTS_GEOJSON, "w") as f:
        json.dump(geojson, f, indent=2, ensure_ascii=False)
    log.info(f"  GeoJSON sauvegardé : {HOTSPOTS_GEOJSON}")

    # Sauvegarder le modèle
    import joblib
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model":       db_model,
        "eps_km":      eps_km,
        "min_samples": min_samples,
        "n_clusters":  n_clusters,
        "hotspots":    df_hotspots.to_dict("records"),
    }, MODELS_DIR / "dbscan_model.pkl")

    # Résumé
    log.info("\n  ── Top 10 Hotspots ──")
    for _, row in df_hotspots.head(10).iterrows():
        log.info(
            f"  #{row['cluster_id']:>3} | {row['region']:<15} | "
            f"{row['accident_count']:>4} acc | "
            f"risque {row['risk_level']:<8} | "
            f"coords ({row['center_lat']:.3f}, {row['center_lon']:.3f})"
        )

    return df_hotspots


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SafeRoads SN — Clustering DBSCAN")
    parser.add_argument("--eps",         type=float, default=1.0,  help="Rayon en km (défaut: 1.0)")
    parser.add_argument("--min-samples", type=int,   default=5,    help="Min points par cluster (défaut: 5)")
    args = parser.parse_args()

    df_hotspots = run_clustering(args.eps, args.min_samples)
    print(f"\n✅ {len(df_hotspots)} hotspots détectés")
    print(df_hotspots[["cluster_id", "region", "accident_count", "risk_level", "peak_hours"]].to_string(index=False))
