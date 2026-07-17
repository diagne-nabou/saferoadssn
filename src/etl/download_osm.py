"""
SafeRoads SN — download_osm.py
Télécharge le réseau routier du Sénégal via OSMnx (OpenStreetMap).
Sauvegarde en GeoPackage (.gpkg) et CSV des segments.

Usage :
    python -m src.etl.download_osm
"""

import sys
import logging
from pathlib import Path

import pandas as pd
import geopandas as gpd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import OSM_RAW_DIR, SENEGAL_CITIES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [OSM] %(message)s")
log = logging.getLogger(__name__)

# ── Fichiers de sortie ──
OSM_GPKG_PATH = OSM_RAW_DIR / "senegal_roads.gpkg"
OSM_CSV_PATH  = OSM_RAW_DIR / "senegal_roads.csv"

# Types de routes à conserver (pertinents pour les accidents)
ROAD_TYPES_KEPT = [
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "motorway_link", "trunk_link", "primary_link", "secondary_link",
    "residential", "unclassified",
]


def download_senegal_roads(force: bool = False) -> gpd.GeoDataFrame:
    """
    Télécharge le réseau routier du Sénégal depuis OpenStreetMap.
    Si déjà téléchargé, charge depuis le cache sauf si force=True.
    """
    try:
        import osmnx as ox
    except ImportError:
        log.error("osmnx non installé. Lancer : pip install osmnx")
        raise

    if OSM_GPKG_PATH.exists() and not force:
        log.info(f"Cache trouvé → chargement depuis {OSM_GPKG_PATH}")
        return load_road_network()

    log.info("Téléchargement réseau routier Sénégal depuis OpenStreetMap...")
    log.info("(Peut prendre 2-5 minutes selon la connexion)")

    try:
        # Télécharger le graphe routier du Sénégal complet
        G = ox.graph_from_place(
            "Senegal",
            network_type="drive",
            simplify=True,
        )

        log.info(f"Graphe téléchargé : {len(G.nodes)} nœuds, {len(G.edges)} arêtes")

        # Convertir en GeoDataFrame des arêtes (segments routiers)
        _, edges = ox.graph_to_gdfs(G)
        edges = edges.reset_index()

        # Garder les colonnes utiles
        cols_keep = [c for c in [
            "osmid", "name", "highway", "length",
            "maxspeed", "lanes", "oneway", "geometry"
        ] if c in edges.columns]

        edges = edges[cols_keep].copy()
        edges = edges.rename(columns={"highway": "road_type", "length": "length_m"})

        # Nettoyer road_type (peut être une liste)
        edges["road_type"] = edges["road_type"].apply(
            lambda x: x[0] if isinstance(x, list) else x
        )

        # Filtrer sur les types pertinents
        edges = edges[edges["road_type"].isin(ROAD_TYPES_KEPT)].copy()

        # Ajouter longueur en km
        edges["length_km"] = (edges["length_m"] / 1000).round(3)

        # Catégorie lisible
        edges["road_category"] = edges["road_type"].map({
            "motorway":      "autoroute",
            "motorway_link": "autoroute",
            "trunk":         "nationale",
            "trunk_link":    "nationale",
            "primary":       "nationale",
            "primary_link":  "nationale",
            "secondary":     "régionale",
            "secondary_link":"régionale",
            "tertiary":      "départementale",
            "residential":   "urbaine",
            "unclassified":  "piste",
        }).fillna("autre")

        # Initialiser les colonnes de risque
        edges["risk_score"]     = 0.0
        edges["accident_count"] = 0

        # Sauvegarder
        OSM_RAW_DIR.mkdir(parents=True, exist_ok=True)
        edges.to_file(str(OSM_GPKG_PATH), driver="GPKG")
        log.info(f"Sauvegardé : {OSM_GPKG_PATH} ({len(edges)} segments)")

        # CSV sans géométrie pour usage rapide
        edges_csv = edges.drop(columns=["geometry"])
        edges_csv.to_csv(OSM_CSV_PATH, index=False)
        log.info(f"CSV sauvegardé : {OSM_CSV_PATH}")

        return edges

    except Exception as e:
        log.error(f"Échec téléchargement OSM : {e}")
        log.info("Tentative de téléchargement par ville...")
        return _download_by_cities()


def _download_by_cities() -> gpd.GeoDataFrame:
    """
    Fallback : télécharge ville par ville si le pays entier échoue.
    """
    import osmnx as ox

    all_edges = []
    for city_info in SENEGAL_CITIES:
        city = city_info["name"]
        try:
            log.info(f"  Téléchargement {city}...")
            G = ox.graph_from_place(
                f"{city}, Senegal",
                network_type="drive",
                dist=15000,  # 15km autour du centre
            )
            _, edges = ox.graph_to_gdfs(G)
            edges = edges.reset_index()
            edges["city_origin"] = city
            all_edges.append(edges)
            log.info(f"  ✓ {city} : {len(edges)} segments")
        except Exception as e:
            log.warning(f"  ✗ {city} ignorée : {e}")

    if not all_edges:
        raise RuntimeError("Impossible de télécharger les données OSM.")

    import pandas as pd
    combined = gpd.GeoDataFrame(
        pd.concat(all_edges, ignore_index=True),
        crs="EPSG:4326"
    )
    combined.to_file(str(OSM_GPKG_PATH), driver="GPKG")
    log.info(f"Téléchargement par villes terminé : {len(combined)} segments")
    return combined


def load_road_network() -> gpd.GeoDataFrame:
    """
    Charge le réseau routier depuis le cache local.
    """
    if not OSM_GPKG_PATH.exists():
        raise FileNotFoundError(
            f"Réseau routier non trouvé : {OSM_GPKG_PATH}\n"
            f"Lancer : python scripts/setup_data.py --osm-only"
        )
    log.info(f"Chargement réseau routier depuis {OSM_GPKG_PATH}")
    gdf = gpd.read_file(str(OSM_GPKG_PATH))
    log.info(f"{len(gdf)} segments chargés")
    return gdf


def get_road_stats() -> dict:
    """
    Retourne des statistiques sur le réseau chargé.
    """
    gdf = load_road_network()
    return {
        "total_segments":   len(gdf),
        "total_length_km":  round(gdf["length_km"].sum(), 1) if "length_km" in gdf.columns else None,
        "road_types":       gdf["road_type"].value_counts().to_dict() if "road_type" in gdf.columns else {},
        "categories":       gdf["road_category"].value_counts().to_dict() if "road_category" in gdf.columns else {},
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Forcer le re-téléchargement")
    parser.add_argument("--stats", action="store_true", help="Afficher les stats")
    args = parser.parse_args()

    if args.stats and OSM_GPKG_PATH.exists():
        stats = get_road_stats()
        print("\n=== Statistiques réseau routier ===")
        for k, v in stats.items():
            print(f"  {k}: {v}")
    else:
        gdf = download_senegal_roads(force=args.force)
        print(f"\n✅ {len(gdf)} segments routiers disponibles")
