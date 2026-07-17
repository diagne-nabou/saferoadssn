"""
extract_osm_senegal.py
======================
Extrait les données OpenStreetMap (routes, bâtiments, lieux habités)
pour les régions manquantes du Sénégal via l'API Overpass.

Régions déjà couvertes (à ne pas re-télécharger) :
  - Dakar       → merged_data_dakar.xlsx
  - Diourbel    → merged_data_dbl.xlsx
  - Kaolack     → merged_data_klk.xlsx

Régions à extraire (ce script) :
  - Ziguinchor, Sédhiou, Kolda        (Casamance / Sud)
  - Saint-Louis, Louga, Matam         (Nord)
  - Thiès                             (Centre-Ouest)
  - Fatick                            (Centre)
  - Kaffrine                          (Centre)
  - Tambacounda, Kédougou             (Est)

Usage :
  pip install requests pandas openpyxl tqdm
  python extract_osm_senegal.py

Sortie : un fichier merged_data_<region>.xlsx par région,
         dans le même format que tes fichiers existants.
"""

import time
import requests
import pandas as pd
from shapely import wkt
from shapely.geometry import Point, LineString, Polygon
from tqdm import tqdm
import os, json, warnings
warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────────
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
    "https://z.overpass-api.de/api/interpreter",
]
OUTPUT_DIR = "osm_regions"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TIMEOUT = 120          # secondes par requête Overpass
RETRY   = 3            # tentatives max
SLEEP   = 8            # secondes entre régions (éviter le rate-limit)

# ── Régions manquantes avec bounding boxes (sud, lat, ouest, est) ────────────
# Format Overpass : (min_lat, min_lon, max_lat, max_lon)
REGIONS = {
    "thies": {
        "label": "Thiès",
        "bbox": (14.40, -17.10, 15.00, -16.20),
    },
    "fatick": {
        "label": "Fatick",
        "bbox": (13.70, -16.80, 14.55, -15.50),
    },
    "kaffrine": {
        "label": "Kaffrine",
        "bbox": (13.50, -15.80, 14.35, -14.30),
    },
    "tambacounda": {
        "label": "Tambacounda",
        "bbox": (12.20, -14.20, 14.00, -11.40),
    },
    "kedougou": {
        "label": "Kédougou",
        "bbox": (11.80, -13.00, 13.20, -11.30),
    },
    "ziguinchor": {
        "label": "Ziguinchor",
        "bbox": (12.20, -16.80, 13.10, -15.30),
    },
    "sedhiou": {
        "label": "Sédhiou",
        "bbox": (12.50, -15.80, 13.30, -14.40),
    },
    "kolda": {
        "label": "Kolda",
        "bbox": (12.20, -15.20, 13.30, -13.50),
    },
    "saint_louis": {
        "label": "Saint-Louis",
        "bbox": (15.40, -16.60, 16.90, -14.80),
    },
    "louga": {
        "label": "Louga",
        "bbox": (14.80, -16.40, 15.80, -14.60),
    },
    "matam": {
        "label": "Matam",
        "bbox": (14.50, -13.80, 15.80, -12.00),
    },
}

# ── Overpass query builder ────────────────────────────────────────────────────
def build_query(bbox, timeout=TIMEOUT):
    s, w, n, e = bbox
    bb = f"{s},{w},{n},{e}"
    return f"""
[out:json][timeout:{timeout}];
(
  way["highway"]({bb});
  node["place"~"city|town|village|hamlet|suburb"]({bb});
  way["building"]({bb});
);
out body geom;
"""

# ── HTTP fetch with retry ─────────────────────────────────────────────────────
def fetch_overpass(query):
    for url in OVERPASS_URLS:
        for attempt in range(1, RETRY + 1):
            try:
                r = requests.post(url, data={"data": query}, timeout=TIMEOUT + 30)
                if r.status_code == 200:
                    return r.json()
                print(f"  HTTP {r.status_code} — retry {attempt}/{RETRY}")
            except requests.exceptions.Timeout:
                print(f"  Timeout sur {url} — retry {attempt}/{RETRY}")
            except Exception as ex:
                print(f"  Erreur: {ex} — retry {attempt}/{RETRY}")
            time.sleep(5 * attempt)
    return None

# ── Geometry helpers ──────────────────────────────────────────────────────────
def nodes_to_coords(nodes):
    return [(n["lon"], n["lat"]) for n in nodes if "lon" in n and "lat" in n]

def element_to_row(el):
    tags = el.get("tags", {})
    osm_id   = el.get("id")
    el_type  = el.get("type")
    geom_wkt = None
    lon = lat = None

    if el_type == "node":
        lon = el.get("lon")
        lat = el.get("lat")
        if lon and lat:
            geom_wkt = Point(lon, lat).wkt

    elif el_type == "way":
        coords = nodes_to_coords(el.get("geometry", []))
        if len(coords) >= 2:
            try:
                if coords[0] == coords[-1] and len(coords) >= 4:
                    geom_wkt = Polygon(coords).wkt
                else:
                    geom_wkt = LineString(coords).wkt
                centroid = wkt.loads(geom_wkt).centroid
                lon, lat = centroid.x, centroid.y
            except Exception:
                pass

    source_file = "roads" if "highway" in tags else \
                  "buildings" if "building" in tags else \
                  "places"

    return {
        "osm_id":       osm_id,
        "name":         tags.get("name"),
        "type":         tags.get("highway") or tags.get("place") or tags.get("building"),
        "source_file":  source_file,
        "geometry_wkt": geom_wkt,
        "population":   tags.get("population"),
        "lon":          lon,
        "lat":          lat,
        "timestamp":    el.get("timestamp"),
        "ref":          tags.get("ref"),
        "oneway":       tags.get("oneway"),
        "bridge":       tags.get("bridge"),
        "maxspeed":     tags.get("maxspeed"),
        "width":        tags.get("width"),
    }

# ── Main loop ─────────────────────────────────────────────────────────────────
def process_region(key, cfg):
    label = cfg["label"]
    out_path = os.path.join(OUTPUT_DIR, f"merged_data_{key}.xlsx")

    if os.path.exists(out_path):
        print(f"[SKIP] {label} — fichier déjà présent : {out_path}")
        return True

    print(f"\n{'='*60}")
    print(f"[{label}]  bbox={cfg['bbox']}")
    print(f"{'='*60}")

    query = build_query(cfg["bbox"])
    print("  → Requête Overpass en cours...")
    data = fetch_overpass(query)

    if data is None:
        print(f"  ✗ Échec pour {label} — passe à la suivante")
        return False

    elements = data.get("elements", [])
    print(f"  → {len(elements)} éléments reçus")

    if not elements:
        print(f"  ✗ Aucun élément — vérifie la bbox")
        return False

    rows = []
    for el in tqdm(elements, desc=f"  Parse {label}", leave=False):
        row = element_to_row(el)
        if row:
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_excel(out_path, index=False, engine="openpyxl")
    size_mb = os.path.getsize(out_path) / 1_000_000
    print(f"  ✓ Sauvegardé : {out_path}  ({len(df):,} lignes, {size_mb:.1f} Mo)")
    return True

def main():
    print("=" * 60)
    print("  Extraction OSM — Régions manquantes du Sénégal")
    print("=" * 60)
    print(f"  Régions à traiter : {len(REGIONS)}")
    print(f"  Dossier de sortie : {os.path.abspath(OUTPUT_DIR)}/")
    print()

    results = {}
    for i, (key, cfg) in enumerate(REGIONS.items(), 1):
        print(f"[{i}/{len(REGIONS)}] Traitement de {cfg['label']}...")
        ok = process_region(key, cfg)
        results[cfg["label"]] = "✓ OK" if ok else "✗ Échec"
        if i < len(REGIONS):
            print(f"  ⏳ Pause {SLEEP}s avant la prochaine région...")
            time.sleep(SLEEP)

    print("\n" + "=" * 60)
    print("  BILAN FINAL")
    print("=" * 60)
    for region, status in results.items():
        print(f"  {status}  {region}")
    print()
    print("  Fichiers générés dans :", os.path.abspath(OUTPUT_DIR))
    print("  → Copie ces fichiers avec tes merged_data_dakar/dbl/klk.xlsx")
    print("    et reviens pour l'étape 5 !")

if __name__ == "__main__":
    main()
