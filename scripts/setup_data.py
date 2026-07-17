"""
SafeRoads SN — setup_data.py
Télécharge automatiquement OSM (réseau routier) + météo Open-Meteo.
Les données accidents doivent être placées manuellement (voir data/raw/accidents/README_format.md).

Usage :
    python scripts/setup_data.py
    python scripts/setup_data.py --osm-only
    python scripts/setup_data.py --weather-only
"""

import argparse
import sys
from pathlib import Path

# Ajouter le dossier racine au path
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    parser = argparse.ArgumentParser(description="SafeRoads SN — Téléchargement des données")
    parser.add_argument("--osm-only",     action="store_true", help="Télécharger uniquement OSM")
    parser.add_argument("--weather-only", action="store_true", help="Télécharger uniquement la météo")
    parser.add_argument("--skip-osm",     action="store_true", help="Ignorer OSM")
    parser.add_argument("--skip-weather", action="store_true", help="Ignorer la météo")
    args = parser.parse_args()

    print("=" * 60)
    print("  SafeRoads SN — Setup données")
    print("=" * 60)

    run_osm     = not args.weather_only and not args.skip_osm
    run_weather = not args.osm_only     and not args.skip_weather

    if run_osm:
        print("\n[1/2] Téléchargement réseau routier OSM (Sénégal)...")
        try:
            from src.etl.download_osm import download_senegal_roads
            download_senegal_roads()
            print("  ✅ OSM téléchargé")
        except Exception as e:
            print(f"  ❌ Erreur OSM : {e}")

    if run_weather:
        print("\n[2/2] Téléchargement météo historique (Open-Meteo)...")
        try:
            from src.etl.download_weather import download_weather_history
            download_weather_history()
            print("  ✅ Météo téléchargée")
        except Exception as e:
            print(f"  ❌ Erreur météo : {e}")

    print("\n" + "=" * 60)
    print("  Étape suivante :")
    print("  1. Placer accidents.csv dans data/raw/accidents/")
    print("  2. Lancer : python scripts/run_etl.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
