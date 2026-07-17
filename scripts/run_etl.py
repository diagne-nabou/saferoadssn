"""
SafeRoads SN -- run_etl.py
Lance le pipeline ETL complet :
  1. Charge et normalise les accidents (CSV)
  2. Charge le reseau routier OSM
  3. Charge la meteo historique
  4. Fusionne tout en un dataset final pret pour le ML

Usage :
    python scripts/run_etl.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    print("=" * 60)
    print("  SafeRoads SN -- Pipeline ETL")
    print("=" * 60)

    steps = [
        ("Chargement accidents",        "src.etl.load_accidents",    "load_accidents"),
        ("Chargement reseau OSM",       "src.etl.download_osm",      "load_road_network"),
        ("Chargement meteo",            "src.etl.download_weather",  "load_weather_features"),
        ("Fusion features",             "src.etl.merge_features",    "merge_all_features"),
    ]

    results = {}
    for label, module_path, func_name in steps:
        print(f"\n  >> {label}...")
        try:
            import importlib
            module = importlib.import_module(module_path)
            func   = getattr(module, func_name)
            result = func()
            results[func_name] = result
            if hasattr(result, '__len__'):
                print(f"    [OK] {len(result)} enregistrements")
            else:
                print(f"    [OK]")
        except FileNotFoundError as e:
            print(f"    [WARN] Fichier manquant : {e}")
            print(f"           -> Lancer d'abord : python scripts/setup_data.py")
        except Exception as e:
            print(f"    [ERR] Erreur : {e}")

    print("\n" + "=" * 60)
    print("  Etape suivante : python src/ml/train.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
