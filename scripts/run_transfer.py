"""
SafeRoads SN — run_transfer.py
Lance le pipeline transfer learning complet :
  1. Mapper dataset global Kaggle → format SafeRoads
  2. Phase 1 : Pré-entraînement sur données globales (132k)
  3. Phase 2 : Fine-tuning sur données sénégalaises
  4. Évaluation comparative Source vs Fine-tuné

Prérequis :
  - data/raw/accidents/accidents.csv  ← dataset Kaggle global téléchargé
  - data/processed/saferoads_dataset.csv ← généré par run_etl.py

Usage :
    python scripts/run_transfer.py
    python scripts/run_transfer.py --skip-mapping   # Si déjà mappé
    python scripts/run_transfer.py --phase 1        # Phase 1 seulement
    python scripts/run_transfer.py --phase 2        # Phase 2 seulement
    python scripts/run_transfer.py --eval           # Évaluation seulement
"""

import sys
import json
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

from src.utils.config import DATA_PROCESSED_DIR, MODELS_DIR


def main():
    parser = argparse.ArgumentParser(description="SafeRoads SN — Transfer Learning Pipeline")
    parser.add_argument("--skip-mapping", action="store_true",
                        help="Ignorer l'étape de mapping (si global_mapped.csv existe déjà)")
    parser.add_argument("--phase", type=int, choices=[1, 2],
                        help="Lancer une seule phase")
    parser.add_argument("--eval", action="store_true",
                        help="Évaluation comparative seulement")
    parser.add_argument("--extra-trees", type=int, default=100,
                        help="Arbres additionnels pour le fine-tuning (défaut: 100)")
    args = parser.parse_args()

    print("=" * 60)
    print("  SafeRoads SN — Pipeline Transfer Learning")
    print("=" * 60)

    # ── Étape 0 : Mapping dataset global ──
    global_mapped = DATA_PROCESSED_DIR / "global_mapped.csv"
    if not args.skip_mapping and not args.eval and args.phase != 2:
        print("\n[0/3] Mapping dataset global Kaggle → SafeRoads...")
        try:
            from src.etl.mapper_global import run_mapping
            df = run_mapping()
            print(f"  ✅ {len(df):,} enregistrements mappés")
        except FileNotFoundError as e:
            print(f"  ❌ {e}")
            print("\n  Télécharger le dataset depuis :")
            print("  https://kaggle.com/datasets/ankushpanday1/global-road-accidents-dataset")
            print("  → Placer dans : data/raw/accidents/accidents.csv")
            return
        except Exception as e:
            print(f"  ❌ Erreur mapping : {e}")
            return
    elif global_mapped.exists():
        import pandas as pd
        n = len(pd.read_csv(global_mapped))
        print(f"\n[0/3] Mapping : cache trouvé ({n:,} enregistrements) → étape ignorée")
    else:
        print(f"\n[0/3] Mapping ignoré (--skip-mapping)")

    # ── Évaluation seule ──
    if args.eval:
        print("\n[Évaluation] Comparaison Source vs Fine-tuné...")
        from src.ml.transfer import evaluate_transfer
        results = evaluate_transfer()
        print(json.dumps(results, indent=2))
        return

    # ── Phase spécifique ──
    if args.phase == 1:
        print("\n[1/1] Phase 1 — Pré-entraînement global...")
        from src.ml.transfer import phase1_pretrain
        result = phase1_pretrain()
        print(f"  ✅ Accuracy source : {result['accuracy']*100:.1f}%")
        return

    if args.phase == 2:
        print("\n[1/1] Phase 2 — Fine-tuning Sénégal...")
        from src.ml.transfer import phase2_finetune
        result = phase2_finetune(args.extra_trees)
        print(f"  ✅ Accuracy fine-tuné : {result['accuracy']*100:.1f}%")
        print(f"  ✅ Stratégie : {result['strategy']}")
        return

    # ── Pipeline complet ──
    from src.ml.transfer import run_full_transfer
    report = run_full_transfer()

    print("\n" + "=" * 60)
    print("  RÉSULTATS FINAUX")
    print("=" * 60)

    if "phase1" in report:
        p1 = report["phase1"]
        print(f"  Phase 1 (global)   → Accuracy : {p1['accuracy']*100:.1f}% | F1 : {p1['f1_weighted']:.3f}")

    if "phase2" in report:
        p2 = report["phase2"]
        print(f"  Phase 2 (Sénégal)  → Accuracy : {p2['accuracy']*100:.1f}% | F1 : {p2['f1_weighted']:.3f}")
        print(f"  Stratégie retenue  : {p2['strategy']}")

    if "evaluation" in report and "gain" in report["evaluation"]:
        g = report["evaluation"]["gain"]
        sign_acc = "+" if g["accuracy"] >= 0 else ""
        sign_f1  = "+" if g["f1_weighted"] >= 0 else ""
        print(f"\n  🎯 Gain transfer learning :")
        print(f"     Accuracy  : {sign_acc}{g['accuracy']*100:.1f}%")
        print(f"     F1-score  : {sign_f1}{g['f1_weighted']:.4f}")

    print(f"\n  📁 Modèles dans : {MODELS_DIR}")
    print(f"     gravity_source.pkl  ← modèle pré-entraîné (global)")
    print(f"     gravity_model.pkl   ← modèle final (fine-tuné Sénégal)")
    print(f"     risk_model.pkl      ← score de risque (Sénégal)")
    print(f"     transfer_report.json← rapport complet")
    print("=" * 60)


if __name__ == "__main__":
    main()
