"""
SafeRoads SN — transfer.py
Transfer learning en 2 phases pour adapter le modèle global
au contexte spécifique du Sénégal.

Stratégie :
  Phase 1 — Pré-entraînement (source domain)
      Dataset global Kaggle (132k records)
      → Apprend les patterns universels : météo, heure, véhicule → gravité
      → Sauvegarde le modèle source

  Phase 2 — Fine-tuning (target domain = Sénégal)
      Données sénégalaises disponibles (même 300-500 records suffisent)
      → Warm-start depuis le modèle source
      → Ré-entraîne en gardant les features géospatiales sénégalaises
      → Sauvegarde le modèle final adapté

  Évaluation :
      Compare source vs fine-tuned sur un test set sénégalais
      Montre le gain du transfer learning

Usage :
    python src/ml/transfer.py
    python src/ml/transfer.py --phase 1          # Pré-entraînement seul
    python src/ml/transfer.py --phase 2          # Fine-tuning seul
    python src/ml/transfer.py --eval             # Évaluation comparative
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
    r2_score, mean_absolute_error,
)

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import DATA_PROCESSED_DIR, MODELS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [TRANSFER] %(message)s")
log = logging.getLogger(__name__)

# ── Chemins ──
GLOBAL_MAPPED_CSV   = DATA_PROCESSED_DIR / "global_mapped.csv"
SENEGAL_DATASET_CSV = DATA_PROCESSED_DIR / "saferoads_dataset.csv"
SOURCE_MODEL_PATH   = MODELS_DIR / "gravity_source.pkl"
TRANSFER_MODEL_PATH = MODELS_DIR / "transfer_model.pkl"   # Modele transfer (ne remplace pas gravity_model.pkl)
FINETUNED_MODEL_PATH= MODELS_DIR / "gravity_model.pkl"    # Reference pour evaluation
RISK_MODEL_PATH     = MODELS_DIR / "risk_model.pkl"
TRANSFER_REPORT     = MODELS_DIR / "transfer_report.json"

# ── Features communes ──
# Subset universel (disponible dans les deux domaines)
UNIVERSAL_FEATURES = [
    "hour", "day_of_week", "month",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "dow_sin", "dow_cos",
    "is_weekend", "is_night",
    "is_rainy",
    "vehicle_type_enc", "cause_enc", "weather_label_enc",
    "road_type_enc", "season_enc", "period_of_day_enc",
    "driver_age", "driver_sex_enc", "driving_experience",
    "road_surface_enc", "light_enc",
    "num_vehicles", "num_victims",
]

# Features additionnelles disponibles uniquement côté Sénégal (après ETL complet)
SENEGAL_EXTRA_FEATURES = [
    "latitude", "longitude",
    "spatial_density", "nearby_accidents",
    "precipitation_mm", "windspeed_kmh",
    "temperature_c", "visibility_km", "humidity_pct",
    "is_holiday_period", "region_enc",
]

ALL_FEATURES = UNIVERSAL_FEATURES + SENEGAL_EXTRA_FEATURES

GRAVITY_LABELS = {0: "léger", 1: "grave", 2: "mortel"}


# ══════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════

def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les features dérivées si elles sont absentes."""

    # Sin/cos temporels
    if "hour" in df.columns:
        df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]  / 24)
        df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]  / 24)
    if "month" in df.columns:
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    if "day_of_week" in df.columns:
        df["dow_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7)
        if "is_weekend" not in df.columns:
            df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    if "hour" in df.columns and "is_night" not in df.columns:
        df["is_night"]  = ((df["hour"] >= 20) | (df["hour"] < 6)).astype(int)
    if "month" in df.columns and "season_enc" not in df.columns:
        df["season_enc"] = df["month"].map(lambda m: 1 if m in [6,7,8,9,10] else 0)

    # Période de la journée
    if "hour" in df.columns and "period_of_day_enc" not in df.columns:
        def period_enc(h):
            if h < 6:  return 0   # nuit
            if h < 10: return 1   # matin
            if h < 14: return 2   # milieu journée
            if h < 18: return 3   # après-midi
            if h < 22: return 4   # soirée
            return 0
        df["period_of_day_enc"] = df["hour"].apply(period_enc)

    # Encodages catégoriels simples si absents
    ENCODINGS = {
        "vehicle_type": {"Voiture":0,"Camion":1,"Moto-Jakarta":2,"Car rapide":3,
                         "Sept-places":4,"Taxi":5,"Pickup":6,"Charette":7,"Autre":8},
        "road_type":    {"autoroute":0,"nationale":1,"régionale":2,
                         "départementale":3,"urbaine":4,"piste":5,"inconnue":6},
        "cause":        {"Excès de vitesse":0,"Somnolence/fatigue":1,
                         "État dégradé route":2,"Téléphone au volant":3,
                         "Alcool":4,"Inconnue":5},
        "weather":      {"Ensoleillé":0,"Nuageux":1,"Pluie légère":2,
                         "Pluie forte":3,"Brouillard":4,"Orage":5,"Inconnu":6},
    }
    for col, enc in ENCODINGS.items():
        enc_col = f"{col}_enc" if col != "weather" else "weather_label_enc"
        if enc_col not in df.columns and col in df.columns:
            df[enc_col] = df[col].map(enc).fillna(0).astype(int)

    # Valeurs manquantes par défaut
    defaults = {
        "driver_age": 35.0, "driver_sex_enc": 0, "driving_experience": 5.0,
        "road_surface_enc": 0, "light_enc": 0, "num_vehicles": 1, "num_victims": 1,
        "is_rainy": 0, "is_holiday_period": 0, "region_enc": 0,
        "spatial_density": 0.0, "nearby_accidents": 0,
        "precipitation_mm": 0.0, "windspeed_kmh": 15.0,
        "temperature_c": 28.0, "visibility_km": 10.0, "humidity_pct": 65.0,
    }
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val

    return df


def _prepare_X_y(df: pd.DataFrame, features: list) -> tuple:
    """Prépare X et y depuis un DataFrame."""
    df = _add_derived_features(df.copy())

    # Cible
    if "gravity_enc" in df.columns:
        y = df["gravity_enc"].astype(int).clip(0, 2)
    elif "gravity" in df.columns:
        y = (df["gravity"] - 1).astype(int).clip(0, 2)
    else:
        raise ValueError("Colonne cible 'gravity' introuvable")

    available = [f for f in features if f in df.columns]
    missing   = [f for f in features if f not in df.columns]
    if missing:
        log.debug(f"Features absentes (remplies à 0) : {missing}")
        for col in missing:
            df[col] = 0

    X = df[features].copy()
    return X, y, available


def _build_pipeline(n_estimators: int = 300, lr: float = 0.05) -> Pipeline:
    if HAS_LGBM:
        model = LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=lr,
            num_leaves=31,
            max_depth=5,
            min_child_samples=10,
            subsample=0.8,
            colsample_bytree=0.7,
            class_weight="balanced",
            random_state=42,
            verbosity=-1,
            n_jobs=-1,
        )
    else:
        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=lr,
            max_depth=5,
            subsample=0.8,
            min_samples_split=20,
            random_state=42,
        )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", model),
    ])


# ══════════════════════════════════════════════════════
# PHASE 1 — PRÉ-ENTRAÎNEMENT SUR DATASET GLOBAL
# ══════════════════════════════════════════════════════

def phase1_pretrain() -> dict:
    """
    Entraîne le modèle source sur le dataset global Kaggle.
    Utilise uniquement les features universelles (disponibles dans les 2 domaines).
    """
    log.info("=" * 55)
    log.info("PHASE 1 — Pré-entraînement (domaine source global)")
    log.info("=" * 55)

    if not GLOBAL_MAPPED_CSV.exists():
        raise FileNotFoundError(
            f"Dataset global mappé non trouvé : {GLOBAL_MAPPED_CSV}\n"
            "Lancer : python -m src.etl.mapper_global"
        )

    df = pd.read_csv(GLOBAL_MAPPED_CSV, low_memory=False)
    log.info(f"Dataset global : {len(df):,} enregistrements")

    X, y, used = _prepare_X_y(df, UNIVERSAL_FEATURES)
    log.info(f"Features utilisées ({len(used)}) : {used}")
    log.info(f"Distribution classes : {pd.Series(y).value_counts().sort_index().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = _build_pipeline(n_estimators=200, lr=0.08)
    log.info("Entraînement Phase 1...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred, average="weighted")

    log.info(f"  Accuracy  : {acc*100:.1f}%")
    log.info(f"  F1-score  : {f1:.4f}")
    log.info(f"\n{classification_report(y_test, y_pred, target_names=['Léger','Grave','Mortel'])}")

    # Sauvegarder le modèle source
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "pipeline":       pipeline,
        "feature_names":  used,
        "domain":         "global",
        "n_train":        len(X_train),
        "accuracy":       round(acc, 4),
        "trained_at":     datetime.now().isoformat(),
    }, SOURCE_MODEL_PATH)
    log.info(f"  Modèle source sauvegardé : {SOURCE_MODEL_PATH}")

    return {
        "phase": 1, "domain": "global",
        "n_samples": len(df), "n_features": len(used),
        "accuracy": round(acc, 4), "f1_weighted": round(f1, 4),
    }


# ══════════════════════════════════════════════════════
# PHASE 2 — FINE-TUNING SUR DONNÉES SÉNÉGAL
# ══════════════════════════════════════════════════════

def phase2_finetune(n_extra_estimators: int = 100) -> dict:
    """
    Fine-tuning via Feature Augmentation :
    - Utilise le modele source (global) pour generer predict_proba sur les donnees Senegal
    - Les 3 probabilites deviennent des features supplementaires (global_prob_leger/grave/mortel)
    - Entraine un nouveau LightGBM sur features Senegal + 3 priors globaux
    - Sauvegarde en transfer_model.pkl (ne remplace PAS gravity_model.pkl)
    """
    log.info("=" * 55)
    log.info("PHASE 2 — Feature Augmentation (domaine cible : Senegal)")
    log.info("=" * 55)

    # ── Charger le modele source ──
    if not SOURCE_MODEL_PATH.exists():
        log.warning("Modele source absent -> lancement Phase 1 d'abord")
        phase1_pretrain()

    source = joblib.load(SOURCE_MODEL_PATH)
    pipeline_source = source["pipeline"]
    source_features = source["feature_names"]
    log.info(f"Modele source charge : {source['n_train']:,} exemples, acc={source['accuracy']}")

    # ── Charger donnees senegalaises ──
    if not SENEGAL_DATASET_CSV.exists():
        raise FileNotFoundError(
            f"Dataset Senegal non trouve : {SENEGAL_DATASET_CSV}\n"
            "Lancer : python scripts/run_etl.py"
        )

    df_sn = pd.read_csv(SENEGAL_DATASET_CSV, low_memory=False)
    log.info(f"Dataset Senegal : {len(df_sn):,} enregistrements")

    # ── Preparer features de base ──
    X_sn, y_sn, used_sn = _prepare_X_y(df_sn, ALL_FEATURES)
    log.info(f"Features Senegal ({len(used_sn)}) : {len(used_sn)} colonnes")

    # ── Feature Augmentation : predict_proba du modele source ──
    log.info("  Generation des priors globaux (predict_proba)...")
    universal_available = [f for f in source_features if f in X_sn.columns]
    missing_source = [f for f in source_features if f not in X_sn.columns]
    X_source_input = X_sn[universal_available].copy()
    for col in missing_source:
        X_source_input[col] = 0
    X_source_input = X_source_input[source_features]

    # Appliquer imputer + predict_proba
    imputer_source = pipeline_source.named_steps["imputer"]
    model_source = pipeline_source.named_steps["model"]
    X_imp = imputer_source.transform(X_source_input)
    probas = model_source.predict_proba(X_imp)

    # Ajouter les 3 colonnes de probabilites
    X_augmented = X_sn.copy()
    X_augmented["global_prob_leger"] = probas[:, 0]
    X_augmented["global_prob_grave"] = probas[:, 1]
    X_augmented["global_prob_mortel"] = probas[:, 2] if probas.shape[1] > 2 else 0.0

    augmented_features = list(X_sn.columns) + ["global_prob_leger", "global_prob_grave", "global_prob_mortel"]
    log.info(f"  Features augmentees : {len(augmented_features)} ({len(used_sn)} + 3 priors)")

    # ── Split ──
    X_train, X_test, y_train, y_test = train_test_split(
        X_augmented[augmented_features], y_sn, test_size=0.2, random_state=42,
        stratify=y_sn if y_sn.nunique() > 1 else None,
    )

    # ── Entrainer le modele augmente ──
    log.info("  Entrainement LightGBM avec features augmentees...")
    pipeline_aug = _build_pipeline(n_estimators=300, lr=0.05)
    pipeline_aug.fit(X_train, y_train)

    y_pred = pipeline_aug.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred).tolist()

    log.info(f"  Accuracy  : {acc*100:.1f}%")
    log.info(f"  F1-score  : {f1:.4f}")
    log.info(f"\n{classification_report(y_test, y_pred, target_names=['Leger','Grave','Mortel'])}")

    # ── Cross-validation ──
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_pipeline = _build_pipeline(n_estimators=300, lr=0.05)
    cv_scores = cross_val_score(cv_pipeline, X_augmented[augmented_features], y_sn, cv=skf, scoring="accuracy")
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    log.info(f"  CV Accuracy : {cv_mean*100:.1f}% +/- {cv_std*100:.1f}%")

    # ── Sauvegarder comme transfer_model.pkl ──
    TRANSFER_MODEL_PATH = MODELS_DIR / "transfer_model.pkl"
    joblib.dump({
        "pipeline":      pipeline_aug,
        "feature_names": augmented_features,
        "classes":       [0, 1, 2],
        "class_labels":  ["leger", "grave", "mortel"],
        "domain":        "senegal_transfer",
        "strategy":      "feature_augmentation",
        "source_model":  str(SOURCE_MODEL_PATH.name),
        "n_train":       len(X_train),
        "accuracy":      round(acc, 4),
        "f1_weighted":   round(f1, 4),
        "trained_at":    datetime.now().isoformat(),
    }, TRANSFER_MODEL_PATH)
    log.info(f"  Modele transfer sauvegarde : {TRANSFER_MODEL_PATH}")

    # ── Sauvegarder metriques ──
    metrics = {
        "model": "LightGBM + Feature Augmentation" if HAS_LGBM else "GBC + Feature Augmentation",
        "accuracy": round(acc, 4),
        "f1_weighted": round(f1, 4),
        "cv_mean": round(cv_mean, 4),
        "cv_std": round(cv_std, 4),
        "confusion_matrix": cm,
        "n_features": len(augmented_features),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "strategy": "feature_augmentation",
        "source_accuracy": source["accuracy"],
    }
    metrics_path = MODELS_DIR / "transfer_model_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"  Metriques sauvegardees : {metrics_path}")

    # ── Modele de risque re-entraine sur Senegal ──
    _retrain_risk_model(df_sn, used_sn)

    return {
        "phase": 2, "domain": "senegal",
        "strategy": "feature_augmentation",
        "n_samples": len(df_sn),
        "n_features": len(augmented_features),
        "accuracy": round(acc, 4),
        "f1_weighted": round(f1, 4),
        "cv_mean": round(cv_mean, 4),
        "source_accuracy": source["accuracy"],
        "gain": round(acc - source["accuracy"], 4),
    }


def _retrain_risk_model(df_sn: pd.DataFrame, features: list):
    """Re-entraîne le modèle de score de risque sur les données sénégalaises."""
    log.info("  Re-entraînement modèle score de risque...")

    df_sn = _add_derived_features(df_sn.copy())

    # Score de risque
    g = df_sn.get("gravity", 2)
    d = df_sn.get("spatial_density", 0)
    d_norm = (d - d.min()) / (d.max() - d.min() + 1e-9)
    r = df_sn.get("is_rainy", False).astype(float)
    n = df_sn.get("is_night", 0).astype(float)
    w = df_sn.get("is_weekend", 0).astype(float)
    y_risk = (40 * (g - 1) / 2 + 30 * d_norm + 10 * r + 8 * n + 7 * w + 5).clip(0, 100)

    available = [f for f in features if f in df_sn.columns]
    X = df_sn[available].fillna(0)
    X_train, _, y_train, _ = train_test_split(X, y_risk, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(n_estimators=150, max_depth=10,
                                        n_jobs=-1, random_state=42)),
    ])
    pipeline.fit(X_train, y_train)

    joblib.dump({
        "pipeline":      pipeline,
        "feature_names": available,
        "target":        "risk_score",
        "domain":        "senegal",
        "trained_at":    datetime.now().isoformat(),
    }, RISK_MODEL_PATH)
    log.info(f"  ✅ Modèle risque sauvegardé : {RISK_MODEL_PATH}")


# ══════════════════════════════════════════════════════
# ÉVALUATION COMPARATIVE
# ══════════════════════════════════════════════════════

def evaluate_transfer() -> dict:
    """
    Compare les performances du modèle source vs fine-tuné
    sur un test set sénégalais.
    """
    log.info("=" * 55)
    log.info("ÉVALUATION — Source vs Fine-tuné sur données Sénégal")
    log.info("=" * 55)

    if not SENEGAL_DATASET_CSV.exists():
        raise FileNotFoundError(f"Dataset Sénégal non trouvé : {SENEGAL_DATASET_CSV}")

    df_sn = pd.read_csv(SENEGAL_DATASET_CSV, low_memory=False)
    results = {}

    for name, path in [("source", SOURCE_MODEL_PATH), ("transfer", TRANSFER_MODEL_PATH), ("finetuned", FINETUNED_MODEL_PATH)]:
        if not path.exists():
            log.warning(f"Modèle {name} absent : {path}")
            continue

        m        = joblib.load(path)
        features = m["feature_names"]
        X, y, _  = _prepare_X_y(df_sn, features)

        _, X_test, _, y_test = train_test_split(X, y, test_size=0.3, random_state=99)

        pipeline = m["pipeline"]
        y_pred   = pipeline.predict(X_test)
        acc      = accuracy_score(y_test, y_pred)
        f1       = f1_score(y_test, y_pred, average="weighted")

        results[name] = {
            "accuracy":    round(acc, 4),
            "f1_weighted": round(f1, 4),
            "domain":      m.get("domain", "unknown"),
            "strategy":    m.get("strategy", "—"),
        }

        log.info(f"\n  [{name}] Accuracy={acc*100:.1f}% | F1={f1:.4f}")
        log.info(f"\n{classification_report(y_test, y_pred, target_names=['Léger','Grave','Mortel'])}")

    # Gain du transfer learning
    if "source" in results and "finetuned" in results:
        gain_acc = results["finetuned"]["accuracy"] - results["source"]["accuracy"]
        gain_f1  = results["finetuned"]["f1_weighted"] - results["source"]["f1_weighted"]
        log.info(f"\n  Gain transfer learning :")
        log.info(f"    Accuracy : {'+' if gain_acc >= 0 else ''}{gain_acc*100:.1f}%")
        log.info(f"    F1-score : {'+' if gain_f1 >= 0 else ''}{gain_f1:.4f}")
        results["gain"] = {"accuracy": round(gain_acc, 4), "f1_weighted": round(gain_f1, 4)}

    return results


# ══════════════════════════════════════════════════════
# PIPELINE COMPLET
# ══════════════════════════════════════════════════════

def run_full_transfer() -> dict:
    """Lance Phase 1 + Phase 2 + Évaluation en séquence."""
    log.info("\n" + "🔄 " * 20)
    log.info("TRANSFER LEARNING COMPLET — SafeRoads SN")
    log.info("🔄 " * 20 + "\n")

    report = {"started_at": datetime.now().isoformat()}

    report["phase1"] = phase1_pretrain()
    report["phase2"] = phase2_finetune()
    report["evaluation"] = evaluate_transfer()
    report["completed_at"] = datetime.now().isoformat()

    # Sauvegarder le rapport
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(TRANSFER_REPORT, "w") as f:
        json.dump(report, f, indent=2)

    log.info(f"\n  📄 Rapport complet : {TRANSFER_REPORT}")
    log.info("\n" + "✅ " * 20)
    log.info("Transfer learning terminé !")
    log.info("✅ " * 20)

    return report


# ══════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SafeRoads SN — Transfer Learning")
    parser.add_argument("--phase", type=int, choices=[1, 2],
                        help="Lancer une seule phase (1=pré-entraînement, 2=fine-tuning)")
    parser.add_argument("--eval", action="store_true",
                        help="Évaluation comparative uniquement")
    parser.add_argument("--extra-trees", type=int, default=100,
                        help="Nombre d'arbres additionnels pour le fine-tuning (défaut: 100)")
    args = parser.parse_args()

    if args.eval:
        results = evaluate_transfer()
        print(json.dumps(results, indent=2))
    elif args.phase == 1:
        result = phase1_pretrain()
        print(json.dumps(result, indent=2))
    elif args.phase == 2:
        result = phase2_finetune(args.extra_trees)
        print(json.dumps(result, indent=2))
    else:
        report = run_full_transfer()
        print(f"\nTransfer learning termine")
        if "evaluation" in report and "gain" in report["evaluation"]:
            g = report["evaluation"]["gain"]
            print(f"   Gain accuracy : {'+' if g['accuracy']>=0 else ''}{g['accuracy']*100:.1f}%")
            print(f"   Gain F1-score : {'+' if g['f1_weighted']>=0 else ''}{g['f1_weighted']:.4f}")
