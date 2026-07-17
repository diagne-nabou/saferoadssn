"""
SafeRoads SN — train.py
Entraine 8 modeles sur le dataset fusionne :
  - XGBoost, LightGBM, RandomForest, SVM, MLP (classification 3 classes)
  - Stacking Ensemble (meta-learner sur les 4 classifieurs)
  - Binaire (mortel vs non-mortel)
  - RandomForest Regression (score de risque)
  - DBSCAN (clustering hotspots)

Usage :
    python src/ml/train.py
    python src/ml/train.py --model gravity
    python src/ml/train.py --model lgbm
    python src/ml/train.py --model stacking
    python src/ml/train.py --model binary
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import (
    GradientBoostingClassifier, RandomForestRegressor,
    RandomForestClassifier, StackingClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score,
    mean_absolute_error, r2_score, mean_squared_error,
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import DATA_PROCESSED_DIR, MODELS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [ML] %(message)s")
log = logging.getLogger(__name__)

FINAL_DATASET = DATA_PROCESSED_DIR / "saferoads_dataset.csv"

# ── Features communes aux deux modèles ──
BASE_FEATURES = [
    # Géospatial
    "latitude", "longitude", "spatial_density", "nearby_accidents",
    # Temporel
    "hour", "day_of_week", "month",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos",
    "is_weekend", "is_night", "is_holiday_period",
    # Météo
    "is_rainy", "precipitation_mm", "windspeed_kmh",
    "temperature_c", "visibility_km", "humidity_pct",
    # Catégorielles encodées
    "vehicle_type_enc", "road_type_enc", "cause_enc",
    "region_enc", "weather_label_enc", "season_enc", "period_of_day_enc",
]

# ── Cibles ──
TARGET_GRAVITY    = "gravity_enc"   # classification  : 0=léger, 1=grave, 2=mortel
TARGET_RISK_SCORE = "risk_score"    # régression      : 0-100


# ══════════════════════════════════════════════════════
# UTILITAIRES
# ══════════════════════════════════════════════════════

def load_dataset() -> pd.DataFrame:
    if not FINAL_DATASET.exists():
        raise FileNotFoundError(
            f"Dataset non trouvé : {FINAL_DATASET}\n"
            "Lancer : python scripts/run_etl.py"
        )
    log.info(f"Chargement dataset : {FINAL_DATASET}")
    df = pd.read_csv(FINAL_DATASET)
    log.info(f"  {len(df):,} lignes × {len(df.columns)} colonnes")
    return df


def select_features(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """Sélectionne uniquement les features disponibles dans le dataset."""
    available = [f for f in features if f in df.columns]
    missing   = [f for f in features if f not in df.columns]
    if missing:
        log.warning(f"  Features absentes (ignorées) : {missing}")
    log.info(f"  {len(available)} features sélectionnées")
    return df[available].copy()


def build_risk_score(df: pd.DataFrame) -> pd.Series:
    """
    Construit un score de risque continu 0-100 à partir des données disponibles.
    Combinaison pondérée : gravité + densité spatiale + conditions météo.
    """
    score = pd.Series(0.0, index=df.index)

    # Gravité (40%)
    if "gravity" in df.columns:
        g_norm = (df["gravity"] - 1) / 2  # 0-1
        score += 40 * g_norm

    # Densité spatiale (30%)
    if "spatial_density" in df.columns:
        d = df["spatial_density"]
        d_norm = (d - d.min()) / (d.max() - d.min() + 1e-9)
        score += 30 * d_norm

    # Conditions météo (15%)
    if "is_rainy" in df.columns:
        score += 10 * df["is_rainy"].astype(float)
    if "windspeed_kmh" in df.columns:
        w_norm = (df["windspeed_kmh"] / 100).clip(0, 1)
        score += 5 * w_norm

    # Période (15%) : nuit et week-end plus risqués
    if "is_night" in df.columns:
        score += 8 * df["is_night"].astype(float)
    if "is_weekend" in df.columns:
        score += 7 * df["is_weekend"].astype(float)

    return score.clip(0, 100).round(2)


def save_metrics(metrics: dict, model_name: str):
    """Sauvegarde les métriques en JSON."""
    path = MODELS_DIR / f"{model_name}_metrics.json"
    metrics["trained_at"] = datetime.now().isoformat()
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"  Métriques sauvegardées : {path}")


def plot_feature_importance(model, feature_names: list, model_name: str, top_n: int = 15):
    """Génère et sauvegarde le graphe d'importance des features."""
    try:
        if hasattr(model, "named_steps"):
            estimator = model.named_steps.get("model", model)
        else:
            estimator = model

        importances = estimator.feature_importances_
        indices = np.argsort(importances)[-top_n:]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(
            [feature_names[i] for i in indices],
            importances[indices],
            color="#2E86AB",
        )
        ax.set_xlabel("Importance")
        ax.set_title(f"Top {top_n} features — {model_name}")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()

        path = MODELS_DIR / f"{model_name}_feature_importance.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        log.info(f"  Graphe sauvegardé : {path}")
    except Exception as e:
        log.warning(f"  Impossible de générer le graphe : {e}")


# ══════════════════════════════════════════════════════
# MODÈLE 1 — GRAVITÉ (Classification)
# ══════════════════════════════════════════════════════

def train_gravity_model(df: pd.DataFrame) -> dict:
    """
    XGBoostClassifier pour prédire la gravité.
    Fallback GradientBoosting si XGBoost non installé.
    Cible : 0=léger, 1=grave, 2=mortel
    """
    log.info("\n" + "=" * 50)
    log.info("MODELE 1 -- Gravite (XGBoost Classification)")
    log.info("=" * 50)

    X = select_features(df, BASE_FEATURES)
    feature_names = list(X.columns)

    # Construire ou récupérer la cible
    if TARGET_GRAVITY in df.columns:
        y = df[TARGET_GRAVITY].astype(int)
    elif "gravity" in df.columns:
        y = (df["gravity"] - 1).astype(int).clip(0, 2)
    else:
        raise ValueError("Colonne cible 'gravity' / 'gravity_enc' introuvable")

    log.info(f"  Distribution classes : {y.value_counts().sort_index().to_dict()}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    log.info(f"  Train : {len(X_train):,} | Test : {len(X_test):,}")

    # Calculer les poids de classes (desequilibre : classe 1/grave tres sous-representee)
    from sklearn.utils.class_weight import compute_sample_weight
    sample_weights = compute_sample_weight("balanced", y_train)
    log.info("  Poids de classes appliques (balanced) pour compenser le desequilibre")

    # Pipeline : imputation + XGBoost (ou fallback GradientBoosting)
    if HAS_XGBOOST:
        # Calculer scale_pos_weight pour chaque classe
        class_counts = y_train.value_counts().sort_index()
        max_count = class_counts.max()
        scale_weights = {c: max_count / max(cnt, 1) for c, cnt in class_counts.items()}
        log.info(f"  Poids par classe : {scale_weights}")

        model_algo = XGBClassifier(
            n_estimators=500,
            learning_rate=0.01,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=5,
            reg_alpha=0.5,
            reg_lambda=2.0,
            gamma=0.2,
            random_state=42,
            use_label_encoder=False,
            eval_metric="mlogloss",
            verbosity=0,
        )
        model_name_str = "XGBClassifier"
        log.info("  Algorithme : XGBoost (optimise)")
    else:
        model_algo = GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            min_samples_split=15,
            random_state=42,
            verbose=0,
        )
        model_name_str = "GradientBoostingClassifier"
        log.info("  Algorithme : GradientBoosting (fallback, xgboost non installe)")

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   model_algo),
    ])

    log.info("  Entrainement en cours...")
    pipeline.fit(X_train, y_train, model__sample_weight=sample_weights)

    # Évaluation
    y_pred = pipeline.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred, average="weighted")

    log.info(f"  Accuracy  : {acc:.4f} ({acc*100:.1f}%)")
    log.info(f"  F1-score  : {f1:.4f}")
    log.info(f"\n{classification_report(y_test, y_pred, target_names=['Leger','Grave','Mortel'])}")

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    log.info(f"  CV Accuracy : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)

    # Sauvegarder
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "gravity_model.pkl"
    joblib.dump({
        "pipeline":      pipeline,
        "feature_names": feature_names,
        "classes":       [0, 1, 2],
        "class_labels":  ["léger", "grave", "mortel"],
        "trained_at":    datetime.now().isoformat(),
    }, model_path)
    log.info(f"  Modele sauvegarde : {model_path}")

    metrics = {
        "model":       model_name_str,
        "task":        "classification",
        "target":      "gravity",
        "n_train":     len(X_train),
        "n_test":      len(X_test),
        "n_features":  len(feature_names),
        "accuracy":    round(acc, 4),
        "f1_weighted": round(f1, 4),
        "cv_mean":     round(cv_scores.mean(), 4),
        "cv_std":      round(cv_scores.std(), 4),
        "confusion_matrix": cm.tolist(),
    }
    save_metrics(metrics, "gravity_model")
    plot_feature_importance(pipeline, feature_names, "gravity_model")

    return metrics


# ══════════════════════════════════════════════════════
# MODÈLE 2 — SCORE DE RISQUE (Régression)
# ══════════════════════════════════════════════════════

def train_risk_model(df: pd.DataFrame) -> dict:
    """
    RandomForestRegressor pour prédire le score de risque géospatial (0-100).
    """
    log.info("\n" + "─" * 50)
    log.info("MODÈLE 2 — Score de risque (RandomForest Régression)")
    log.info("─" * 50)

    X = select_features(df, BASE_FEATURES)
    feature_names = list(X.columns)

    # Construire le score de risque
    log.info("  Construction du score de risque...")
    y = build_risk_score(df)
    log.info(f"  Score moyen : {y.mean():.1f} | Écart-type : {y.std():.1f}")
    log.info(f"  Min : {y.min():.1f} | Max : {y.max():.1f}")

    # Split (pas de stratify pour la régression)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    log.info(f"  Train : {len(X_train):,} | Test : {len(X_test):,}")

    # Pipeline
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features="sqrt",
            n_jobs=-1,
            random_state=42,
        )),
    ])

    log.info("  Entraînement en cours...")
    pipeline.fit(X_train, y_train)

    # Évaluation
    y_pred = pipeline.predict(X_test)
    r2     = r2_score(y_test, y_pred)
    mae    = mean_absolute_error(y_test, y_pred)
    rmse   = np.sqrt(mean_squared_error(y_test, y_pred))

    log.info(f"  R²   : {r2:.4f}")
    log.info(f"  MAE  : {mae:.2f} points")
    log.info(f"  RMSE : {rmse:.2f} points")

    # Cross-validation
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="r2", n_jobs=-1)
    log.info(f"  CV R² : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Sauvegarder
    model_path = MODELS_DIR / "risk_model.pkl"
    joblib.dump({
        "pipeline":      pipeline,
        "feature_names": feature_names,
        "target":        "risk_score",
        "score_range":   [0, 100],
        "trained_at":    datetime.now().isoformat(),
    }, model_path)
    log.info(f"  Modèle sauvegardé : {model_path}")

    metrics = {
        "model":      "RandomForestRegressor",
        "task":       "regression",
        "target":     "risk_score",
        "n_train":    len(X_train),
        "n_test":     len(X_test),
        "n_features": len(feature_names),
        "r2":         round(r2, 4),
        "mae":        round(mae, 2),
        "rmse":       round(rmse, 2),
        "cv_r2_mean": round(cv_scores.mean(), 4),
        "cv_r2_std":  round(cv_scores.std(), 4),
    }
    save_metrics(metrics, "risk_model")
    plot_feature_importance(pipeline, feature_names, "risk_model")

    return metrics


# ══════════════════════════════════════════════════════
# MODÈLE 3 — MLP Deep Learning (Classification)
# ══════════════════════════════════════════════════════

def train_mlp_model(df: pd.DataFrame) -> dict:
    """
    MLPClassifier (réseau de neurones) pour prédire la gravité.
    Architecture : 2 couches cachées (128, 64) avec activation ReLU.
    Cible : 0=léger, 1=grave, 2=mortel
    """
    log.info("\n" + "=" * 50)
    log.info("MODELE 3 -- Gravite (MLP / Deep Learning)")
    log.info("=" * 50)

    X = select_features(df, BASE_FEATURES)
    feature_names = list(X.columns)

    # Construire la cible
    if TARGET_GRAVITY in df.columns:
        y = df[TARGET_GRAVITY].astype(int)
    elif "gravity" in df.columns:
        y = (df["gravity"] - 1).astype(int).clip(0, 2)
    else:
        raise ValueError("Colonne cible 'gravity' / 'gravity_enc' introuvable")

    log.info(f"  Distribution classes : {y.value_counts().sort_index().to_dict()}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    log.info(f"  Train : {len(X_train):,} | Test : {len(X_test):,}")

    # Pipeline : imputation + normalisation + MLP
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            solver="adam",
            alpha=0.01,
            learning_rate="adaptive",
            learning_rate_init=0.001,
            max_iter=800,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=30,
            batch_size=32,
            random_state=42,
            verbose=False,
        )),
    ])

    # MLP ne supporte pas sample_weight -> sureechantillonnage de la classe minoritaire
    from sklearn.utils import resample
    X_train_res = X_train.copy()
    X_train_res["_target"] = y_train.values
    class_counts = y_train.value_counts()
    max_count = class_counts.max()
    frames = []
    for cls in class_counts.index:
        cls_data = X_train_res[X_train_res["_target"] == cls]
        if len(cls_data) < max_count:
            cls_upsampled = resample(cls_data, replace=True, n_samples=max_count, random_state=42)
            frames.append(cls_upsampled)
        else:
            frames.append(cls_data)
    X_train_bal = pd.concat(frames).sample(frac=1, random_state=42)
    y_train_bal = X_train_bal.pop("_target")
    log.info(f"  Sureechantillonnage : {len(X_train)} -> {len(X_train_bal)} (classes equilibrees)")

    log.info("  Entrainement MLP (256-128-64 neurones, ReLU, Adam, balanced)...")
    pipeline.fit(X_train_bal, y_train_bal)

    # Évaluation
    y_pred = pipeline.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred, average="weighted")

    log.info(f"  Accuracy  : {acc:.4f} ({acc*100:.1f}%)")
    log.info(f"  F1-score  : {f1:.4f}")
    log.info(f"\n{classification_report(y_test, y_pred, target_names=['Leger','Grave','Mortel'])}")

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    log.info(f"  CV Accuracy : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)

    # Nombre d'époques effectuées
    mlp_model = pipeline.named_steps["model"]
    n_epochs = mlp_model.n_iter_
    log.info(f"  Epoques : {n_epochs}")

    # Sauvegarder
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "mlp_model.pkl"
    joblib.dump({
        "pipeline":      pipeline,
        "feature_names": feature_names,
        "classes":       [0, 1, 2],
        "class_labels":  ["léger", "grave", "mortel"],
        "architecture":  "MLP (256, 128, 64)",
        "trained_at":    datetime.now().isoformat(),
    }, model_path)
    log.info(f"  Modele sauvegarde : {model_path}")

    metrics = {
        "model":       "MLPClassifier",
        "task":        "classification",
        "target":      "gravity",
        "architecture": "(256, 128, 64) ReLU",
        "n_train":     len(X_train),
        "n_test":      len(X_test),
        "n_features":  len(feature_names),
        "accuracy":    round(acc, 4),
        "f1_weighted": round(f1, 4),
        "cv_mean":     round(cv_scores.mean(), 4),
        "cv_std":      round(cv_scores.std(), 4),
        "n_epochs":    int(n_epochs),
        "confusion_matrix": cm.tolist(),
    }
    save_metrics(metrics, "mlp_model")

    return metrics


# ══════════════════════════════════════════════════════
# MODÈLE 4 — LightGBM (Classification)
# ══════════════════════════════════════════════════════

def train_lightgbm_model(df: pd.DataFrame) -> dict:
    """LightGBM pour predire la gravite. Leaf-wise, class_weight balanced."""
    log.info("\n" + "=" * 50)
    log.info("MODELE 4 -- Gravite (LightGBM)")
    log.info("=" * 50)

    if not HAS_LGBM:
        log.warning("  LightGBM non installe -> skip")
        return {}

    X = select_features(df, BASE_FEATURES)
    feature_names = list(X.columns)

    if TARGET_GRAVITY in df.columns:
        y = df[TARGET_GRAVITY].astype(int)
    elif "gravity" in df.columns:
        y = (df["gravity"] - 1).astype(int).clip(0, 2)
    else:
        raise ValueError("Colonne cible 'gravity' introuvable")

    log.info(f"  Distribution classes : {y.value_counts().sort_index().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    log.info(f"  Train : {len(X_train)} | Test : {len(X_test)}")

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=4,
            min_child_samples=10,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.3,
            reg_lambda=1.5,
            class_weight="balanced",
            random_state=42,
            verbosity=-1,
            n_jobs=-1,
        )),
    ])

    log.info("  Entrainement LightGBM...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred)

    log.info(f"  Accuracy  : {acc:.4f} ({acc*100:.1f}%)")
    log.info(f"  F1-score  : {f1:.4f}")
    log.info(f"\n{classification_report(y_test, y_pred, target_names=['Leger','Grave','Mortel'])}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    log.info(f"  CV Accuracy : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "lightgbm_model.pkl"
    joblib.dump({
        "pipeline": pipeline, "feature_names": feature_names,
        "classes": [0, 1, 2], "class_labels": ["leger", "grave", "mortel"],
        "trained_at": datetime.now().isoformat(),
    }, model_path)
    log.info(f"  Modele sauvegarde : {model_path}")

    metrics = {
        "model": "LGBMClassifier", "task": "classification", "target": "gravity",
        "n_train": len(X_train), "n_test": len(X_test), "n_features": len(feature_names),
        "accuracy": round(acc, 4), "f1_weighted": round(f1, 4),
        "cv_mean": round(cv_scores.mean(), 4), "cv_std": round(cv_scores.std(), 4),
        "confusion_matrix": cm.tolist(),
    }
    save_metrics(metrics, "lightgbm_model")
    plot_feature_importance(pipeline, feature_names, "lightgbm_model")
    return metrics


# ══════════════════════════════════════════════════════
# MODÈLE 5 — RandomForest (Classification)
# ══════════════════════════════════════════════════════

def train_rf_classifier_model(df: pd.DataFrame) -> dict:
    """RandomForest pour predire la gravite. balanced_subsample."""
    log.info("\n" + "=" * 50)
    log.info("MODELE 5 -- Gravite (RandomForest)")
    log.info("=" * 50)

    X = select_features(df, BASE_FEATURES)
    feature_names = list(X.columns)

    if TARGET_GRAVITY in df.columns:
        y = df[TARGET_GRAVITY].astype(int)
    elif "gravity" in df.columns:
        y = (df["gravity"] - 1).astype(int).clip(0, 2)
    else:
        raise ValueError("Colonne cible 'gravity' introuvable")

    log.info(f"  Distribution classes : {y.value_counts().sort_index().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    log.info(f"  Train : {len(X_train)} | Test : {len(X_test)}")

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=500,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    log.info("  Entrainement RandomForest...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred)

    log.info(f"  Accuracy  : {acc:.4f} ({acc*100:.1f}%)")
    log.info(f"  F1-score  : {f1:.4f}")
    log.info(f"\n{classification_report(y_test, y_pred, target_names=['Leger','Grave','Mortel'])}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    log.info(f"  CV Accuracy : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "rf_classifier_model.pkl"
    joblib.dump({
        "pipeline": pipeline, "feature_names": feature_names,
        "classes": [0, 1, 2], "class_labels": ["leger", "grave", "mortel"],
        "trained_at": datetime.now().isoformat(),
    }, model_path)
    log.info(f"  Modele sauvegarde : {model_path}")

    metrics = {
        "model": "RandomForestClassifier", "task": "classification", "target": "gravity",
        "n_train": len(X_train), "n_test": len(X_test), "n_features": len(feature_names),
        "accuracy": round(acc, 4), "f1_weighted": round(f1, 4),
        "cv_mean": round(cv_scores.mean(), 4), "cv_std": round(cv_scores.std(), 4),
        "confusion_matrix": cm.tolist(),
    }
    save_metrics(metrics, "rf_classifier_model")
    plot_feature_importance(pipeline, feature_names, "rf_classifier_model")
    return metrics


# ══════════════════════════════════════════════════════
# MODÈLE 6 — Stacking Ensemble
# ══════════════════════════════════════════════════════

def train_stacking_model(df: pd.DataFrame) -> dict:
    """Stacking : XGBoost + LightGBM + RF + SVM -> LogisticRegression meta-learner."""
    log.info("\n" + "=" * 50)
    log.info("MODELE 6 -- Gravite (Stacking Ensemble)")
    log.info("=" * 50)

    X = select_features(df, BASE_FEATURES)
    feature_names = list(X.columns)

    if TARGET_GRAVITY in df.columns:
        y = df[TARGET_GRAVITY].astype(int)
    elif "gravity" in df.columns:
        y = (df["gravity"] - 1).astype(int).clip(0, 2)
    else:
        raise ValueError("Colonne cible 'gravity' introuvable")

    log.info(f"  Distribution classes : {y.value_counts().sort_index().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    log.info(f"  Train : {len(X_train)} | Test : {len(X_test)}")

    # Construire les estimateurs de base
    from sklearn.utils.class_weight import compute_sample_weight
    sample_weights = compute_sample_weight("balanced", y_train)

    estimators = []

    if HAS_XGBOOST:
        estimators.append(("xgb", XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=3,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
            reg_alpha=0.5, reg_lambda=2.0, gamma=0.2,
            random_state=42, use_label_encoder=False,
            eval_metric="mlogloss", verbosity=0,
        )))

    if HAS_LGBM:
        estimators.append(("lgbm", LGBMClassifier(
            n_estimators=200, learning_rate=0.05, num_leaves=31,
            max_depth=4, min_child_samples=10, subsample=0.8,
            colsample_bytree=0.7, reg_alpha=0.3, reg_lambda=1.5,
            class_weight="balanced", random_state=42, verbosity=-1,
        )))

    estimators.append(("rf", RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_split=10,
        min_samples_leaf=5, max_features="sqrt",
        class_weight="balanced_subsample", random_state=42, n_jobs=-1,
    )))

    estimators.append(("svm", make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced",
            probability=True, random_state=42),
    )))

    log.info(f"  Base estimators : {[e[0] for e in estimators]}")

    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(
            C=1.0, max_iter=1000, class_weight="balanced", random_state=42,
        ),
        cv=5,
        stack_method="predict_proba",
        n_jobs=-1,
    )

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", stacking),
    ])

    log.info("  Entrainement Stacking Ensemble...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred)

    log.info(f"  Accuracy  : {acc:.4f} ({acc*100:.1f}%)")
    log.info(f"  F1-score  : {f1:.4f}")
    log.info(f"\n{classification_report(y_test, y_pred, target_names=['Leger','Grave','Mortel'])}")

    cv_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv_fold, scoring="accuracy", n_jobs=-1)
    log.info(f"  CV Accuracy : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "stacking_model.pkl"
    joblib.dump({
        "pipeline": pipeline, "feature_names": feature_names,
        "classes": [0, 1, 2], "class_labels": ["leger", "grave", "mortel"],
        "trained_at": datetime.now().isoformat(),
    }, model_path)
    log.info(f"  Modele sauvegarde : {model_path}")

    metrics = {
        "model": "StackingClassifier", "task": "classification", "target": "gravity",
        "n_train": len(X_train), "n_test": len(X_test), "n_features": len(feature_names),
        "accuracy": round(acc, 4), "f1_weighted": round(f1, 4),
        "cv_mean": round(cv_scores.mean(), 4), "cv_std": round(cv_scores.std(), 4),
        "confusion_matrix": cm.tolist(),
    }
    save_metrics(metrics, "stacking_model")
    return metrics


# ══════════════════════════════════════════════════════
# MODÈLE 7 — Binaire (mortel vs non-mortel)
# ══════════════════════════════════════════════════════

def train_binary_gravity_model(df: pd.DataFrame) -> dict:
    """Classification binaire : mortel (1) vs non-mortel (0)."""
    log.info("\n" + "=" * 50)
    log.info("MODELE 7 -- Binaire (mortel vs non-mortel)")
    log.info("=" * 50)

    X = select_features(df, BASE_FEATURES)
    feature_names = list(X.columns)

    if "gravity" in df.columns:
        y = (df["gravity"] == 3).astype(int)  # 1=mortel, 0=non-mortel
    elif TARGET_GRAVITY in df.columns:
        y = (df[TARGET_GRAVITY] == 2).astype(int)
    else:
        raise ValueError("Colonne cible 'gravity' introuvable")

    log.info(f"  Distribution : non-mortel={int((y==0).sum())}, mortel={int((y==1).sum())}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    log.info(f"  Train : {len(X_train)} | Test : {len(X_test)}")

    if HAS_XGBOOST:
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        model_algo = XGBClassifier(
            n_estimators=300, learning_rate=0.03, max_depth=3,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
            scale_pos_weight=n_neg / max(n_pos, 1),
            reg_alpha=0.5, reg_lambda=2.0, gamma=0.2,
            random_state=42, use_label_encoder=False,
            eval_metric="logloss", verbosity=0,
        )
    else:
        model_algo = GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.03, max_depth=3,
            random_state=42,
        )

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", model_algo),
    ])

    log.info("  Entrainement classification binaire...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred)

    log.info(f"  Accuracy  : {acc:.4f} ({acc*100:.1f}%)")
    log.info(f"  F1-score  : {f1:.4f}")
    log.info(f"\n{classification_report(y_test, y_pred, target_names=['Non-mortel','Mortel'])}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    log.info(f"  CV Accuracy : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "binary_gravity_model.pkl"
    joblib.dump({
        "pipeline": pipeline, "feature_names": feature_names,
        "classes": [0, 1], "class_labels": ["non-mortel", "mortel"],
        "trained_at": datetime.now().isoformat(),
    }, model_path)
    log.info(f"  Modele sauvegarde : {model_path}")

    metrics = {
        "model": "XGBoost_Binary", "task": "binary_classification",
        "target": "mortel_vs_non", "n_train": len(X_train), "n_test": len(X_test),
        "n_features": len(feature_names),
        "accuracy": round(acc, 4), "f1_weighted": round(f1, 4),
        "cv_mean": round(cv_scores.mean(), 4), "cv_std": round(cv_scores.std(), 4),
        "confusion_matrix": cm.tolist(),
    }
    save_metrics(metrics, "binary_gravity_model")
    plot_feature_importance(pipeline, feature_names, "binary_gravity_model")
    return metrics


# ══════════════════════════════════════════════════════
# CLUSTERING DBSCAN (Hotspots)
# ══════════════════════════════════════════════════════

def train_clustering(df: pd.DataFrame) -> dict:
    """
    DBSCAN géospatial pour identifier les zones à forte concentration d'accidents.
    """
    from sklearn.cluster import DBSCAN

    log.info("\n" + "─" * 50)
    log.info("MODÈLE 3 — Hotspots (DBSCAN Géospatial)")
    log.info("─" * 50)

    coords = df[["latitude", "longitude"]].dropna().values
    coords_rad = np.deg2rad(coords)

    # eps = 1km en radians (1/6371)
    eps_km  = 1.0
    eps_rad = eps_km / 6371.0

    db = DBSCAN(
        eps=eps_rad,
        min_samples=5,
        algorithm="ball_tree",
        metric="haversine",
        n_jobs=-1,
    )
    labels = db.fit_predict(coords_rad)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()
    log.info(f"  Clusters trouvés : {n_clusters}")
    log.info(f"  Points isolés    : {n_noise}")

    # Construire le tableau des hotspots avec score composite
    total_clustered = (labels != -1).sum()
    hotspots = []
    for cluster_id in sorted(set(labels)):
        if cluster_id == -1:
            continue
        mask     = labels == cluster_id
        cluster  = df[mask]
        center_lat = cluster["latitude"].mean()
        center_lon = cluster["longitude"].mean()
        count      = len(cluster)
        avg_gravity = cluster["gravity"].mean() if "gravity" in cluster.columns else 2.0

        # Rayon du cluster en km
        dists = np.sqrt(
            (cluster["latitude"] - center_lat)**2 +
            (cluster["longitude"] - center_lon)**2
        ) * 111
        radius_km = max(dists.max(), 0.1)

        # Score de risque composite : densite(30%) + gravite(40%) + frequence(20%) + proximite(10%)
        area_km2 = max(np.pi * radius_km ** 2, 0.01)
        density_norm = min((count / area_km2) / 50.0, 1.0)
        gravity_norm = (avg_gravity - 1.0) / 2.0
        freq_norm = min(count / max(total_clustered, 1), 1.0)
        risk_score = round(30 * density_norm + 40 * gravity_norm + 20 * freq_norm, 1)

        if risk_score >= 70:
            risk_level = "critique"
        elif risk_score >= 50:
            risk_level = "eleve"
        elif risk_score >= 30:
            risk_level = "moyen"
        else:
            risk_level = "faible"

        hotspots.append({
            "cluster_id":    int(cluster_id),
            "center_lat":    round(center_lat, 5),
            "center_lon":    round(center_lon, 5),
            "accident_count": int(count),
            "avg_gravity":   round(float(avg_gravity), 2),
            "risk_score":    risk_score,
            "risk_level":    risk_level,
            "region":        cluster["region"].mode()[0] if "region" in cluster.columns else "Inconnue",
        })

    df_hotspots = pd.DataFrame(hotspots).sort_values("risk_score", ascending=False)

    # Sauvegarder
    hotspots_path = DATA_PROCESSED_DIR / "hotspots.csv"
    df_hotspots.to_csv(hotspots_path, index=False)

    model_path = MODELS_DIR / "dbscan_model.pkl"
    joblib.dump({
        "model":      db,
        "labels":     labels,
        "hotspots":   df_hotspots.to_dict("records"),
        "trained_at": datetime.now().isoformat(),
    }, model_path)

    log.info(f"  Top 5 hotspots :")
    for _, row in df_hotspots.head(5).iterrows():
        log.info(f"    Cluster {row['cluster_id']} — {row['region']} — "
                 f"{row['accident_count']} accidents — risque {row['risk_level']}")

    log.info(f"  Hotspots sauvegardés : {hotspots_path}")

    metrics = {
        "model":       "DBSCAN",
        "task":        "clustering",
        "eps_km":      eps_km,
        "min_samples": 5,
        "n_clusters":  n_clusters,
        "n_noise":     int(n_noise),
        "n_points":    len(coords),
    }
    save_metrics(metrics, "dbscan_model")
    return metrics


# ══════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ══════════════════════════════════════════════════════

def select_best_gravity_model(all_metrics: dict):
    """Auto-selectionne le meilleur modele 3-classes et le copie en gravity_model.pkl."""
    clf_models = {
        "gravity": "gravity_model.pkl",
        "lgbm": "lightgbm_model.pkl",
        "rf_clf": "rf_classifier_model.pkl",
        "stacking": "stacking_model.pkl",
        "mlp": "mlp_model.pkl",
    }
    best_key, best_f1 = None, -1
    for key in clf_models:
        m = all_metrics.get(key, {})
        if m and m.get("f1_weighted", 0) > best_f1:
            best_f1 = m["f1_weighted"]
            best_key = key

    if best_key and best_key != "gravity":
        src = MODELS_DIR / clf_models[best_key]
        dst = MODELS_DIR / "gravity_model.pkl"
        if src.exists():
            import shutil
            shutil.copy2(src, dst)
            log.info(f"  [BEST] {all_metrics[best_key]['model']} copie -> gravity_model.pkl (F1={best_f1:.3f})")
    elif best_key:
        log.info(f"  [BEST] XGBoost reste le meilleur (F1={best_f1:.3f})")


def main():
    parser = argparse.ArgumentParser(description="SafeRoads SN -- Entrainement ML")
    parser.add_argument(
        "--model",
        choices=["gravity", "lgbm", "rf_clf", "stacking", "binary",
                 "risk", "mlp", "clustering", "all"],
        default="all",
        help="Modele a entrainer (defaut: all)"
    )
    args = parser.parse_args()

    log.info("=" * 55)
    log.info("  SafeRoads SN -- Entrainement des modeles ML")
    log.info("=" * 55)

    df = load_dataset()

    all_metrics = {}

    if args.model in ("gravity", "all"):
        all_metrics["gravity"] = train_gravity_model(df)

    if args.model in ("lgbm", "all"):
        m = train_lightgbm_model(df)
        if m:
            all_metrics["lgbm"] = m

    if args.model in ("rf_clf", "all"):
        all_metrics["rf_clf"] = train_rf_classifier_model(df)

    if args.model in ("mlp", "all"):
        all_metrics["mlp"] = train_mlp_model(df)

    if args.model in ("stacking", "all"):
        all_metrics["stacking"] = train_stacking_model(df)

    if args.model in ("binary", "all"):
        all_metrics["binary"] = train_binary_gravity_model(df)

    if args.model in ("risk", "all"):
        all_metrics["risk"] = train_risk_model(df)

    if args.model in ("clustering", "all"):
        all_metrics["clustering"] = train_clustering(df)

    # Rapport final
    log.info("\n" + "=" * 55)
    log.info("  RESUME DES PERFORMANCES")
    log.info("=" * 55)

    clf_keys = ["gravity", "lgbm", "rf_clf", "mlp", "stacking"]
    log.info("\n  -- CLASSIFICATION 3 CLASSES (gravite) --")
    for key in clf_keys:
        m = all_metrics.get(key, {})
        if m and "accuracy" in m:
            log.info(f"  {m['model']:<25} Acc={m['accuracy']*100:.1f}%  F1={m['f1_weighted']:.3f}  CV={m['cv_mean']:.3f}")

    if "binary" in all_metrics:
        m = all_metrics["binary"]
        log.info(f"\n  -- CLASSIFICATION BINAIRE (mortel vs non-mortel) --")
        log.info(f"  {m['model']:<25} Acc={m['accuracy']*100:.1f}%  F1={m['f1_weighted']:.3f}  CV={m['cv_mean']:.3f}")

    if "risk" in all_metrics:
        m = all_metrics["risk"]
        log.info(f"\n  -- REGRESSION (score de risque) --")
        log.info(f"  Risque     -> R2: {m['r2']:.4f} | MAE: {m['mae']:.2f} | RMSE: {m['rmse']:.2f}")

    if "clustering" in all_metrics:
        m = all_metrics["clustering"]
        log.info(f"\n  -- CLUSTERING --")
        log.info(f"  DBSCAN     -> {m['n_clusters']} hotspots detectes")

    # Auto-selectionner le meilleur modele 3-classes
    if args.model == "all":
        select_best_gravity_model(all_metrics)

    log.info("\n" + "=" * 55)
    log.info("  Etape suivante : python src/ml/transfer.py")
    log.info("=" * 55)

    # Sauvegarder le rapport global
    report_path = MODELS_DIR / "training_report.json"
    with open(report_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    log.info(f"\n  Rapport complet : {report_path}")


if __name__ == "__main__":
    main()
