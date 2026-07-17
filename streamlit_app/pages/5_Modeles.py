"""
SafeRoads SN -- Page Comparaison des Modeles
Compare les performances de tous les modeles ML entraines.
Affiche metriques, matrices de confusion, feature importance.
"""

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Modeles -- SafeRoads SN", page_icon="brain", layout="wide")

MODELS_DIR  = Path(__file__).parent.parent.parent / "models"
CLASSES     = ["Leger", "Grave", "Mortel"]

# ══════════════════════════════════════════════════════
# CHARGEMENT DES METRIQUES
# ══════════════════════════════════════════════════════

MODEL_REGISTRY = [
    ("gravity_model",          "XGBoost",                 "#2E86AB"),
    ("mlp_model",              "MLP (Deep Learning)",      "#A23B72"),
    ("lightgbm_model",         "LightGBM",                 "#4CAF50"),
    ("rf_classifier_model",    "RandomForest",              "#FF9800"),
    ("stacking_model",         "Stacking Ensemble",         "#9C27B0"),
    ("binary_gravity_model",   "Binaire (Mortel/Non)",      "#E91E63"),
    ("transfer_model",         "Transfer Learning",         "#00BCD4"),
]


@st.cache_data(ttl=120)
def load_all_metrics():
    """Charge toutes les metriques des modeles depuis les fichiers JSON."""
    metrics = {}
    for name, _, _ in MODEL_REGISTRY:
        path = MODELS_DIR / f"{name}_metrics.json"
        if path.exists():
            with open(path) as f:
                metrics[name] = json.load(f)
    # Modeles historiques
    for name in ["risk_model", "dbscan_model"]:
        path = MODELS_DIR / f"{name}_metrics.json"
        if path.exists():
            with open(path) as f:
                metrics[name] = json.load(f)
    # Rapport global
    report_path = MODELS_DIR / "training_report.json"
    if report_path.exists():
        with open(report_path) as f:
            metrics["_report"] = json.load(f)
    return metrics


@st.cache_data(ttl=120)
def load_training_report():
    path = MODELS_DIR / "training_report.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


# ══════════════════════════════════════════════════════
# INTERFACE
# ══════════════════════════════════════════════════════

st.title("Comparaison des modeles ML")
st.caption("Performances de tous les modeles entraines | Metriques, matrices de confusion, feature importance")

metrics = load_all_metrics()
report  = load_training_report()

if not metrics:
    st.warning("Aucun modele entraine. Lancez : `python src/ml/train.py --model all`")
    st.stop()

# ── Section 1 : KPIs globaux ──
st.subheader("Vue d'ensemble")

gm = metrics.get("gravity_model", {})
mm = metrics.get("mlp_model", {})
rm = metrics.get("risk_model", {})
dm = metrics.get("dbscan_model", {})
lm = metrics.get("lgbm_model", {})
sm = metrics.get("stacking_model", {})
tm = metrics.get("transfer_model", {})

# Trouver le meilleur modele 3-classes
best_model_name = "—"
best_cv = 0
for key, label, _ in MODEL_REGISTRY:
    m = metrics.get(key, {})
    cv = m.get("cv_mean", 0)
    if cv > best_cv and key not in ("binary_gravity_model",):
        best_cv = cv
        best_model_name = label

k1, k2, k3, k4 = st.columns(4)
k1.metric(
    "Meilleur modele (CV)",
    f"{best_cv*100:.1f}%",
    help=f"{best_model_name} — Cross-validation 5-folds"
)
k2.metric(
    "Modeles entraines",
    f"{sum(1 for k, _, _ in MODEL_REGISTRY if k in metrics)}",
    help="Nombre total de modeles"
)
k3.metric(
    "Score Risque (R2)",
    f"{rm.get('r2', 0):.3f}",
    help="R2 du RandomForestRegressor"
)
k4.metric(
    "Hotspots DBSCAN",
    f"{dm.get('n_clusters', 0)} zones",
    help=f"{dm.get('n_noise', 0)} points isoles"
)

st.divider()

# ══════════════════════════════════════════════════════
# Section 2 : COMPARAISON CLASSIFICATION (tous modeles 3-classes)
# ══════════════════════════════════════════════════════

st.subheader("Comparaison des modeles de classification (Gravite)")

comparison_data = []
for key, label, color in MODEL_REGISTRY:
    m = metrics.get(key, {})
    if m and key != "binary_gravity_model":
        comparison_data.append({
            "Modele":        label,
            "Algorithme":    m.get("model", "—"),
            "Accuracy (%)":  round(m.get("accuracy", 0) * 100, 1),
            "F1-Score":      round(m.get("f1_weighted", 0), 4),
            "CV Accuracy":   round(m.get("cv_mean", 0) * 100, 1),
            "CV Ecart-type": round(m.get("cv_std", 0) * 100, 1),
            "N features":    m.get("n_features", 0),
            "_key":          key,
            "_color":        color,
        })

if comparison_data:
    df_comp = pd.DataFrame(comparison_data)

    # Highlight le meilleur
    best_idx = df_comp["CV Accuracy"].idxmax()
    st.success(f"Meilleur modele de classification : **{df_comp.loc[best_idx, 'Modele']}** (CV Accuracy = {df_comp.loc[best_idx, 'CV Accuracy']:.1f}%)")

    st.dataframe(
        df_comp.drop(columns=["_key", "_color"]),
        use_container_width=True,
        hide_index=True,
        column_config={
            "CV Accuracy": st.column_config.ProgressColumn(
                "CV Accuracy", min_value=0, max_value=100, format="%.1f%%"
            ),
        },
    )

    # Graphique barres comparatif
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("#### Accuracy & F1-Score")
        fig = go.Figure()
        for i, row in df_comp.iterrows():
            fig.add_trace(go.Bar(
                name=row["Modele"],
                x=["Accuracy (%)", "F1-Score (x100)", "CV Accuracy (%)"],
                y=[row["Accuracy (%)"], row["F1-Score"] * 100, row["CV Accuracy"]],
                marker_color=row["_color"],
                text=[f"{row['Accuracy (%)']:.1f}", f"{row['F1-Score']*100:.1f}", f"{row['CV Accuracy']:.1f}"],
                textposition="outside",
            ))
        fig.update_layout(
            barmode="group", height=400,
            margin=dict(t=10, b=10, l=0, r=0),
            plot_bgcolor="#0D1117", paper_bgcolor="#0D1117",
            font_color="#C9D1D9",
            legend=dict(orientation="h", y=1.15),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_chart2:
        st.markdown("#### Stabilite (Cross-Validation)")
        fig2 = go.Figure()
        for i, row in df_comp.iterrows():
            fig2.add_trace(go.Bar(
                name=row["Modele"],
                x=[row["Modele"]],
                y=[row["CV Accuracy"]],
                error_y=dict(type="data", array=[row["CV Ecart-type"]]),
                marker_color=row["_color"],
                text=[f"{row['CV Accuracy']:.1f}%"],
                textposition="outside",
            ))
        fig2.update_layout(
            height=400, showlegend=False,
            margin=dict(t=10, b=10, l=0, r=0),
            plot_bgcolor="#0D1117", paper_bgcolor="#0D1117",
            font_color="#C9D1D9",
            yaxis=dict(title="Accuracy CV (%)"),
        )
        st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════
# Section 3 : MATRICES DE CONFUSION (onglets)
# ══════════════════════════════════════════════════════

st.subheader("Matrices de confusion")

# Creer les onglets pour chaque modele avec une matrice
models_with_cm = [(key, label, color) for key, label, color in MODEL_REGISTRY
                  if key in metrics and metrics[key].get("confusion_matrix") and key != "binary_gravity_model"]

if models_with_cm:
    tab_labels = [label for _, label, _ in models_with_cm]
    tabs = st.tabs(tab_labels)

    for tab, (key, label, color) in zip(tabs, models_with_cm):
        m = metrics[key]
        cm = m["confusion_matrix"]
        color_scale = {
            "gravity_model": "Blues", "mlp_model": "Purples",
            "lightgbm_model": "Greens", "rf_classifier_model": "Oranges",
            "stacking_model": "RdPu", "transfer_model": "Teal",
        }.get(key, "Blues")

        with tab:
            cm_arr = np.array(cm)
            cm_pct = (cm_arr / cm_arr.sum(axis=1, keepdims=True) * 100).round(1)
            text_vals = [[f"{cm_arr[i][j]}<br>({cm_pct[i][j]:.0f}%)"
                          for j in range(len(CLASSES))] for i in range(len(CLASSES))]

            fig = go.Figure(data=go.Heatmap(
                z=cm_arr,
                x=[f"Predit {c}" for c in CLASSES],
                y=[f"Reel {c}" for c in CLASSES],
                colorscale=color_scale,
                text=text_vals,
                texttemplate="%{text}",
                showscale=False,
            ))
            fig.update_layout(
                height=350,
                margin=dict(t=10, b=10, l=0, r=0),
                plot_bgcolor="#0D1117", paper_bgcolor="#0D1117",
                font_color="#C9D1D9",
                xaxis=dict(side="bottom"),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Metriques sous la matrice
            c1, c2, c3 = st.columns(3)
            c1.metric("Accuracy", f"{m.get('accuracy', 0)*100:.1f}%")
            c2.metric("F1-Score", f"{m.get('f1_weighted', 0):.4f}")
            c3.metric("CV Accuracy", f"{m.get('cv_mean', 0)*100:.1f}%")
else:
    st.info("Aucune matrice de confusion disponible. Relancez l'entrainement.")

st.divider()

# ══════════════════════════════════════════════════════
# Section 4 : MODELE BINAIRE (mortel vs non-mortel)
# ══════════════════════════════════════════════════════

bm = metrics.get("binary_gravity_model", {})
if bm:
    st.subheader("Classification binaire (Mortel vs Non-mortel)")

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Accuracy", f"{bm.get('accuracy', 0)*100:.1f}%")
    b2.metric("F1-Score", f"{bm.get('f1_weighted', 0):.4f}")
    b3.metric("CV Accuracy", f"{bm.get('cv_mean', 0)*100:.1f}%")
    b4.metric("Rappel Mortel", f"{bm.get('recall_mortel', bm.get('accuracy', 0))*100:.1f}%")

    cm_bin = bm.get("confusion_matrix")
    if cm_bin:
        cm_arr = np.array(cm_bin)
        bin_classes = ["Non-mortel", "Mortel"]
        cm_pct = (cm_arr / cm_arr.sum(axis=1, keepdims=True) * 100).round(1)
        text_vals = [[f"{cm_arr[i][j]}<br>({cm_pct[i][j]:.0f}%)"
                      for j in range(2)] for i in range(2)]

        fig = go.Figure(data=go.Heatmap(
            z=cm_arr,
            x=[f"Predit {c}" for c in bin_classes],
            y=[f"Reel {c}" for c in bin_classes],
            colorscale="Reds",
            text=text_vals,
            texttemplate="%{text}",
            showscale=False,
        ))
        fig.update_layout(
            height=300,
            margin=dict(t=10, b=10, l=0, r=0),
            plot_bgcolor="#0D1117", paper_bgcolor="#0D1117",
            font_color="#C9D1D9",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

# ══════════════════════════════════════════════════════
# Section 5 : TRANSFER LEARNING
# ══════════════════════════════════════════════════════

if tm:
    st.subheader("Transfer Learning (Feature Augmentation)")

    st.markdown("""
    **Strategie** : Le modele global (5000+ records) genere des probabilites de gravite
    qui sont utilisees comme 3 features supplementaires pour le modele Senegal.
    """)

    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Accuracy", f"{tm.get('accuracy', 0)*100:.1f}%")
    t2.metric("F1-Score", f"{tm.get('f1_weighted', 0):.4f}")
    t3.metric("CV Accuracy", f"{tm.get('cv_mean', 0)*100:.1f}%")
    src_acc = tm.get("source_accuracy", 0)
    gain = tm.get("accuracy", 0) - src_acc if src_acc else 0
    t4.metric("Gain vs Source", f"{'+' if gain >= 0 else ''}{gain*100:.1f}%")

    st.divider()

# ══════════════════════════════════════════════════════
# Section 6 : FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════

st.subheader("Importance des features (XGBoost)")

fi_path = MODELS_DIR / "gravity_model_feature_importance.png"
if fi_path.exists():
    st.image(str(fi_path), use_container_width=True)
else:
    st.info("Graphe d'importance non disponible.")

st.divider()

# ══════════════════════════════════════════════════════
# Section 7 : MODELE DE RISQUE
# ══════════════════════════════════════════════════════

st.subheader("Modele de Score de Risque (Regression)")

if rm:
    r1, r2_col, r3 = st.columns(3)
    r1.metric("R2", f"{rm.get('r2', 0):.4f}")
    r2_col.metric("MAE", f"{rm.get('mae', 0):.2f} pts")
    r3.metric("RMSE", f"{rm.get('rmse', 0):.2f} pts")

    st.markdown(f"""
    | Propriete | Valeur |
    |---|---|
    | Algorithme | RandomForestRegressor |
    | Echantillons (train/test) | {rm.get('n_train', '?')} / {rm.get('n_test', '?')} |
    | Features | {rm.get('n_features', '?')} |
    | CV R2 | {rm.get('cv_r2_mean', 0):.4f} +/- {rm.get('cv_r2_std', 0):.4f} |
    """)

    fi_risk_path = MODELS_DIR / "risk_model_feature_importance.png"
    if fi_risk_path.exists():
        st.image(str(fi_risk_path), use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════
# Section 8 : CLUSTERING
# ══════════════════════════════════════════════════════

st.subheader("Clustering DBSCAN")

if dm:
    d1, d2, d3 = st.columns(3)
    d1.metric("Clusters detectes", dm.get("n_clusters", 0))
    d2.metric("Points isoles", dm.get("n_noise", 0))
    d3.metric("Points totaux", dm.get("n_points", 0))

    pct = round((1 - dm.get("n_noise", 0) / max(dm.get("n_points", 1), 1)) * 100, 1)
    st.progress(pct / 100, text=f"{pct}% des accidents dans un cluster")

    st.markdown(f"""
    | Parametre | Valeur |
    |---|---|
    | Rayon (eps) | {dm.get('eps_km', 1.0)} km |
    | Min. echantillons | {dm.get('min_samples', 5)} |
    | Metrique | Haversine (geospatiale) |
    """)

st.divider()

# ══════════════════════════════════════════════════════
# Section 9 : RESUME METHODOLOGIQUE
# ══════════════════════════════════════════════════════

st.subheader("Resume methodologique")

st.markdown("""
| Modele | Type | Usage | Algorithme |
|---|---|---|---|
| **Gravite (XGBoost)** | Classification 3-classes | Predire la severite d'un accident | XGBoost (Gradient Boosting) |
| **Gravite (MLP)** | Classification 3-classes | Comparaison Deep Learning | MLP 3 couches (256, 128, 64) |
| **Gravite (LightGBM)** | Classification 3-classes | Leaf-wise boosting, class_weight balanced | LightGBM |
| **Gravite (RF)** | Classification 3-classes | Ensemble bagging | RandomForest (balanced_subsample) |
| **Stacking Ensemble** | Classification 3-classes | Combine XGB+LGBM+RF+SVM | StackingClassifier + LogReg meta |
| **Binaire** | Classification 2-classes | Mortel vs Non-mortel | XGBoost (scale_pos_weight) |
| **Transfer Learning** | Classification 3-classes | Feature augmentation depuis modele global | LightGBM + priors globaux |
| **Score de risque** | Regression | Score continu 0-100 par zone | RandomForest Regressor |
| **Hotspots** | Clustering | Detection zones dangereuses | DBSCAN geospatial (Haversine) |

**Pipeline d'entrainement** :
1. Chargement du dataset preprocesse (387 accidents, 29 features)
2. Split 80/20 avec stratification
3. Entrainement avec imputation des valeurs manquantes (mediane)
4. Evaluation : accuracy, F1-score, cross-validation 5-folds
5. Auto-selection du meilleur modele (copie vers `gravity_model.pkl`)
6. Transfer learning : pre-entrainement global (5000+ records) + feature augmentation
""")

# Bouton re-entrainement
st.divider()
with st.expander("Re-entrainer les modeles"):
    st.warning("Cette action va re-entrainer tous les modeles. Cela peut prendre quelques minutes.")
    st.code("python src/ml/train.py --model all", language="bash")
    st.code("python src/ml/transfer.py", language="bash")
    st.caption("Lancez ces commandes dans le terminal du projet.")
