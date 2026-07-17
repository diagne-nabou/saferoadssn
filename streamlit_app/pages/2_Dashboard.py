"""
SafeRoads SN — Page Dashboard
Statistiques et graphes interactifs Plotly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.db import query_df, get_stats, test_connection

st.set_page_config(page_title="Dashboard — SafeRoads SN", page_icon="📊", layout="wide")

# ── Chargement données ──
@st.cache_data(ttl=120, show_spinner="Chargement des données...")
def load_data():
    ok, _ = test_connection()
    if ok:
        df = query_df("SELECT * FROM accidents")
        if not df.empty:
            return df
    # Fallback CSV
    csv = Path(__file__).parent.parent.parent / "data/processed/saferoads_dataset.csv"
    if csv.exists():
        return pd.read_csv(csv)
    return _generate_demo_data()


def _generate_demo_data():
    """Données de démo si aucune source disponible."""
    np.random.seed(42)
    n = 1200
    regions = ["Dakar","Thiès","Kaolack","Diourbel","Mbour","Saint-Louis",
               "Louga","Ziguinchor","Tambacounda","Kolda"]
    vehicles = ["Car rapide","Moto-Jakarta","Camion","Voiture","Sept-places","Charette"]
    causes   = ["Excès de vitesse","Somnolence/fatigue","État dégradé route",
                "Téléphone au volant","Alcool","Inconnue"]
    weather  = ["Ensoleillé","Nuageux","Pluie légère","Pluie forte"]

    return pd.DataFrame({
        "region":       np.random.choice(regions, n, p=[.25,.15,.10,.09,.08,.08,.07,.07,.06,.05]),
        "hour":         np.random.choice(range(24), n),
        "day_of_week":  np.random.choice(range(7), n),
        "month":        np.random.choice(range(1,13), n),
        "gravity":      np.random.choice([1,2,3], n, p=[.56,.34,.10]),
        "vehicle_type": np.random.choice(vehicles, n, p=[.28,.22,.18,.20,.08,.04]),
        "cause":        np.random.choice(causes, n, p=[.38,.24,.18,.12,.05,.03]),
        "weather":      np.random.choice(weather, n, p=[.55,.25,.15,.05]),
        "is_rainy":     np.random.choice([False,True], n, p=[.80,.20]),
    })


GRAVITY_LABELS = {1:"Léger",2:"Grave",3:"Mortel"}
GRAVITY_COLORS = {"Léger":"#28a745","Grave":"#fd7e14","Mortel":"#dc3545"}
DAYS           = ["Lun","Mar","Mer","Jeu","Ven","Sam","Dim"]
MONTHS         = ["Jan","Fév","Mar","Avr","Mai","Jun","Jul","Aoû","Sep","Oct","Nov","Déc"]

# ══════════════════════════════════════════════════════
# INTERFACE
# ══════════════════════════════════════════════════════

st.title("📊 Dashboard analytique — Accidents routiers")
st.caption("Données : PostgreSQL · Fallback : CSV processsed")

df = load_data()

# ── Sidebar filtres ──
with st.sidebar:
    st.markdown("### 🎛️ Filtres")
    regions_dispo = ["Toutes"] + sorted(df["region"].dropna().unique().tolist()) if "region" in df.columns else ["Toutes"]
    region_sel    = st.selectbox("Région", regions_dispo)

    gravity_sel = st.multiselect("Gravité", [1,2,3], default=[1,2,3],
                                  format_func=lambda x: GRAVITY_LABELS[x])

    if "month" in df.columns:
        months_dispo = sorted(df["month"].dropna().unique().astype(int).tolist())
        month_sel    = st.multiselect("Mois", months_dispo,
                                       default=months_dispo,
                                       format_func=lambda x: MONTHS[x-1])
    else:
        month_sel = list(range(1,13))

    if st.button("🔄 Actualiser", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# Appliquer filtres
df_f = df.copy()
if region_sel != "Toutes" and "region" in df_f.columns:
    df_f = df_f[df_f["region"] == region_sel]
if "gravity" in df_f.columns:
    df_f = df_f[df_f["gravity"].isin(gravity_sel)]
if "month" in df_f.columns:
    df_f = df_f[df_f["month"].isin(month_sel)]

# ── KPIs ──
total = len(df_f)
mort  = len(df_f[df_f["gravity"] == 3]) if "gravity" in df_f.columns else 0
grave = len(df_f[df_f["gravity"] == 2]) if "gravity" in df_f.columns else 0
leger = len(df_f[df_f["gravity"] == 1]) if "gravity" in df_f.columns else 0
pct_grave = round((mort + grave) / max(total, 1) * 100, 1)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("🚗 Total accidents",  f"{total:,}")
k2.metric("🟢 Légers",           f"{leger:,}")
k3.metric("🟠 Graves",           f"{grave:,}")
k4.metric("🔴 Mortels",          f"{mort:,}")
k5.metric("⚠️ Taux grave+mortel", f"{pct_grave}%")

st.divider()

# ── Graphes ligne 1 ──
col1, col2 = st.columns(2)

with col1:
    # Afficher le graphe horaire SEULEMENT si les heures sont reelles (pas -1)
    has_real_hours = "hour" in df_f.columns and (df_f["hour"] >= 0).any() and df_f["hour"].nunique() > 3
    if has_real_hours:
        st.subheader("Accidents par heure de la journée")
        hourly = df_f[df_f["hour"] >= 0].groupby("hour").size().reset_index(name="count")
        fig = px.area(hourly, x="hour", y="count",
                      color_discrete_sequence=["#238636"],
                      labels={"hour":"Heure","count":"Accidents"})
        fig.add_vrect(x0=20, x1=24, fillcolor="#dc3545", opacity=0.08, line_width=0,
                      annotation_text="Nuit", annotation_position="top left")
        fig.add_vrect(x0=0,  x1=6,  fillcolor="#dc3545", opacity=0.08, line_width=0)
        fig.update_layout(height=280, margin=dict(t=10,b=10,l=0,r=0),
                          plot_bgcolor="#0D1117", paper_bgcolor="#0D1117",
                          font_color="#C9D1D9", xaxis=dict(dtick=2))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.subheader("Gravité par région")
        if "region" in df_f.columns and "gravity" in df_f.columns:
            df_rg = df_f.copy()
            df_rg["gravity_label"] = df_rg["gravity"].map(GRAVITY_LABELS)
            cross = df_rg.groupby(["region", "gravity_label"]).size().reset_index(name="count")
            fig = px.bar(cross, x="region", y="count", color="gravity_label",
                         color_discrete_map=GRAVITY_COLORS,
                         labels={"region":"Région","count":"Accidents","gravity_label":"Gravité"})
            fig.update_layout(height=280, margin=dict(t=10,b=10,l=0,r=0),
                              plot_bgcolor="#0D1117", paper_bgcolor="#0D1117",
                              font_color="#C9D1D9", barmode="stack",
                              xaxis=dict(tickangle=-30, tickfont=dict(size=11)))
            st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Répartition par gravité")
    if "gravity" in df_f.columns:
        grav = df_f["gravity"].map(GRAVITY_LABELS).value_counts().reset_index()
        grav.columns = ["Gravité", "Count"]
        fig = px.pie(grav, names="Gravité", values="Count",
                     color="Gravité", color_discrete_map=GRAVITY_COLORS,
                     hole=0.45)
        fig.update_layout(height=280, margin=dict(t=10,b=10,l=0,r=0),
                          paper_bgcolor="#0D1117", font_color="#C9D1D9",
                          legend=dict(orientation="v"))
        st.plotly_chart(fig, use_container_width=True)

# ── Graphes ligne 2 ──
col3, col4 = st.columns(2)

with col3:
    st.subheader("Accidents par région")
    if "region" in df_f.columns:
        reg = df_f["region"].value_counts().head(10).reset_index()
        reg.columns = ["Région","Count"]
        fig = px.bar(reg.sort_values("Count"), x="Count", y="Région",
                     orientation="h", color="Count",
                     color_continuous_scale=["#238636","#ffc107","#dc3545"])
        fig.update_layout(height=300, margin=dict(t=10,b=10,l=0,r=0),
                          plot_bgcolor="#0D1117", paper_bgcolor="#0D1117",
                          font_color="#C9D1D9", coloraxis_showscale=False,
                          yaxis=dict(tickfont=dict(size=11)))
        st.plotly_chart(fig, use_container_width=True)

with col4:
    st.subheader("Accidents par jour de la semaine")
    if "day_of_week" in df_f.columns:
        dow = df_f["day_of_week"].value_counts().sort_index().reset_index()
        dow.columns = ["Jour","Count"]
        dow["Jour"] = dow["Jour"].map(lambda x: DAYS[x] if 0 <= x < 7 else str(x))
        fig = px.bar(dow, x="Jour", y="Count",
                     color="Count",
                     color_continuous_scale=["#1f6feb","#388bfd","#58a6ff"],
                     labels={"Count":"Accidents"})
        fig.update_layout(height=300, margin=dict(t=10,b=10,l=0,r=0),
                          plot_bgcolor="#0D1117", paper_bgcolor="#0D1117",
                          font_color="#C9D1D9", coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

# ── Graphes ligne 3 ──
col5, col6 = st.columns(2)

with col5:
    st.subheader("Types de véhicules impliqués")
    if "vehicle_type" in df_f.columns:
        veh = df_f["vehicle_type"].value_counts().head(8).reset_index()
        veh.columns = ["Véhicule","Count"]
        fig = px.bar(veh, x="Véhicule", y="Count",
                     color="Count",
                     color_continuous_scale="Oranges",
                     labels={"Count":"Accidents"})
        fig.update_layout(height=280, margin=dict(t=10,b=10,l=0,r=0),
                          plot_bgcolor="#0D1117", paper_bgcolor="#0D1117",
                          font_color="#C9D1D9", coloraxis_showscale=False,
                          xaxis=dict(tickangle=-30, tickfont=dict(size=11)))
        st.plotly_chart(fig, use_container_width=True)

with col6:
    st.subheader("Accidents par mois (saisonnalité)")
    if "month" in df_f.columns:
        monthly = df_f.groupby("month").size().reset_index(name="count")
        monthly["label"] = monthly["month"].map(lambda x: MONTHS[x-1])
        fig = px.line(monthly, x="label", y="count", markers=True,
                      color_discrete_sequence=["#58a6ff"],
                      labels={"label":"Mois","count":"Accidents"})
        fig.add_hrect(y0=0, y1=monthly["count"].max(),
                      x0="Jun", x1="Oct",
                      fillcolor="#ffc107", opacity=0.07, line_width=0,
                      annotation_text="☔ Hivernage", annotation_position="top left")
        fig.update_layout(height=280, margin=dict(t=10,b=10,l=0,r=0),
                          plot_bgcolor="#0D1117", paper_bgcolor="#0D1117",
                          font_color="#C9D1D9")
        st.plotly_chart(fig, use_container_width=True)

# ── Heatmap ou graphe alternatif ──
st.divider()
if has_real_hours:
    st.subheader("Heatmap : Heure x Jour de la semaine")
    pivot = df_f[df_f["hour"] >= 0].pivot_table(
        index="day_of_week", columns="hour",
        values="gravity" if "gravity" in df_f.columns else "hour",
        aggfunc="count", fill_value=0)
    pivot.index = [DAYS[i] if 0 <= i < 7 else str(i) for i in pivot.index]

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[f"{h}h" for h in pivot.columns],
        y=pivot.index.tolist(),
        colorscale=[[0,"#0D1117"],[0.3,"#1f6feb"],[0.6,"#fd7e14"],[1.0,"#dc3545"]],
        showscale=True,
        colorbar=dict(title="Accidents", tickfont=dict(color="#C9D1D9")),
    ))
    fig.update_layout(height=280, margin=dict(t=10,b=10,l=0,r=0),
                      paper_bgcolor="#0D1117", plot_bgcolor="#0D1117",
                      font_color="#C9D1D9", xaxis=dict(dtick=2))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.subheader("Heatmap : Mois x Jour de la semaine")
    if "month" in df_f.columns and "day_of_week" in df_f.columns:
        pivot = df_f.pivot_table(
            index="day_of_week", columns="month",
            values="gravity" if "gravity" in df_f.columns else "month",
            aggfunc="count", fill_value=0)
        pivot.index = [DAYS[i] if 0 <= i < 7 else str(i) for i in pivot.index]
        pivot.columns = [MONTHS[m-1] if 1 <= m <= 12 else str(m) for m in pivot.columns]

        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale=[[0,"#0D1117"],[0.3,"#1f6feb"],[0.6,"#fd7e14"],[1.0,"#dc3545"]],
            showscale=True,
            colorbar=dict(title="Accidents", tickfont=dict(color="#C9D1D9")),
        ))
        fig.update_layout(height=280, margin=dict(t=10,b=10,l=0,r=0),
                          paper_bgcolor="#0D1117", plot_bgcolor="#0D1117",
                          font_color="#C9D1D9")
        st.plotly_chart(fig, use_container_width=True)
