"""
SafeRoads SN -- Page Carte
Affiche les accidents individuels + hotspots DBSCAN sur une carte Folium.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
from utils.ml import get_hotspots_from_models
from utils.db import query_df, test_connection

st.set_page_config(page_title="Carte -- SafeRoads SN", page_icon="map", layout="wide")

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"

GRAVITY_COLORS = {1: "#28a745", 2: "#fd7e14", 3: "#dc3545"}
GRAVITY_LABELS = {1: "Leger", 2: "Grave", 3: "Mortel"}
RISK_COLORS = {"critique":"#dc3545","eleve":"#fd7e14","moyen":"#ffc107","faible":"#28a745"}
RISK_LABELS = {"critique":"Critique","eleve":"Eleve","moyen":"Moyen","faible":"Faible"}

# ══════════════════════════════════════════════════════
# CHARGEMENT DES DONNEES
# ══════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner="Chargement des donnees...")
def load_accidents():
    """Charge les accidents depuis le dataset final."""
    csv_path = DATA_DIR / "saferoads_dataset.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df = df[df["latitude"].notna() & df["longitude"].notna()]
        return df
    return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner="Chargement des hotspots...")
def load_hotspots():
    ok, _ = test_connection()
    if ok:
        df = query_df("SELECT * FROM hotspots ORDER BY accident_count DESC")
        if not df.empty:
            return df
    hs = get_hotspots_from_models()
    if hs:
        return pd.DataFrame(hs)
    csv_path = DATA_DIR / "hotspots.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


# =============================================
# MISE EN PAGE
# =============================================

st.title("Carte des accidents -- Senegal")
st.caption("387 accidents reels avec hotspots DBSCAN | 14 regions administratives")

df_acc = load_accidents()
df_hs = load_hotspots()

if df_acc.empty:
    st.warning("Aucune donnee d'accident. Lancez : `python scripts/run_etl.py`")
    st.stop()

# Normaliser risk_level (accents)
if not df_hs.empty and "risk_level" in df_hs.columns:
    df_hs["risk_level"] = df_hs["risk_level"].str.replace("eleve", "eleve").str.replace("\u00e9lev\u00e9", "eleve")

# -- Filtres sidebar --
all_regions = sorted(df_acc["region"].dropna().unique().tolist())

with st.sidebar:
    st.markdown("### Filtres carte")
    region_filter = st.multiselect("Region", all_regions, default=all_regions)
    gravity_filter = st.multiselect(
        "Gravite",
        [1, 2, 3],
        default=[1, 2, 3],
        format_func=lambda x: GRAVITY_LABELS.get(x, str(x)),
    )
    show_hotspots = st.checkbox("Afficher zones hotspots", value=True)
    show_accidents = st.checkbox("Afficher accidents individuels", value=True)
    map_style = st.selectbox("Fond de carte", [
        "CartoDB positron", "OpenStreetMap", "CartoDB dark_matter",
    ])

# Appliquer filtres
df_filtered = df_acc.copy()
if region_filter:
    df_filtered = df_filtered[df_filtered["region"].isin(region_filter)]
if gravity_filter:
    df_filtered = df_filtered[df_filtered["gravity"].isin(gravity_filter)]

# -- KPIs --
total_acc = len(df_filtered)
n_regions = df_filtered["region"].nunique()
n_mortel = len(df_filtered[df_filtered["gravity"] == 3])
n_grave = len(df_filtered[df_filtered["gravity"] == 2])
avg_grav = round(df_filtered["gravity"].mean(), 1) if total_acc > 0 else 0

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total accidents", f"{total_acc}")
k2.metric("Regions", f"{n_regions}")
k3.metric("Mortels", f"{n_mortel}")
k4.metric("Graves", f"{n_grave}")
k5.metric("Gravite moyenne", f"{avg_grav}/3")

st.divider()

# -- Carte Folium --
m = folium.Map(
    location=[14.5, -15.0],
    zoom_start=7,
    tiles=map_style,
)

# ── Couche 1 : Accidents individuels (MarkerCluster) ──
if show_accidents and not df_filtered.empty:
    cluster = MarkerCluster(name="Accidents individuels").add_to(m)

    for _, row in df_filtered.iterrows():
        lat = row["latitude"]
        lon = row["longitude"]
        grav = int(row.get("gravity", 1))
        color = GRAVITY_COLORS.get(grav, "#888")
        label = GRAVITY_LABELS.get(grav, "?")
        region = row.get("region", "?")
        ville = row.get("ville", "")
        date = str(row.get("date", row.get("datetime", "")))[:10]
        vehicle = row.get("vehicle_type", "")
        morts = int(row.get("num_victims", 0)) if pd.notna(row.get("num_victims")) else ""

        popup_html = f"""
        <div style="font-family:sans-serif;min-width:180px;padding:4px">
          <b>{region}</b>{f' - {ville}' if ville and str(ville) != 'nan' else ''}<br>
          <span style="background:{color};color:#fff;padding:2px 8px;border-radius:8px;font-size:11px">{label}</span>
          <hr style="margin:6px 0;border-color:#ddd">
          Date : {date}<br>
          {f'Vehicule : {vehicle}<br>' if vehicle and str(vehicle) != 'nan' else ''}
          {f'Victimes : {morts}<br>' if morts != '' else ''}
          Position : {lat:.4f}, {lon:.4f}
        </div>
        """

        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            weight=1,
            popup=folium.Popup(popup_html, max_width=220),
            tooltip=f"{region} | {label} | {date}",
        ).add_to(cluster)

# ── Couche 2 : Hotspots DBSCAN ──
if show_hotspots and not df_hs.empty:
    for _, row in df_hs.iterrows():
        lat = row.get("center_lat", row.get("latitude"))
        lon = row.get("center_lon", row.get("longitude"))
        if pd.isna(lat) or pd.isna(lon):
            continue
        level = row.get("risk_level", "faible")
        color = RISK_COLORS.get(level, "#888")
        count = int(row.get("accident_count", 0))
        region = row.get("region", "Zone")
        gravity = row.get("avg_gravity", 2.0)
        label = RISK_LABELS.get(level, level)
        risk_sc = row.get("risk_score", "")

        # Zone de danger (cercle transparent)
        radius_m = max(5000, min(count * 800, 25000))
        folium.Circle(
            location=[lat, lon],
            radius=radius_m,
            color=color,
            fill=True,
            fill_opacity=0.15,
            weight=2,
            opacity=0.5,
        ).add_to(m)

        # Marqueur hotspot avec nombre d'accidents
        icon_html = f'''
        <div style="
            background:{color};
            color:{"#000" if level == "moyen" else "#fff"};
            border-radius:50%;
            width:36px;height:36px;
            display:flex;align-items:center;justify-content:center;
            font-weight:bold;font-size:13px;
            border:2px solid #fff;
            box-shadow:0 2px 6px rgba(0,0,0,0.4);
        ">{count}</div>
        '''

        risk_sc_html = f"Score de risque : <b>{risk_sc:.0f}/100</b><br>" if risk_sc != "" and not pd.isna(risk_sc) else ""

        popup_html = f"""
        <div style="font-family:sans-serif;min-width:200px;padding:4px">
          <b style="font-size:15px">Hotspot -- {region}</b><br>
          <span style="background:{color};color:{"#000" if level=="moyen" else "#fff"};
                padding:3px 10px;border-radius:10px;font-size:12px;display:inline-block;margin:4px 0">
            {label}
          </span>
          <hr style="margin:8px 0;border-color:#ddd">
          <b>{count}</b> accidents dans cette zone<br>
          Gravite moyenne : <b>{gravity:.1f}</b> / 3<br>
          {risk_sc_html}Position : {lat:.3f}, {lon:.3f}
        </div>
        """

        folium.Marker(
            location=[lat, lon],
            icon=folium.DivIcon(
                html=icon_html,
                icon_size=(36, 36),
                icon_anchor=(18, 18),
            ),
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"Hotspot {region} -- {count} accidents -- {label}",
        ).add_to(m)

# Afficher la carte
st_folium(m, height=560, use_container_width=True)

# -- Legende sous la carte --
st.markdown("""
<div style="display:flex;gap:15px;justify-content:center;padding:10px 0;font-size:13px;flex-wrap:wrap">
  <b>Gravite :</b>
  <span><span style="color:#28a745;font-size:16px">&#9679;</span> Leger</span>
  <span><span style="color:#fd7e14;font-size:16px">&#9679;</span> Grave</span>
  <span><span style="color:#dc3545;font-size:16px">&#9679;</span> Mortel</span>
  <b style="margin-left:15px">Hotspots :</b>
  <span><span style="color:#dc3545;font-size:16px">&#9673;</span> Critique</span>
  <span><span style="color:#fd7e14;font-size:16px">&#9673;</span> Eleve</span>
  <span><span style="color:#ffc107;font-size:16px">&#9673;</span> Moyen</span>
  <span><span style="color:#28a745;font-size:16px">&#9673;</span> Faible</span>
</div>
""", unsafe_allow_html=True)

# -- Statistiques par region --
st.divider()
st.subheader(f"Statistiques par region ({n_regions} regions)")

if not df_filtered.empty:
    stats = df_filtered.groupby("region").agg(
        accidents=("gravity", "count"),
        gravite_moy=("gravity", "mean"),
        mortels=("gravity", lambda x: (x == 3).sum()),
        graves=("gravity", lambda x: (x == 2).sum()),
    ).round(1).sort_values("accidents", ascending=False).reset_index()

    max_acc_r = int(stats["accidents"].max()) if not stats.empty else 1
    st.dataframe(
        stats,
        use_container_width=True,
        hide_index=True,
        column_config={
            "region":      "Region",
            "accidents":   st.column_config.ProgressColumn("Accidents", min_value=0, max_value=max_acc_r),
            "gravite_moy": st.column_config.NumberColumn("Gravite moy.", format="%.1f"),
            "mortels":     "Mortels",
            "graves":      "Graves",
        },
    )
