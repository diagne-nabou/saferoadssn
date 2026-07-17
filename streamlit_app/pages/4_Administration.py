"""
SafeRoads SN — Page Administration
Import CSV accidents → normalisation → PostgreSQL.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from utils.db import (test_connection, init_schema, insert_accidents_df,
                      insert_hotspots_df, get_stats, query_df, get_conn)

st.set_page_config(page_title="Administration — SafeRoads SN", page_icon="⚙️", layout="wide")

st.title("⚙️ Administration — Gestion des données")

# ══════════════════════════════════════════════════════
# SECTION 1 — Connexion PostgreSQL
# ══════════════════════════════════════════════════════

st.subheader("🔌 Connexion PostgreSQL")

with st.expander("Configuration de la connexion", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        db_host = st.text_input("Hôte",     value="localhost")
        db_port = st.text_input("Port",     value="5432")
        db_name = st.text_input("Base",     value="saferoads")
    with col2:
        db_user = st.text_input("Utilisateur", value="saferoads_user")
        db_pass = st.text_input("Mot de passe", type="password", value="")

    if st.button("🔗 Tester la connexion", use_container_width=True):
        ok, msg = test_connection()
        if ok:
            st.success(msg)
        else:
            st.error(msg)
            st.code("""
# Installer PostgreSQL (Ubuntu/Debian)
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql

# Créer la base et l'utilisateur
sudo -u postgres psql -c "CREATE USER saferoads_user WITH PASSWORD 'votre_mdp';"
sudo -u postgres psql -c "CREATE DATABASE saferoads OWNER saferoads_user;"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE saferoads TO saferoads_user;"
            """, language="bash")

    if st.button("🏗️ Initialiser le schéma (créer les tables)", use_container_width=True):
        if init_schema():
            st.success("✅ Tables créées : accidents, hotspots, predictions_log")
        else:
            st.error("❌ Échec — vérifier la connexion")

# ── Statistiques BD ──
st.divider()
st.subheader("📈 État de la base de données")
ok, msg = test_connection()

if ok:
    stats = get_stats()
    s1, s2, s3, s4, s5, s6 = st.columns(6)
    s1.metric("🚗 Total accidents",  f"{stats.get('total',0):,}")
    s2.metric("🔴 Mortels",          f"{stats.get('mortels',0):,}")
    s3.metric("🟠 Graves",           f"{stats.get('graves',0):,}")
    s4.metric("🟢 Légers",           f"{stats.get('legers',0):,}")
    s5.metric("🗺️ Hotspots actifs",  f"{stats.get('hotspots',0):,}")
    s6.metric("🔮 Prédictions log",  f"{stats.get('preds',0):,}")
else:
    st.warning(f"PostgreSQL non connecté : {msg}")

# ══════════════════════════════════════════════════════
# HELPERS (defini avant utilisation)
# ══════════════════════════════════════════════════════

def _normalize(df_raw: pd.DataFrame, fmt: str) -> pd.DataFrame:
    """Normalisation rapide pour l'import admin."""
    import unicodedata

    def _strip_accents(s):
        if not isinstance(s, str): return s
        return "".join(c for c in unicodedata.normalize("NFD", s)
                      if unicodedata.category(c) != "Mn")

    if fmt == "senegal_real":
        df = df_raw.copy()
        rng = np.random.default_rng(42)

        if "flag_aberrant" in df.columns:
            df = df[df["flag_aberrant"] == 0]

        df["datetime"] = pd.to_datetime(df["date"], errors="coerce")
        # Recuperer lignes sans date depuis annee/mois
        mask_no_dt = df["datetime"].isna()
        if mask_no_dt.any():
            y = pd.to_numeric(df.loc[mask_no_dt, "annee"], errors="coerce").fillna(2025).astype(int)
            m = pd.to_numeric(df.loc[mask_no_dt, "mois"], errors="coerce").fillna(1).astype(int)
            df.loc[mask_no_dt, "datetime"] = pd.to_datetime(
                y.astype(str) + "-" + m.astype(str) + "-15", errors="coerce")
        year_raw = pd.to_numeric(df.get("annee", pd.Series(dtype=float)), errors="coerce")
        df["year"] = year_raw.fillna(df["datetime"].dt.year).fillna(2025).astype(int)
        month_raw = pd.to_numeric(df.get("mois", pd.Series(dtype=float)), errors="coerce")
        df["month"] = month_raw.fillna(df["datetime"].dt.month).fillna(1).astype(int)

        day_map = {"Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,
                   "Friday":4,"Saturday":5,"Sunday":6}
        if "jour_sem" in df.columns:
            df["day_of_week"] = df["jour_sem"].map(day_map).fillna(0).astype(int)
        else:
            df["day_of_week"] = df["datetime"].dt.dayofweek

        # Heure NON DISPONIBLE dans les donnees reelles
        df["hour"] = -1

        df["nb_morts"]   = pd.to_numeric(df.get("nb_morts", 0), errors="coerce").fillna(0)
        df["nb_blesses"] = pd.to_numeric(df.get("nb_blesses", 0), errors="coerce").fillna(0)
        df["gravity"] = 1
        df.loc[df["nb_blesses"] > 0, "gravity"] = 2
        df.loc[df["nb_morts"] > 0, "gravity"] = 3

        ville_map = {
            "Dakar":"Dakar","Pikine":"Dakar","Rufisque":"Dakar",
            "Thies":"Thies","Tivaouane":"Thies",
            "Mbour":"Mbour","Saly":"Mbour",
            "Kaolack":"Kaolack","Nioro":"Kaolack","Kaffrine":"Kaolack","Fatick":"Kaolack",
            "Saint-Louis":"Saint-Louis","Matam":"Saint-Louis",
            "Diourbel":"Diourbel","Touba":"Diourbel",
            "Ziguinchor":"Ziguinchor","Bignona":"Ziguinchor","Sedhiou":"Ziguinchor",
            "Tambacounda":"Tambacounda","Kedougou":"Tambacounda",
            "Louga":"Louga","Linguere":"Louga",
            "Kolda":"Kolda","Velingara":"Kolda",
        }
        ville_map_norm = {_strip_accents(k): v for k, v in ville_map.items()}
        df["region"] = df["ville"].apply(
            lambda v: ville_map_norm.get(_strip_accents(v), "Dakar") if isinstance(v, str) else "Dakar"
        )

        df["latitude"]  = pd.to_numeric(df.get("lat", np.nan), errors="coerce")
        df["longitude"] = pd.to_numeric(df.get("lon", np.nan), errors="coerce")

        reg_coords = {
            "Dakar":(14.6937,-17.4441),"Thies":(14.7886,-16.9260),
            "Mbour":(14.3850,-16.9653),"Kaolack":(14.1652,-16.0726),
            "Saint-Louis":(16.0179,-16.4896),"Diourbel":(14.6550,-16.2323),
            "Ziguinchor":(12.5681,-16.2719),"Tambacounda":(13.7707,-13.6673),
            "Louga":(15.6172,-16.2240),"Kolda":(12.8983,-14.9412),
        }
        mask_no = df["latitude"].isna() | df["longitude"].isna()
        for idx in df[mask_no].index:
            coords = reg_coords.get(df.at[idx, "region"], (14.6937, -17.4441))
            df.at[idx, "latitude"]  = coords[0] + rng.uniform(-0.2, 0.2)
            df.at[idx, "longitude"] = coords[1] + rng.uniform(-0.2, 0.2)
        df["geo_source"] = "real"
        df.loc[mask_no, "geo_source"] = "geocoded"

        veh_map = {"Voiture":"Voiture","4x4":"Voiture","Camion":"Camion",
                   "Camionnette":"Camion","Camion-citerne":"Camion",
                   "Bus":"Car rapide","Minicar":"Car rapide","Car":"Car rapide",
                   "Car rapide":"Car rapide","Moto":"Moto-Jakarta",
                   "Moto-taxi / Jakarta":"Moto-Jakarta","Deux-roues":"Moto-Jakarta",
                   "Taxi":"Taxi","7 places":"Sept-places","Pick-up":"Pickup",
                   "Charrette":"Charette"}
        if "type_vehicule" in df.columns:
            df["vehicle_type"] = df["type_vehicule"].map(veh_map).fillna("Autre")
        else:
            df["vehicle_type"] = "Autre"

        road_map = {"motorway":"autoroute","trunk":"nationale","primary":"nationale",
                    "secondary":"regionale","tertiary":"departementale",
                    "residential":"urbaine","unclassified":"piste"}
        if "road_type" in df.columns:
            df["road_type"] = df["road_type"].map(road_map).fillna("urbaine")
        else:
            df["road_type"] = "urbaine"

        cond_map = {"sec":"Ensoleille","pluie_legere":"Pluie legere",
                    "pluie_moderee":"Pluie legere","pluie_forte":"Pluie forte",
                    "pluie_tres_forte":"Orage"}
        if "condition_pluie" in df.columns:
            df["weather"] = df["condition_pluie"].map(cond_map).fillna("Inconnu")
            df["is_rainy"] = df["condition_pluie"].str.contains("pluie", na=False)
        else:
            df["weather"] = "Inconnu"
            df["is_rainy"] = False

        df["cause"]  = "Inconnue"
        df["source"] = "senegal_real"
        df["num_victims"]  = (df["nb_morts"] + df["nb_blesses"]).clip(1, 999).astype(int)
        df["num_vehicles"] = 2

    elif fmt == "kaggle":
        df = df_raw.copy()
        gravity_map = {"Slight Injury":1,"Serious Injury":2,"Fatal injury":3}
        weather_map = {"Normal":"Ensoleille","Raining":"Pluie legere",
                       "Raining and Windy":"Pluie forte","Cloudy":"Nuageux",
                       "Fog or mist":"Brouillard","Other":"Inconnu","Unknown":"Inconnu"}
        vehicle_map = {"Automobile":"Voiture","Lorry (11-40Q)":"Camion",
                       "Lorry (40Q+)":"Camion","Motorcycle":"Moto-Jakarta",
                       "Public (> 45 seats)":"Car rapide","Public (12-45 seats)":"Sept-places",
                       "Taxi":"Taxi","Other":"Autre"}

        df["gravity"]      = df.get("Casualty_severity","").map(gravity_map).fillna(1).astype(int)
        df["weather"]      = df.get("Weather_conditions","").map(weather_map).fillna("Inconnu")
        df["vehicle_type"] = df.get("Type_of_vehicle","").map(vehicle_map).fillna("Autre")
        df["cause"]        = df.get("Cause_of_accident","Inconnue").fillna("Inconnue")
        df["num_vehicles"] = pd.to_numeric(df.get("Number_of_vehicles_involved",1), errors="coerce").fillna(1).astype(int)
        df["num_victims"]  = pd.to_numeric(df.get("Number_of_casualties",1), errors="coerce").fillna(1).astype(int)

        if "Time" in df.columns:
            df["hour"] = pd.to_datetime(df["Time"], format="%H:%M:%S", errors="coerce").dt.hour.fillna(12).astype(int)
        else:
            df["hour"] = 12

        rng = np.random.default_rng(42)
        regions = [
            {"region":"Dakar","lat":14.6937,"lon":-17.4441,"w":0.25},
            {"region":"Thies","lat":14.7886,"lon":-16.9260,"w":0.15},
            {"region":"Kaolack","lat":14.1652,"lon":-16.0726,"w":0.10},
            {"region":"Diourbel","lat":14.6550,"lon":-16.2323,"w":0.09},
            {"region":"Mbour","lat":14.3850,"lon":-16.9653,"w":0.08},
            {"region":"Saint-Louis","lat":16.0179,"lon":-16.4896,"w":0.08},
            {"region":"Louga","lat":15.6172,"lon":-16.2240,"w":0.07},
            {"region":"Ziguinchor","lat":12.5681,"lon":-16.2719,"w":0.07},
            {"region":"Tambacounda","lat":13.7707,"lon":-13.6673,"w":0.06},
            {"region":"Kolda","lat":12.8983,"lon":-14.9412,"w":0.05},
        ]
        weights = [r["w"] for r in regions]
        chosen  = rng.choice(len(regions), size=len(df), p=weights)
        df["latitude"]  = [regions[i]["lat"] + rng.uniform(-0.3,0.3) for i in chosen]
        df["longitude"] = [regions[i]["lon"] + rng.uniform(-0.3,0.3) for i in chosen]
        df["region"]    = [regions[i]["region"] for i in chosen]
        df["geo_source"] = "simulated"
        df["datetime"]  = pd.Timestamp("2022-01-01")
        df["year"]   = 2022
        df["month"]  = rng.integers(1,13, size=len(df))
        df["day_of_week"] = rng.integers(0,7, size=len(df))
        df["is_rainy"] = df["weather"].isin(["Pluie legere","Pluie forte","Brouillard"])
        df["source"] = "kaggle"

    else:  # native
        df = df_raw.copy()
        df["datetime"]   = pd.to_datetime(df["datetime"], errors="coerce")
        df["year"]       = df["datetime"].dt.year
        df["month"]      = df["datetime"].dt.month
        df["day_of_week"]= df["datetime"].dt.dayofweek
        df["hour"]       = df["datetime"].dt.hour
        df["gravity"]    = pd.to_numeric(df["gravity"], errors="coerce").fillna(1).astype(int).clip(1,3)
        df["geo_source"] = "real"
        df["source"]     = "native"

    return df


# ══════════════════════════════════════════════════════
# SECTION 2 -- Import CSV accidents
# ══════════════════════════════════════════════════════

st.divider()
st.subheader("📥 Import de données accidents (CSV)")

tab1, tab2, tab3 = st.tabs(["📁 Upload CSV", "📂 Fichier local", "🗑️ Gestion table"])

with tab1:
    st.markdown("""
    **Formats acceptes :**
    - **Donnees reelles Senegal** (`accidents_senegal_meteo_final.csv` / `.xlsx`)
    - Dataset Kaggle [Road Traffic Accidents](https://kaggle.com/datasets/saurabhshahane/road-traffic-accidents)
    - CSV SafeRoads natif (colonnes : `datetime, latitude, longitude, gravity`)

    Le script detecte automatiquement le format et normalise les colonnes.
    """)

    uploaded = st.file_uploader("Choisir un fichier CSV ou XLSX", type=["csv", "xlsx", "xls"])

    if uploaded:
        # Charger selon le type
        if uploaded.name.endswith((".xlsx", ".xls")):
            df_raw = pd.read_excel(uploaded)
        else:
            df_raw = pd.read_csv(uploaded, low_memory=False)

        st.info(f"Fichier : {len(df_raw):,} lignes x {len(df_raw.columns)} colonnes")
        st.dataframe(df_raw.head(5), use_container_width=True)

        # Detection format
        kaggle_cols  = {"Casualty_severity", "Cause_of_accident", "Type_of_vehicle"}
        native_cols  = {"datetime", "latitude", "longitude", "gravity"}
        senegal_cols = {"date", "ville", "nb_morts", "nb_blesses"}

        if kaggle_cols.issubset(set(df_raw.columns)):
            fmt = "kaggle"
        elif native_cols.issubset(set(df_raw.columns)):
            fmt = "native"
        elif senegal_cols.issubset(set(df_raw.columns)):
            fmt = "senegal_real"
        else:
            fmt = "inconnu"

        fmt_labels = {
            "kaggle": "Kaggle (Road Traffic Accidents)",
            "native": "SafeRoads natif",
            "senegal_real": "Donnees reelles Senegal",
            "inconnu": "inconnu",
        }
        st.info(f"Format detecte : **{fmt_labels.get(fmt, fmt)}**")

        if fmt == "inconnu":
            st.warning("Format non reconnu. Colonnes minimum requises : "
                       "`datetime, latitude, longitude, gravity` (natif) "
                       "ou `date, ville, nb_morts, nb_blesses` (Senegal)")

        elif st.button("Normaliser et importer", use_container_width=True):
            with st.spinner("Normalisation en cours..."):
                df_clean = _normalize(df_raw, fmt)

            st.success(f"{len(df_clean):,} lignes normalisees")
            st.dataframe(df_clean.head(5), use_container_width=True)

            if ok:
                with st.spinner("Insertion PostgreSQL..."):
                    n_ok, n_err = insert_accidents_df(df_clean)
                st.success(f"{n_ok:,} accidents inseres | {n_err} erreurs")
            else:
                # Sauvegarder en CSV local
                save_path = Path(__file__).parent.parent.parent / "data/processed/accidents_clean.csv"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                df_clean.to_csv(save_path, index=False)
                st.info(f"PostgreSQL non disponible -- sauvegarde en CSV : {save_path}")

with tab2:
    st.markdown("**Charger le CSV depuis le pipeline ETL local :**")
    local_path = Path(__file__).parent.parent.parent / "data/processed/accidents_clean.csv"

    if local_path.exists():
        df_local = pd.read_csv(local_path)
        st.info(f"Trouvé : `{local_path}` — {len(df_local):,} lignes")
        st.dataframe(df_local.head(5), use_container_width=True)

        if st.button("📤 Importer dans PostgreSQL", use_container_width=True):
            if ok:
                with st.spinner("Insertion..."):
                    n_ok, n_err = insert_accidents_df(df_local)
                st.success(f"✅ {n_ok:,} accidents insérés | {n_err} erreurs")
            else:
                st.error("PostgreSQL non connecté")
    else:
        st.info(f"Aucun fichier trouvé : `{local_path}`\nLancer d'abord : `python scripts/run_etl.py`")

    # Hotspots
    st.markdown("**Charger les hotspots DBSCAN :**")
    hs_path = Path(__file__).parent.parent.parent / "data/processed/hotspots.csv"
    if hs_path.exists():
        df_hs = pd.read_csv(hs_path)
        st.info(f"Hotspots : {len(df_hs)} clusters")
        if st.button("📤 Importer hotspots dans PostgreSQL", use_container_width=True):
            if ok:
                n = insert_hotspots_df(df_hs)
                st.success(f"✅ {n} hotspots importés")
            else:
                st.error("PostgreSQL non connecté")

with tab3:
    st.markdown("### 🗑️ Opérations sur les tables")
    st.warning("⚠️ Ces opérations sont irréversibles.")

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🗑️ Vider accidents", use_container_width=True):
            if ok:
                conn = get_conn()
                conn.cursor().execute("TRUNCATE TABLE accidents RESTART IDENTITY")
                conn.commit(); conn.close()
                st.success("Table accidents vidée")
    with c2:
        if st.button("🗑️ Vider hotspots", use_container_width=True):
            if ok:
                conn = get_conn()
                conn.cursor().execute("TRUNCATE TABLE hotspots RESTART IDENTITY")
                conn.commit(); conn.close()
                st.success("Table hotspots vidée")
    with c3:
        if st.button("🗑️ Vider prédictions", use_container_width=True):
            if ok:
                conn = get_conn()
                conn.cursor().execute("TRUNCATE TABLE predictions_log RESTART IDENTITY")
                conn.commit(); conn.close()
                st.success("Table predictions_log vidée")

    st.divider()
    st.subheader("🔍 Aperçu des tables")
    table_sel = st.selectbox("Table", ["accidents", "hotspots", "predictions_log"])
    if ok:
        df_preview = query_df(f"SELECT * FROM {table_sel} ORDER BY id DESC LIMIT 20")
        if not df_preview.empty:
            st.dataframe(df_preview, use_container_width=True, hide_index=True)
        else:
            st.info("Table vide.")


