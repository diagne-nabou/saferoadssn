"""
SafeRoads SN — Page Prédiction
Formulaire → prédiction ML gravité + score de risque → log PostgreSQL.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import folium
from streamlit_folium import st_folium
from datetime import datetime
from utils.ml import predict, RISK_COLORS, GRAVITY_COLORS
from utils.db import log_prediction, query_df, test_connection

st.set_page_config(page_title="Prédiction — SafeRoads SN", page_icon="🔮", layout="wide")

st.title("🔮 Simulateur de prédiction")
st.caption("Renseignez les paramètres d'une situation routière pour obtenir une prédiction de risque.")

# ══════════════════════════════════════════════════════
# FORMULAIRE
# ══════════════════════════════════════════════════════

with st.form("prediction_form"):
    st.subheader("📍 Localisation & Contexte")
    c1, c2, c3 = st.columns(3)

    with c1:
        region = st.selectbox("Région", [
            "Dakar","Thies","Diourbel","Kaolack","Saint-Louis","Matam",
            "Fatick","Kaffrine","Tambacounda","Kedougou",
            "Ziguinchor","Sedhiou","Kolda","Louga",
        ])
        lat_defaults = {"Dakar":14.6937,"Thies":14.7886,"Kaolack":14.1652,
                        "Saint-Louis":16.0179,"Diourbel":14.6550,"Ziguinchor":12.5681,
                        "Tambacounda":13.7707,"Louga":15.6172,"Kolda":12.8983,
                        "Matam":15.6559,"Fatick":14.3390,"Kaffrine":14.1058,
                        "Kedougou":12.5605,"Sedhiou":12.7083}
        lon_defaults = {"Dakar":-17.4441,"Thies":-16.9260,"Kaolack":-16.0726,
                        "Saint-Louis":-16.4896,"Diourbel":-16.2323,"Ziguinchor":-16.2719,
                        "Tambacounda":-13.6673,"Louga":-16.2240,"Kolda":-14.9412,
                        "Matam":-13.2554,"Fatick":-16.4111,"Kaffrine":-15.5508,
                        "Kedougou":-12.1747,"Sedhiou":-15.5569}
        latitude  = st.number_input("Latitude",  value=lat_defaults[region], format="%.4f")
        longitude = st.number_input("Longitude", value=lon_defaults[region], format="%.4f")

    with c2:
        road_type    = st.selectbox("Type de route", [
            "nationale","autoroute","régionale","départementale","urbaine","piste"
        ])
        vehicle_type = st.selectbox("Type de véhicule", [
            "Voiture","Car rapide","Moto-Jakarta","Camion",
            "Sept-places","Taxi","Pickup","Charette","Autre"
        ])
        cause = st.selectbox("Cause présumée", [
            "Inconnue","Excès de vitesse","Somnolence/fatigue",
            "État dégradé route","Téléphone au volant","Alcool",
        ])

    with c3:
        nearby_accidents = st.slider("Accidents recensés dans la zone (5 km)", 0, 100, 5)
        spatial_density  = st.slider("Densité acc./km²", 0.0, 2.0, 0.1, 0.05)

    st.subheader("🕐 Temporalité")
    t1, t2, t3 = st.columns(3)
    with t1:
        hour = st.slider("Heure", 0, 23, datetime.now().hour)
    with t2:
        day_of_week = st.selectbox("Jour", range(7),
                                    format_func=lambda x: ["Lundi","Mardi","Mercredi","Jeudi","Vendredi","Samedi","Dimanche"][x])
    with t3:
        month = st.selectbox("Mois", range(1,13),
                              format_func=lambda x: ["Jan","Fév","Mar","Avr","Mai","Jun","Jul","Aoû","Sep","Oct","Nov","Déc"][x-1])

    st.subheader("🌤️ Conditions météo")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        is_rainy       = st.checkbox("Pluie en cours", value=False)
        weather_label  = "Pluie légère" if is_rainy else "Ensoleillé"
    with m2:
        precipitation  = st.number_input("Précipitations (mm)", 0.0, 100.0, 0.0, step=0.5)
    with m3:
        windspeed      = st.slider("Vent (km/h)", 0, 120, 15)
    with m4:
        visibility     = st.slider("Visibilité (km)", 0.0, 20.0, 10.0, 0.5)

    humidity = st.slider("Humidité (%)", 0, 100, 65)

    submitted = st.form_submit_button("🚀 Lancer la prédiction", use_container_width=True)

# ══════════════════════════════════════════════════════
# RÉSULTATS (stockés en session_state pour survivre aux reruns)
# ══════════════════════════════════════════════════════

if submitted:
    params = {
        "latitude": latitude, "longitude": longitude,
        "hour": hour, "day_of_week": day_of_week, "month": month,
        "is_rainy": is_rainy, "precipitation_mm": precipitation,
        "windspeed_kmh": windspeed, "temperature_c": 28.0,
        "visibility_km": visibility, "humidity_pct": humidity,
        "vehicle_type": vehicle_type, "road_type": road_type,
        "region": region, "cause": cause,
        "weather": weather_label,
        "nearby_accidents": nearby_accidents,
        "spatial_density": spatial_density,
    }

    with st.spinner("Calcul en cours..."):
        result = predict(params)

    st.session_state["prediction_result"] = result
    st.session_state["prediction_params"] = params

    # Sauvegarder dans PostgreSQL
    ok, _ = test_connection()
    if ok:
        log_prediction({
            "latitude":     latitude,
            "longitude":    longitude,
            "hour":         hour,
            "region":       region,
            "is_rainy":     is_rainy,
            "vehicle_type": vehicle_type,
            "gravity_label": result.get("gravity_label", "—"),
            "risk_score":   result.get("risk_score", 0),
            "risk_level":   result.get("risk_level", "—"),
        })

if "prediction_result" in st.session_state:
    result = st.session_state["prediction_result"]
    params = st.session_state["prediction_params"]

    st.divider()
    st.subheader("📊 Résultats de la prédiction")

    # ── Métriques principales ──
    r1, r2, r3 = st.columns(3)

    grav_color  = result.get("gravity_color",  "#888")
    risk_color  = result.get("risk_color",     "#888")
    grav_label  = result.get("gravity_label",  "—")
    risk_score  = result.get("risk_score",     0)
    risk_level  = result.get("risk_level",     "—")

    with r1:
        st.markdown(f"""
        <div style="background:#161B22;border:2px solid {grav_color};border-radius:10px;
                    padding:20px;text-align:center">
          <div style="font-size:13px;color:#8B949E;margin-bottom:6px">GRAVITE PREDITE</div>
          <div style="font-size:36px;font-weight:700;color:{grav_color}">{grav_label.upper()}</div>
        </div>
        """, unsafe_allow_html=True)

    with r2:
        bar_width = int(risk_score)
        st.markdown(f"""
        <div style="background:#161B22;border:2px solid {risk_color};border-radius:10px;
                    padding:20px;text-align:center">
          <div style="font-size:13px;color:#8B949E;margin-bottom:6px">SCORE DE RISQUE</div>
          <div style="font-size:36px;font-weight:700;color:{risk_color}">{risk_score}/100</div>
          <div style="background:#21262D;border-radius:4px;height:8px;margin-top:8px">
            <div style="background:{risk_color};width:{bar_width}%;height:8px;border-radius:4px"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with r3:
        st.markdown(f"""
        <div style="background:#161B22;border:2px solid {risk_color};border-radius:10px;
                    padding:20px;text-align:center">
          <div style="font-size:13px;color:#8B949E;margin-bottom:6px">NIVEAU DE RISQUE</div>
          <div style="font-size:36px;font-weight:700;color:{risk_color}">{risk_level.upper()}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Probabilités ──
    st.markdown("<br>", unsafe_allow_html=True)
    if "probabilities" in result:
        st.subheader("Probabilités par classe de gravité")
        proba = result["probabilities"]
        p1, p2, p3 = st.columns(3)
        p1.metric("Léger",  f"{proba.get('léger',0)*100:.1f}%")
        p2.metric("Grave",  f"{proba.get('grave',0)*100:.1f}%")
        p3.metric("Mortel", f"{proba.get('mortel',0)*100:.1f}%")

    # ── Recommandations ──
    st.subheader("Recommandations")
    for rec in result.get("recommendations", []):
        st.info(rec)

    # ── Mini-carte ──
    st.subheader("Localisation analysée")
    pred_lat = params["latitude"]
    pred_lon = params["longitude"]
    pred_region = params["region"]
    map_pred = folium.Map(location=[pred_lat, pred_lon], zoom_start=12)
    folium.CircleMarker(
        location=[pred_lat, pred_lon],
        radius=18,
        color=risk_color,
        fill=True,
        fill_color=risk_color,
        fill_opacity=0.7,
        popup=f"{pred_region} — Risque {risk_level} — Score {risk_score}",
        tooltip=f"{pred_region} | Risque : {risk_level}",
    ).add_to(map_pred)
    folium.Circle(
        location=[pred_lat, pred_lon],
        radius=5000,
        color=risk_color,
        fill=True,
        fill_opacity=0.1,
        weight=1,
    ).add_to(map_pred)
    st_folium(map_pred, height=300, use_container_width=True)

    ok, _ = test_connection()
    if ok:
        st.success("Prédiction enregistrée dans PostgreSQL")
    else:
        st.caption("PostgreSQL non connecté — prédiction non sauvegardée")

# ── Historique des prédictions ──
st.divider()
st.subheader("🕑 Historique des 10 dernières prédictions")
ok, _ = test_connection()
if ok:
    df_log = query_df("""
        SELECT created_at, region, hour, vehicle_type,
               gravity_label, risk_score, risk_level
        FROM predictions_log
        ORDER BY created_at DESC
        LIMIT 10
    """)
    if not df_log.empty:
        st.dataframe(df_log, use_container_width=True, hide_index=True,
                     column_config={
                         "created_at":    "Date/heure",
                         "region":        "Région",
                         "hour":          "Heure",
                         "vehicle_type":  "Véhicule",
                         "gravity_label": "Gravité",
                         "risk_score":    st.column_config.ProgressColumn("Score", min_value=0, max_value=100),
                         "risk_level":    "Niveau",
                     })
    else:
        st.info("Aucune prédiction enregistrée pour l'instant.")
else:
    st.info("Connectez PostgreSQL pour voir l'historique (page Administration).")
