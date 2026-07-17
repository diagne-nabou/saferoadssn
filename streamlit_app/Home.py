"""
SafeRoads SN — Application Streamlit
Point d'entrée principal.

Usage :
    streamlit run streamlit_app/Home.py
"""

import streamlit as st

st.set_page_config(
    page_title="SafeRoads SN",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS global ──
st.markdown("""
<style>
  /* Police */
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');
  html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: #0D1117;
    border-right: 1px solid #21262D;
  }
  [data-testid="stSidebar"] * { color: #E6EDF3 !important; }

  /* Metric cards */
  [data-testid="metric-container"] {
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 8px;
    padding: 16px;
  }

  /* Boutons */
  .stButton > button {
    background: #238636;
    color: white;
    border: none;
    border-radius: 6px;
    font-weight: 600;
    padding: 8px 20px;
    transition: background 0.2s;
  }
  .stButton > button:hover { background: #2EA043; }

  /* Headers */
  h1 { color: #E6EDF3; font-weight: 700; }
  h2, h3 { color: #C9D1D9; }

  /* Risk badges */
  .badge-critique { background:#FF0000;color:#fff;padding:3px 10px;border-radius:12px;font-size:12px;font-weight:600; }
  .badge-eleve    { background:#FF8C00;color:#fff;padding:3px 10px;border-radius:12px;font-size:12px;font-weight:600; }
  .badge-moyen    { background:#FFD700;color:#000;padding:3px 10px;border-radius:12px;font-size:12px;font-weight:600; }
  .badge-faible   { background:#00AA00;color:#fff;padding:3px 10px;border-radius:12px;font-size:12px;font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──
with st.sidebar:
    st.markdown("## 🛣️ SafeRoads SN")
    st.caption("Prédiction accidents routiers · Sénégal")
    st.divider()
    st.markdown("""
    **Navigation**
    - 🏠 Accueil ← *vous êtes ici*
    - 🗺️ **Carte** — hotspots
    - 📊 **Dashboard** — statistiques
    - 🔮 **Prédiction** — simulateur
    - ⚙️ **Administration** — import données
    """)
    st.divider()
    st.caption("Gindima Group · Dakar, Sénégal")

# ── Page d'accueil ──
st.markdown("# 🛣️ SafeRoads SN")
st.markdown("### Système intelligent d'analyse et de prédiction des accidents routiers au Sénégal")
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    #### 🗺️ Carte interactive
    Visualisez les zones à risque sur la carte du Sénégal.
    Hotspots détectés par clustering géospatial DBSCAN.
    """)
    if st.button("Ouvrir la carte", use_container_width=True):
        st.switch_page("pages/1_Carte.py")

with col2:
    st.markdown("""
    #### 📊 Dashboard analytique
    Explorez les statistiques d'accidents par région,
    heure, gravité, type de véhicule et météo.
    """)
    if st.button("Voir le dashboard", use_container_width=True):
        st.switch_page("pages/2_Dashboard.py")

with col3:
    st.markdown("""
    #### 🔮 Simulateur de prédiction
    Saisissez les paramètres d'une situation routière
    et obtenez une prédiction de gravité + score de risque.
    """)
    if st.button("Lancer le simulateur", use_container_width=True):
        st.switch_page("pages/3_Prediction.py")

st.divider()

# ── Info stack ──
st.markdown("#### Stack technique")
c1, c2, c3, c4 = st.columns(4)
c1.info("**ML** · GradientBoosting\nRandomForest · DBSCAN")
c2.info("**Carte** · Folium\nLeaflet.js · OSM tiles")
c3.info("**Base** · PostgreSQL\npsycopg2 · SQLAlchemy")
c4.info("**Données** · OpenStreetMap\nOpen-Meteo · Kaggle")
