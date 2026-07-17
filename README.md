# SafeRoads SN

**Cartographie prédictive des risques routiers au Sénégal** — de la collecte de données non
structurées (presse en ligne) jusqu'à un dashboard cartographique pour décideurs.

> Pipeline complet : scraping presse → extraction par règles → enrichissement OSM/météo →
> ETL → ML (classification de gravité + transfer learning) → clustering spatial → dashboard.

Projet réalisé **en binôme** dans le cadre de la certification Data Science & IA — DIT AI Hub
Sénégal, 2025-2026.

---

## Contexte

Le Sénégal ne dispose pas de jeu de données ouvert et géolocalisé sur les accidents de la
route. Ce projet en **construit un** à partir de la presse en ligne, puis l'exploite pour :

- prédire la **gravité** d'un accident (léger / grave / mortel) ;
- identifier les **zones accidentogènes** par clustering spatial (DBSCAN) ;
- calculer un **score de risque** par localisation ;
- offrir un **dashboard interactif** à des décideurs non techniques.

**387 accidents** (2017-2024), construits à partir de **~382 articles de presse**.

---

## Données : une construction, pas un téléchargement

C'est le cœur du projet. Le jeu de données n'existait pas — il a été **collecté et structuré**.

1. **Scraping presse** (Selenium + BeautifulSoup) — `src/collect/`
   - Le Quotidien (`lequotidien.sn`), avec contournement de la protection anti-bot CleanTalk ;
   - E-Médias (`emedias.sn`) ; Dakaractu (appoint).
2. **Extraction d'information par règles** (regex + normalisation lexicale)
   - nombre de morts/blessés (mapping mots→chiffres), ville (liste de localités),
     filtre de pertinence « accident routier au Sénégal », gravité dérivée du bilan humain.
   - *Extraction à base de règles — pas d'apprentissage automatique.*
3. **Enrichissement géospatial et météo**
   - réseau routier via OpenStreetMap / Overpass (`src/collect/extract_osm_senegal.py`,
     `src/etl/download_osm.py`) ;
   - météo historique via Open-Meteo (`src/etl/download_weather.py`) ;
   - fusion en 36 features (`src/etl/merge_features.py`).

**Limites assumées :** couverture = ce que la presse rapporte (biais vers les accidents
graves/mortels) ; géolocalisation **géocodée à la ville**, pas GPS exact ; gravité **dérivée**
du bilan humain, pas un label officiel de police.

---

## Modèles

Classification de la gravité (3 classes) + variantes. Métriques issues des fichiers
`models/*_metrics.json`.

| Modèle | Accuracy | Validation croisée (5-fold) |
|---|---|---|
| XGBoost | 65,4 % | 65,6 % ± 5,8 % |
| RandomForest | 62,8 % | 64,6 % ± 3,9 % |
| MLP | 60,3 % | 59,7 % ± 5,5 % |
| LightGBM | 57,7 % | 56,0 % ± 8,5 % |
| Stacking | 52,6 % | 44,7 % ± 7,4 % |
| Binaire (mortel / non) | 67,9 % | 67,4 % ± 6,4 % |
| **Transfer Learning** | 88,5 % (test, n=78) | **82,2 % ± 3,8 %** |

### À lire honnêtement

- **Baseline classe majoritaire = 61,7 %** (prédire « mortel » à chaque fois). La plupart des
  classifieurs directs sont à peine au-dessus. **Seul le transfer learning s'en détache
  nettement : 82,2 % (CV) contre 61,7 %.**
- Le chiffre robuste du transfer est la **moyenne en validation croisée (82,2 %)** ;
  l'accuracy de 88,5 % porte sur un seul split de 78 points, donc optimiste.
- La classe **« grave » (22 obs sur 387)** est trop rare : les modèles ne la prédisent
  quasiment jamais correctement. Tous utilisent `class_weight="balanced"`, mais le déséquilibre
  reste la limite principale.

### Distribution des classes

| Classe | Effectif | Proportion |
|---|---|---|
| Mortel | 239 | 61,7 % |
| Léger | 126 | 32,6 % |
| Grave | 22 | 5,7 % |

---

## Architecture

```
saferoads/
├── src/
│   ├── collect/     ← scraping presse + extraction par règles + OSM (collecte des données)
│   ├── etl/         ← download OSM/météo, chargement, fusion des features
│   ├── ml/          ← entraînement, transfer learning, prédiction, clustering
│   ├── api/         ← API FastAPI
│   └── utils/       ← config, connexion PostgreSQL/PostGIS
├── streamlit_app/   ← dashboard 5 pages (carte, dashboard, prédiction, admin, modèles)
├── data/            ← raw / processed / external
├── models/          ← modèles .pkl + métriques .json
├── infra/           ← Docker, PostgreSQL/PostGIS, Nginx
├── scripts/         ← ETL, setup
└── docker-compose.yml
```

---

## Stack

Python 3.10+ · scikit-learn, XGBoost, LightGBM · Selenium, BeautifulSoup · GeoPandas, OSMnx,
Folium · Open-Meteo, Overpass · Streamlit, Plotly · FastAPI · PostgreSQL 16 + PostGIS 3.4 ·
Docker Compose.

---

## Démarrage rapide

```bash
git clone https://github.com/diagne-nabou/saferoadssn.git
cd saferoadssn
cp .env.example .env            # placeholders, à ajuster ; aucune clé requise pour Open-Meteo
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Dashboard
streamlit run streamlit_app/Home.py

# API REST (docs sur http://localhost:8000/docs)
uvicorn src.api.main:app --reload --port 8000
```

Ré-entraînement des modèles : `python src/ml/train.py --model all` puis
`python src/ml/transfer.py`.

---

## Contributions

Projet réalisé à deux.

- **Seynabou Diagne** — collecte des données (scraping presse, extraction par règles),
  enrichissement OSM/météo, pipeline ETL, dataset final.
- **Mouhamadou Tall** — modèle source (transfer learning sur données Kaggle) et son
  adaptation sur le dataset réel sénégalais, contributions à la modélisation.

Le transfer learning livré combine le modèle source du binôme (données Kaggle, 5000 lignes)
adapté par **feature augmentation** sur les 387 accidents réels issus de la collecte presse.

---

## Limites & suite

Petit dataset (387 obs), classes déséquilibrées, géolocalisation à la ville : les métriques
sont à lire avec prudence — la validation croisée stratifiée et les intervalles de confiance
larges sont donnés pour cette raison. Le projet reste **en cours de développement**.
