# Collecte des données — presse en ligne sénégalaise

Cette étape amont a produit le dataset d'accidents utilisé par le reste du projet.
Elle n'était pas dans le repo d'origine ; elle est ajoutée ici pour rendre la démarche
vérifiable de bout en bout.

## Démarche

1. **Scraping presse** (Selenium + BeautifulSoup)
   - `scraping_lequotidien.ipynb` — Le Quotidien (`lequotidien.sn`), avec contournement
     de la protection anti-bot CleanTalk, pagination et extraction du contenu par article.
   - `scraping_emedia.ipynb` — E-Médias (`emedias.sn`), recherche « accident ».
   - Dakaractu — appoint (peu d'articles exploitables).

2. **Extraction d'information par règles** (regex + normalisation lexicale)
   - Nombre de morts / blessés : mapping mots→chiffres (« deux morts » → 2).
   - Ville : détection contre une liste de localités sénégalaises.
   - Filtre de pertinence : ne garder que les accidents **routiers au Sénégal**.
   - Gravité dérivée d'une règle : morts > 0 → mortel ; blessés > 0 → grave.
   - ⚠️ Extraction **à base de règles**, pas d'apprentissage automatique (ni spaCy, ni NER).

3. **Enrichissement géospatial** — `extract_osm_senegal.py`
   - Réseau routier / bâtiments / lieux habités via l'API Overpass (OpenStreetMap),
     par région du Sénégal.

## Sortie

~382 articles collectés → **387 accidents structurés** après nettoyage, alimentant
`data/raw/accidents/accidents.csv` (colonne `source` : `lequotidien_scrape`, `emedias`).

## Limites assumées

- Couverture = ce que la presse rapporte (biais vers les accidents graves/mortels).
- Géolocalisation **géocodée à la ville**, pas GPS exact.
- Gravité **dérivée** du bilan humain, pas un label officiel de police.

## Reproduire

Les notebooks sont fournis **sans sorties d'exécution**. Dépendances : `selenium`,
`beautifulsoup4`, `pandas`, `requests`, `shapely`, `tqdm`. Le scraping dépend de la
structure HTML des sites à la date d'exécution ; les articles peuvent avoir changé
ou disparu depuis.
