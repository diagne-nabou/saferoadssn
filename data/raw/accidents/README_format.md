# Format attendu — accidents.csv

Placer votre fichier CSV dans ce dossier sous le nom `accidents.csv`.

## Colonnes minimales requises

| Colonne | Type | Description | Exemple |
|---------|------|-------------|---------|
| `datetime` | string | Date et heure de l'accident | `2022-03-15 14:30:00` |
| `latitude` | float | Latitude GPS | `14.6937` |
| `longitude` | float | Longitude GPS | `-17.4441` |
| `gravity` | int | Gravité : 1=léger, 2=grave, 3=mortel | `2` |

## Colonnes optionnelles (enrichissent le modèle)

| Colonne | Type | Description |
|---------|------|-------------|
| `region` | string | Région du Sénégal |
| `road_type` | string | nationale / urbaine / piste |
| `vehicle_type` | string | car_rapide / camion / moto / voiture |
| `cause` | string | vitesse / somnolence / alcool / autre |
| `weather` | string | ensoleillé / pluie / nuageux |
| `num_vehicles` | int | Nombre de véhicules impliqués |
| `num_victims` | int | Nombre de victimes |

## Sources compatibles

- **Kaggle** : https://kaggle.com/datasets/saurabhshahane/road-traffic-accidents
  (Éthiopie — colonnes à mapper via `src/etl/load_accidents.py`)
- **ANSD Sénégal** : Format variable selon le rapport — mapping manuel requis
- **Dataset personnalisé** : Respecter les colonnes minimales ci-dessus

## Mapping automatique Kaggle → SafeRoads

Le script `src/etl/load_accidents.py` contient un mapper Kaggle intégré.
Il convertit automatiquement les colonnes du dataset éthiopien vers le
format SafeRoads, en conservant la structure géospatiale.
