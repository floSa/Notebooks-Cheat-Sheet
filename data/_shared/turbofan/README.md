# NASA C-MAPSS Turbofan (FD001)

Dataset de référence pour la maintenance prédictive (RUL).
Utilisé par `TS_Maintenance_Predictive`.

## Chargement automatique

Le notebook télécharge et met en cache **ici** (`data/_shared/turbofan/`) les 3 fichiers FD001
via la fonction `load_cmapss_fd001()`. Aucune action manuelle nécessaire si le réseau est dispo.
Un **fallback synthétique** prend le relais hors-ligne.

## Fichiers (sous-dataset FD001)

| Fichier | Contenu |
|---|---|
| `train_FD001.txt` | 100 moteurs suivis jusqu'à la panne (run-to-failure), 20 631 lignes |
| `test_FD001.txt`  | 100 moteurs tronqués avant la panne, 13 096 lignes |
| `RUL_FD001.txt`   | Vraie RUL au dernier cycle observé des 100 moteurs test |

26 colonnes : `engine_id, cycle, op_1..op_3, sensor_1..sensor_21`.

## Téléchargement manuel (si besoin)

```bash
MIRROR=https://raw.githubusercontent.com/edwardzjl/CMAPSSData/master
for f in train_FD001.txt test_FD001.txt RUL_FD001.txt; do
  curl -L -o "$f" "$MIRROR/$f"
done
```

## Sources

- NASA PCoE : <https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/>
- Mirror : <https://github.com/edwardzjl/CMAPSSData>
- Saxena et al. (2008), PHM08 challenge.
