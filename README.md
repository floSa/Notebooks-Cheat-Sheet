# Notebooks-Cheat-Sheet

Organisation du dépôt en 3 familles + docs.

```
0_Documentation/                  Consignes, contrat de travail, statuts, références
                         → 00_workflow_contract.md = LE contrat à lire en premier
                         → 00_REFERENCE_ORIGINAUX.md = inventaire des vrais originaux
                         → 00_REFAITS_A_REVERIFIER.md = refaits bâtis sur base gutée

1_Old_Notebooks/         FAMILLE 1 — mes vrais notebooks d'origine (source de vérité)
  Notebooks.zip            source maître (committée)
  ipynb/                   44 originaux dézippés (gitignoré, régénérable)
  md/                      conversion jupytext des 44 originaux
  00_REFERENCE_ORIGINAUX.md

2_New_Notebooks/              FAMILLE 2 — travail récent (refontes + sujets neufs)
  ipynb/                   notebooks produits (ex-04_notebooks_finaux)
  md/                      markdown de travail (ex-03_md_ameliores)
  plans/                   plans des sujets neufs (ex-05_nouveaux_notebooks)

3_Sessions_Ratees/      FAMILLE 3 — anciennes sorties IA gutées (mauvaises)
  ipynb/                   64 notebooks réduits d'une session ratée
                           ⚠️ NE PAS réutiliser comme base — voir incident ci-dessous

scripts/                 outillage (restore, conversion jupytext, check_format)
data/                    datasets de travail
apps/                    apps Flask/FastAPI/Streamlit
```

## ⚠️ Incident 2026-06-01 (à connaître)

Les vrais originaux (`1_Old_Notebooks/`) n'étaient pas présents sur la machine lors des
sessions de refonte. `scripts/restore_originaux.sh` restaurait à tort depuis la branche
`main` (gutée, ~20 cellules au lieu de 200+), peuplant `3_Sessions_Ratees/`. Les refontes
de `2_New_Notebooks/` ont donc été bâties sur ce contenu réduit → voir
`0_Documentation/00_REFAITS_A_REVERIFIER.md` pour la liste des notebooks à recontrôler contre les
vrais originaux.

**Règle** : toute refonte lit `1_Old_Notebooks/ipynb/<nom>.ipynb` (les vrais), jamais
`main` ni `3_Sessions_Ratees/`.

## Restaurer les originaux après clone

```bash
bash scripts/restore_originaux.sh   # dézippe 1_Old_Notebooks/Notebooks.zip → ipynb/
```
