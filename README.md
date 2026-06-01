# Notebooks-Cheat-Sheet

Organisation du dépôt en 3 familles + docs.

```
0_DOCS/                  Consignes, contrat de travail, statuts, références
                         → 00_workflow_contract.md = LE contrat à lire en premier
                         → 00_REFERENCE_ORIGINAUX.md = inventaire des vrais originaux
                         → 00_REFAITS_A_REVERIFIER.md = refaits bâtis sur base gutée

1_MES_NOTEBOOKS/         FAMILLE 1 — mes vrais notebooks d'origine (source de vérité)
  Notebooks.zip            source maître (committée)
  ipynb/                   44 originaux dézippés (gitignoré, régénérable)
  md/                      conversion jupytext des 44 originaux
  00_REFERENCE_ORIGINAUX.md

2_NOUVEAUX/              FAMILLE 2 — travail récent (refontes + sujets neufs)
  ipynb/                   notebooks produits (ex-04_notebooks_finaux)
  md/                      markdown de travail (ex-03_md_ameliores)
  plans/                   plans des sujets neufs (ex-05_nouveaux_notebooks)

3_SESSIONS_RATEES/      FAMILLE 3 — anciennes sorties IA gutées (mauvaises)
  ipynb/                   64 notebooks réduits d'une session ratée
                           ⚠️ NE PAS réutiliser comme base — voir incident ci-dessous

scripts/                 outillage (restore, conversion jupytext, check_format)
data/                    datasets de travail
apps/                    apps Flask/FastAPI/Streamlit
```

## ⚠️ Incident 2026-06-01 (à connaître)

Les vrais originaux (`1_MES_NOTEBOOKS/`) n'étaient pas présents sur la machine lors des
sessions de refonte. `scripts/restore_originaux.sh` restaurait à tort depuis la branche
`main` (gutée, ~20 cellules au lieu de 200+), peuplant `3_SESSIONS_RATEES/`. Les refontes
de `2_NOUVEAUX/` ont donc été bâties sur ce contenu réduit → voir
`0_DOCS/00_REFAITS_A_REVERIFIER.md` pour la liste des notebooks à recontrôler contre les
vrais originaux.

**Règle** : toute refonte lit `1_MES_NOTEBOOKS/ipynb/<nom>.ipynb` (les vrais), jamais
`main` ni `3_SESSIONS_RATEES/`.

## Restaurer les originaux après clone

```bash
bash scripts/restore_originaux.sh   # dézippe 1_MES_NOTEBOOKS/Notebooks.zip → ipynb/
```
