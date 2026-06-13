# Notebooks-Cheat-Sheet

Collection de cheat-sheets data science en notebooks Jupyter, organisée **par génération**.

## Structure

```
Notebook_2018-2021/   Génération d'origine (2018-2021)
  Notebooks.zip         archive source — committée (source de vérité)
  ipynb/                originaux dézippés (gitignoré, régénérable)
  md/                   conversion jupytext des originaux

Notebook_2026/        Génération 2026 — refonte complète, BASE COURANTE
  ipynb/                44 notebooks à jour
  md/                   markdown de travail (source jupytext)
  plans/                plans des sujets neufs
  _doc/                 journal de la refonte 2026 (suivi, décisions)
  COLAB_READINESS.md    état Colab par notebook

Documentation/        Méthode réutilisable pour les prochaines mises à jour
                        (consignes de format, contrat de travail, stratégie datasets, blueprint DL)
scripts/              outillage (conversion jupytext, check_format, download_data, restore)
data/                 datasets de travail
apps/                 démos Flask / FastAPI / Streamlit
```

## Convention de versionnement

Chaque génération vit dans son propre dossier `Notebook_<période>`. La plus récente
(`Notebook_2026`) est la **base courante**. Pour une mise à jour future (dans un an ou
plus) : repartir de la base courante vers un nouveau `Notebook_<année>`, et garder les
générations précédentes comme archives. La méthode de travail réutilisable est dans
`Documentation/`.

## Utilisation

Restaurer les originaux après un clone :

```bash
bash scripts/restore_originaux.sh   # dézippe Notebook_2018-2021/Notebooks.zip -> ipynb/
```

Télécharger les jeux de données des case studies (les autres sont chargés à la volée
via sklearn / seaborn / HuggingFace) :

```bash
bash scripts/download_data.sh
```

Convertir entre `.md` (source éditable) et `.ipynb` (généré) via jupytext :

```bash
bash scripts/02_ipynb_to_md.sh                 # ipynb -> md (originaux)
bash scripts/04_md_to_ipynb.sh [fichier.md]    # md -> ipynb (génération 2026)
```

Auditer le format d'un notebook avant/après conversion :

```bash
python scripts/check_format.py --both Notebook_2026/md/NOM.md Notebook_2026/ipynb/NOM.ipynb
```

## Tester sur Google Colab

Les notebooks récupèrent leurs données par programme (sklearn / seaborn / HuggingFace) ;
un seul (`TS_Maintenance_Predictive`) demande un fichier à télécharger manuellement.
Le détail par notebook (données, `pip install` éventuel) est dans
[`Notebook_2026/COLAB_READINESS.md`](Notebook_2026/COLAB_READINESS.md).
