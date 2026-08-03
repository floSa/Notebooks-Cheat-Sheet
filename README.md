# Notebooks-Cheat-Sheet

**Collection de cheat-sheets data science en notebooks Jupyter, organisée par génération — 44 fiches couvrant structures Python, EDA, machine learning, deep learning, NLP, séries temporelles et bases de données.**

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![uv](https://img.shields.io/badge/uv-package_manager-DE5FE9?logo=uv&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-notebooks-F37626?logo=jupyter&logoColor=white)
![jupytext](https://img.shields.io/badge/jupytext-1.19-5C2D91)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?logo=scikitlearn&logoColor=white)

## Index des cheat-sheets

Base courante : `Notebook_2026/ipynb/` (**44 notebooks**). L'état Colab par notebook (données,
`pip install` éventuel) est dans [`Notebook_2026/COLAB_READINESS.md`](Notebook_2026/COLAB_READINESS.md).

**Structures & Python**
- [Structure_Python](Notebook_2026/ipynb/Structure_Python.ipynb)
- [Structures_L_T_D_Cheat_Sheet](Notebook_2026/ipynb/Structures_L_T_D_Cheat_Sheet.ipynb)
- [Structures_DataFrame](Notebook_2026/ipynb/Structures_DataFrame.ipynb)
- [Structure_BDD_DataFrame](Notebook_2026/ipynb/Structure_BDD_DataFrame.ipynb)
- [Structures_Preprocessing](Notebook_2026/ipynb/Structures_Preprocessing.ipynb)
- [Structure_Generer_Donnees_Classification](Notebook_2026/ipynb/Structure_Generer_Donnees_Classification.ipynb)

**EDA & visualisation**
- [EDA_Visualisation_Introduction](Notebook_2026/ipynb/EDA_Visualisation_Introduction.ipynb)
- [EDA_Stats_Analyse_Desc_Visual](Notebook_2026/ipynb/EDA_Stats_Analyse_Desc_Visual.ipynb)
- [EDA_Analyse_Multivarie](Notebook_2026/ipynb/EDA_Analyse_Multivarie.ipynb)

**Machine learning**
- [ML_Regression_Classification_Multiple](Notebook_2026/ipynb/ML_Regression_Classification_Multiple.ipynb)
- [ML_Regression_Classification_CV_GridSearch](Notebook_2026/ipynb/ML_Regression_Classification_CV_GridSearch.ipynb)
- [ML_Bagging_Boosting](Notebook_2026/ipynb/ML_Bagging_Boosting.ipynb)
- [ML_Optimisation_de_Modeles](Notebook_2026/ipynb/ML_Optimisation_de_Modeles.ipynb)
- [ML_Explication_Feature_Importance_Selection](Notebook_2026/ipynb/ML_Explication_Feature_Importance_Selection.ipynb)
- [ML_MLFlow_Bench](Notebook_2026/ipynb/ML_MLFlow_Bench.ipynb)
- [ML_Apprentissage_par_Renforcement](Notebook_2026/ipynb/ML_Apprentissage_par_Renforcement.ipynb)
- [INRIA_SKLearn_MOOC](Notebook_2026/ipynb/INRIA_SKLearn_MOOC.ipynb)

**Deep learning**
- [DL_Deep_Learning_Maths](Notebook_2026/ipynb/DL_Deep_Learning_Maths.ipynb)
- [DL_Frameworks_Comparatif](Notebook_2026/ipynb/DL_Frameworks_Comparatif.ipynb)
- [DL_PyTorch](Notebook_2026/ipynb/DL_PyTorch.ipynb)
- [DL_TensorFlow](Notebook_2026/ipynb/DL_TensorFlow.ipynb)
- [DL_Keras](Notebook_2026/ipynb/DL_Keras.ipynb)
- [DL_JAX](Notebook_2026/ipynb/DL_JAX.ipynb)
- [DL_KAN_Kolmogorov_Arnold](Notebook_2026/ipynb/DL_KAN_Kolmogorov_Arnold.ipynb)

**NLP**
- [NLP_Classification_Supervisee](Notebook_2026/ipynb/NLP_Classification_Supervisee.ipynb)
- [NLP_Classification_Smote](Notebook_2026/ipynb/NLP_Classification_Smote.ipynb)
- [NLP_NER](Notebook_2026/ipynb/NLP_NER.ipynb)
- [NLP_NER_BiLSTM_CRF](Notebook_2026/ipynb/NLP_NER_BiLSTM_CRF.ipynb)
- [NLP_Transformers](Notebook_2026/ipynb/NLP_Transformers.ipynb)
- [NLP_Recherche_d_informations](Notebook_2026/ipynb/NLP_Recherche_d_informations.ipynb)

**Séries temporelles & signal**
- [TS_Time_Series_Intro](Notebook_2026/ipynb/TS_Time_Series_Intro.ipynb)
- [TS_Time_Series_Overview](Notebook_2026/ipynb/TS_Time_Series_Overview.ipynb)
- [TS_ARIMA](Notebook_2026/ipynb/TS_ARIMA.ipynb)
- [TS_Generer_Sequence](Notebook_2026/ipynb/TS_Generer_Sequence.ipynb)
- [TS_Maintenance_Predictive](Notebook_2026/ipynb/TS_Maintenance_Predictive.ipynb)
- [TdS_Introduction_Traitement_Signal](Notebook_2026/ipynb/TdS_Introduction_Traitement_Signal.ipynb)

**Bases de données**
- [BDD_DuckDB](Notebook_2026/ipynb/BDD_DuckDB.ipynb)
- [BDD_Vectorielles](Notebook_2026/ipynb/BDD_Vectorielles.ipynb)

**Qualité & robustesse des données**
- [Detection_Outliers](Notebook_2026/ipynb/Detection_Outliers.ipynb)
- [Test_donnees_manquante_modeles](Notebook_2026/ipynb/Test_donnees_manquante_modeles.ipynb)

**Ingénierie documentaire**
- [DE_Docling](Notebook_2026/ipynb/DE_Docling.ipynb)

**APIs & apps** (versions exécutables dans [`apps/`](apps/README.md))
- [Flask_API](Notebook_2026/ipynb/Flask_API.ipynb)
- [FastAPI_API](Notebook_2026/ipynb/FastAPI_API.ipynb)
- [Streamlit_brique](Notebook_2026/ipynb/Streamlit_brique.ipynb)

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
