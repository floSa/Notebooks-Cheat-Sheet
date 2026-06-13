# 🔍 Vérification : quels refait se sont basés sur les VRAIS originaux ?

> Généré par `scripts/verifier_base.py`. Compare les « signatures » (noms de fonctions/classes
> + titres de sections) présentes dans l'original mais ABSENTES du guté. Si le refait reprend
> ce signal « original-only », il s'est basé sur le vrai original.
>
> ⚠️ **Borne basse** : le workflow renommait souvent fonctions/titres. Un verdict ❌ peut donc
> masquer un refait qui a puisé dans l'original en renommant. Les ✅ sont sûrs ; les ❌ sont
> « probables » et à recontrôler visuellement.

## Résultat (ce portable, 2026-06-01)

| Verdict | Refait | reprise |
|---|---|---|
| ✅ basé sur l'original | DL_Deep_Learning_Maths | 63 % |
| ✅ basé sur l'original | EDA_Stats_Analyse_Desc_Visual | 27 % |
| 🟠 partiellement | DE_Docling | 16 % |
| 🟠 partiellement | Detection_Outliers | 14 % |
| ⚪ indéterminé (original ≈ guté, trop petit) | Flask_API, ML_MLFlow_Bench, NLP_Transformers, Test_donnees_manquante_modeles, TS_Generer_Sequence | — |
| ❌ guté seulement (à refaire/recontrôler) | BDD_DuckDB, BDD_Vectorielles, DL_KAN, DL_Keras, DL_TensorFlow, EDA_Analyse_Multivarie, EDA_Visualisation_Introduction, INRIA_SKLearn_MOOC, ML_Apprentissage_par_Renforcement, ML_Bagging_Boosting, ML_Explication_Feature_Importance_Selection, ML_Optimisation_de_Modeles, ML_Regression_Classification_CV_GridSearch, ML_Regression_Classification_Multiple, NLP_Classification_Smote, NLP_Classification_Supervisee, NLP_NER, NLP_NER_BiLSTM_CRF, NLP_Recherche_d_informations, Streamlit_brique, Structure_BDD_DataFrame, Structure_Generer_Donnees_Classification, Structure_Python, Structures_DataFrame, Structures_L_T_D_Cheat_Sheet, Structures_Preprocessing, TdS_Introduction_Traitement_Signal, TS_ARIMA, TS_Maintenance_Predictive, TS_Time_Series_Intro, TS_Time_Series_Overview | 0–6 % |

## Rejouer la vérif (depuis n'importe quelle machine, ex. PC fixe)

```bash
# 1. Récupérer la branche
git clone git@github.com-perso:floSa/Notebooks-Cheat-Sheet.git   # ou git pull
cd Notebooks-Cheat-Sheet
git checkout feat/restart-jupytext-workflow

# 2. Restaurer les vrais originaux depuis le zip (gitignorés, régénérables)
bash scripts/restore_originaux.sh

# 3. Lancer la vérif
uv run python scripts/verifier_base.py
```

Le script lit `Notebook_2018-2021/ipynb` (originaux), `3_Sessions_Ratees/ipynb` (gutés)
et `Notebook_2026/ipynb` (refait). Ajuste la table `ALIAS` dans le script si un renommage
n'est pas reconnu.
