# 📦 Palette de datasets

Datasets d'origine perdus → on remplace par une palette mutualisée. Principes :

- **Chargement programmatique** prioritaire (`sklearn`, `seaborn`, `datasets` HuggingFace, `statsmodels`) → rien à committer.
- **Fichiers téléchargés** uniquement si pas d'alternative (NASA, NYC Taxi parquet) → dans `data/_shared/<nom>/` avec un `README.md` qui donne la commande exacte.
- **Mutualisation** : un même dataset sert plusieurs notebooks d'un domaine (continuité pédagogique).

---

## 🗂️ Palette canonique

| Dataset | Source / chargement | Notebooks couverts | Volume |
|---|---|---|---|
| **Titanic** | `seaborn.load_dataset('titanic')` ou `sklearn.datasets.fetch_openml('titanic')` | EDA Visu, EDA Stats, Détection Outliers, Preprocessing, Test_données_manquante, ML classif baseline, AAA_Test_ML, **MLFlow_Bench**, Streamlit | <100 KB en mémoire |
| **California Housing** | `sklearn.datasets.fetch_california_housing()` | Régression Multiple, CV_GridSearch (regr), Bagging_Boosting (regr), Optimisation, **Feature_Importance_Selection** | ~700 KB |
| **UCI Adult (Income)** | `sklearn.datasets.fetch_openml('adult')` | Preprocessing (catégorielles), Generer_Donnees_Classification (compare synth vs réel), DL tabulaire pour comparatif | ~5 MB |
| **MNIST** | `sklearn.datasets.fetch_openml('mnist_784')` ou `torchvision.datasets.MNIST` | **DL_Frameworks_Comparatif**, DL_PyTorch, DL_TensorFlow, DL_Keras, DL_JAX (image) | ~15 MB |
| **Fashion-MNIST** (alternative) | idem | Variante MNIST si trop classique | ~30 MB |
| **20 Newsgroups** | `sklearn.datasets.fetch_20newsgroups()` | NLP_Classification_Supervisee, NLP_Classification_Smote, NLP_Recherche_d_informations (baseline TF-IDF) | ~14 MB |
| **IMDB Reviews** | `datasets.load_dataset('imdb')` | NLP_Classification_Smote (LSTM/BERT), NLP_Transformers (fine-tuning sentiment) | ~80 MB |
| **CoNLL-2003** | `datasets.load_dataset('conll2003')` | NLP_NER (transformers-first), NLP_NER_BiLSTM_CRF (wiki historique) | ~5 MB |
| **Wikipedia subset** | `datasets.load_dataset('wikipedia', '20220301.simple', split='train[:1%]')` | NLP_Recherche_d_informations (BERT/dense), **BDD_Vectorielles** (FAISS/Qdrant/LanceDB/pgvector), retrieval/RAG | ~50 MB |
| **Air Passengers** | `statsmodels.datasets.get_rdataset('AirPassengers')` | TS_Time_Series_Intro, TS_Time_Series_Overview, **TS_ARIMA** | <10 KB |
| **M4 (subset)** | `datasets.load_dataset('LeoTungAnh/m4_hourly')` OU CSV depuis github | TS_Time_Series_Overview (cas multi-séries), TS_Generer_Sequence (split/window) | ~10 MB |
| **NASA Turbofan (C-MAPSS)** | Download depuis NASA PCoE — `data/_shared/turbofan/README.md` | **TS_Maintenance_Predictive** (le case study sera dessus) | ~10 MB |
| **NYC Taxi (parquet sample)** | `wget https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet` | **BDD_DuckDB** (cas démo SQL/analytics) | ~50 MB |
| **librosa samples** | `librosa.example('trumpet')`, `librosa.example('libri1')` | **TdS_Introduction_Traitement_Signal** | bundled |
| **Sample PDFs** | `datasets.load_dataset('pdfa-eng-wds')` ou PDFs publics téléchargés | **DE_Docling** | ~variable |
| **OpenAI Gym / Gymnasium** | `import gymnasium as gym; gym.make('CartPole-v1')` | ML_Apprentissage_par_Renforcement (si extension RL au-delà des bandits) | n/a (environnement) |
| **Synthetic data** | `sklearn.datasets.make_*` | Structure_Generer_Donnees_Classification (déjà fait), DL_Deep_Learning_Maths (XOR/moons) | n/a |

---

## 🔁 Couverture multi-notebooks (vue inverse)

Qui partage quoi :

- **Titanic** → 8 notebooks (le plus mutualisé : EDA → preprocessing → ML classif → MLFlow)
- **California Housing** → 5 notebooks (toute la branche régression + interprétation)
- **MNIST** → 5 notebooks (toute la branche DL framework)
- **20 Newsgroups + IMDB** → 4 notebooks (NLP classique)
- **Wikipedia subset** → 3 notebooks (NLP retrieval + Vector DB)
- **Air Passengers** → 3 notebooks (TS classique)

---

## 📁 Structure `data/` proposée

```
data/
├── _shared/
│   ├── turbofan/           # NASA C-MAPSS — README.md avec download
│   │   ├── README.md
│   │   └── (CSV après téléchargement)
│   ├── nyc_taxi/           # NYC Taxi parquet
│   │   ├── README.md
│   │   └── (parquet après téléchargement)
│   └── pdfs_sample/        # PDFs pour Docling
│       └── README.md
└── <notebook_name>/        # uniquement si dataset spécifique non couvert ci-dessus
```

Pour tous les autres datasets (Titanic, California Housing, MNIST, 20News, IMDB, CoNLL, Wikipedia, Air Passengers, M4) → **chargement programmatique direct dans le notebook**, rien à committer.

---

## ⚠️ Points d'attention

- **HuggingFace `datasets`** : peut nécessiter `pip install datasets` (ajouter au pyproject quand on attaque la vague NLP).
- **`fetch_openml`** : premier appel télécharge dans `~/scikit_learn_data/` — penser à le mentionner dans les notebooks (cache).
- **Turbofan NASA** : URL change parfois — vérifier au moment de la refonte du TS_Maintenance.
- **NYC Taxi** : préférer un mois récent (Jan 2024+) pour le format parquet stable.
- **Wikipedia subset** : prendre le `simple` (anglais simplifié) pour avoir un volume raisonnable.
