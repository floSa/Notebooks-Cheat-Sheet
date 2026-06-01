---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

<!-- #region -->
# 🎈 Streamlit — Apps data en pur Python
<!-- #endregion -->

<!-- #region -->
Notebook **Cheat-sheet + Tutoriel** sur **Streamlit** : framework Python pour construire des **apps web data** sans HTML/CSS/JS, en quelques lignes.

Idéal pour : **prototypes**, **dashboards internes**, **demos ML**, **outils data team**.

Couverture :

1. **Concept** : exécution top-to-bottom + state.
2. **Inputs & outputs** widgets (sliders, selectbox, file upload, ...).
3. **Layout** : columns, tabs, expander, sidebar.
4. **Caching** : `@st.cache_data` et `@st.cache_resource`.
5. **Session state** : préserver entre interactions.
6. **Chatbot** (LLM) — pattern 2026.
7. **DataFrames interactifs** : `st.dataframe`, `st.data_editor`.
8. **Visualisations** : matplotlib, plotly, altair, st-charts natif.
9. **Multi-page apps**.
10. **Déploiement** : Streamlit Community Cloud, Hugging Face Spaces, Docker.
<!-- #endregion -->

<!-- #region -->
## 1. Concept Streamlit
<!-- #endregion -->

<!-- #region -->
**Particularité** : à chaque interaction utilisateur (clic, slider, input), Streamlit **réexécute le script Python du haut au bas**. Pas de callbacks, pas d'event handlers — c'est le script qui définit l'état à chaque run.

**Avantages** :

- Code linéaire, lisible, simple.
- Pas de gestion d'event loop.
- Hot-reload natif au save du fichier.

**Conséquences** :

- Tout calcul lourd doit être **caché** (`@st.cache_data`) pour ne pas refaire à chaque re-run.
- **Session state** explicite pour préserver l'état entre re-runs.
<!-- #endregion -->

<!-- #region -->
## 2. Hello + inputs basiques
<!-- #endregion -->

<!-- #region -->
```python
"""
# Sauvegarder dans app.py puis : streamlit run app.py
import streamlit as st
import pandas as pd
import numpy as np

st.title("Mon premier app Streamlit")
st.subheader("Sous-titre")
st.write("Markdown supporté : **gras**, `code`, $LaTeX$.")

# Inputs
name = st.text_input("Ton nom", value="Anonyme")
age = st.slider("Ton âge", 0, 120, 30)
fav_color = st.selectbox("Couleur préférée", ["rouge", "bleu", "vert"])
agree = st.checkbox("J'accepte les CGU")
n = st.number_input("Nombre", min_value=0, max_value=1000, value=10, step=1)
date = st.date_input("Date de naissance")
files = st.file_uploader("Upload (multiple)", accept_multiple_files=True)

# Bouton
if st.button("Valider"):
    st.success(f"Salut {name} ({age} ans, couleur {fav_color}) !")
    st.balloons()
"""
```
<!-- #endregion -->

<!-- #region -->
## 3. Layout
<!-- #endregion -->

<!-- #region -->
```python
"""
# Columns
col1, col2, col3 = st.columns([2, 1, 1])  # ratios
with col1:
    st.write("Large")
with col2:
    st.write("Petit")
with col3:
    st.write("Petit")

# Tabs
tab1, tab2 = st.tabs(["Données", "Modèle"])
with tab1:
    st.dataframe(df)
with tab2:
    st.write("Métriques")

# Expander
with st.expander("Détails techniques"):
    st.code("import sklearn ...")

# Sidebar (navigation, options globales)
with st.sidebar:
    section = st.radio("Section", ["EDA", "Modèle", "Prédiction"])
    st.write(f"Version : 1.0")
"""
```
<!-- #endregion -->

<!-- #region -->
## 4. Caching — `@st.cache_data` et `@st.cache_resource`
<!-- #endregion -->

<!-- #region -->
**Critique en Streamlit** — sans cache, ton DataFrame de 100 MB est rechargé à chaque slider bougé.

| Decorator | Pour |
|---|---|
| `@st.cache_data` | Données sérialisables (DataFrame, list, dict, fichiers downloadés) |
| `@st.cache_resource` | Objets globaux non-sérialisables (DB connection, ML model, LLM) |

```python
"""
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)   # exécuté une seule fois par valeur de path

@st.cache_resource
def load_model(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)   # chargé une seule fois, conservé entre re-runs

df = load_data("data.csv")           # rapide à chaque re-run grâce au cache
model = load_model("all-MiniLM-L6-v2")
"""
```
<!-- #endregion -->

<!-- #region -->
## 5. Session state
<!-- #endregion -->

<!-- #region -->
Pour des valeurs qui **persistent entre interactions** (compteur, messages chat, choix utilisateur) :

```python
"""
if "count" not in st.session_state:
    st.session_state.count = 0

if st.button("Incrément"):
    st.session_state.count += 1

st.write(f"Compteur : {st.session_state.count}")
"""
```

**Pattern courant** : initialiser dans le top du script, modifier via les callbacks ou inline.
<!-- #endregion -->

<!-- #region -->
## 6. Chatbot LLM — pattern 2026
<!-- #endregion -->

<!-- #region -->
Streamlit fournit des widgets `st.chat_message` et `st.chat_input` depuis 2023 — devenu LE standard pour les UIs chatbot LLM.

```python
"""
import streamlit as st
from transformers import pipeline

@st.cache_resource
def get_chatbot():
    return pipeline("text-generation", model="HuggingFaceTB/SmolLM2-135M-Instruct")


st.title("Chat LLM démo")

# Init historique
if "messages" not in st.session_state:
    st.session_state.messages = []

# Afficher l'historique
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input utilisateur
if prompt := st.chat_input("Pose ta question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Génération + streaming
    chatbot = get_chatbot()
    with st.chat_message("assistant"):
        # Pour streaming : st.write_stream(generator_qui_yield_tokens)
        response = chatbot(prompt, max_new_tokens=100, do_sample=False)[0]["generated_text"]
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
"""
```
<!-- #endregion -->

<!-- #region -->
## 7. DataFrames interactifs
<!-- #endregion -->

<!-- #region -->
```python
"""
df = pd.DataFrame(...)

# Read-only : table affichable, scroll, sort
st.dataframe(df, use_container_width=True, height=400)

# Éditable : data editor (Streamlit 1.21+) → renvoie le DataFrame modifié
edited = st.data_editor(df, num_rows="dynamic")

# Sélection de lignes (depuis 1.35)
event = st.dataframe(df, on_select="rerun", selection_mode="multi-row")
selected_rows = event.selection.rows
"""
```
<!-- #endregion -->

<!-- #region -->
## 8. Visualisations
<!-- #endregion -->

<!-- #region -->
```python
"""
# Matplotlib
fig, ax = plt.subplots()
ax.scatter(df["x"], df["y"])
st.pyplot(fig)

# Plotly (interactif)
import plotly.express as px
fig_px = px.scatter(df, x="x", y="y", color="cat")
st.plotly_chart(fig_px, use_container_width=True)

# Altair (recommandé pour stat plots)
import altair as alt
chart = alt.Chart(df).mark_circle().encode(x="x", y="y", color="cat")
st.altair_chart(chart)

# Streamlit charts natifs (super simples)
st.line_chart(df.set_index("date"))
st.bar_chart(df["cat"].value_counts())
st.map(df_with_lat_lon)
"""
```
<!-- #endregion -->

<!-- #region -->
## 9. Multi-page apps
<!-- #endregion -->

<!-- #region -->
**Méthode moderne 2024+** :

```python
"""
# app.py
import streamlit as st

pages = [
    st.Page("pages/eda.py", title="EDA", icon="📊"),
    st.Page("pages/model.py", title="Modèle", icon="🤖"),
    st.Page("pages/predict.py", title="Prédiction", icon="🔮"),
]
pg = st.navigation(pages)
pg.run()
"""
```

Ou la version **historique** (fichiers dans `pages/`, auto-routing) :

```
my_app/
├── app.py             # page d'accueil
└── pages/
    ├── 1_EDA.py
    ├── 2_Model.py
    └── 3_Predict.py
```
<!-- #endregion -->

<!-- #region -->
## 10. Déploiement
<!-- #endregion -->

<!-- #region -->
| Plateforme | Notes |
|---|---|
| **Streamlit Community Cloud** | Free, lié à un repo GitHub, parfait pour démos publiques |
| **Hugging Face Spaces** | Free, lié à un repo HF, populaire pour démos ML |
| **Railway / Render / Fly.io** | PaaS, free tier généreux |
| **Docker + n'importe quel cloud** | Plus de contrôle |
| **GCP Cloud Run / AWS App Runner** | Sans gérer les serveurs |

Dockerfile minimal :

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
```
<!-- #endregion -->

<!-- #region -->
## 11. Bonnes pratiques 2026
<!-- #endregion -->

<!-- #region -->
- **Cacher systématiquement** les chargements lourds (`@st.cache_data` / `@st.cache_resource`).
- **Utiliser session state** pour les widgets dont l'état doit survivre.
- **Sidebar** pour la nav globale + paramètres ; **tabs/columns** pour la nav locale.
- Pour les **dashboards qui doivent recharger en temps réel** : `st.rerun()` ou `time.sleep + st.empty()` (placeholder).
- **Auth** : pas natif Streamlit. Options : `streamlit-authenticator`, OAuth via Auth0 ou Cloudflare Access devant.
- **Tests** : `streamlit-app-test` (lib communautaire), ou tests d'intégration via `playwright`.
- **Alternative moderne** : **Gradio** (équivalent Streamlit centré ML demos, plus simple pour file-in / file-out).
<!-- #endregion -->

<!-- #region -->
## 12. Streamlit vs Gradio vs Dash vs Reflex
<!-- #endregion -->

<!-- #region -->
| Framework | Focus | Quand |
|---|---|---|
| **Streamlit** | Data apps généralistes | Démos, dashboards internes |
| **Gradio** | ML demos (HF native) | Demos modèles HF, partage rapide |
| **Dash (Plotly)** | Dashboards prod | Apps multi-utilisateurs, plus de contrôle CSS |
| **Reflex** | Full-stack Python → web | Quand tu veux du contrôle React-like sans JS |
| **Solara** | Apps Python pour Jupyter & web | Mix notebook + app |
<!-- #endregion -->

<!-- #region -->
## 13. Sources
<!-- #endregion -->

<!-- #region -->
- [Streamlit — docs](https://docs.streamlit.io/)
- [Streamlit gallery](https://streamlit.io/gallery)
- [Awesome Streamlit](https://github.com/MarcSkovMadsen/awesome-streamlit)
- [Gradio — docs](https://www.gradio.app/docs)
- Notebooks liés : `Flask_API`, `FastAPI_API`, `ML_MLFlow_Bench`.
<!-- #endregion -->
