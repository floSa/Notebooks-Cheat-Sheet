---
jupyter:
  jupytext:
    notebook_metadata_filter: -jupytext.text_representation.jupytext_version
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
  kernelspec:
    display_name: Python 3
    name: python3
---

<!-- #region id="_tbGMmk0QtTe" -->
# Briques
<!-- #endregion -->

<!-- #region id="2FxO7qJNmhrZ" -->
## Bouton asynchrone
<!-- #endregion -->

```python id="A9Vh0zbAQWnc"
import streamlit as st

if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked = False
    st.session_state.cpt = 0  # Initialisation de cpt dans session_state

def click_button():
    st.session_state.button_clicked = True
    st.session_state.cpt += 1  # Incrémentation de cpt dans session_state

st.button("Cliquez-moi", on_click=click_button)

if st.session_state.button_clicked:
    st.write("Le bouton a été cliqué ! cpt = " + str(st.session_state.cpt) )
else:
    st.write("Le bouton non cliqué ! cpt = "+ str(st.session_state.cpt) )

st.write("fin de code")
```

<!-- #region id="OaXh5UbiQRuq" -->
## Test Autocomplétion
<!-- #endregion -->

<!-- #region id="aLPQvjYJbqHj" -->
### Chat message
<!-- #endregion -->

```python id="dUpE8f9udHAo"
! pip install streamlit-textcomplete -q
```

```python id="moB2WVHabsGd"
import streamlit as st
from textcomplete import textcomplete, StrategyProps

####################################################################

# CSS personnalisé pour ajuster la position de la fenêtre de suggestions
st.markdown(
    """
    <style>
    .textcomplete-dropdown {
        position: absolute !important;
        z-index: 1000;
        left: auto !important;
        top: auto !important;
    }
    </style>
    """, unsafe_allow_html=True
)

ma_liste = "'toto', 'tata', 'tutu'"

# Définir une stratégie d'autocomplétion pour les mentions
mention_strategy = StrategyProps(
    id="mentionUser",
    match=r"\B@(\w*)$",  # Regex pour détecter '@' suivi d'un mot
    search="async (term, callback) => callback(["+ma_liste+"].filter(name => name.toLowerCase().includes(term.toLowerCase())))",
    replace="(username) => `${username}`",
    template="(username) => `${username}`",
)

textcomplete(
    area_label="Envoyer un message...",
    strategies=[mention_strategy],
    max_count=3,
)

# Entrée utilisateur
if user_input := st.chat_input("Envoyer un message..."):
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

```

<!-- #region id="DRHUffNXlvMn" -->
### Text Area
<!-- #endregion -->

```python id="sW0l4t77lzu_"
import streamlit as st
from textcomplete import textcomplete, StrategyProps

# CSS personnalisé pour ajuster la position de la fenêtre de suggestions
st.markdown(
    """
    <style>
    .textcomplete-dropdown {
        position: absolute !important;
        z-index: 1000;
        left: auto !important;
        top: auto !important;
    }
    </style>
    """, unsafe_allow_html=True
)

# Définir un label pour la zone de texte
label = "Exemple d'autocomplétion ajusté"

# Initialiser une zone de texte avec Streamlit
texte = st.text_area(
    label=label,
    key="zone_text_autocomplete",
)

# Définir une stratégie simple d'autocomplétion pour les mentions
mention_strategy = StrategyProps(
    id="mentionUser",
    match=r"\B@(\w*)$",  # Regex pour détecter '@' suivi d'un mot
    search="async (term, callback) => callback(['pomme', 'poire', 'banane', 'cerise', 'fraise', 'orange', 'mandarine', 'kiwi', 'ananas', 'mangue', 'raisin', 'myrtille', 'framboise', 'cassis', 'groseille', 'prune', 'abricot', 'pêche', 'nectarine', 'figue','citron', 'pamplemousse', 'avocat', 'passion', 'goyave', 'litchi', 'durian', 'papaye', 'grenade', 'mûre'].filter(name => name.toLowerCase().includes(term.toLowerCase())))",
    replace="(username) => `${username}`",
    template="(username) => `👤 ${username}`",
)

# Initialiser le composant d'autocomplétion avec la stratégie
textcomplete(
    area_label=label,
    strategies=[mention_strategy],
    max_count=3,  # Limiter le nombre de suggestions
)

```

<!-- #region id="f0KbXJnv7H6t" -->
## ChatBot CBA
<!-- #endregion -->

```python id="qnzkEng67L6D"
import random
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from textcomplete import textcomplete, StrategyProps
import numpy as np

# Initialisation de l'état de session

if "messages" not in st.session_state:
    st.session_state.messages = []
if "df_received" not in st.session_state:
    st.session_state.df_received = False
if "plots" not in st.session_state:
    st.session_state.plots = []

####################################################################

# Fonction simulant la réponse d'un backend Flask
def simulate_flask_response(input_text):
    if not input_text:
        return None
    response_type = random.choice([0, 1])

    if response_type == 0:
        return {"type": 0, "content": f"Réponse du bot pour: {input_text}"}
    else:
        n_rows = random.randint(5, 20)
        categories = ['A', 'B', 'C']
        data = {
            'Cat1': [random.choice(categories) for _ in range(n_rows)],
            'Cat2': [random.choice(categories) for _ in range(n_rows)],
            'Cat3': [random.choice(categories) for _ in range(n_rows)],
            'Quantitative1': np.random.uniform(0, 20, n_rows),
            'Quantitative2': np.random.uniform(0, 20, n_rows),
            'DateTime': pd.date_range(start='2022-01-01', periods=n_rows)
        }
        return {"type": 1, "content": data}

####################################################################

def plot_qualitative_vs_qualitative(df, x_col, y_col):
    fig, ax = plt.subplots(figsize=(6, 4))
    crosstab = pd.crosstab(df[x_col], df[y_col])
    crosstab.plot(kind='bar', stacked=True, ax=ax, color=plt.cm.tab10.colors)
    ax.set_title(f"Comptes croisés de {x_col} et {y_col}")
    ax.set_xlabel(x_col)
    ax.set_ylabel("Compte")
    return fig

def plot_qualitative_vs_quantitative(df, x_col, y_col):
    # Créer un graphique circulaire (camembert)
    fig, ax = plt.subplots(figsize=(6, 4))
    # Agréger les données : somme des valeurs quantitatives pour chaque catégorie qualitative
    agg_data = df.groupby(x_col)[y_col].sum().sort_values(ascending=False)
    # Tracer le camembert (pie chart) avec les données triées
    ax.pie(agg_data, labels=agg_data.index, autopct='%1.1f%%', startangle=90)
    # Ajouter un titre
    ax.set_title(f'{y_col} par {x_col}')
    # S'assurer que le graphique est bien un cercle
    ax.axis('equal')  # Égaliser les axes pour faire un cercle parfait
    return fig

def plot_quantitative_vs_quantitative(df, x_col, y_col):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df[x_col], df[y_col], alpha=0.7, color='blue')
    ax.set_title(f'{y_col} en fonction de {x_col}')
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    return fig

def plot_qualitative_vs_time(df, x_col, y_col):
    fig, ax = plt.subplots(figsize=(6, 4))
    df = df.sort_values(x_col)
    categories = df[y_col].unique()
    colors = plt.cm.tab10.colors[:len(categories)]
    for cat, color in zip(categories, colors):
        subset = df[df[y_col] == cat]
        ax.scatter(subset[x_col], [cat] * len(subset), label=cat, color=color, alpha=0.7)
    ax.set_title(f'{y_col} en fonction de {x_col}')
    ax.set_xlabel(x_col)
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories)
    ax.legend(title=y_col)
    plt.xticks(rotation=45)
    return fig

def plot_quantitative_vs_time(df, x_col, y_col):
    fig, ax = plt.subplots(figsize=(6, 4))
    df = df.sort_values(x_col)
    ax.scatter(df[x_col], df[y_col], alpha=0.7, color='blue')
    ax.set_title(f'{y_col} en fonction de {x_col}')
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    plt.xticks(rotation=45)
    return fig

def reorder_columns(df, x_col, y_col):
    x_type = "datetime" if pd.api.types.is_datetime64_any_dtype(df[x_col]) else \
             "numeric" if pd.api.types.is_numeric_dtype(df[x_col]) else "categorical"
    y_type = "datetime" if pd.api.types.is_datetime64_any_dtype(df[y_col]) else \
             "numeric" if pd.api.types.is_numeric_dtype(df[y_col]) else "categorical"

    if x_type == "categorical" and y_type == "numeric":
        return x_col, y_col
    elif x_type == "numeric" and y_type == "categorical":
        return y_col, x_col
    elif x_type == "datetime" or (x_type == "numeric" and y_type == "numeric"):
        return x_col, y_col
    elif y_type == "datetime":
        return y_col, x_col
    else:
        return x_col, y_col

def generate_plot(df, x_col, y_col):
    try:
        x_col, y_col = reorder_columns(df, x_col, y_col)

        x_type = "datetime" if pd.api.types.is_datetime64_any_dtype(df[x_col]) else \
                 "numeric" if pd.api.types.is_numeric_dtype(df[x_col]) else "categorical"
        y_type = "datetime" if pd.api.types.is_datetime64_any_dtype(df[y_col]) else \
                 "numeric" if pd.api.types.is_numeric_dtype(df[y_col]) else "categorical"

        if x_type == "categorical" and y_type == "categorical":
            return plot_qualitative_vs_qualitative(df, x_col, y_col), None
        elif x_type == "categorical" and y_type == "numeric":
            return plot_qualitative_vs_quantitative(df, x_col, y_col), None
        elif x_type == "numeric" and y_type == "numeric":
            return plot_quantitative_vs_quantitative(df, x_col, y_col), None
        elif x_type == "datetime" and y_type == "categorical":
            return plot_qualitative_vs_time(df, x_col, y_col), None
        elif x_type == "datetime" and y_type == "numeric":
            return plot_quantitative_vs_time(df, x_col, y_col), None
        else:
            return None, "Type de données non pris en charge"
    except Exception as e:
        return None, str(e)

####################################################################

# Initialisation de l'application Streamlit
st.title("Chatbot CBA Front testV0.2 18/11/2024")

# Charger le fichier d'aide
help_table = pd.read_csv("research_help.csv", sep=';', encoding='latin-1', header=0)

# Ajout de la barre de recherche dans la barre latérale
search_term = st.sidebar.text_input("Research help", "")

# Filtrer le DataFrame selon la chaîne de recherche
if search_term:
    mask = help_table.apply(lambda row: row.astype(str).str.contains(search_term, case=False).any(), axis=1)
    filtered_df = help_table[mask]
else:
    filtered_df = help_table

# Affichage du DataFrame filtré
st.dataframe(filtered_df, width=800, height=110, hide_index=True)

####################################################################

# CSS personnalisé pour ajuster la position de la fenêtre de suggestions
st.markdown(
    """
    <style>
    .textcomplete-dropdown {
        position: absolute !important;
        z-index: 1000;
        left: auto !important;
        top: auto !important;
    }
    </style>
    """, unsafe_allow_html=True
)

ma_liste = "'toto', 'tata', 'tutu'"

# Définir une stratégie d'autocomplétion pour les mentions
mention_strategy = StrategyProps(
    id="mentionUser",
    match=r"\B@(\w*)$",  # Regex pour détecter '@' suivi d'un mot
    search="async (term, callback) => callback(["+ma_liste+"].filter(name => name.toLowerCase().includes(term.toLowerCase())))",
    replace="(username) => `${username}`",
    template="(username) => `${username}`",
)

textcomplete(
    area_label="Envoyer un message...",
    strategies=[mention_strategy],
    max_count=3,
)

####################################################################

# Affichage des messages de l'historique
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if isinstance(message["content"], pd.DataFrame):
            st.dataframe(message["content"], width=800, height=200, hide_index=True)
        elif isinstance(message["content"], plt.Figure):
            st.pyplot(message["content"])
        else:
            st.markdown(message["content"])

# Entrée utilisateur
if user_input := st.chat_input("Envoyer un message..."):
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Simuler une réponse de l'assistant
    response = simulate_flask_response(user_input)
    if response["type"] == 0:  # Réponse texte
        assistant_response = response["content"]
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        with st.chat_message("assistant"):
            st.markdown(assistant_response)
    elif response["type"] == 1:  # Réponse DataFrame
        df = pd.DataFrame(response["content"])
        st.session_state.messages.append({"role": "assistant", "content": df})
        st.session_state.df_received = True
        with st.chat_message("assistant"):
            st.dataframe(df, width=800, height=200, hide_index=True)

####################################################################

# Si un DataFrame est reçu, l'enregistrer dans l'état de session pour une réutilisation ultérieure
if st.session_state.df_received and isinstance(st.session_state.messages[-1]["content"], pd.DataFrame):
    st.session_state.last_df = st.session_state.messages[-1]["content"]

# Options de visualisation toujours disponibles si un DataFrame existe
if "last_df" in st.session_state and not st.session_state.last_df.empty:
    st.sidebar.header("Options de Visualisation")
    df = st.session_state.last_df  # Récupérer le DataFrame enregistré dans l'état de session
    x_col = st.sidebar.selectbox("Variable X", df.columns)
    y_col = st.sidebar.selectbox("Variable Y", df.columns)

    if st.sidebar.button("Générer le graphique"):
        fig, error = generate_plot(df, x_col, y_col)
        if fig:
            st.session_state.messages.append({"role": "assistant", "content": fig})  # Ajouter le graphique aux messages
            with st.chat_message("assistant"):
                st.pyplot(fig)
        else:
            st.error(f"Erreur : {error}")

```
