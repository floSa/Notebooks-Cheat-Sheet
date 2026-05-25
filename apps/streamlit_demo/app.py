"""Streamlit — démo multi-features exécutable.

Lance :
    uv run streamlit run apps/streamlit_demo/app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------- Page config ----------
st.set_page_config(page_title="Demo Streamlit", page_icon="🎈", layout="wide")
st.title("🎈 Demo Streamlit — Cal Housing")

# ---------- Sidebar : paramètres ----------
with st.sidebar:
    st.header("Paramètres")
    test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05)
    seed = st.number_input("Random seed", 0, 9999, 42)
    feature = st.selectbox("Feature à visualiser", ["MedInc", "HouseAge", "AveRooms", "Latitude"])


# ---------- Cache lourd ----------
@st.cache_data
def load_data() -> pd.DataFrame:
    data = fetch_california_housing(as_frame=True)
    df = data.data.copy()
    df["target"] = data.target
    return df


@st.cache_resource
def train_model(X_train, y_train) -> LinearRegression:
    return LinearRegression().fit(X_train, y_train)


df = load_data()

# ---------- Body ----------
tab1, tab2, tab3 = st.tabs(["📊 Données", "📈 Visualisation", "🤖 Modèle"])

with tab1:
    st.subheader("Aperçu")
    st.dataframe(df.head(50), use_container_width=True)
    st.write(f"Shape : {df.shape}")
    st.write("Stats résumées :")
    st.dataframe(df.describe().T, use_container_width=True)

with tab2:
    st.subheader(f"Distribution de `{feature}`")
    col1, col2 = st.columns(2)
    with col1:
        st.bar_chart(df[feature].value_counts().sort_index().head(50))
    with col2:
        st.line_chart(df[feature].sort_values().reset_index(drop=True))

with tab3:
    st.subheader("Régression linéaire — target = MedHouseVal")
    features = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population"]

    X = df[features]
    y = df["target"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=int(seed))

    scaler = StandardScaler().fit(X_tr)
    model = train_model(scaler.transform(X_tr), y_tr)
    preds = model.predict(scaler.transform(X_te))

    col1, col2, col3 = st.columns(3)
    col1.metric("R²", f"{r2_score(y_te, preds):.4f}")
    col2.metric("N train", len(X_tr))
    col3.metric("N test", len(X_te))

    st.subheader("Coefficients")
    coef_df = pd.DataFrame({"feature": features, "coef": model.coef_})
    st.bar_chart(coef_df.set_index("feature"))

    st.subheader("Prédiction interactive")
    inputs = {}
    cols = st.columns(len(features))
    for col, feat in zip(cols, features):
        with col:
            inputs[feat] = st.number_input(
                feat,
                value=float(df[feat].median()),
                min_value=float(df[feat].min()),
                max_value=float(df[feat].max()),
            )

    if st.button("Prédire"):
        x_in = scaler.transform(pd.DataFrame([inputs])[features])
        pred = model.predict(x_in)[0]
        st.success(f"Prédiction MedHouseVal : **{pred:.3f}** (en 100k$)")
