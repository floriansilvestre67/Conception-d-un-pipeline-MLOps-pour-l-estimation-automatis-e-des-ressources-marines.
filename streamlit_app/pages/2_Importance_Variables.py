"""
pages/2_Importance_Variables.py — Graphiques d'importance des variables.
"""

import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import IMP_PHASE1_PATH, IMP_PHASE2_PATH, FEATURE_LABELS

st.set_page_config(page_title="Importance des variables", page_icon="📊", layout="wide")
st.title("Importance des variables")
st.markdown(
    "Les graphiques ci-dessous montrent l'importance relative de chaque variable "
    "pour chacun des deux étages du pipeline Hurdle."
)


@st.cache_data
def load_importance(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["feature", "importance"])
    df = pd.read_csv(path)
    # Ajouter les labels lisibles
    df["label"] = df["feature"].map(lambda f: FEATURE_LABELS.get(f, f))
    return df.sort_values("importance", ascending=True)


def plot_importance(df: pd.DataFrame, title: str):
    if df.empty:
        st.warning("Données d'importance non disponibles pour ce modèle.")
        return

    top_n = st.slider(
        f"Nombre de variables à afficher ({title})",
        min_value=3,
        max_value=len(df),
        value=min(10, len(df)),
        key=f"slider_{title}",
    )

    df_plot = df.tail(top_n)

    fig = px.bar(
        df_plot,
        x="importance",
        y="label",
        orientation="h",
        title=title,
        labels={"importance": "Importance", "label": "Variable"},
        color="importance",
        color_continuous_scale="Blues",
        text_auto=".3f",
    )
    fig.update_layout(
        height=max(350, top_n * 35),
        coloraxis_showscale=False,
        yaxis_title=None,
        xaxis_title="Importance relative",
        margin=dict(l=10, r=30, t=50, b=30),
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)


tab1, tab2 = st.tabs(["Phase 1 — Classification (présence)", "Phase 2 — Régression (quantité)"])

with tab1:
    df1 = load_importance(IMP_PHASE1_PATH)
    st.markdown(
        "Importance des variables pour le modèle `HistGradientBoostingClassifier` "
        "qui prédit la **présence ou l'absence** de biomasse (NASC > 0)."
    )
    plot_importance(df1, "Importance — Détection de présence")

with tab2:
    df2 = load_importance(IMP_PHASE2_PATH)
    st.markdown(
        "Importance des variables pour le modèle `HistGradientBoostingRegressor` "
        "qui prédit la **quantité de biomasse** (NASC, espace Yeo-Johnson)."
    )
    plot_importance(df2, "Importance — Estimation de la biomasse")
