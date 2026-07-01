"""
app.py — Page d'accueil de l'application Streamlit.
"""

import json
import os

import streamlit as st

from config import APP_TITLE, APP_DESCRIPTION, METRICS_PATH

# ── Configuration de la page ──────────────────────────────────────────────────
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Titre ─────────────────────────────────────────────────────────────────────
st.title(APP_TITLE)
st.markdown(APP_DESCRIPTION)

st.divider()

# ── Indicateurs de performance ────────────────────────────────────────────────
st.subheader("Performance du pipeline")

metrics = {"f1": None, "roc_auc": None, "r2": None}
if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)

col1, col2, col3 = st.columns(3)

def metric_card(col, label, key, fmt=".3f", help_text=""):
    val = metrics.get(key)
    with col:
        if val is not None:
            st.metric(label=label, value=f"{val:{fmt}}", help=help_text)
        else:
            st.metric(label=label, value="—", help="Données non disponibles")

metric_card(
    col1, "F1-Score (classification)", "f1",
    help_text="F1-Score de la phase de détection de présence (seuil 0.2)."
)
metric_card(
    col2, "ROC-AUC (classification)", "roc_auc",
    help_text="Aire sous la courbe ROC pour la classification présence/absence."
)
metric_card(
    col3, "R² (régression)", "r2",
    help_text="Coefficient de détermination de la régression NASC (données de test)."
)

st.divider()

# ── Navigation ────────────────────────────────────────────────────────────────
st.subheader("Navigation")

c1, c2, c3 = st.columns(3)

with c1:
    st.info("**Prédiction**\n\nSaisissez des variables environnementales et obtenez une prédiction de présence et de biomasse.")
    st.page_link("pages/1_Prediction.py", label="Aller à la prédiction")

with c2:
    st.info("**Importance des variables**\n\nVisualisez les variables les plus influentes pour chaque étage du pipeline.")
    st.page_link("pages/2_Importance_Variables.py", label="Voir l'importance")

with c3:
    st.info("**Carte spatiale**\n\nExplorez la distribution géographique des observations de présence/absence.")
    st.page_link("pages/3_Carte_Spatiale.py", label="Voir la carte")

