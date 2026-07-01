"""
pages/1_Prediction.py — Formulaire de saisie et résultats de prédiction.
"""

import json
import math
import os
import traceback

import numpy as np
import pandas as pd
import streamlit as st

# Remonter d'un niveau pour importer config et predict
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    FEATURES_ORDER, FEATURE_LABELS, DEFAULT_THRESHOLD,
    MEDIANES_PATH, FEATURES_SIMPLE_PATH, FEATURES_AVANCE_PATH,
)
from predict import load_artifacts, predict_hurdle, build_csv_row

# ── Configuration ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Prédiction NASC", page_icon="🎯", layout="wide")
st.title("Prédiction de biomasse")
st.markdown(
    "Renseignez les variables environnementales pour obtenir une prédiction de présence "
    "et de biomasse acoustique de l'anchois (NASC, m²/nmi²)."
)

# ── Chargement des données de référence ──────────────────────────────────────
@st.cache_data
def load_reference_data():
    medianes = {}
    if os.path.exists(MEDIANES_PATH):
        with open(MEDIANES_PATH) as f:
            medianes = json.load(f)

    features_simple = FEATURES_ORDER
    if os.path.exists(FEATURES_SIMPLE_PATH):
        with open(FEATURES_SIMPLE_PATH) as f:
            features_simple = json.load(f)

    features_avance = FEATURES_ORDER
    if os.path.exists(FEATURES_AVANCE_PATH):
        with open(FEATURES_AVANCE_PATH) as f:
            features_avance = json.load(f)

    return medianes, features_simple, features_avance

medianes, features_simple, features_avance = load_reference_data()

# ── Chargement des modèles ────────────────────────────────────────────────────
@st.cache_resource
def get_models():
    return load_artifacts()

try:
    clf, reg, yeo, scaler_clf, scaler_reg = get_models()
    models_loaded = True
except Exception as e:
    st.error(f"Impossible de charger les modèles : {e}")
    st.code(traceback.format_exc())
    models_loaded = False
    st.stop()

# ── Paramètres ────────────────────────────────────────────────────────────────
st.sidebar.header("Paramètres")

mode = st.sidebar.radio(
    "Mode de saisie",
    options=["Simple", "Avancé", "Personnalisé"],
    index=0,
    help="Le mode simple n'affiche que les variables les plus influentes. Le mode personnalisé vous laisse choisir vos variables.",
)

threshold = st.sidebar.slider(
    "Seuil de décision (classification)",
    min_value=0.0,
    max_value=1.0,
    value=DEFAULT_THRESHOLD,
    step=0.01,
    help="Si la probabilité de présence dépasse ce seuil, une présence est prédite.",
)

st.sidebar.divider()
st.sidebar.info(
    "Les champs sont pré-remplis avec les **médianes d'entraînement**. "
    "Modifiez les valeurs pour explorer différents scénarios."
)

# ── Sélection des features selon le mode ─────────────────────────────────────
FEATURES_MOIS = {"Mois_sin", "Mois_cos"}

# Liste de toutes les features saisissables (sans Mois_sin/Mois_cos)
all_saisie = [f for f in FEATURES_ORDER if f not in FEATURES_MOIS]

if mode == "Simple":
    active_features = [f for f in features_simple if f in FEATURES_ORDER and f not in FEATURES_MOIS]
elif mode == "Avancé":
    active_features = [f for f in features_avance if f in FEATURES_ORDER and f not in FEATURES_MOIS]
else:  # Personnalisé
    selected = st.sidebar.multiselect(
        "Variables à renseigner",
        options=all_saisie,
        default=all_saisie[:5],
        format_func=lambda f: FEATURE_LABELS.get(f, f),
        help="Les variables non sélectionnées seront remplacées par leur médiane d'entraînement.",
    )
    active_features = selected

# ── Formulaire de saisie ──────────────────────────────────────────────────────
NOMS_MOIS = [
    "Janvier", "Février", "Mars", "Avril", "Mai", "Juin",
    "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre",
]

if mode == "Personnalisé":
    st.subheader("Saisie des variables — Mode Personnalisé")
    if not active_features:
        st.warning("Sélectionnez au moins une variable dans la barre latérale.")
        st.stop()
    vars_auto = [f for f in all_saisie if f not in active_features]
    if vars_auto:
        with st.expander(f"{len(vars_auto)} variable(s) complétée(s) automatiquement par la médiane"):
            st.write(", ".join(FEATURE_LABELS.get(f, f) for f in vars_auto))
else:
    st.subheader(f"Saisie des variables — Mode {mode}")

cols_per_row = 3
feature_values = {}

# Sélecteur de mois en première position dans la grille
col_mois, col2, col3 = st.columns(cols_per_row)
with col_mois:
    mois_choisi = st.selectbox(
        "Mois",
        options=list(range(1, 13)),
        format_func=lambda m: f"{m} — {NOMS_MOIS[m - 1]}",
        index=0,
        help="Mois_sin et Mois_cos sont calculés automatiquement.",
    )

mois_sin = math.sin(2 * math.pi * mois_choisi / 12)
mois_cos = math.cos(2 * math.pi * mois_choisi / 12)

# Remplir col2 et col3 avec les deux premières features de la liste
remaining_features = list(active_features)
for col, feat in zip([col2, col3], remaining_features[:2]):
    with col:
        default_val = float(medianes.get(feat, 0.0))
        feature_values[feat] = st.number_input(
            label=FEATURE_LABELS.get(feat, feat),
            value=default_val,
            format="%.4f",
            key=f"input_{feat}",
        )
remaining_features = remaining_features[2:]

# Reste des features en grille de 3
feature_groups = [remaining_features[i:i+cols_per_row] for i in range(0, len(remaining_features), cols_per_row)]

for group in feature_groups:
    cols = st.columns(cols_per_row)
    for i, feat in enumerate(group):
        default_val = float(medianes.get(feat, 0.0))
        with cols[i]:
            feature_values[feat] = st.number_input(
                label=FEATURE_LABELS.get(feat, feat),
                value=default_val,
                format="%.4f",
                key=f"input_{feat}",
            )

# Compléter toutes les features : saisies + médianes pour les absentes + mois calculé
full_features = {}
for feat in FEATURES_ORDER:
    if feat == "Mois_sin":
        full_features[feat] = mois_sin
    elif feat == "Mois_cos":
        full_features[feat] = mois_cos
    elif feat in feature_values:
        full_features[feat] = feature_values[feat]
    else:
        full_features[feat] = float(medianes.get(feat, 0.0))

# ── Bouton Prédire ────────────────────────────────────────────────────────────
st.divider()

if st.button("Prédire", type="primary", use_container_width=True):
    with st.spinner("Calcul en cours…"):
        result = predict_hurdle(full_features, clf, reg, yeo,
                                scaler_clf, scaler_reg, threshold=threshold)

    st.divider()
    st.subheader("Résultats")

    proba    = result["proba_presence"]
    presence = result["presence"]
    pred_nasc = result["pred_nasc"]

    col_p, col_r = st.columns(2)

    with col_p:
        st.metric(
            label="Probabilité de présence",
            value=f"{proba:.1%}",
            delta=f"Seuil : {threshold:.0%}",
            delta_color="off",
        )

    with col_r:
        if presence:
            st.success("Présence détectée (probabilité ≥ seuil)")
        else:
            st.warning("Absence prédite (probabilité < seuil)")

    # Barre de probabilité visuelle
    st.progress(proba, text=f"Probabilité de présence : {proba:.1%}")

    # ── Résultat NASC ──────────────────────────────────────────────────────────
    if presence:
        st.divider()
        nasc_mediane = float(medianes.get("__nasc_mediane__", 500.0))

        st.metric(
            label="NASC prédit (m²/nmi²)",
            value=f"{pred_nasc:,.1f}",
        )

        # Message interprétatif
        if pred_nasc < 100:
            interpretation = "Biomasse **très faible** — présence détectée mais densité négligeable."
        elif pred_nasc < 500:
            interpretation = "Biomasse **faible** — densité en dessous de la médiane observée."
        elif pred_nasc < 2000:
            interpretation = "Biomasse **modérée** — densité dans la gamme habituelle."
        elif pred_nasc < 5000:
            interpretation = "Biomasse **élevée** — densité supérieure à la médiane observée."
        else:
            interpretation = "Biomasse **très élevée** — densité exceptionnelle."

        st.info(interpretation)

    else:
        st.info("Aucune valeur de NASC prédite : absence détectée à ce seuil de décision.")

    # ── Export JSON ────────────────────────────────────────────────────────────
    st.divider()
    import json as _json
    export_data = {
        "proba_presence": result["proba_presence"],
        "presence":       result["presence"],
        "pred_nasc":      result["pred_nasc"],
        "seuil_decision": threshold,
        "variables":      full_features,
    }
    json_bytes = _json.dumps(export_data, indent=2, ensure_ascii=False).encode("utf-8")

    st.download_button(
        label="Télécharger la prédiction (JSON)",
        data=json_bytes,
        file_name="prediction_nasc.json",
        mime="application/json",
        use_container_width=True,
    )
