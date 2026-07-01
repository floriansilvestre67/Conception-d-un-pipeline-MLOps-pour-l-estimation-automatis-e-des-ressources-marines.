"""
pages/3_Carte_Spatiale.py — Carte de distribution spatiale des observations.
"""

import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import SAMPLE_LOC_PATH

st.set_page_config(page_title="Carte spatiale", page_icon="🗺️", layout="wide")
st.title("Distribution spatiale des observations")
st.markdown(
    "Cette carte représente la distribution géographique des points d'observation "
    "avec leur statut de présence ou d'absence de biomasse d'anchois."
)


@st.cache_data
def load_locations() -> pd.DataFrame:
    if not os.path.exists(SAMPLE_LOC_PATH):
        return pd.DataFrame(columns=["Latitude", "Longitude", "presence"])
    df = pd.read_csv(SAMPLE_LOC_PATH)
    df["Statut"] = df["presence"].map({1: "Présence", 0: "Absence"})
    return df


df = load_locations()

if df.empty:
    st.warning("Fichier `sample_locations.csv` introuvable ou vide.")
    st.stop()

# ── Filtres ───────────────────────────────────────────────────────────────────
col_f1, col_f2 = st.columns(2)
with col_f1:
    show_presence = st.checkbox("Afficher les présences", value=True)
with col_f2:
    show_absence = st.checkbox("Afficher les absences", value=True)

mask = pd.Series([False] * len(df))
if show_presence:
    mask |= df["presence"] == 1
if show_absence:
    mask |= df["presence"] == 0

df_filtered = df[mask]

st.markdown(
    f"**{len(df_filtered)}** points affichés "
    f"({df_filtered['presence'].sum()} présences, "
    f"{(df_filtered['presence'] == 0).sum()} absences)"
)

# ── Carte ─────────────────────────────────────────────────────────────────────
color_map = {"Présence": "#2196F3", "Absence": "#FF5722"}

fig = px.scatter_mapbox(
    df_filtered,
    lat="Latitude",
    lon="Longitude",
    color="Statut",
    color_discrete_map=color_map,
    hover_data={"Latitude": ":.3f", "Longitude": ":.3f", "presence": False, "Statut": True},
    zoom=4,
    height=600,
    title="Présence / Absence de biomasse d'anchois",
)
fig.update_layout(
    mapbox_style="open-street-map",
    legend_title_text="Statut",
    margin=dict(l=0, r=0, t=40, b=0),
)

st.plotly_chart(fig, use_container_width=True)

# ── Statistiques rapides ──────────────────────────────────────────────────────
with st.expander("📋 Statistiques de l'échantillon"):
    total = len(df)
    n_pres = int(df["presence"].sum())
    n_abs  = total - n_pres
    st.markdown(
        f"- **Total d'observations** : {total}\n"
        f"- **Présences** : {n_pres} ({n_pres/total:.1%})\n"
        f"- **Absences** : {n_abs} ({n_abs/total:.1%})\n"
        f"- **Latitude** : [{df['Latitude'].min():.2f}°, {df['Latitude'].max():.2f}°]\n"
        f"- **Longitude** : [{df['Longitude'].min():.2f}°, {df['Longitude'].max():.2f}°]"
    )
    st.dataframe(df.head(20), use_container_width=True)
