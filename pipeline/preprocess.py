"""
pipeline/preprocess.py
──────────────────────
Étape 1 du pipeline Hurdle : lecture, nettoyage et préparation des données.

Entrée  : /data/raw/NASC_anchois.xls
Sorties : /shared/data/df_processed.parquet
          /shared/data/medianes_train.json
          /shared/data/sample_locations.csv
"""

import json
import math
import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer
import joblib

# ── Flag données fictives / réelles ──────────────────────────────────────────
# Changer USE_FAKE_DATA à "true" dans docker-compose.yml pour utiliser les
# données fictives (demo GitHub) ou "false" pour les données réelles.
USE_FAKE_DATA = os.environ.get("USE_FAKE_DATA", "false").lower() == "true"

if USE_FAKE_DATA:
    RAW_PATH = os.path.join(os.path.dirname(os.environ.get("RAW_DATA_PATH", "/data/raw/NASC_anchois.xls")), "Fake_NASC_anchois.xls")
    print("[preprocess] ⚠️  MODE DEMO — données fictives utilisées")
else:
    RAW_PATH = os.environ.get("RAW_DATA_PATH", "/data/raw/NASC_anchois.xls")
SHARED_DATA   = os.environ.get("SHARED_DATA_PATH", "/shared/data")
SHARED_MODELS = os.environ.get("SHARED_MODELS_PATH", "/shared/models")

# Colonnes à supprimer (hors des 21 features du pipeline final)
TO_DROP = [
    "Ara", "Cal", "Diatoms", "Dino", "Grenn algae", "Nano",
    "No3", "O2", "Phyc", "Pico", "Po4", "Nppv", "Prokar",
    "Secchi", "Si", "Spco2", "Vgos",
]

FEATURES_ORDER = [
    "Bathy", "SST", "So_mean", "Density", "SH", "Wind",
    "Uo", "Vo", "Ue", "Ve", "Ugos",
    "pH", "Alk", "Fe", "Chla",
    "Hapto", "Micro", "Prochlo", "SM",
    "Mois_sin", "Mois_cos",
]

FEATURES_SIMPLE = ["Fe", "Chla", "Bathy", "SST", "SH", "pH", "Wind", "So_mean"]


def main():
    print("[preprocess] Lecture des données...")
    df = pd.read_excel(RAW_PATH)
    print(f"[preprocess] Shape brut : {df.shape}")

    # ── Ordre exact du notebook ───────────────────────────────────────────────
    # 1. Suppression des lignes sans NASC uniquement 
    df = df.dropna(subset=["NASC_Anchois"]).reset_index(drop=True)

    # 2. Suppression des doublons
    df = df.drop_duplicates().reset_index(drop=True)

    # 3. Filtre valeurs aberrantes
    df = df[df["NASC_Anchois"] < 100000].reset_index(drop=True)

    # 4. Suppression des colonnes inutiles
    df = df.drop(columns=[c for c in TO_DROP if c in df.columns])

    # 5. Suppression des NaN sur les features uniquement (StandardScaler l'exige)
    features_present = [f for f in FEATURES_ORDER if f not in ("Mois_sin", "Mois_cos")]
    df = df.dropna(subset=features_present).reset_index(drop=True)

    print(f"[preprocess] Shape après nettoyage : {df.shape}")

    # ── 3. Encodage cyclique du mois ──────────────────────────────────────────
    df["Mois_sin"] = df["Mois"].apply(lambda m: math.sin(2 * math.pi * m / 12))
    df["Mois_cos"] = df["Mois"].apply(lambda m: math.cos(2 * math.pi * m / 12))

    # ── 4. Transformation Yeo-Johnson sur NASC_Anchois ────────────────────────
    pt = PowerTransformer(method="yeo-johnson")
    df["yeo_NASC"] = pt.fit_transform(df[["NASC_Anchois"]])

    # ── 5. Colonne présence (basée sur yeo_NASC > min, fidèle au notebook) ───
    df["presence"] = (df["yeo_NASC"] > df["yeo_NASC"].min()).astype(int)

    print(f"[preprocess] Présences : {df['presence'].sum()} / {len(df)} "
          f"({df['presence'].mean():.1%})")

    # ── 6. Sauvegarde ─────────────────────────────────────────────────────────
    os.makedirs(SHARED_DATA, exist_ok=True)
    os.makedirs(SHARED_MODELS, exist_ok=True)

    df.to_parquet(os.path.join(SHARED_DATA, "df_processed.parquet"), index=False)
    print("[preprocess] df_processed.parquet sauvegardé")

    # Transformer Yeo-Johnson sauvegardé (fit sur toutes les données)
    joblib.dump(pt, os.path.join(SHARED_MODELS, "yeo_transformer.joblib"))
    print("[preprocess] yeo_transformer.joblib sauvegardé")

    # Médianes pour le formulaire Streamlit
    medianes = df[FEATURES_ORDER].median().to_dict()
    with open(os.path.join(SHARED_DATA, "medianes_train.json"), "w") as f:
        json.dump(medianes, f, indent=2)
    print("[preprocess] medianes_train.json sauvegardé")

    # Listes de features
    with open(os.path.join(SHARED_DATA, "features_simple.json"), "w") as f:
        json.dump(FEATURES_SIMPLE, f, indent=2)
    with open(os.path.join(SHARED_DATA, "features_avance.json"), "w") as f:
        json.dump(FEATURES_ORDER, f, indent=2)

    # Échantillon géographique pour la carte Streamlit
    geo = df[["Latitude", "Longitude", "presence"]].sample(
        n=min(500, len(df)), random_state=42
    ).reset_index(drop=True)
    geo.to_csv(os.path.join(SHARED_DATA, "sample_locations.csv"), index=False)
    print("[preprocess] sample_locations.csv sauvegardé")

    print("[preprocess] Terminé.")


if __name__ == "__main__":
    main()
