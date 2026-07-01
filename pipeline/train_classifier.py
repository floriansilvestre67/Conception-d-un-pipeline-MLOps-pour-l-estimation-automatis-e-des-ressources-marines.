"""
pipeline/train_classifier.py
─────────────────────────────
Étape 2 : entraînement du classificateur de présence/absence.

Entrée  : /shared/data/df_processed.parquet
Sorties : /shared/models/clf_presence.joblib
          /shared/models/scaler_clf.joblib
"""

import json
import os

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

SHARED_DATA   = os.environ.get("SHARED_DATA_PATH", "/shared/data")
SHARED_MODELS = os.environ.get("SHARED_MODELS_PATH", "/shared/models")

FEATURES_ORDER = [
    "Bathy", "SST", "So_mean", "Density", "SH", "Wind",
    "Uo", "Vo", "Ue", "Ve", "Ugos",
    "pH", "Alk", "Fe", "Chla",
    "Hapto", "Micro", "Prochlo", "SM",
    "Mois_sin", "Mois_cos",
]


def main():
    print("[train_classifier] Chargement des données...")
    df = pd.read_parquet(os.path.join(SHARED_DATA, "df_processed.parquet"))

    X1 = df[FEATURES_ORDER]
    y1 = df["presence"]

    print(f"[train_classifier] X1 shape : {X1.shape} | "
          f"Positifs : {y1.sum()} ({y1.mean():.1%})")

    # ── Chargement des hyperparamètres optimaux ───────────────────────────────
    params_path = os.path.join(SHARED_DATA, "best_params_clf.json")
    if os.path.exists(params_path):
        with open(params_path) as f:
            params = json.load(f)
        print(f"[train_classifier] Params Optuna chargés : {params}")
    else:
        params = {
            "max_iter": 498, "max_depth": 11, "max_leaf_nodes": 40,
            "min_samples_leaf": 29, "learning_rate": 0.08458161037695616,
            "l2_regularization": 0.16835731396150172, "max_bins": 71,
        }
        print("[train_classifier] Params Optuna (défaut) utilisés")

    # ── StandardScaler fit sur toutes les données (X1) ────────────────────────
    scaler_clf = StandardScaler()
    X1_scaled = scaler_clf.fit_transform(X1)

    # ── Entraînement ─────────────────────────────────────────────────────────
    print("[train_classifier] Entraînement HistGradientBoostingClassifier...")
    clf = HistGradientBoostingClassifier(**params)
    clf.fit(X1_scaled, y1)

    # ── Sauvegarde ────────────────────────────────────────────────────────────
    os.makedirs(SHARED_MODELS, exist_ok=True)
    joblib.dump(clf,        os.path.join(SHARED_MODELS, "clf_presence.joblib"))
    joblib.dump(scaler_clf, os.path.join(SHARED_MODELS, "scaler_clf.joblib"))

    print("[train_classifier] clf_presence.joblib sauvegardé")
    print("[train_classifier] scaler_clf.joblib sauvegardé")
    print("[train_classifier] Terminé.")


if __name__ == "__main__":
    main()
