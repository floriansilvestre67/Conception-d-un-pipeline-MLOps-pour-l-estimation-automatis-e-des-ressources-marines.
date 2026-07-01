"""
pipeline/train_regressor.py
────────────────────────────
Étape 3 : entraînement du régresseur sur les données de présence uniquement.

Entrée  : /shared/data/df_processed.parquet
Sorties : /shared/models/reg_biomasse.joblib
          /shared/models/scaler_reg.joblib
"""

import json
import os

import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

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
    print("[train_regressor] Chargement des données...")
    df = pd.read_parquet(os.path.join(SHARED_DATA, "df_processed.parquet"))

    # Sous-ensemble présence uniquement
    df_pres = df[df["presence"] == 1].reset_index(drop=True)
    X_reg = df_pres[FEATURES_ORDER]
    y_reg = df_pres["yeo_NASC"]

    print(f"[train_regressor] X_reg shape : {X_reg.shape}")

    # ── Chargement des hyperparamètres optimaux ───────────────────────────────
    params_path = os.path.join(SHARED_DATA, "best_params_reg.json")
    if os.path.exists(params_path):
        with open(params_path) as f:
            params = json.load(f)
        print(f"[train_regressor] Params Optuna chargés : {params}")
    else:
        params = {
            "max_iter": 114, "max_depth": 10, "max_leaf_nodes": 46,
            "min_samples_leaf": 19, "learning_rate": 0.0796,
            "l2_regularization": 0.629, "max_bins": 87,
        }
        print("[train_regressor] Params Optuna (défaut) utilisés")

    # ── StandardScaler fit sur les données de présence uniquement ─────────────
    scaler_reg = StandardScaler()
    X_reg_scaled = scaler_reg.fit_transform(X_reg)

    # ── Entraînement ─────────────────────────────────────────────────────────
    print("[train_regressor] Entraînement HistGradientBoostingRegressor...")
    reg = HistGradientBoostingRegressor(**params)
    reg.fit(X_reg_scaled, y_reg)

    # ── Sauvegarde ────────────────────────────────────────────────────────────
    os.makedirs(SHARED_MODELS, exist_ok=True)
    joblib.dump(reg,        os.path.join(SHARED_MODELS, "reg_biomasse.joblib"))
    joblib.dump(scaler_reg, os.path.join(SHARED_MODELS, "scaler_reg.joblib"))

    print("[train_regressor] reg_biomasse.joblib sauvegardé")
    print("[train_regressor] scaler_reg.joblib sauvegardé")
    print("[train_regressor] Terminé.")


if __name__ == "__main__":
    main()
