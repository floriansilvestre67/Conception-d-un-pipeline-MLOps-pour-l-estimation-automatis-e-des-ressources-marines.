"""
pipeline/optimize.py
─────────────────────
Étape 2 : optimisation des hyperparamètres du pipeline Hurdle via Optuna.

Entrée  : /shared/data/df_processed.parquet
Sorties : /shared/data/best_params_clf.json
          /shared/data/best_params_reg.json
"""

import json
import os

import numpy as np
import pandas as pd
import optuna
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, r2_score

SHARED_DATA = os.environ.get("SHARED_DATA_PATH", "/shared/data")

FEATURES_ORDER = [
    "Bathy", "SST", "So_mean", "Density", "SH", "Wind",
    "Uo", "Vo", "Ue", "Ve", "Ugos",
    "pH", "Alk", "Fe", "Chla",
    "Hapto", "Micro", "Prochlo", "SM",
    "Mois_sin", "Mois_cos",
]


# ── Objectif Optuna — Classificateur ─────────────────────────────────────────
def objective_clf(trial):
    params = {
        "max_iter":          trial.suggest_int("max_iter", 100, 500),
        "max_depth":         trial.suggest_int("max_depth", 3, 12),
        "max_leaf_nodes":    trial.suggest_int("max_leaf_nodes", 20, 60),
        "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 10, 50),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 1.0),
        "max_bins":          trial.suggest_int("max_bins", 32, 128),
        "random_state":      42,
    }
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X1)
    clf = HistGradientBoostingClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X_scaled, y1, cv=cv, scoring="roc_auc", n_jobs=-1)
    return scores.mean()


# ── Objectif Optuna — Régresseur ─────────────────────────────────────────────
def objective_reg(trial):
    params = {
        "max_iter":          trial.suggest_int("max_iter", 100, 500),
        "max_depth":         trial.suggest_int("max_depth", 3, 12),
        "max_leaf_nodes":    trial.suggest_int("max_leaf_nodes", 20, 60),
        "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 10, 50),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 1.0),
        "max_bins":          trial.suggest_int("max_bins", 32, 128),
        "random_state":      42,
    }
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reg)
    reg = HistGradientBoostingRegressor(**params)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(reg, X_scaled, y_reg, cv=cv, scoring="r2", n_jobs=-1)
    return scores.mean()


def main():
    global X1, y1, X_reg, y_reg

    print("[optimize] Chargement des données...")
    df = pd.read_parquet(os.path.join(SHARED_DATA, "df_processed.parquet"))
    X1 = df[FEATURES_ORDER]
    y1 = df["presence"]
    df_pres = df[df["presence"] == 1]
    X_reg = df_pres[FEATURES_ORDER]
    y_reg = df_pres["yeo_NASC"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # ── Optimisation classificateur ───────────────────────────────────────────
    print("[optimize] Optimisation classificateur (100 trials)...")
    study_clf = optuna.create_study(direction="maximize", study_name="hurdle_clf")
    study_clf.optimize(objective_clf, n_trials=100, show_progress_bar=True)
    best_params_clf = study_clf.best_params
    print(f"[optimize] Meilleur ROC-AUC clf : {study_clf.best_value:.4f}")
    print(f"[optimize] Meilleurs params clf : {best_params_clf}")

    # ── Optimisation régresseur ───────────────────────────────────────────────
    print("[optimize] Optimisation régresseur (100 trials)...")
    study_reg = optuna.create_study(direction="maximize", study_name="hurdle_reg")
    study_reg.optimize(objective_reg, n_trials=100, show_progress_bar=True)
    best_params_reg = study_reg.best_params
    print(f"[optimize] Meilleur R² reg : {study_reg.best_value:.4f}")
    print(f"[optimize] Meilleurs params reg : {best_params_reg}")

    # ── Sauvegarde ────────────────────────────────────────────────────────────
    os.makedirs(SHARED_DATA, exist_ok=True)

    with open(os.path.join(SHARED_DATA, "best_params_clf.json"), "w") as f:
        json.dump(best_params_clf, f, indent=2)
    print(f"[optimize] best_params_clf.json sauvegardé")

    with open(os.path.join(SHARED_DATA, "best_params_reg.json"), "w") as f:
        json.dump(best_params_reg, f, indent=2)
    print(f"[optimize] best_params_reg.json sauvegardé")

    print("[optimize] Terminé.")


# ══════════════════════════════════════════════════════════════════════════════
#  PARAMÈTRES DE RÉFÉRENCE (issus de l'étude Optuna initiale sur données réelles)
#  Utilisés par train_classifier.py et train_regressor.py si les JSON
#  ne sont pas encore disponibles (premier run sans optimize).
# ══════════════════════════════════════════════════════════════════════════════

# REFERENCE_PARAMS_CLF = {
#     "max_iter":           498,
#     "max_depth":          11,
#     "max_leaf_nodes":     40,
#     "min_samples_leaf":   29,
#     "learning_rate":      0.08458161037695616,
#     "l2_regularization":  0.16835731396150172,
#     "max_bins":           71,
#     "random_state":       42,
# }

# REFERENCE_PARAMS_REG = {
#     "max_iter":           114,
#     "max_depth":          10,
#     "max_leaf_nodes":     46,
#     "min_samples_leaf":   19,
#     "learning_rate":      0.0796,
#     "l2_regularization":  0.629,
#     "max_bins":           87,
#     "random_state":       42,
# }


if __name__ == "__main__":
    main()
