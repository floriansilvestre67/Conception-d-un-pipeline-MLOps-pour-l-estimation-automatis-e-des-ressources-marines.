"""
pipeline/optimize.py
─────────────────────
Étape 2 : optimisation des hyperparamètres du pipeline Hurdle.

Le code Optuna ci-dessous est commenté pour ne pas ralentir le pipeline.
Les meilleurs paramètres trouvés lors de l'étude Optuna sont définis
directement comme variables et exportés en JSON pour les étapes suivantes.

Entrée  : /shared/data/df_processed.parquet
Sorties : /shared/data/best_params_clf.json
          /shared/data/best_params_reg.json
"""

import json
import os

import numpy as np
import pandas as pd

SHARED_DATA = os.environ.get("SHARED_DATA_PATH", "/shared/data")

# ══════════════════════════════════════════════════════════════════════════════
#  CODE OPTUNA — COMMENTÉ (trop long pour un pipeline automatisé)
#  Décommenter pour relancer l'optimisation complète (~2-4h sur CPU)
# ══════════════════════════════════════════════════════════════════════════════

# import optuna
# from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
# from sklearn.metrics import roc_auc_score, r2_score

# FEATURES_ORDER = [
#     "Bathy", "SST", "So_mean", "Density", "SH", "Wind",
#     "Uo", "Vo", "Ue", "Ve", "Ugos",
#     "pH", "Alk", "Fe", "Chla",
#     "Hapto", "Micro", "Prochlo", "SM",
#     "Mois_sin", "Mois_cos",
# ]

# df = pd.read_parquet(os.path.join(SHARED_DATA, "df_processed.parquet"))
# X1 = df[FEATURES_ORDER]
# y1 = df["presence"]
# df_pres = df[df["presence"] == 1]
# X_reg = df_pres[FEATURES_ORDER]
# y_reg = df_pres["yeo_NASC"]

# ── Objectif Optuna — Classificateur ─────────────────────────────────────────
# def objective_clf(trial):
#     params = {
#         "max_iter":        trial.suggest_int("max_iter", 100, 500),
#         "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
#         "max_depth":       trial.suggest_int("max_depth", 3, 10),
#         "min_samples_leaf":trial.suggest_int("min_samples_leaf", 10, 50),
#         "l2_regularization":trial.suggest_float("l2_regularization", 0.0, 1.0),
#         "random_state":    42,
#     }
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X1)
#     clf = HistGradientBoostingClassifier(**params)
#     cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#     scores = cross_val_score(clf, X_scaled, y1, cv=cv, scoring="roc_auc", n_jobs=-1)
#     return scores.mean()

# ── Objectif Optuna — Régresseur ──────────────────────────────────────────────
# def objective_reg(trial):
#     params = {
#         "max_iter":        trial.suggest_int("max_iter", 100, 500),
#         "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
#         "max_depth":       trial.suggest_int("max_depth", 3, 10),
#         "min_samples_leaf":trial.suggest_int("min_samples_leaf", 10, 50),
#         "l2_regularization":trial.suggest_float("l2_regularization", 0.0, 1.0),
#         "random_state":    42,
#     }
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X_reg)
#     reg = HistGradientBoostingRegressor(**params)
#     cv = KFold(n_splits=5, shuffle=True, random_state=42)
#     scores = cross_val_score(reg, X_scaled, y_reg, cv=cv, scoring="r2", n_jobs=-1)
#     return scores.mean()

# ── Lancement des études Optuna ───────────────────────────────────────────────
# optuna.logging.set_verbosity(optuna.logging.WARNING)

# study_clf = optuna.create_study(direction="maximize", study_name="hurdle_clf")
# study_clf.optimize(objective_clf, n_trials=100, show_progress_bar=True)
# best_params_clf = study_clf.best_params
# print(f"[optimize] Meilleur ROC-AUC clf : {study_clf.best_value:.4f}")
# print(f"[optimize] Meilleurs params clf : {best_params_clf}")

# study_reg = optuna.create_study(direction="maximize", study_name="hurdle_reg")
# study_reg.optimize(objective_reg, n_trials=100, show_progress_bar=True)
# best_params_reg = study_reg.best_params
# print(f"[optimize] Meilleur R² reg : {study_reg.best_value:.4f}")
# print(f"[optimize] Meilleurs params reg : {best_params_reg}")

# ══════════════════════════════════════════════════════════════════════════════
#  MEILLEURS PARAMÈTRES ISSUS DE L'ÉTUDE OPTUNA
#  Remplacer ces valeurs par les résultats réels de study_clf.best_params
#  et study_reg.best_params après une exécution complète.
# ══════════════════════════════════════════════════════════════════════════════

best_params_clf = {
    "max_iter":           498,
    "max_depth":          11,
    "max_leaf_nodes":     40,
    "min_samples_leaf":   29,
    "learning_rate":      0.08458161037695616,
    "l2_regularization":  0.16835731396150172,
    "max_bins":           71,
    "random_state":       42,
}

best_params_reg = {
    "max_iter":           114,
    "max_depth":          10,
    "max_leaf_nodes":     46,
    "min_samples_leaf":   19,
    "learning_rate":      0.0796,
    "l2_regularization":  0.629,
    "max_bins":           87,
    "random_state":       42,
}


def main():
    print("[optimize] Export des hyperparamètres optimaux...")

    os.makedirs(SHARED_DATA, exist_ok=True)

    with open(os.path.join(SHARED_DATA, "best_params_clf.json"), "w") as f:
        json.dump(best_params_clf, f, indent=2)
    print(f"[optimize] best_params_clf.json : {best_params_clf}")

    with open(os.path.join(SHARED_DATA, "best_params_reg.json"), "w") as f:
        json.dump(best_params_reg, f, indent=2)
    print(f"[optimize] best_params_reg.json : {best_params_reg}")

    print("[optimize] Terminé.")


if __name__ == "__main__":
    main()
