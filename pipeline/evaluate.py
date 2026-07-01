"""
pipeline/evaluate.py
─────────────────────
Étape 4 : évaluation du pipeline Hurdle et export des métriques et importances.

Évaluation classification : StratifiedKFold(5) — fidèle au notebook.
Évaluation régression     : KFold(5).
Importances               : permutation sur dataset complet (n_repeats=10).

Entrées : /shared/data/df_processed.parquet + modèles dans /shared/models/
Sorties : /shared/data/metrics.json
          /shared/data/importance_phase1.csv
          /shared/data/importance_phase2.csv
"""

import json
import os

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, roc_auc_score, r2_score
from sklearn.model_selection import StratifiedKFold, KFold

SHARED_DATA   = os.environ.get("SHARED_DATA_PATH", "/shared/data")
SHARED_MODELS = os.environ.get("SHARED_MODELS_PATH", "/shared/models")

FEATURES_ORDER = [
    "Bathy", "SST", "So_mean", "Density", "SH", "Wind",
    "Uo", "Vo", "Ue", "Ve", "Ugos",
    "pH", "Alk", "Fe", "Chla",
    "Hapto", "Micro", "Prochlo", "SM",
    "Mois_sin", "Mois_cos",
]

# Hyperparamètres Optuna de référence (utilisés si les JSON ne sont pas trouvés)
DEFAULT_PARAMS_CLF = {
    "max_iter": 498, "max_depth": 11, "max_leaf_nodes": 40,
    "min_samples_leaf": 29, "learning_rate": 0.08458161037695616,
    "l2_regularization": 0.16835731396150172, "max_bins": 71,
}
DEFAULT_PARAMS_REG = {
    "max_iter": 114, "max_depth": 10, "max_leaf_nodes": 46,
    "min_samples_leaf": 19, "learning_rate": 0.0796,
    "l2_regularization": 0.629, "max_bins": 87,
}


def load_params(path, default):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default


def main():
    print("[evaluate] Chargement des données et modèles...")
    df = pd.read_parquet(os.path.join(SHARED_DATA, "df_processed.parquet"))

    # Modèles entraînés sur données complètes (utilisés pour l'importance)
    clf        = joblib.load(os.path.join(SHARED_MODELS, "clf_presence.joblib"))
    reg        = joblib.load(os.path.join(SHARED_MODELS, "reg_biomasse.joblib"))
    scaler_clf = joblib.load(os.path.join(SHARED_MODELS, "scaler_clf.joblib"))
    scaler_reg = joblib.load(os.path.join(SHARED_MODELS, "scaler_reg.joblib"))

    # Hyperparamètres
    params_clf = load_params(os.path.join(SHARED_DATA, "best_params_clf.json"), DEFAULT_PARAMS_CLF)
    params_reg = load_params(os.path.join(SHARED_DATA, "best_params_reg.json"), DEFAULT_PARAMS_REG)

    # Données classification
    X1 = df[FEATURES_ORDER]
    y1 = df["presence"]

    # Données régression (présences uniquement)
    df_pres = df[df["presence"] == 1].reset_index(drop=True)
    X_reg   = df_pres[FEATURES_ORDER]
    y_reg   = df_pres["yeo_NASC"]

    # Scaling avec les scalers fit sur données complètes (fidèle au notebook)
    X1_scaled    = scaler_clf.transform(X1)
    X_reg_scaled = scaler_reg.transform(X_reg)

    # ── Métriques classification — StratifiedKFold(5) ─────────────────────────
    print("[evaluate] Évaluation classification (StratifiedKFold 5 folds)...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_all  = []
    y_true_all  = []
    y_proba_all = []

    # Même objet réutilisé à chaque fold — fidèle au notebook
    eval_clf = HistGradientBoostingClassifier(**{**params_clf, "random_state": 42})

    for fold, (train_idx, test_idx) in enumerate(skf.split(X1_scaled, y1), 1):
        X_tr, X_te = X1_scaled[train_idx], X1_scaled[test_idx]
        y_tr = y1.iloc[train_idx]
        y_te = y1.iloc[test_idx]

        eval_clf.fit(X_tr, y_tr)
        y_pred_all.extend(eval_clf.predict(X_te))
        y_true_all.extend(y_te)
        y_proba_all.extend(eval_clf.predict_proba(X_te)[:, 1])
        print(f"  Fold {fold}/5 ✓")

    y_pred_all  = np.array(y_pred_all)
    y_true_all  = np.array(y_true_all)
    y_proba_all = np.array(y_proba_all)

    f1      = float(f1_score(y_true_all, y_pred_all))
    roc_auc = float(roc_auc_score(y_true_all, y_proba_all))
    print(f"  F1={f1:.4f} | ROC-AUC={roc_auc:.4f}")

    # ── Métriques régression — KFold(5) ──────────────────────────────────────
    print("[evaluate] Évaluation régression (KFold 5 folds)...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    r2_scores = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_reg_scaled), 1):
        X_tr_r, X_te_r = X_reg_scaled[train_idx], X_reg_scaled[test_idx]
        y_tr_r = y_reg.iloc[train_idx]
        y_te_r = y_reg.iloc[test_idx]

        fold_reg = HistGradientBoostingRegressor(**{**params_reg, "random_state": 42})
        fold_reg.fit(X_tr_r, y_tr_r)
        r2_scores.append(float(r2_score(y_te_r, fold_reg.predict(X_te_r))))
        print(f"  Fold {fold}/5 ✓")

    r2 = float(np.mean(r2_scores))
    print(f"  R² moyen={r2:.4f}")

    # ── Sauvegarde métriques ──────────────────────────────────────────────────
    metrics = {"f1": f1, "roc_auc": roc_auc, "r2": r2}
    with open(os.path.join(SHARED_DATA, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("[evaluate] metrics.json sauvegardé")

    # ── Importance Phase 1 — permutation sur données complètes (n_repeats=10) ─
    print("[evaluate] Calcul importance Phase 1 (classification)...")
    perm_clf = permutation_importance(
        clf, X1_scaled, y1,
        n_repeats=10, random_state=42, scoring="f1"
    )
    imp_clf_df = pd.DataFrame({
        "feature":    FEATURES_ORDER,
        "importance": perm_clf.importances_mean,
    }).sort_values("importance", ascending=False)
    imp_clf_df.to_csv(os.path.join(SHARED_DATA, "importance_phase1.csv"), index=False)
    print("[evaluate] importance_phase1.csv sauvegardé")

    # ── Importance Phase 2 — permutation sur données complètes (n_repeats=10) ─
    print("[evaluate] Calcul importance Phase 2 (régression)...")
    perm_reg = permutation_importance(
        reg, X_reg_scaled, y_reg,
        n_repeats=10, random_state=42, scoring="r2"
    )
    imp_reg_df = pd.DataFrame({
        "feature":    FEATURES_ORDER,
        "importance": perm_reg.importances_mean,
    }).sort_values("importance", ascending=False)
    imp_reg_df.to_csv(os.path.join(SHARED_DATA, "importance_phase2.csv"), index=False)
    print("[evaluate] importance_phase2.csv sauvegardé")

    print(f"\n[evaluate] ✓ F1={f1:.3f} | ROC-AUC={roc_auc:.3f} | R²={r2:.3f}")


if __name__ == "__main__":
    main()
