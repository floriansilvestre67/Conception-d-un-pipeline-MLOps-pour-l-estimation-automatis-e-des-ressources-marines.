"""
predict.py — Logique de prédiction du pipeline Hurdle.

Pipeline exact reproduit depuis la Section 6 du notebook :
  1. StandardScaler (scaler_clf) → clf.predict_proba()
  2. Si présence : StandardScaler (scaler_reg) → reg.predict() → yeo.inverse_transform()
"""

import joblib
import numpy as np
import pandas as pd
from functools import lru_cache

from config import (
    CLF_PATH, REG_PATH, YEO_PATH,
    SCALER_CLF_PATH, SCALER_REG_PATH,
    FEATURES_ORDER,
)


@lru_cache(maxsize=1)
def load_artifacts():
    """Charge les cinq artefacts joblib une seule fois (mis en cache)."""
    clf        = joblib.load(CLF_PATH)
    reg        = joblib.load(REG_PATH)
    yeo        = joblib.load(YEO_PATH)
    scaler_clf = joblib.load(SCALER_CLF_PATH)
    scaler_reg = joblib.load(SCALER_REG_PATH)
    return clf, reg, yeo, scaler_clf, scaler_reg


def predict_hurdle(features: dict, clf, reg, yeo, scaler_clf, scaler_reg,
                   threshold: float = 0.5) -> dict:
    """
    Exécute le pipeline Hurdle à deux étages en reproduisant exactement
    le pipeline du notebook (Section 6) :

      Phase 1 : scaler_clf.transform(X) → clf.predict_proba()
      Phase 2 : scaler_reg.transform(X) → reg.predict() → yeo.inverse_transform()

    Parameters
    ----------
    features   : dict {nom_feature: valeur}
    clf        : HistGradientBoostingClassifier (clf_presence)
    reg        : HistGradientBoostingRegressor (reg_biomasse)
    yeo        : PowerTransformer Yeo-Johnson fitté sur NASC_Anchois
    scaler_clf : StandardScaler fitté sur X1 (toutes les observations)
    scaler_reg : StandardScaler fitté sur X_reg (présences uniquement)
    threshold  : seuil de décision classification

    Returns
    -------
    dict :
        "proba_presence" : float  — probabilité de présence [0, 1]
        "presence"       : int    — 0 ou 1
        "pred_nasc"      : float  — valeur NASC estimée (0 si absence)
    """
    # Réordonner les colonnes dans l'ordre exact utilisé à l'entraînement
    X = pd.DataFrame([features])[FEATURES_ORDER]

    # ── Phase 1 : classification ──────────────────────────────────────────────
    X_clf = scaler_clf.transform(X)
    proba = float(clf.predict_proba(X_clf)[0, 1])
    presence = int(proba >= threshold)
    pred_nasc = 0.0

    # ── Phase 2 : régression (si présence prédite) ────────────────────────────
    if presence:
        X_reg = scaler_reg.transform(X)
        yeo_pred = reg.predict(X_reg)[0]
        pred_nasc = float(yeo.inverse_transform([[yeo_pred]])[0, 0])
        pred_nasc = max(pred_nasc, 0.0)  # sécurité : NASC toujours ≥ 0

    return {
        "proba_presence": proba,
        "presence": presence,
        "pred_nasc": pred_nasc,
    }


def build_csv_row(features: dict, result: dict) -> pd.DataFrame:
    """Construit un DataFrame d'une ligne pour l'export CSV."""
    row = {**features, **result}
    return pd.DataFrame([row])
