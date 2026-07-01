"""
config.py — Constantes globales de l'application.
"""

import os

# ── Chemins ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR   = os.path.join(BASE_DIR, "data")

CLF_PATH          = os.path.join(MODELS_DIR, "clf_presence.joblib")
REG_PATH          = os.path.join(MODELS_DIR, "reg_biomasse.joblib")
YEO_PATH          = os.path.join(MODELS_DIR, "yeo_transformer.joblib")
SCALER_CLF_PATH   = os.path.join(MODELS_DIR, "scaler_clf.joblib")
SCALER_REG_PATH   = os.path.join(MODELS_DIR, "scaler_reg.joblib")

MEDIANES_PATH         = os.path.join(DATA_DIR, "medianes_train.json")
FEATURES_SIMPLE_PATH  = os.path.join(DATA_DIR, "features_simple.json")
FEATURES_AVANCE_PATH  = os.path.join(DATA_DIR, "features_avance.json")
IMP_PHASE1_PATH       = os.path.join(DATA_DIR, "importance_phase1.csv")
IMP_PHASE2_PATH       = os.path.join(DATA_DIR, "importance_phase2.csv")
METRICS_PATH          = os.path.join(DATA_DIR, "metrics.json")
SAMPLE_LOC_PATH       = os.path.join(DATA_DIR, "sample_locations.csv")

# ── Ordre exact des features utilisé à l'entraînement ────────────────────────
# ⚠️  Cet ordre DOIT correspondre exactement aux colonnes vues par clf et reg.
#
# Dérivé de la Section 6 du notebook (cell 54) :
#   df_final.drop(columns=['yeo_NASC', 'Mois', 'Annee']) après suppression de
#   ['NASC_Anchois', 'Longitude', 'Latitude', 'Phyc', 'Vgos', 'O2', 'Po4',
#    'No3', 'Spco2', 'Ara', 'Cal', 'Si', 'Grenn algae', 'Secchi', 'Prokar',
#    'Pico', 'Nppv', 'Dino', 'Diatoms', 'Nano']
#
# ⚠️  O2 et Spco2 sont EXCLUS (figuraient dans to_drop_final).
FEATURES_ORDER = [
    "Bathy",
    "SST",
    "So_mean",
    "Density",
    "SH",
    "Wind",
    "Uo",
    "Vo",
    "Ue",
    "Ve",       # composante méridionale du courant d'Ekman
    "Ugos",
    "pH",
    "Alk",
    "Fe",
    "Chla",
    "Hapto",    # Haptophytes
    "Micro",    # Micro-phytoplancton
    "Prochlo",  # Prochlorococcus
    "SM",       # Small phytoplankton
    "Mois_sin",
    "Mois_cos",
]

# ── Descriptions des features (pour les labels du formulaire) ─────────────────
FEATURE_LABELS = {
    "Bathy":    "Bathymétrie (m)",
    "SST":      "Température de surface (°C)",
    "So_mean":  "Salinité moyenne (PSU)",
    "Density":  "Densité de l'eau (kg/m³)",
    "SH":       "Hauteur de la colonne d'eau (m)",
    "Wind":     "Vitesse du vent (m/s)",
    "Uo":       "Courant zonal de surface U (m/s)",
    "Vo":       "Courant méridional de surface V (m/s)",
    "Ue":       "Courant zonal d'Ekman Ue (m/s)",
    "Ve":       "Courant méridional d'Ekman Ve (m/s)",
    "Ugos":     "Vitesse géostrophique (m/s)",
    "pH":       "pH",
    "Alk":      "Alcalinité totale (µmol/kg)",
    "Fe":       "Concentration en fer (nmol/L)",
    "Chla":     "Chlorophylle-a (mg/m³)",
    "Hapto":    "Haptophytes (mg/m³)",
    "Micro":    "Micro-phytoplancton (mg/m³)",
    "Prochlo":  "Prochlorococcus (mg/m³)",
    "SM":       "Small phytoplankton (mg/m³)",
    "Mois_sin": "Mois — composante sinus",
    "Mois_cos": "Mois — composante cosinus",
}

# ── Seuil de décision par défaut ──────────────────────────────────────────────
DEFAULT_THRESHOLD = 0.2

# ── Métadonnées ───────────────────────────────────────────────────────────────
APP_TITLE       = "Prédiction de Biomasse Halieutique — Anchois"
APP_DESCRIPTION = (
    "Cette application expose un pipeline Hurdle à deux étages pour prédire "
    "la biomasse acoustique de l'anchois (valeur NASC) à partir de variables "
    "environnementales. Le modèle prédit d'abord la **présence** de biomasse, "
    "puis, si elle est détectée, **la quantité** (NASC, m²/nmi²)."
)
