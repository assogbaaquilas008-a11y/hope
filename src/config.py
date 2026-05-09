"""
SIPT Pro v4 — Configuration centrale
Toutes les constantes, chemins, seuils partagés.
"""

# ── Fichiers de données ────────────────────────────────────────────────────────
DATASET_FILE       = "electricity_fraud_dataset.csv"
META_FILE          = "client_metadata.csv"
FEATURES_FILE      = "features_debug.csv"
FEEDER_MAP_FILE    = "client_feeder_map.csv"
FEEDER_INJ_FILE    = "feeder_injection.csv"
REGION_INJ_FILE    = "region_injection.csv"


# False positive management
AWAY_LOW_THRESHOLD = 0.5          # kWh
AWAY_RETURN_THRESHOLD = 1.0
AWAY_MIN_DAYS = 7
SENSOR_FAULT_VAR = 0.02
DATA_QUALITY_MIN = 0.7
EMA_ALPHA = 0.05

# ── Artefacts ML ──────────────────────────────────────────────────────────────
IF_MODEL_FILE      = "isolation_forest.pkl"
SCALER_FILE        = "scaler.pkl"
FEATURE_NAMES_FILE = "feature_names.pkl"
BG_SAMPLES_FILE    = "background_samples.pkl"
LSTM_MODEL_FILE    = "lstm_autoencoder.keras"
LSTM_THRESH_FILE   = "lstm_threshold.pkl"
SEED_FILE          = "dataset_seed.pkl"

# ── Registre ──────────────────────────────────────────────────────────────────
REGISTRY_FILE      = "fraud_registry.json"

# ── Simulation ────────────────────────────────────────────────────────────────
WINDOW_SIZE        = 60
HISTORY_DAYS       = 60
SIM_DAYS           = 7
CONFIRM_THRESHOLD  = 2       # détections pour "avéré"

# ── Réseau ────────────────────────────────────────────────────────────────────
REGIONS = ["NORD", "SUD", "EST", "OUEST", "CENTRE"]

# ── Seuils détection ──────────────────────────────────────────────────────────
ENERGY_BALANCE_WARN_PROB  = 0.50   # prob piquage → warning
ENERGY_BALANCE_ALERT_PROB = 0.70   # prob piquage → alerte
FEEDER_BALANCE_THRESH     = 0.12   # écart normalisé sur feeder
AWAY_LOW_THRESHOLD        = 0.5    # kWh/j → "away"
AWAY_RETURN_THRESHOLD     = 1.0    # kWh/j → retour
AWAY_MIN_DAYS             = 7      # jours consécutifs
SENSOR_FAULT_VAR          = 0.02   # variance min
DATA_QUALITY_MIN          = 0.70   # ratio valides
EMA_ALPHA                 = 0.05   # lissage baseline

# ── CUSUM ─────────────────────────────────────────────────────────────────────
CUSUM_K_FACTOR    = 0.5    # k = K_FACTOR * sigma
CUSUM_H_FACTOR    = 5.0    # h = H_FACTOR * sigma

# ── Couleurs SCADA v4 (haute visibilité) ──────────────────────────────────────

COLOR_BG        = "#F5F5F4"
COLOR_SURFACE   = "#FFFFFF"

COLOR_TEXT      = "#D6E2F0"
COLOR_TEXT_DARK = "#0F172A"

COLOR_ALERT     = "#FF4C4C"
COLOR_WARN      = "#FFB347"
COLOR_INFO      = "#5FAFD7"
COLOR_SUCCESS   = "#00FFA5"
COLOR_ACCENT    = "#A855F7"
COLOR_LSTM      = "#F97316"

COLOR_PANEL     = "#0D1A35"
COLOR_PANEL_2   = "#132347"

COLOR_BORDER    = "#223A5E"
COLOR_BORDER_SOFT = "rgba(80,120,200,.22)"

CLIENT_PROFILES = {
    "RESIDENTIEL": {
        "base_range": (3, 18), "weekend_factor": 1.25,
        "noise_std": 0.12, "seasonal": True,
        "description": "Foyer domestique – consommation modérée, sensible aux week-ends",
    },
    "INDUSTRIEL": {
        "base_range": (80, 250), "weekend_factor": 0.60,
        "noise_std": 0.04, "seasonal": False,
        "description": "Site industriel – forte consommation, chute le weekend",
    },
    "COMMERCIAL": {
        "base_range": (25, 90), "weekend_factor": 0.45,
        "noise_std": 0.09, "seasonal": True,
        "description": "Commerce/bureau – pic en semaine, quasi-nul le weekend",
    },
    "AGRICOLE": {
        "base_range": (8, 40), "weekend_factor": 1.05,
        "noise_std": 0.18, "seasonal": True,
        "description": "Exploitation agricole – consommation saisonnière",
    },
}
PROFILE_WEIGHTS = [0.55, 0.15, 0.20, 0.10]

FEATURE_NAMES = [
    "Conso moyenne (kWh/j)",
    "Volatilité (écart-type)",
    "Tendance linéaire (slope)",
    "Variation journalière moy",
    "Ratio 1ère/2ème période",
    "Amplitude max-min",
    "Coefficient de variation",
    "Jours très faible conso (<0.5 kWh)",
    "Ratio Weekend/Weekday",
    "Volatilité Weekend vs Weekday",
    "Autocorrélation temporelle (lag-1)",
    "Instabilité directionnelle",
]
