"""
SIPT Pro v4 — Entrypoint Streamlit
Assemblage de tous les onglets via les modules src/.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")

# S'assurer que le répertoire racine est dans le path
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

from src.config import (
    DATASET_FILE, META_FILE, FEEDER_MAP_FILE, FEEDER_INJ_FILE, REGION_INJ_FILE,
    IF_MODEL_FILE, SCALER_FILE, WINDOW_SIZE, CONFIRM_THRESHOLD,
    COLOR_BG, COLOR_TEXT, COLOR_INFO, COLOR_ALERT, COLOR_WARN, COLOR_SUCCESS,
)
from src.models.loaders     import load_if_artifacts, load_lstm_artifacts, load_shap_explainer
from src.registry.manager   import load_registry
from src.ui.components      import inject_css, scada_header, kpi_grid, section_header
from src.ui.pages           import (
    vue_reseau, surveillance, fraudsters, suspects, analytics, exclusions, model_info
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SIPT Pro v4 — Surveillance Réseau",
    layout="wide", page_icon="⚡",
    initial_sidebar_state="expanded",
)
inject_css()

# ── Chargement artefacts ──────────────────────────────────────────────────────
iso, scaler, feature_names, bg_samples = load_if_artifacts()
lstm_model, lstm_threshold             = load_lstm_artifacts()
shap_explainer                         = load_shap_explainer(iso, bg_samples)

# ── Seuil IF ──────────────────────────────────────────────────────────────────
threshold_if = -0.10  # valeur par défaut
if iso is not None:
    try:
        _feat_file = "features_debug.csv"
        if os.path.exists(_feat_file):
            _fd = pd.read_csv(_feat_file)
            threshold_if = float(np.percentile(_fd["score_IF"].values, 5))
    except Exception:
        pass

# ── Données ────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    if not os.path.exists(DATASET_FILE):
        return None, None
    df   = pd.read_csv(DATASET_FILE, parse_dates=["timestamp"])
    meta = pd.read_csv(META_FILE) if os.path.exists(META_FILE) else pd.DataFrame()
    return df, meta

@st.cache_data(show_spinner=False)
def load_topology_data():
    fmap = pd.read_csv(FEEDER_MAP_FILE) if os.path.exists(FEEDER_MAP_FILE) else pd.DataFrame()
    finj = pd.read_csv(FEEDER_INJ_FILE) if os.path.exists(FEEDER_INJ_FILE) else pd.DataFrame()
    rinj = pd.read_csv(REGION_INJ_FILE) if os.path.exists(REGION_INJ_FILE) else pd.DataFrame()
    return fmap, finj, rinj

df, meta = load_data()
feeder_map, feeder_injection, region_injection = load_topology_data()

# ── Registre ──────────────────────────────────────────────────────────────────
if "registry" not in st.session_state:
    st.session_state.registry = load_registry()
registry = st.session_state.registry

lstm_ready = lstm_model is not None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f'<div style="font-family:Share Tech Mono;font-size:14px;color:{COLOR_INFO};'
        f'letter-spacing:2px;margin-bottom:16px">⚡ SIPT PRO v4</div>',
        unsafe_allow_html=True,
    )
    pres_mode = st.toggle("🖥 Mode Présentation", value=False)
    st.divider()

    # Statuts
    data_ok  = df is not None
    model_ok = iso is not None
    st.markdown(
        f'<div style="font-size:11px;font-family:Share Tech Mono">'
        f'<div style="color:{"#00FFA5" if data_ok else "#FF4C4C"}">{"✓" if data_ok else "✗"} Dataset</div>'
        f'<div style="color:{"#00FFA5" if model_ok else "#FF4C4C"}">{"✓" if model_ok else "✗"} Isolation Forest</div>'
        f'<div style="color:{"#00FFA5" if lstm_ready else "#FFB347"}">{"✓" if lstm_ready else "⚠"} LSTM Autoencoder</div>'
        f'<div style="color:{"#00FFA5" if shap_explainer else "#FFB347"}">{"✓" if shap_explainer else "⚠"} SHAP Explainer</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    if not data_ok:
        st.error("Données absentes. Lancez : `python train.py`")
    if not model_ok:
        st.warning("Modèles absents. Lancez : `python train.py`")

    st.markdown(
        f'<div style="font-size:10px;color:#4a7aaa;font-family:Share Tech Mono;margin-top:8px">'
        f'IF seuil : {threshold_if:.3f}<br>'
        f'LSTM seuil : {f"{lstm_threshold:.4f}" if lstm_threshold is not None else "N/A"}<br>'
        f'Confirmation : {CONFIRM_THRESHOLD} det.<br>'
        f'Fenêtre : {WINDOW_SIZE}j</div>',
        unsafe_allow_html=True,
    )

    if st.button("🗑 Vider registre", use_container_width=True):
        st.session_state.registry = {}
        from src.registry.manager import save_registry
        save_registry({})
        st.rerun()

# ── Header ────────────────────────────────────────────────────────────────────
clock_str = datetime.now().strftime("%H:%M:%S · %d/%m/%Y")
scada_header(clock_str, pres_mode)

# ── KPI bar ───────────────────────────────────────────────────────────────────
if data_ok:
    n_clients   = df["client_id"].nunique()
    n_confirmed = sum(1 for e in registry.values() if e.get("is_confirmed"))
    n_suspects  = sum(1 for e in registry.values() if not e.get("is_confirmed") and e.get("detections"))
    n_away      = sum(1 for e in registry.values() if e.get("away_active"))
    n_sf        = sum(1 for e in registry.values() if e.get("sensor_fault_active"))
    total_det   = sum(len(e.get("detections", [])) for e in registry.values())

    kpi_grid([
        (n_clients,   "Clients surveillés", "info"),
        (n_confirmed, "Fraudeurs avérés",   "alert"),
        (n_suspects,  "Suspects actifs",    "warn"),
        (total_det,   "Détections totales", "lstm"),
        (n_away,      "Clients absents",    "ok"),
        (n_sf,        "Capteurs en panne",  "purple"),
    ])

# ── Tabs ──────────────────────────────────────────────────────────────────────
(tab_reseau, tab_surv, tab_fraud, tab_susp,
 tab_anal, tab_excl, tab_info) = st.tabs([
    "🌐 Vue Réseau",
    "📡 Surveillance",
    "🔴 Fraudeurs",
    "⚡ Suspects",
    "📊 Analytique",
    "🛡 Exclusions",
    "ℹ️ Modèles",
])

if not data_ok:
    for t in [tab_reseau, tab_surv, tab_fraud, tab_susp, tab_anal, tab_excl]:
        with t:
            st.error("Dataset introuvable. Lancez `python train.py` pour générer les données et entraîner les modèles.")
    with tab_info:
        model_info.render(threshold_if, lstm_threshold, lstm_ready)
    st.stop()

with tab_reseau:
    vue_reseau.render(df, registry, region_injection if not region_injection.empty else None)

with tab_surv:
    surveillance.render(
        df, meta, registry,
        iso, scaler, feature_names, shap_explainer,
        lstm_model, lstm_threshold,
        threshold_if, pres_mode,
    )
    st.session_state.registry = registry

with tab_fraud:
    fraudsters.render(registry, df, scaler, shap_explainer, feature_names)

with tab_susp:
    suspects.render(registry, df, scaler, shap_explainer, feature_names)

with tab_anal:
    analytics.render(
        df, meta, registry, iso, scaler, feature_names,
        feeder_map if not feeder_map.empty else None,
        feeder_injection if not feeder_injection.empty else None,
    )

with tab_excl:
    exclusions.render(registry, df)
    st.session_state.registry = registry

with tab_info:
    model_info.render(threshold_if, lstm_threshold, lstm_ready)
