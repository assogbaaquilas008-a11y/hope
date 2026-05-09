"""
SIPT Pro v4 — Chargement des artefacts ML
Isolation Forest, LSTM, Scaler, SHAP explainer.
"""
import os
import joblib
import numpy as np
import streamlit as st

from src.config import (
    IF_MODEL_FILE, SCALER_FILE, FEATURE_NAMES_FILE,
    BG_SAMPLES_FILE, LSTM_MODEL_FILE, LSTM_THRESH_FILE,
)


@st.cache_resource(show_spinner=False)
def load_if_artifacts():
    """Charge IsolationForest + scaler + feature_names + background pour SHAP."""
    if not os.path.exists(IF_MODEL_FILE):
        return None, None, None, None
    iso    = joblib.load(IF_MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    fnames = joblib.load(FEATURE_NAMES_FILE)
    bg     = joblib.load(BG_SAMPLES_FILE) if os.path.exists(BG_SAMPLES_FILE) else None
    return iso, scaler, fnames, bg


@st.cache_resource(show_spinner=False)
def load_lstm_artifacts():
    """Charge le modèle LSTM et son seuil MSE. Retourne (None, None) si absent."""
    if not os.path.exists(LSTM_MODEL_FILE) or not os.path.exists(LSTM_THRESH_FILE):
        return None, None
    try:
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
        from tensorflow.keras.models import load_model
        model  = load_model(LSTM_MODEL_FILE, compile=False)
        thresh = joblib.load(LSTM_THRESH_FILE)
        return model, thresh
    except Exception:
        return None, None


@st.cache_resource(show_spinner=False)
def load_shap_explainer(_iso, bg):
    """Crée un TreeExplainer SHAP pour l'Isolation Forest."""
    if _iso is None or bg is None:
        return None
    try:
        import shap
        return shap.TreeExplainer(_iso, data=bg, feature_perturbation="interventional")
    except Exception:
        return None
