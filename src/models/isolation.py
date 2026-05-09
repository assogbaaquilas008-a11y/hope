"""
SIPT Pro v4 — Isolation Forest helpers
extract_features, SHAP explain, niveau combiné, risk_pct.
"""
import numpy as np


def extract_features(s: np.ndarray) -> np.ndarray:
    """12 features statistiques sur une fenêtre de consommation."""
    s       = np.array(s, dtype=float)
    n       = len(s)
    wk_vals = s[np.arange(n) % 7 >= 5]
    wd_vals = s[np.arange(n) % 7 <  5]
    diff_s  = np.diff(s)
    mean_s  = np.mean(s)   if n > 0 else 0.0
    std_s   = np.std(s)    if n > 1 else 0.0
    trend   = np.polyfit(np.arange(n), s, 1)[0] if n >= 2 else 0.0
    h1      = np.mean(s[:n // 2]) if n >= 2 else mean_s
    h2      = np.mean(s[n // 2:]) if n >= 2 else mean_s
    mean_wk = np.mean(wk_vals) if len(wk_vals) > 0 else mean_s
    mean_wd = np.mean(wd_vals) if len(wd_vals) > 0 else mean_s
    std_wk  = np.std(wk_vals)  if len(wk_vals) > 1 else std_s
    std_wd  = np.std(wd_vals)  if len(wd_vals) > 1 else std_s
    if n > 2 and np.std(s[:-1]) > 0 and np.std(s[1:]) > 0:
        autocorr = float(np.corrcoef(s[:-1], s[1:])[0, 1])
    else:
        autocorr = 0.0
    return np.array([
        mean_s,
        std_s,
        trend,
        np.mean(diff_s)  if n > 1 else 0.0,
        h1 / h2          if h2 > 0 else 1.0,
        np.max(s) - np.min(s),
        std_s / mean_s   if mean_s > 0 else 0.0,
        float(np.sum(s < 0.5)),
        mean_wk / mean_wd if mean_wd > 0 else 1.0,
        std_wk  / std_wd  if std_wd  > 0 else 1.0,
        autocorr,
        float(np.sum(diff_s[1:] * diff_s[:-1] < 0)) if n > 2 else 0.0,
    ])


def shap_explain(explainer, feat_vec: np.ndarray, feature_names: list, top_n: int = None) -> list:
    """
    Retourne [(feature_name, shap_value)] triés par |valeur| décroissant.
    Si top_n est spécifié, retourne seulement les top_n.
    Retourne [] si explainer indisponible.
    """
    if explainer is None:
        return []
    try:
        sv = explainer.shap_values(feat_vec.reshape(1, -1))[0]
        pairs = sorted(zip(feature_names, sv), key=lambda x: abs(x[1]), reverse=True)
        if top_n is not None and top_n > 0:
            pairs = pairs[:top_n]
        return pairs
    except Exception:
        return []

def combined_level(if_score: float, threshold_if: float, lstm_anom: bool, cusum_alarm: bool = False):
    """
    Retourne (level, label, detector_name)
    level : str (DOUBLE_ALERT, ALERT_IF, SUSPECT_LSTM, WARNING_CUSUM, NORMAL)
    """
    if_alert = if_score < threshold_if   # plus négatif = anormal
    if if_alert and (lstm_anom or cusum_alarm):
        return ("DOUBLE_ALERT", "Double alerte", "IF+LSTM")
    if if_alert:
        return ("ALERT_IF", "Alerte Isolation Forest", "IF")
    if lstm_anom:
        return ("SUSPECT_LSTM", "Suspect LSTM", "LSTM")
    if cusum_alarm:
        return ("WARNING_CUSUM", "Micro-dérive CUSUM", "CUSUM")
    return ("NORMAL", "Normal", "")

def risk_pct(score_if: float, score_lstm: float | None,
             threshold_if: float, threshold_lstm: float | None,
             cusum_score: float = 0.0) -> float:
    """
    Calcule un risque [0, 1] combinant IF, LSTM et CUSUM.
    Pondération : IF 50%, LSTM 35%, CUSUM 15%.
    """
    # Composante IF : normalise le score autour du seuil
    # Plus le score est négatif (anormal), plus risk_if est élevé
    if_range = abs(threshold_if) if threshold_if != 0 else 0.5
    risk_if  = float(np.clip((threshold_if - score_if) / (if_range * 2), 0, 1))

    # Composante LSTM
    if score_lstm is not None and threshold_lstm and threshold_lstm > 0:
        risk_lstm = float(np.clip(score_lstm / (threshold_lstm * 2), 0, 1))
    else:
        risk_lstm = 0.0

    # Composante CUSUM (normalisée entre 0 et 1, déjà fournie par le détecteur)
    risk_cusum = float(np.clip(cusum_score, 0, 1))

    return risk_if * 0.70 + risk_lstm * 0.30 #+ risk_cusum * 0.15
