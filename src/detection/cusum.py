"""
SIPT Pro v4 — Détecteur CUSUM pour micro-dérives
Détecte une chute négative persistante de la consommation (bypass lent).
"""
import numpy as np
from src.config import CUSUM_K_FACTOR, CUSUM_H_FACTOR


def compute_cusum(vals: np.ndarray, k_factor: float = CUSUM_K_FACTOR,
                  h_factor: float = CUSUM_H_FACTOR) -> dict:
    """
    Algorithme CUSUM unilatéral (côté négatif) :
      k = k_factor * sigma    (slack / référence)
      h = h_factor * sigma    (seuil de déclenchement)

    S_n = max(0, S_{n-1} + (mu - x_n - k))

    Retourne :
      alarm        : bool
      cusum_score  : float normalisé [0, 1]
      cusum_series : array des valeurs S_n
      max_deviation: float (S_n max)
    """
    vals  = np.array(vals, dtype=float)
    n     = len(vals)
    mu    = np.mean(vals)
    sigma = np.std(vals) if np.std(vals) > 1e-6 else 1.0

    k = k_factor * sigma
    h = h_factor * sigma

    S      = np.zeros(n)
    S_prev = 0.0
    for i in range(n):
        S[i]   = max(0.0, S_prev + (mu - vals[i] - k))
        S_prev = S[i]

    max_S      = float(np.max(S))
    alarm      = max_S > h
    # Score normalisé : max_S / h → 1 = alarme déclenchée
    cusum_norm = float(np.clip(max_S / h, 0.0, 2.0) / 2.0)  # borne à 1

    return {
        "alarm":        alarm,
        "cusum_score":  cusum_norm,
        "cusum_series": S,
        "max_deviation": max_S,
        "threshold_h":  h,
        "mu":           mu,
    }


def cusum_for_client(vals: np.ndarray, window: int = 30) -> dict:
    """
    Applique CUSUM sur la fenêtre glissante d'un client.
    Si données insuffisantes, retourne alarm=False.
    """
    if len(vals) < max(10, window // 3):
        return {"alarm": False, "cusum_score": 0.0, "cusum_series": np.zeros(len(vals)),
                "max_deviation": 0.0, "threshold_h": 1.0, "mu": 0.0}
    return compute_cusum(vals[-window:])
