"""
SIPT Pro v4 — Gestion des faux positifs
Away detection, capteur bloqué, qualité données, baseline EMA.
Utilisation d'une classe pour conserver l'état par client.
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any

from src.config import (
    AWAY_LOW_THRESHOLD, AWAY_RETURN_THRESHOLD, AWAY_MIN_DAYS,
    SENSOR_FAULT_VAR, DATA_QUALITY_MIN, EMA_ALPHA,
)


# ----------------------------------------------------------------------
# Fonctions utilitaires (inchangées, mais appelées par la classe)
# ----------------------------------------------------------------------
def _detect_away_status(vals: np.ndarray) -> dict:
    """Détection automatique d'absence."""
    if len(vals) < AWAY_MIN_DAYS:
        return {"is_away": False, "away_days": 0}
    recent = vals[-AWAY_MIN_DAYS:]
    away = bool(np.all(recent < AWAY_LOW_THRESHOLD))
    days = int(np.sum(vals < AWAY_LOW_THRESHOLD))
    return {"is_away": away, "away_days": days}


def _detect_sensor_fault(vals: np.ndarray, window: int = 7) -> dict:
    """Détection automatique de capteur bloqué."""
    if len(vals) < window:
        return {"sensor_fault": False, "reason": ""}
    recent = vals[-window:]
    var_r = float(np.var(recent))
    all_eq = bool(np.all(recent == recent[0]))
    fault = var_r < SENSOR_FAULT_VAR or all_eq
    reason = "Variance nulle" if all_eq else (f"Variance faible ({var_r:.4f})" if fault else "")
    return {"sensor_fault": fault, "reason": reason}


def _data_quality_score(vals: np.ndarray, window: int = 30) -> dict:
    """Ratio de valeurs valides."""
    if len(vals) == 0:
        return {"quality": 0.0, "skip": True}
    w = vals[-window:] if len(vals) >= window else vals
    valid = float(np.sum((~np.isnan(w)) & (w >= 0)))
    quality = valid / len(w)
    return {"quality": round(quality, 3), "skip": quality < DATA_QUALITY_MIN}


def _update_ema_baseline(current_ema: Optional[float], new_value: float, alpha: float = EMA_ALPHA) -> float:
    if current_ema is None:
        return float(new_value)
    return float(alpha * new_value + (1 - alpha) * current_ema)


# ----------------------------------------------------------------------
# Classe principale utilisée dans l'application
# ----------------------------------------------------------------------
class FalsePositiveManager:
    """
    Gère l'état des clients pour :
    - Absences (away) automatiques ou manuelles
    - Défauts capteur (sensor fault)
    - Baseline EMA (consommation normale glissante)
    - Qualité des données
    """
    def __init__(self):
        # Structure : client_id -> dict avec clés 'away', 'sensor_fault', 'baseline'
        self._clients: Dict[str, Dict[str, Any]] = {}

    def _ensure_client(self, client_id: str):
        if client_id not in self._clients:
            self._clients[client_id] = {
                "away_active": False,
                "away_start": None,
                "sensor_fault_active": False,
                "sensor_fault_start": None,
                "baseline_ema": None,
                "last_window": None,
            }

    # --- Gestion away ---
    def update_away(self, client_id: str, window_vals: list, current_date) -> bool:
        """
        Met à jour le statut "away" en fonction des consommations récentes.
        Retourne True si le client est actuellement considéré absent.
        """
        self._ensure_client(client_id)
        client = self._clients[client_id]
        vals = np.array(window_vals)

        # Vérifier d'abord les déclarations manuelles (via registre)
        # Ici on se base uniquement sur la détection auto ; les manuels seront gérés via le registre
        auto = _detect_away_status(vals)
        if auto["is_away"] and not client["away_active"]:
            # Début d'absence automatique
            client["away_active"] = True
            client["away_start"] = current_date
        elif not auto["is_away"] and client["away_active"]:
            # Fin d'absence automatique (retour à une conso normale)
            client["away_active"] = False
            # On conserve la date de fin dans une clé séparée si besoin
        return client["away_active"]

    # --- Gestion capteur ---
    def update_sensor_fault(self, client_id: str, window_vals: list) -> bool:
        """
        Détecte automatiquement un défaut capteur.
        Retourne True si un défaut est actif.
        """
        self._ensure_client(client_id)
        client = self._clients[client_id]
        vals = np.array(window_vals)
        auto = _detect_sensor_fault(vals)
        if auto["sensor_fault"] and not client["sensor_fault_active"]:
            client["sensor_fault_active"] = True
            client["sensor_fault_start"] = pd.Timestamp.now()
        elif not auto["sensor_fault"] and client["sensor_fault_active"]:
            # Réparation automatique
            client["sensor_fault_active"] = False
        return client["sensor_fault_active"]

    # --- Qualité données ---
    def data_quality(self, window_vals: list, window: int = 30) -> float:
        """Retourne le ratio de qualité (0..1)."""
        vals = np.array(window_vals)
        dq = _data_quality_score(vals, window)
        return dq["quality"]

    # --- Baseline EMA ---
    def update_baseline(self, client_id: str, current_consumption: float) -> float:
        """Met à jour l'EMA du client et retourne la nouvelle valeur."""
        self._ensure_client(client_id)
        client = self._clients[client_id]
        new_ema = _update_ema_baseline(client["baseline_ema"], current_consumption)
        client["baseline_ema"] = new_ema
        return new_ema

    def get_baseline(self, client_id: str) -> Optional[float]:
        self._ensure_client(client_id)
        return self._clients[client_id]["baseline_ema"]

    # --- Statut global d'exonération ---
    def is_suppressed(self, client_id: str) -> bool:
        """
        Retourne True si le client doit être exclu des détections (away ou capteur défaillant).
        """
        self._ensure_client(client_id)
        client = self._clients[client_id]
        return client["away_active"] or client["sensor_fault_active"]

    # --- (Optionnel) Lecture/écriture depuis le registre pour sauvegarde ---
    def load_from_registry(self, registry: dict):
        """Synchronise l'état du manager avec le registre persistant."""
        for cid, entry in registry.items():
            self._ensure_client(cid)
            self._clients[cid]["away_active"] = entry.get("away_active", False)
            self._clients[cid]["sensor_fault_active"] = entry.get("sensor_fault_active", False)
            # Ne pas écraser la baseline si elle existe déjà
            if "baseline_ema" in entry and entry["baseline_ema"] is not None:
                self._clients[cid]["baseline_ema"] = entry["baseline_ema"]

    def save_to_registry(self, registry: dict):
        """Met à jour le registre avec l'état actuel du manager."""
        for cid, client in self._clients.items():
            if cid not in registry:
                registry[cid] = {}
            registry[cid]["away_active"] = client["away_active"]
            registry[cid]["sensor_fault_active"] = client["sensor_fault_active"]
            if client["baseline_ema"] is not None:
                registry[cid]["baseline_ema"] = client["baseline_ema"]