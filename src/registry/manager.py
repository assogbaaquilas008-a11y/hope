"""
SIPT Pro v4 — Gestionnaire du registre de fraude
Persistance JSON, ajout de détections, gestion des événements.
"""
import os
import json
from datetime import date, datetime
from src.config import REGISTRY_FILE, CONFIRM_THRESHOLD


# ── Helpers date ───────────────────────────────────────────────────────────────
def _today() -> str:
    return date.today().isoformat()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ── Chargement / Sauvegarde ────────────────────────────────────────────────────
def load_registry() -> dict:
    """Charge le registre depuis le disque. Retourne {} si absent."""
    if os.path.exists(REGISTRY_FILE):
        try:
            with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_registry(registry: dict) -> None:
    """Sauvegarde le registre sur disque (atomique via fichier temp)."""
    tmp = REGISTRY_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)
    os.replace(tmp, REGISTRY_FILE)


# ── Ajout d'une détection ──────────────────────────────────────────────────────
def add_detection(registry: dict, client_id: str, level: str,
                  risk_pct: float, profile: str, region: str,
                  timestamp: str) -> dict:
    """
    Enregistre une détection pour un client.
    Si le client atteint CONFIRM_THRESHOLD détections → is_confirmed = True.
    """
    if client_id not in registry:
        registry[client_id] = {
            "client_id":      client_id,
            "profile":        profile,
            "region":         region,
            "detections":     [],
            "is_confirmed":   False,
            "first_seen":     timestamp,
            "last_seen":      timestamp,
            "away_periods":   [],
            "sensor_faults":  [],
            "false_positives":[],
            "away_active":    False,
            "sensor_fault_active": False,
            "ema_baseline":   None,
        }
    entry = registry[client_id]
    entry["detections"].append({
        "timestamp": timestamp,
        "level":     level,
        "risk_pct":  round(risk_pct * 100, 1),
    })
    entry["last_seen"] = timestamp
    if len(entry["detections"]) >= CONFIRM_THRESHOLD:
        entry["is_confirmed"] = True
    return registry


# ── Événements Away ────────────────────────────────────────────────────────────
def start_away_period(registry: dict, client_id: str,
                      start: str = None) -> dict:
    """Marque un client comme absent (manuellement)."""
    start = start or _today()
    entry = registry.setdefault(client_id, _empty_entry(client_id))
    entry["away_active"] = True
    entry["away_periods"].append({"start": start, "end": None})
    return registry


def end_away_period(registry: dict, client_id: str, end: str = None) -> dict:
    """Clôture la période d'absence active."""
    end = end or _today()
    entry = registry.get(client_id)
    if entry is None:
        return registry
    entry["away_active"] = False
    for p in reversed(entry.get("away_periods", [])):
        if p.get("end") is None:
            p["end"] = end
            break
    return registry


# ── Événements Capteur ────────────────────────────────────────────────────────
def declare_sensor_fault(registry: dict, client_id: str) -> dict:
    """Déclare une panne de capteur pour un client."""
    entry = registry.setdefault(client_id, _empty_entry(client_id))
    entry["sensor_fault_active"] = True
    entry.setdefault("sensor_faults", []).append({"start": _today(), "end": None})
    return registry


def resolve_sensor_fault(registry: dict, client_id: str) -> dict:
    """Clôture la panne capteur active."""
    entry = registry.get(client_id)
    if entry is None:
        return registry
    entry["sensor_fault_active"] = False
    for f in reversed(entry.get("sensor_faults", [])):
        if f.get("end") is None:
            f["end"] = _today()
            break
    return registry


# ── Faux positifs ─────────────────────────────────────────────────────────────
def add_false_positive(registry: dict, client_id: str,
                       detection_type: str, reason: str) -> dict:
    """Enregistre un faux positif confirmé par l'opérateur."""
    entry = registry.setdefault(client_id, _empty_entry(client_id))
    entry.setdefault("false_positives", []).append({
        "date":           _today(),
        "detection_type": detection_type,
        "reason":         reason,
    })
    return registry


# ── Helper interne ────────────────────────────────────────────────────────────
def _empty_entry(client_id: str) -> dict:
    return {
        "client_id":           client_id,
        "profile":             "?",
        "region":              "?",
        "detections":          [],
        "is_confirmed":        False,
        "first_seen":          _now(),
        "last_seen":           _now(),
        "away_periods":        [],
        "sensor_faults":       [],
        "false_positives":     [],
        "away_active":         False,
        "sensor_fault_active": False,
        "ema_baseline":        None,
    }
