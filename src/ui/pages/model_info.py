"""
SIPT Pro v4 — Tab : Info Modèles (pédagogique)
Architecture hybride IF + LSTM + CUSUM + modules complémentaires.
"""
import streamlit as st
from src.config import (
    REGISTRY_FILE, WINDOW_SIZE, CONFIRM_THRESHOLD,
    COLOR_INFO, COLOR_ALERT, COLOR_WARN, COLOR_SUCCESS, COLOR_ACCENT, COLOR_LSTM,
    FEATURE_NAMES,
)
from src.ui.components import section_header


def render(threshold_if: float, lstm_threshold: float | None, lstm_ready: bool):
    section_header("ARCHITECTURE DÉTECTION HYBRIDE SIPT PRO v4")

    lstm_status_html = (
        f'<span style="color:{COLOR_LSTM};font-weight:700">ACTIF</span> — '
        f'Seuil MSE : <code style="color:{COLOR_LSTM}">{lstm_threshold:.4f}</code>'
        if lstm_ready and lstm_threshold
        else f'<span style="color:{COLOR_ALERT}">INACTIF</span> (exécuter train.py avec TensorFlow)'
    )

    st.markdown(
        f'<div style="background:#0d1a35;border:1px solid #1a3a6e;border-radius:10px;'
        f'padding:20px;font-size:13px;color:#c8d8f0;margin-bottom:16px">'

        f'<h4 style="color:{COLOR_INFO};font-family:Share Tech Mono;letter-spacing:2px">① ISOLATION FOREST (IF)</h4>'
        f'<p>Détecte les outliers dans l\'espace des 12 features statistiques agrégées sur la fenêtre {WINDOW_SIZE}j. '
        f'Score ∈ [-0.5, 0.5] : plus négatif = plus anormal. '
        f'Seuil actuel : <b style="color:{COLOR_ALERT};font-family:Share Tech Mono">{threshold_if:.3f}</b></p>'

        f'<h4 style="color:{COLOR_LSTM};font-family:Share Tech Mono;letter-spacing:2px;margin-top:16px">'
        f'② LSTM AUTOENCODER — {lstm_status_html}</h4>'
        f'<p>Entraîné uniquement sur des clients normaux. Reconstruction d\'une fenêtre '
        f'de {WINDOW_SIZE} jours (z-score par client). '
        f'Pattern frauduleux → <b>MSE élevée</b> = anomalie temporelle.</p>'
        f'<pre style="background:#080d1a;border:1px solid #1a3a6e;padding:10px;'
        f'font-size:11px;color:{COLOR_ACCENT};border-radius:6px">'
        f'Input (30,1) → LSTM(64) → LSTM(32) → [latent 32D]\n'
        f'                                  |\n'
        f'Output(30,1) ← Dense(1) ← LSTM(64) ← LSTM(32) ← RepeatVector</pre>'

        f'<h4 style="color:#8b5cf6;font-family:Share Tech Mono;letter-spacing:2px;margin-top:16px">③ CUSUM (micro-dérives)</h4>'
        f'<p>Algorithme CUSUM unilatéral : détecte une baisse négative persistante '
        f'de la consommation (bypass lent). Paramètres : k = 0.5σ, h = 5σ. '
        f'Déclenche un niveau "warning" seul, amplifie le niveau si combiné à IF/LSTM.</p>'

        f'<h4 style="color:{COLOR_WARN};font-family:Share Tech Mono;letter-spacing:2px;margin-top:16px">④ BILAN ÉNERGÉTIQUE (piquage direct)</h4>'
        f'<p>Compare l\'injection régionale aux consommations comptabilisées + pertes (5%). '
        f'Prob. piquage = écart / (0.2 × injection). Seuil alerte : 70%.</p>'

        f'<h4 style="color:{COLOR_SUCCESS};font-family:Share Tech Mono;letter-spacing:2px;margin-top:16px">⑤ TOPOLOGIE RÉSEAU (branchement avant compteur)</h4>'
        f'<p>Bilan par feeder (segment réseau). Alerte persistante si écart > 12% sur ≥ 3 jours.</p>'

        f'<h4 style="color:{COLOR_INFO};font-family:Share Tech Mono;letter-spacing:2px;margin-top:16px">⑥ GESTION FAUX POSITIFS</h4>'
        f'<p>Away auto ({"< 0.5 kWh"} × 7j), capteur bloqué (variance {"< 0.02"}), '
        f'qualité données ({"< 70%"} valides → skip), baseline EMA (α=0.05).</p>'

        f'<h4 style="color:{COLOR_SUCCESS};font-family:Share Tech Mono;letter-spacing:2px;margin-top:16px">SCORE HYBRIDE</h4>'
        f'<table style="width:100%;font-size:12px;border-collapse:collapse">'
        f'<tr style="background:#1a3a6e;color:{COLOR_INFO};font-family:Share Tech Mono">'
        f'<th style="padding:7px">Combinaison</th><th style="padding:7px">Niveau</th><th style="padding:7px">Pondération</th></tr>'
        f'<tr style="background:#0a1628"><td style="padding:6px">IF + LSTM ou IF + CUSUM</td>'
        f'<td style="padding:6px;color:{COLOR_ALERT}">DOUBLE ALERTE</td><td style="padding:6px">IF 50% + LSTM 35% + CUSUM 15%</td></tr>'
        f'<tr style="background:#0d1a35"><td style="padding:6px">IF seul</td>'
        f'<td style="padding:6px;color:#ff6b6b">ALERTE IF</td><td style="padding:6px">Basé IF</td></tr>'
        f'<tr style="background:#0a1628"><td style="padding:6px">LSTM seul</td>'
        f'<td style="padding:6px;color:{COLOR_WARN}">SUSPECT LSTM</td><td style="padding:6px">Basé LSTM</td></tr>'
        f'<tr style="background:#0d1a35"><td style="padding:6px">CUSUM seul</td>'
        f'<td style="padding:6px;color:#a78bfa">MICRO-DÉRIVE</td><td style="padding:6px">CUSUM score</td></tr>'
        f'</table>'

        f'<h4 style="color:{COLOR_ACCENT};font-family:Share Tech Mono;letter-spacing:2px;margin-top:16px">REGISTRE PERSISTANT</h4>'
        f'<p>Fichier <code style="color:{COLOR_ACCENT}">{REGISTRY_FILE}</code>. '
        f'Client confirmé à {CONFIRM_THRESHOLD} détections. Survit aux redémarrages.</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Table features
    section_header("FEATURES ISOLATION FOREST")
    feat_info = [
        ("Conso moyenne",            "mean(fenêtre)",            "Très faible vs profil"),
        ("Volatilité (std)",          "std(fenêtre)",             "Anormalement élevée ou nulle"),
        ("Tendance linéaire",         "polyfit deg.1",            "Chute brutale persistante"),
        ("Variation journalière moy", "mean(diff)",               "Sauts brusques répétés"),
        ("Ratio 1ère/2ème période",   "mean[:15] / mean[15:]",    "Chute soudaine en 2ème moitié"),
        ("Amplitude max-min",         "max - min",                "Très faible = plateau anormal"),
        ("Coefficient de variation",  "std / mean",               "CV élevé = erratique"),
        ("Jours très faible conso",   "count(< 0.5 kWh)",         "Beaucoup de jours quasi-nuls"),
        ("Ratio Weekend/Weekday",     "mean_wk / mean_wd",        "Pas de variation typique"),
        ("Volatilité Wk/Wd",          "std_wk / std_wd",          "Asymétrie anormale"),
        ("Autocorrélation lag-1",     "corr(s[:-1], s[1:])",      "Faible = comportement aléatoire"),
        ("Instabilité directionnelle","count(changements dir.)",   "Élevé = oscillations suspectes"),
    ]
    table = '<table style="width:100%;border-collapse:collapse;font-size:11px"><thead>'
    table += f'<tr style="background:#1a3a6e;color:{COLOR_INFO};font-family:Share Tech Mono">'
    table += '<th style="padding:7px;text-align:left">Feature</th>'
    table += '<th style="padding:7px;text-align:left">Calcul</th>'
    table += '<th style="padding:7px;text-align:left">Signal fraude si…</th></tr></thead><tbody>'
    for i, (name, calc, sig) in enumerate(feat_info):
        bg = "#0a1628" if i % 2 == 0 else "#0d1a35"
        table += (
            f'<tr style="background:{bg};color:#c8d8f0">'
            f'<td style="padding:5px 8px;color:{COLOR_ACCENT};font-family:Share Tech Mono">{name}</td>'
            f'<td style="padding:5px 8px;color:#8aa8cc;font-family:monospace">{calc}</td>'
            f'<td style="padding:5px 8px;color:{COLOR_WARN}">{sig}</td></tr>'
        )
    table += "</tbody></table>"
    st.markdown(table, unsafe_allow_html=True)
