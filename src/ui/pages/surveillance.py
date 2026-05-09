"""
SIPT Pro v4 — Onglet Surveillance
Simulation semaine glissante avec Isolation Forest + LSTM.
Affichage en direct des alertes et des métriques.
"""
import streamlit as st
import pandas as pd
import numpy as np
import time
import json
from datetime import datetime

from src.config import HISTORY_DAYS, SIM_DAYS, CONFIRM_THRESHOLD
from src.models.isolation import extract_features, shap_explain, combined_level, risk_pct
from src.models.lstm_utils import batch_lstm_score
from src.detection.false_positive import FalsePositiveManager
from src.registry.manager import add_detection, save_registry


def render(df, meta, registry, iso, scaler, feature_names, shap_explainer,
          lstm_model, lstm_threshold, threshold_if, sim_speed, win_size,
          monitored_clients, pres_mode=False):
    """
    Surveillance en semaine glissante.
    Les paramètres dynamiques sont passés depuis la sidebar.
    """
    st.markdown('<div class="section-header">◈ SURVEILLANCE — SEMAINE GLISSANTE</div>',
                unsafe_allow_html=True)

    # Affichage des paramètres actuels (dépliable)
    with st.expander("🔧 Paramètres simulation", expanded=False):
        st.write(f"**Clients surveillés** : {len(monitored_clients)}")
        st.write(f"**Seuil IF** : {threshold_if:.3f}")
        st.write(f"**Vitesse** : {sim_speed} s/jour")
        st.write(f"**Fenêtre glissante** : {win_size} jours")
        st.write(f"**Historique pré-chargé** : {HISTORY_DAYS} jours")
        st.write(f"**Jours simulés** : {SIM_DAYS} jours")
        st.write(f"**Confirmation après** : {CONFIRM_THRESHOLD} détections")

    if not monitored_clients:
        st.warning("Aucun client sélectionné. Vérifiez les filtres dans la sidebar.")
        return registry

    # ── Contrôles ──────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        start = st.button("▶ DÉMARRER", type="primary", use_container_width=True)
    with col2:
        stop = st.button("■ ARRÊTER", use_container_width=True)
    with col3:
        reset = st.button("↺ RESET", use_container_width=True)

    # ── État de session ────────────────────────────────────────────────────
    if "surv_active" not in st.session_state:
        st.session_state.surv_active = False
    if "surv_step" not in st.session_state:
        st.session_state.surv_step = 0
    if "client_windows" not in st.session_state:
        st.session_state.client_windows = {}
    if "client_scores" not in st.session_state:
        st.session_state.client_scores = {}
    if "client_lstm_mse" not in st.session_state:
        st.session_state.client_lstm_mse = {}
    if "fp_manager" not in st.session_state:
        st.session_state.fp_manager = FalsePositiveManager()

    # ── Gestion boutons ────────────────────────────────────────────────────
    if reset:
        st.session_state.surv_active = False
        st.session_state.surv_step = 0
        st.session_state.client_windows = {}
        st.session_state.client_scores = {}
        st.session_state.client_lstm_mse = {}
        st.rerun()
    if stop:
        st.session_state.surv_active = False
        st.rerun()
    if start:
        st.session_state.surv_active = True
        st.session_state.surv_step = 0
        st.session_state.client_windows = {}
        st.session_state.client_scores = {}
        st.session_state.client_lstm_mse = {}
        st.rerun()

    if not st.session_state.surv_active:
        st.info("Cliquez sur **DÉMARRER** pour lancer la surveillance sur la semaine glissante.")
        return registry

    # ── Préparation des données pour les clients filtrés ───────────────────
    client_data = {}
    for cid in monitored_clients:
        cdf = df[df["client_id"] == cid].sort_values("timestamp")
        if len(cdf) == 0:
            continue
        client_data[cid] = cdf["consumption_kwh"].values

    if not client_data:
        st.error("Aucune donnée de consommation trouvée pour les clients sélectionnés.")
        st.session_state.surv_active = False
        return registry

    max_days = max(len(v) for v in client_data.values())
    if max_days < HISTORY_DAYS + SIM_DAYS:
        st.warning(f"Pas assez de données : besoin de {HISTORY_DAYS + SIM_DAYS} jours, disponible = {max_days}")
        st.session_state.surv_active = False
        return registry

    # ── Chargement de l'historique silencieux ──────────────────────────────
    if st.session_state.surv_step == 0:
        for cid, vals in client_data.items():
            st.session_state.client_windows[cid] = list(vals[:HISTORY_DAYS])
        st.session_state.surv_step = HISTORY_DAYS

    # ── Placeholders pour l'interface ──────────────────────────────────────
    progress_bar = st.progress(0)
    metrics_placeholder = st.empty()       # pour les 3 métriques en direct
    ticker_placeholder = st.empty()        # pour les alertes
    day_info = st.empty()                  # pour le texte du jour

    # ── Boucle de simulation ───────────────────────────────────────────────
    for offset in range(SIM_DAYS):
        if not st.session_state.surv_active:
            break

        current_day = st.session_state.surv_step + offset
        if current_day >= max_days:
            st.warning("Fin des données atteinte plus tôt que prévu.")
            break

        # Date pour affichage
        first_cid = list(client_data.keys())[0]
        date_label = df[df["client_id"] == first_cid]["timestamp"].iloc[current_day].strftime("%d/%m/%Y")
        day_info.info(f"📍 **Jour {offset+1}/{SIM_DAYS}** – {date_label}")

        # ── Collecte des fenêtres ─────────────────────────────────────────
        ready_clients = []
        windows_batch = []
        feats_batch = []

        for cid, vals in client_data.items():
            if current_day >= len(vals):
                continue
            # Ajouter la nouvelle valeur
            st.session_state.client_windows[cid].append(vals[current_day])
            # Garder les win_size derniers jours
            if len(st.session_state.client_windows[cid]) > win_size:
                st.session_state.client_windows[cid].pop(0)
            win = np.array(st.session_state.client_windows[cid])
            if len(win) < win_size:
                continue
            # Vérifications qualité et faux positifs
            fp_mgr = st.session_state.fp_manager
            if fp_mgr.data_quality(win) < 0.7:
                continue
            if fp_mgr.update_sensor_fault(cid, win):
                continue
            ready_clients.append(cid)
            windows_batch.append(win)
            feats_batch.append(extract_features(win))

        if not ready_clients:
            progress_bar.progress((offset+1)/SIM_DAYS)
            time.sleep(sim_speed)
            continue

        # ── Scores IF ─────────────────────────────────────────────────────
        X_feat = np.array(feats_batch)
        X_scaled = scaler.transform(X_feat)
        if_scores = iso.decision_function(X_scaled)

        # ── Scores LSTM (batch) ───────────────────────────────────────────
        if lstm_model is not None and lstm_threshold is not None:
            lstm_mses, lstm_anoms = batch_lstm_score(lstm_model, lstm_threshold, np.array(windows_batch))
        else:
            lstm_mses = np.zeros(len(ready_clients))
            lstm_anoms = np.zeros(len(ready_clients), dtype=bool)

        # ── Traitement des alertes ────────────────────────────────────────
        new_alerts = []
        for idx, cid in enumerate(ready_clients):
            if_score = float(if_scores[idx])
            lstm_mse = float(lstm_mses[idx])
            lstm_anom = bool(lstm_anoms[idx])

            st.session_state.client_scores[cid] = if_score
            st.session_state.client_lstm_mse[cid] = lstm_mse

            lvl, lbl, detected_by = combined_level(if_score, threshold_if, lstm_anom)

            # Mise à jour away / capteur
            fp_mgr = st.session_state.fp_manager
            fp_mgr.update_baseline(cid, windows_batch[idx][-1])
            away = fp_mgr.update_away(cid, st.session_state.client_windows[cid], current_day)
            if away or fp_mgr.is_suppressed(cid):
                continue

            if lvl in ("alert", "alert_high"):
                # Métadonnées
                client_meta = meta[meta["client_id"] == cid]
                if not client_meta.empty:
                    region = client_meta.iloc[0]["region"]
                    profile = client_meta.iloc[0]["profile"]
                else:
                    region = profile = "?"

                # SHAP
                shap_reasons = []
                if shap_explainer is not None:
                    shap_reasons = shap_explain(shap_explainer, feature_names, X_scaled[[idx]], top_n=5)

                # Risque (formule hybride)
                risk = risk_pct(if_score, threshold_if, lstm_mse, lstm_threshold)

                narrative = f"[{detected_by}] IF={if_score:.4f}"
                if lstm_mse > 0:
                    narrative += f" | LSTM MSE={lstm_mse:.4f}"

                # Enregistrement dans le registre
                registry = add_detection(
                    registry, cid, detected_by, risk, profile, region, date_label
                )
                entry = registry.get(cid)
                if entry:
                    entry["last_if_score"] = if_score
                    entry["last_lstm_mse"] = lstm_mse
                    entry["last_shap"] = shap_reasons
                    entry["last_narrative"] = narrative
                save_registry(registry)

                new_alerts.append({
                    "cid": cid, "region": region,
                    "score": if_score, "lstm_mse": lstm_mse,
                    "detected_by": detected_by,
                    "date": date_label,
                    "confirmed": registry[cid].get("is_confirmed", False),
                    "count": len(registry[cid].get("detections", []))
                })

        # ── Affichage des métriques en direct ──────────────────────────────
        total_detections = sum(len(e.get("detections", [])) for e in registry.values())
        confirmed_now = sum(1 for e in registry.values() if e.get("is_confirmed"))
        with metrics_placeholder.container():
            m1, m2, m3 = st.columns(3)
            m1.metric("📊 Alertes aujourd'hui", len(new_alerts))
            m2.metric("🔔 Détections totales", total_detections)
            m3.metric("✅ Fraudeurs avérés", confirmed_now)

        # ── Ticker d'alertes ───────────────────────────────────────────────
        if new_alerts:
            ticker_html = ""
            for a in new_alerts[-5:]:
                badge = "ticker-badge"
                if a["detected_by"] == "BOTH":
                    badge = "ticker-badge-both"
                elif a["detected_by"] == "LSTM":
                    badge = "ticker-badge-lstm"
                confirmed_tag = ' <span style="color:#FF4C4C">[AVÉRÉ]</span>' if a["confirmed"] else ""
                ticker_html += f'''
                <div class="alert-ticker">
                    <span class="{badge}">{a["detected_by"]}</span>
                    <span>{a["cid"]} | {a["region"]}{confirmed_tag}</span>
                    <span class="ticker-meta">IF={a["score"]:.4f} | MSE={a["lstm_mse"]:.4f} | {a["date"]} | #{a["count"]}</span>
                </div>
                '''
            ticker_placeholder.markdown(ticker_html, unsafe_allow_html=True)

        # ── Progression ────────────────────────────────────────────────────
        progress_bar.progress((offset+1)/SIM_DAYS)
        time.sleep(sim_speed)

    # ── Fin de simulation ──────────────────────────────────────────────────
    st.session_state.surv_active = False
    st.success(f"✅ Simulation terminée – {confirmed_now} fraudeur(s) avéré(s) détecté(s).")

    # Bouton pour exporter le registre
    registry_json = json.dumps(registry, indent=2, default=str)
    st.download_button(
        label="📥 Télécharger le registre (JSON)",
        data=registry_json,
        file_name=f"fraud_registry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True
    )

    # Force le rechargement complet de l'application pour mettre à jour les KPI globaux
    st.rerun()
    return registry