import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime

from src.config import WINDOW_SIZE, HISTORY_DAYS, SIM_DAYS, CONFIRM_THRESHOLD
from src.models.isolation import extract_features, shap_explain, combined_level, risk_pct
from src.models.lstm_utils import batch_lstm_score
from src.detection.false_positive import FalsePositiveManager
from src.registry.manager import add_detection, save_registry

def render(df, meta, registry, iso, scaler, feature_names, shap_explainer,
          lstm_model, lstm_threshold, threshold_if, pres_mode=False):

    st.markdown('<div class="section-header">◈ SURVEILLANCE — SEMAINE GLISSANTE</div>',
                unsafe_allow_html=True)

    # ── Filtres région / profil (à adapter selon ton besoin) ──────────────
    regions = sorted(df["region"].unique())
    sel_region = st.selectbox("Région", ["Toutes"] + regions, key="surv_region")
    if sel_region != "Toutes":
        clients = df[df["region"] == sel_region]["client_id"].unique()
    else:
        clients = df["client_id"].unique()

    # Profils optionnels
    profiles = sorted(meta["profile"].unique()) if not meta.empty else []
    sel_profiles = st.multiselect("Profil", profiles, default=profiles, key="surv_prof")
    if sel_profiles:
        valid = meta[meta["profile"].isin(sel_profiles)]["client_id"].tolist()
        clients = [c for c in clients if c in valid]

    if len(clients) == 0:
        st.warning("Aucun client correspondant aux filtres.")
        return registry

    # ── Layout des contrôles ──────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns([1,1,1,3])
    with col1:
        start = st.button("▶ DÉMARRER", type="primary", use_container_width=True)
    with col2:
        stop = st.button("■ ARRÊTER", use_container_width=True)
    with col3:
        reset = st.button("↺ RESET", use_container_width=True)
    with col4:
        st.markdown(
            f'<div style="font-family:Share Tech Mono;font-size:12px;color:#00E5FF;padding-top:8px">'
            f'Simulation : {SIM_DAYS} jours · Fenêtre {WINDOW_SIZE}j</div>',
            unsafe_allow_html=True
        )

    # Initialisation session state pour la surveillance
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

    # Gestion des boutons
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
        # Réinitialiser les fenêtres si on redémarre
        st.session_state.client_windows = {}
        st.session_state.client_scores = {}
        st.session_state.client_lstm_mse = {}
        st.rerun()

    # ── Si surveillance active, exécuter la boucle jour par jour ──────────
    if st.session_state.surv_active and len(clients) > 0:
        # Préparer les données clients
        client_data = {
            cid: df[df["client_id"] == cid].sort_values("timestamp")["consumption_kwh"].values
            for cid in clients
        }
        max_days = max(len(v) for v in client_data.values())
        # Historique silencieux (HISTORY_DAYS jours)
        if st.session_state.surv_step == 0:
            for cid in clients:
                vals = client_data[cid][:HISTORY_DAYS]
                st.session_state.client_windows[cid] = list(vals)
            st.session_state.surv_step = HISTORY_DAYS

        # Boucle sur les jours de surveillance (SIM_DAYS)
        progress_bar = st.progress(0)
        status_placeholder = st.empty()
        ticker_placeholder = st.empty()

        for offset in range(SIM_DAYS):
            if not st.session_state.surv_active:
                break
            current_day = st.session_state.surv_step + offset
            if current_day >= max_days:
                st.warning("Données insuffisantes pour certains clients.")
                break

            date_label = df[df["client_id"] == clients[0]]["timestamp"].iloc[current_day].strftime("%d/%m/%Y")

            # ── Batch : collecter les fenêtres ────────────────────────────
            ready_clients = []
            windows_batch = []
            feats_batch = []

            for cid in clients:
                vals = client_data[cid]
                if current_day >= len(vals):
                    continue
                # Ajouter la nouvelle valeur
                st.session_state.client_windows[cid].append(vals[current_day])
                # Garder une fenêtre de taille WINDOW_SIZE
                if len(st.session_state.client_windows[cid]) > WINDOW_SIZE:
                    st.session_state.client_windows[cid].pop(0)
                win = np.array(st.session_state.client_windows[cid])
                if len(win) < WINDOW_SIZE:
                    continue
                # Vérifier qualité données & faux positifs
                fp_mgr = st.session_state.fp_manager
                if fp_mgr.data_quality(win) < 0.7:
                    continue
                if fp_mgr.update_sensor_fault(cid, win):
                    # Capteur défaillant : on ignore cette fenêtre
                    continue
                ready_clients.append(cid)
                windows_batch.append(win)
                feats_batch.append(extract_features(win))

            if not ready_clients:
                # Aucune fenêtre valide, on passe au jour suivant
                progress_bar.progress((offset+1)/SIM_DAYS)
                time.sleep(0.05)
                continue

            # ── IF scores batch ──────────────────────────────────────────
            X_feat = np.array([f for f in feats_batch if f is not None])
            if len(X_feat) == 0:
                continue
            X_scaled = scaler.transform(X_feat)
            if_scores = iso.decision_function(X_scaled)

            # ── LSTM batch ───────────────────────────────────────────────
            if lstm_model is not None and lstm_threshold is not None:
                lstm_mses, lstm_anoms = batch_lstm_score(lstm_model, lstm_threshold, np.array(windows_batch))
            else:
                lstm_mses = np.zeros(len(ready_clients))
                lstm_anoms = np.zeros(len(ready_clients), dtype=bool)

            # ── Traitement par client ────────────────────────────────────
            new_alerts = []
            for idx, cid in enumerate(ready_clients):
                if_score = float(if_scores[idx])
                lstm_mse = float(lstm_mses[idx])
                lstm_anom = bool(lstm_anoms[idx])

                st.session_state.client_scores[cid] = if_score
                st.session_state.client_lstm_mse[cid] = lstm_mse

                lvl, lbl, detected_by = combined_level(if_score, threshold_if, lstm_anom, False)

                # Mise à jour de la baseline (pour CUSUM, etc.)
                baseline = st.session_state.fp_manager.update_baseline(cid, windows_batch[idx][-1])

                # Vérifier absence (vacances) – exonération
                away = st.session_state.fp_manager.update_away(cid, st.session_state.client_windows[cid], current_day)
                if away or st.session_state.fp_manager.is_suppressed(cid):
                    # Si en away ou capteur défaillant, on ignore l'alerte
                    continue

                if lvl in ("DOUBLE_ALERT", "ALERT_IF", "SUSPECT_LSTM"):
                    # Récupérer métadonnées
                    client_meta = meta[meta["client_id"] == cid]
                    if not client_meta.empty:
                        region = client_meta.iloc[0]["region"]
                        profile = client_meta.iloc[0]["profile"]
                    else:
                        region = profile = "?"

                    # SHAP
                    
                    shap_reasons = shap_explain(shap_explainer, X_scaled[[idx]], feature_names, top_n=5)

                    # Calcul risque %
                    risk = risk_pct(if_score, threshold_if, lstm_mse, lstm_threshold)

                    narrative = f"[{detected_by}] IF={if_score:.4f}"
                    if lstm_mse > 0:
                        narrative += f" | LSTM MSE={lstm_mse:.4f}"

                    # Ajout au registre
                    registry = add_detection(
                            registry=registry,
                            client_id=cid,
                            level=detected_by,       # "IF", "LSTM", "IF+LSTM"
                            risk_pct=risk,
                            profile=profile,
                            region=region,
                            timestamp=date_label
)
                    # Puis enrichissez l'entrée avec les détails
                    entry = registry.get(cid)
                    if entry:
                        entry["last_if_score"] = if_score
                        entry["last_lstm_mse"] = lstm_mse
                        entry["last_shap"] = shap_reasons
                        save_registry(registry)   # persistance immédiate

                    new_alerts.append({
                        "cid": cid, "region": region,
                        "score": if_score, "lstm_mse": lstm_mse,
                        "detected_by": detected_by,
                        "date": date_label,
                        "confirmed": registry[cid].get("is_confirmed", False),
                        "count": registry[cid].get("detection_count", len(registry[cid].get("detections", [])))
                    })

            # ── Affichage du ticker (dernières alertes) ───────────────────
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

            # ── Mise à jour de la barre de progression ───────────────────
            progress_bar.progress((offset+1)/SIM_DAYS)
            status_placeholder.info(f"📍 Jour {offset+1}/{SIM_DAYS} – {date_label} – {len(ready_clients)} clients analysés – {len(new_alerts)} nouvelles alertes")
            time.sleep(0.15)   # simulation vitesse (modifiable)

        # Fin de la simulation
        st.session_state.surv_active = False
        st.success(f"✅ Simulation terminée. {len([v for v in registry.values() if v.get('is_confirmed')])} fraudeurs avérés détectés.")

        # ── BOUTON DE TÉLÉCHARGEMENT DU REGISTRE ──────────────────────────
        import json
        registry_json = json.dumps(registry, indent=2, default=str)
        st.download_button(
            label="📥 Télécharger le registre",
            data=registry_json,
            file_name=f"fraud_registry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )

    # Affichage du résumé des clients pendant la surveillance (optionnel)
    # Tu peux conserver l'affichage des expanders par région si souhaité

    return registry