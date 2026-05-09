"""
SIPT Pro v4 — Tab : Exclusions & Événements
Gestion des périodes d'absence, pannes capteurs, faux positifs.
"""
import streamlit as st
import pandas as pd
from src.ui.components import section_header
from src.registry.manager import (
    start_away_period, end_away_period,
    declare_sensor_fault, resolve_sensor_fault,
    add_false_positive, save_registry,
)
from src.config import COLOR_WARN, COLOR_INFO, COLOR_ALERT, COLOR_SUCCESS


def render(registry: dict, df: pd.DataFrame):
    section_header("EXCLUSIONS & GESTION DES ÉVÉNEMENTS")

    tab_away, tab_sensor, tab_fp = st.tabs([
        "🏖 Périodes absence", "📡 Pannes capteurs", "⚠️ Faux positifs"
    ])

    # ── Away ──────────────────────────────────────────────────────────────────
    with tab_away:
        st.markdown(
            f'<div style="font-family:Share Tech Mono;font-size:11px;color:#8aa8cc;margin-bottom:12px">'
            f'Clients absents : alertes supprimées automatiquement durant la période.</div>',
            unsafe_allow_html=True,
        )

        # Actifs
        away_clients = [(cid, e) for cid, e in registry.items() if e.get("away_active")]
        if away_clients:
            section_header("CLIENTS ACTUELLEMENT ABSENTS")
            for cid, entry in away_clients:
                col1, col2 = st.columns([3, 1])
                with col1:
                    last_away = next(
                        (p for p in reversed(entry.get("away_periods", [])) if not p.get("end")),
                        {}
                    )
                    st.markdown(
                        f'<div style="font-family:Share Tech Mono;font-size:12px;color:{COLOR_WARN}">'
                        f'🏖 {cid} — absent depuis {last_away.get("start","?")} '
                        f'· {entry.get("profile","?")} · {entry.get("region","?")}</div>',
                        unsafe_allow_html=True,
                    )
                with col2:
                    if st.button(f"Clôturer", key=f"end_away_{cid}"):
                        end_away_period(registry, cid)
                        save_registry(registry)
                        st.success(f"Période d'absence clôturée pour {cid}")
                        st.rerun()
        else:
            st.markdown(
                f'<div style="color:{COLOR_SUCCESS};font-family:Share Tech Mono;font-size:12px">'
                f'✓ Aucun client marqué absent.</div>',
                unsafe_allow_html=True,
            )

        # Déclarer manuellement
        st.divider()
        section_header("DÉCLARER UNE ABSENCE MANUELLE")
        all_clients = sorted(df["client_id"].unique().tolist())
        sel_away = st.selectbox("Client", all_clients, key="away_client")
        if st.button("🏖 Déclarer absent", key="btn_away"):
            start_away_period(registry, sel_away)
            save_registry(registry)
            st.success(f"{sel_away} marqué absent. Les alertes sont suspendues.")
            st.rerun()

    # ── Sensor faults ──────────────────────────────────────────────────────────
    with tab_sensor:
        st.markdown(
            f'<div style="font-family:Share Tech Mono;font-size:11px;color:#8aa8cc;margin-bottom:12px">'
            f'Clients avec panne capteur déclarée : scores IF/LSTM non calculés.</div>',
            unsafe_allow_html=True,
        )

        fault_clients = [(cid, e) for cid, e in registry.items() if e.get("sensor_fault_active")]
        if fault_clients:
            section_header("CAPTEURS EN PANNE")
            for cid, entry in fault_clients:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(
                        f'<div style="font-family:Share Tech Mono;font-size:12px;color:{COLOR_ALERT}">'
                        f'📡 {cid} — Capteur bloqué · {entry.get("region","?")}</div>',
                        unsafe_allow_html=True,
                    )
                with col2:
                    if st.button("Résoudre", key=f"resolve_sf_{cid}"):
                        resolve_sensor_fault(registry, cid)
                        save_registry(registry)
                        st.success(f"Panne résolue pour {cid}")
                        st.rerun()
        else:
            st.markdown(
                f'<div style="color:{COLOR_SUCCESS};font-family:Share Tech Mono;font-size:12px">'
                f'✓ Aucune panne capteur active.</div>',
                unsafe_allow_html=True,
            )

        st.divider()
        section_header("DÉCLARER UNE PANNE CAPTEUR")
        all_clients = sorted(df["client_id"].unique().tolist())
        sel_sf = st.selectbox("Client", all_clients, key="sf_client")
        if st.button("📡 Déclarer panne capteur", key="btn_sf"):
            declare_sensor_fault(registry, sel_sf)
            save_registry(registry)
            st.success(f"Panne capteur déclarée pour {sel_sf}.")
            st.rerun()

    # ── Faux positifs ──────────────────────────────────────────────────────────
    with tab_fp:
        section_header("JOURNAL DES FAUX POSITIFS")

        fp_rows = []
        for cid, entry in registry.items():
            for fp in entry.get("false_positives", []):
                fp_rows.append({
                    "Client":    cid,
                    "Date":      fp.get("date", "?"),
                    "Type":      fp.get("detection_type", "?"),
                    "Raison":    fp.get("reason", "?"),
                    "Région":    entry.get("region", "?"),
                })
        if fp_rows:
            st.dataframe(pd.DataFrame(fp_rows), use_container_width=True, hide_index=True)
        else:
            st.markdown(
                f'<div style="color:{COLOR_SUCCESS};font-family:Share Tech Mono;font-size:12px">'
                f'✓ Aucun faux positif enregistré.</div>',
                unsafe_allow_html=True,
            )

        st.divider()
        section_header("ENREGISTRER UN FAUX POSITIF")
        all_detected = sorted([cid for cid, e in registry.items() if e.get("detections")])
        if all_detected:
            sel_fp   = st.selectbox("Client détecté", all_detected, key="fp_client")
            det_type = st.selectbox("Type détection", ["IF", "LSTM", "CUSUM", "DOUBLE", "Bilan"], key="fp_type")
            reason   = st.text_input("Raison (ex : installation temporaire, travaux)", key="fp_reason")
            if st.button("⚠️ Enregistrer faux positif", key="btn_fp"):
                if reason:
                    add_false_positive(registry, sel_fp, det_type, reason)
                    save_registry(registry)
                    st.success(f"Faux positif enregistré pour {sel_fp}.")
                    st.rerun()
                else:
                    st.warning("Veuillez saisir une raison.")
        else:
            st.info("Aucun client détecté dans le registre.")
