"""
SIPT Pro v4 — Tab : Fraudeurs avérés
Rétrocompatible avec le registre v3 (detected_by / date / shap 3-tuples)
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.ui.components import section_header
from src.models.isolation import extract_features, shap_explain
from src.config import WINDOW_SIZE


def _shap_html(reasons: list) -> str:
    if not reasons:
        return '<p style="color:#4a7aaa;font-size:11px">SHAP non disponible</p>'
    html    = ""
    # Compatible 2-tuples (v4) ET 3-tuples (v3 : name, val, abs_v)
    max_abs = max(abs(item[1]) for item in reasons) or 1
    for item in reasons:
        name, val = item[0], item[1]
        abs_v     = abs(val)
        pct       = round(abs_v / max_abs * 100)
        cls_val   = "shap-val-neg" if val < 0 else "shap-val-pos"
        cls_fill  = "shap-fill-neg" if val < 0 else "shap-fill-pos"
        direction = "▼ Anomalement bas" if val < 0 else "▲ Anomalement élevé"
        html += (
            f'<div class="shap-row">'
            f'<span class="shap-feat">{name}</span>'
            f'<div class="shap-bar"><div class="{cls_fill}" style="width:{pct}%"></div></div>'
            f'<span class="{cls_val}">{val:+.3f}</span>'
            f'</div>'
            f'<div style="font-size:9px;color:#6a6a8a;padding:1px 0 3px 0">{direction}</div>'
        )
    return html


def _badge_color(d: dict) -> str:
    """Compatible level (v4) et detected_by (v3)."""
    lvl = d.get("level") or d.get("detected_by", "?")
    if lvl in ("DOUBLE", "BOTH"):   return "#f97316"
    if lvl == "LSTM":               return "#ffaa00"
    if lvl == "CUSUM":              return "#8b5cf6"
    return "#ff3b3b"


def _det_timestamp(d: dict) -> str:
    return str(d.get("timestamp") or d.get("date") or "")[:10]


def render(registry: dict, df: pd.DataFrame, scaler, shap_explainer, feature_names):
    registry = st.session_state.get("registry", registry)

    section_header("FRAUDEURS AVÉRÉS — REGISTRE CONFIRMÉ")

    # ── Collecter uniquement les AVÉRÉS ───────────────────────────────────────
    confirmed = {}
    for cid, entry in registry.items():
        if not entry.get("is_confirmed", False):
            continue
        detections = entry.get("detections", [])
        if not detections:
            continue

        # Compteurs : valeur stockée OU recalcul fallback v3/v4
        entry["detection_count"] = entry.get("detection_count") or len(detections)
        entry["max_risk_pct"]    = entry.get("max_risk_pct") or max(
            (d.get("risk_pct", 0) for d in detections), default=0
        )
        entry["both_count"] = entry.get("both_count") or sum(
            1 for d in detections
            if d.get("level") == "DOUBLE" or d.get("detected_by") == "BOTH"
        )
        entry["lstm_count"] = entry.get("lstm_count") or sum(
            1 for d in detections
            if d.get("level") in ("LSTM", "DOUBLE")
            or d.get("detected_by") in ("LSTM", "BOTH")
        )
        confirmed[cid] = entry

    if not confirmed:
        st.markdown(
            '<div style="color:#00FFA5;font-family:Share Tech Mono;font-size:13px;padding:20px">'
            '✓ Aucun fraudeur avéré dans le registre actuel.</div>',
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f'<div style="color:#FF4C4C;font-family:Share Tech Mono;font-size:13px;margin-bottom:12px">'
        f'● {len(confirmed)} FRAUDEUR(S) AVÉRÉ(S) DANS LE REGISTRE</div>',
        unsafe_allow_html=True,
    )

    # ── Filtres ────────────────────────────────────────────────────────────────
    all_regions  = sorted({v.get("region",  "?") for v in confirmed.values()})
    all_profiles = sorted({v.get("profile", "?") for v in confirmed.values()})
    f1, f2, f3 = st.columns(3)
    with f1:
        f_regions  = st.multiselect("Région",  all_regions,  key="fa_reg",  placeholder="Toutes")
    with f2:
        f_profiles = st.multiselect("Profil",  all_profiles, key="fa_prof", placeholder="Tous")
    with f3:
        sort_by = st.selectbox(
            "Trier par",
            ["Nb détections ↓", "Risque max ↓", "Dernière détection", "Client ID"],
            key="fa_sort",
        )

    items = list(confirmed.items())
    if f_regions:
        items = [(cid, e) for cid, e in items if e.get("region") in f_regions]
    if f_profiles:
        items = [(cid, e) for cid, e in items if e.get("profile") in f_profiles]
    if sort_by == "Nb détections ↓":
        items = sorted(items, key=lambda x: -x[1]["detection_count"])
    elif sort_by == "Risque max ↓":
        items = sorted(items, key=lambda x: -x[1]["max_risk_pct"])
    elif sort_by == "Dernière détection":
        items = sorted(items, key=lambda x: x[1].get("last_seen", ""), reverse=True)
    elif sort_by == "Client ID":
        items = sorted(items, key=lambda x: x[0])

    st.markdown(
        f'<div style="font-family:Share Tech Mono,monospace;font-size:11px;'
        f'color:#4a7aaa;margin-bottom:12px">{len(items)} fraudeurs affichés</div>',
        unsafe_allow_html=True,
    )

    # ── Cards ──────────────────────────────────────────────────────────────────
    for cid, entry in items:
        detections = entry.get("detections", [])
        n_det      = entry["detection_count"]
        n_both     = entry["both_count"]
        n_lstm     = entry["lstm_count"]
        max_risk   = entry["max_risk_pct"]
        first_seen = str(entry.get("first_seen", "?"))[:10]
        last_seen  = str(entry.get("last_seen",  "?"))[:10]
        region     = entry.get("region",  "?")
        profile    = entry.get("profile", "?")
        min_score  = entry.get("min_score_if")
        max_lstm   = entry.get("max_lstm_mse", 0)

        risk_color = "#ff3b3b" if max_risk > 70 else ("#ffaa00" if max_risk > 40 else "#00ff9d")

        # Timeline des 5 dernières détections
        timeline = ""
        for d in detections[-5:]:
            col = _badge_color(d)
            ts  = _det_timestamp(d)
            lvl = d.get("level") or d.get("detected_by", "?")
            timeline += (
                f'<span style="font-size:9px;background:#1a0820;'
                f'border:1px solid {col};color:{col};'
                f'border-radius:3px;padding:1px 5px;margin-right:4px">'
                f'{ts} [{lvl}]</span>'
            )

        badge_lstm = (
            f'<span class="detection-badge detection-badge-lstm">{n_lstm} LSTM</span>'
            if n_lstm > 0 else ""
        )
        badge_both = (
            f'<span class="detection-badge" style="background:#2a0840;'
            f'border-color:#9a2060;color:#f97316">{n_both} doubles</span>'
            if n_both > 0 else ""
        )

        lstm_section = ""
        if max_lstm and max_lstm > 0:
            lstm_section = (
                f'<div style="font-size:10px;color:#f97316;margin-top:4px">'
                f'🟠 MSE LSTM max : <span style="font-family:Share Tech Mono,monospace">'
                f'{max_lstm:.4f}</span>'
                f'&nbsp;|&nbsp; Détections doubles : <b>{n_both}</b></div>'
            )

        score_block = ""
        if min_score is not None:
            score_block = (
                f'<div style="font-family:Share Tech Mono,monospace;font-size:11px;'
                f'color:#8aa8cc;margin-top:4px">IF min : {min_score:.4f}</div>'
            )

        # SHAP : stocké dans le registre OU recalculé à la volée
        shap_pairs = entry.get("shap_reasons", [])
        if not shap_pairs and shap_explainer is not None and scaler is not None:
            cdf  = df[df["client_id"] == cid].sort_values("timestamp")
            vals = cdf["consumption_kwh"].values
            if len(vals) >= WINDOW_SIZE:
                feat    = extract_features(vals[-WINDOW_SIZE:])
                feat_sc = scaler.transform(feat.reshape(1, -1))
                shap_pairs = shap_explain(shap_explainer, feat_sc[0], feature_names)
        shap_block = _shap_html(shap_pairs)

        st.markdown(
            f'<div class="fraud-card-confirmed">'
            f'<div style="display:flex;justify-content:space-between;align-items:flex-start">'
            f'<div>'
            f'<div class="fraud-id">⚠ {cid} '
            f'<span class="confirmed-badge" style="margin-left:8px">AVÉRÉ</span></div>'
            f'<div class="fraud-meta">'
            f'Région : <b>{region}</b> &nbsp;|&nbsp; Profil : <b>{profile}</b>'
            f'&nbsp;|&nbsp; 1ère det. : {first_seen} &nbsp;|&nbsp; Dernière : {last_seen}'
            f'</div>'
            f'<div style="margin:4px 0">'
            f'<span class="detection-badge">{n_det} détections IF</span>'
            f'{badge_lstm}{badge_both}</div>'
            f'<div style="margin-top:6px">{timeline}</div>'
            f'{lstm_section}'
            f'</div>'
            f'<div style="text-align:right">'
            f'<div style="font-family:Share Tech Mono,monospace;font-size:24px;color:{risk_color}">'
            f'{max_risk:.0f}%</div>'
            f'<div style="font-size:9px;color:#4a7aaa">RISQUE MAX</div>'
            f'{score_block}'
            f'</div>'
            f'</div>'
            f'<div style="margin-top:12px">'
            f'<div style="font-size:9px;color:#a855f7;letter-spacing:1px;margin-bottom:5px">'
            f'🧠 SHAP — FACTEURS DÉTERMINANTS</div>'
            f'{shap_block}'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        with st.expander(f"📊 Courbe consommation — {cid}"):
            cdf = df[df["client_id"] == cid].sort_values("timestamp")
            fig = go.Figure(go.Scatter(
                x=cdf["timestamp"], y=cdf["consumption_kwh"],
                mode="lines", line=dict(color="#ff3b3b", width=1.5),
                fill="tozeroy", fillcolor="rgba(255,59,59,0.06)",
            ))
            fig.add_hline(y=0.5, line_color="#ffaa00", line_dash="dot",
                          annotation_text="Seuil faible conso")
            fig.update_layout(
                template="plotly_dark", paper_bgcolor="#0d1a35", plot_bgcolor="#080d1a",
                font_color="#c8d8f0", height=280, margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Tableau récapitulatif ──────────────────────────────────────────────────
    section_header("TABLEAU RÉCAPITULATIF")
    recap_rows = [
        {
            "Client":         cid,
            "Région":         e.get("region",  "?"),
            "Profil":         e.get("profile", "?"),
            "Détections":     e["detection_count"],
            "Double IF+LSTM": e["both_count"],
            "Risque max %":   e["max_risk_pct"],
            "Score IF min":   e.get("min_score_if"),
            "Dernière det.":  e.get("last_seen", "?"),
        }
        for cid, e in items
    ]
    if recap_rows:
        rdf = pd.DataFrame(recap_rows)
        st.dataframe(
            rdf, use_container_width=True, hide_index=True,
            column_config={
                "Risque max %": st.column_config.ProgressColumn(min_value=0, max_value=100),
                "Score IF min": st.column_config.NumberColumn(format="%.4f"),
            },
        )