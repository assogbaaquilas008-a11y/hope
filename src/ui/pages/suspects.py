"""
SIPT Pro v4 — Tab : Suspects
Rétrocompatible avec le registre v3 (detected_by / date / shap 3-tuples)
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from src.ui.components import section_header
from src.config import CONFIRM_THRESHOLD


def _badge_color(d: dict) -> str:
    """
    Retourne la couleur du badge d'une détection.
    Supporte level (v4 : DOUBLE/LSTM/CUSUM) et detected_by (v3 : BOTH/LSTM/IF).
    """
    lvl = d.get("level") or d.get("detected_by", "?")
    if lvl in ("DOUBLE", "BOTH"):
        return "#f97316"
    if lvl == "LSTM":
        return "#ffaa00"
    if lvl == "CUSUM":
        return "#8b5cf6"
    return "#ffaa00"


def _det_timestamp(d: dict) -> str:
    """Retourne la date de détection quelle que soit la clé (timestamp v4 ou date v3)."""
    return str(d.get("timestamp") or d.get("date") or "")[:10]


def render(registry: dict, df: pd.DataFrame, scaler=None, shap_explainer=None, feature_names=None):
    # Toujours lire depuis session_state (paramètre peut être stale)
    registry = st.session_state.get("registry", registry)

    section_header("SUSPECTS ACTIFS — SURVEILLANCE EN COURS")

    suspects = []
    for cid, entry in registry.items():
        if entry.get("is_confirmed", False):
            continue
        detections = entry.get("detections", [])
        if not detections:
            continue

        # FIX compteurs : valeur stockée OU recalculée avec fallback v3/v4
        n_det    = entry.get("detection_count") or len(detections)
        max_risk = entry.get("max_risk_pct") or max(
            (d.get("risk_pct", 0) for d in detections), default=0
        )
        suspects.append({
            "client_id":       cid,
            "region":          entry.get("region",  "?"),
            "profile":         entry.get("profile", "?"),
            "first_seen":      str(entry.get("first_seen", "?"))[:10],
            "detections":      detections,
            "detection_count": n_det,
            "max_risk_pct":    max_risk,
        })

    if not suspects:
        st.markdown(
            '<div style="color:#00FFA5;font-family:Share Tech Mono;font-size:13px;padding:20px">'
            '✓ Aucun suspect dans le registre actuel.</div>',
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f'<div style="color:#FFB347;font-family:Share Tech Mono;font-size:13px;margin-bottom:12px">'
        f'● {len(suspects)} CLIENT(S) SUSPECT(S) · MOINS DE {CONFIRM_THRESHOLD} DÉTECTIONS</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="font-size:11px;color:#4a7aaa;margin-bottom:12px">'
        f'Ces clients ont déclenché au moins une alerte mais pas encore '
        f'le seuil de confirmation ({CONFIRM_THRESHOLD} détections). '
        f'Ils restent sous surveillance.</div>',
        unsafe_allow_html=True,
    )

    suspects.sort(key=lambda x: -x["max_risk_pct"])

    for s in suspects[:30]:
        cid      = s["client_id"]
        n_det    = s["detection_count"]
        max_risk = s["max_risk_pct"]
        risk_col = "#ffaa00" if max_risk > 40 else "#8aa8cc"

        progress      = min(n_det / CONFIRM_THRESHOLD, 1.0)
        bar_color     = "#ffaa00" if progress < 0.8 else "#ff3b3b"
        bar_width_str = f"{progress * 100:.0f}%"

        # Badges des 3 dernières détections avec couleur par niveau (v3 + v4)
        det_str = ""
        for d in s["detections"][-3:]:
            badge_col = _badge_color(d)
            ts = _det_timestamp(d)
            lvl = d.get("level") or d.get("detected_by", "")
            label = f"{ts}" + (f" [{lvl}]" if lvl and lvl != "—" else "")
            det_str += (
                f'<span style="font-size:9px;background:#1a1800;'
                f'border:1px solid {badge_col};color:{badge_col};'
                f'border-radius:3px;padding:1px 5px;margin-right:3px">'
                f'{label}</span>'
            )

        st.markdown(
            f'<div class="fraud-card-suspect">'
            f'<div style="display:flex;justify-content:space-between;align-items:center">'
            f'<div style="flex:1">'
            f'<div class="suspect-id">⚡ {cid}</div>'
            f'<div class="fraud-meta">'
            f'{s["region"]} | {s["profile"]} | 1ère det. {s["first_seen"]}'
            f'</div>'
            f'<div style="margin-top:4px">{det_str}</div>'
            f'<div style="background:#1a1a2e;border-radius:4px;padding:2px;margin:6px 0 2px 0">'
            f'<div style="background:{bar_color};width:{bar_width_str};height:6px;border-radius:4px"></div>'
            f'</div>'
            f'<div style="font-size:9px;color:#aabbcc">'
            f'{n_det}/{CONFIRM_THRESHOLD} détections vers confirmation'
            f'</div>'
            f'</div>'
            f'<div style="text-align:right;min-width:75px;margin-left:12px">'
            f'<div style="font-family:Share Tech Mono,monospace;font-size:22px;color:{risk_col}">'
            f'{max_risk:.0f}%</div>'
            f'<div style="font-size:9px;color:#4a7aaa">RISQUE</div>'
            f'</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Mini-graphe évolution du risque (si ≥ 2 détections)
        if n_det >= 2:
            dates = [_det_timestamp(d) for d in s["detections"]]
            risks = [d.get("risk_pct", 0) for d in s["detections"]]
            fig = go.Figure(go.Scatter(
                x=dates, y=risks,
                mode="lines+markers",
                line=dict(color="#ffaa00", width=2),
                marker=dict(size=6, color="#ffaa00"),
                fill="tozeroy",
                fillcolor="rgba(255,170,0,0.08)",
            ))
            fig.update_layout(
                height=120,
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(visible=False),
                yaxis=dict(visible=False, range=[0, 105]),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})