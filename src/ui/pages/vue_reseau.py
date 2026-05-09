"""
SIPT Pro v4 — Tab : Vue Réseau
Bilan énergétique régional, probabilité de piquage, tendance déviation.
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from src.config import REGIONS, COLOR_ALERT, COLOR_WARN, COLOR_INFO, COLOR_SUCCESS, COLOR_PANEL, COLOR_BORDER
from src.detection.energy_balance import EnergyBalanceDetector
from src.ui.components import section_header


def render(df: pd.DataFrame, registry: dict, region_injection: pd.DataFrame = None):
    section_header("VUE RÉSEAU — BILAN ÉNERGÉTIQUE PAR RÉGION")

    detector = EnergyBalanceDetector()

    # Dernier jour disponible dans df
    last_date = df["timestamp"].dt.date.max()
    day_df    = df[df["timestamp"].dt.date == last_date]

    cols = st.columns(len(REGIONS))

    for i, region in enumerate(REGIONS):
        # Injection régionale : du fichier ou estimée
        if region_injection is not None and not region_injection.empty:
            row = region_injection[
                (region_injection["region"] == region) &
                (region_injection["date"] == str(last_date))
            ]
            inj_kwh = float(row["injection_kwh"].iloc[0]) if not row.empty else _estimate_injection(day_df, region)
        else:
            inj_kwh = _estimate_injection(day_df, region)

        bal = detector.compute_region_balance(region, day_df, inj_kwh)

        # Historique déviation
        hist_df = detector.compute_historical_balances(df, region_injection, region, n_days=14)

        # Style carte
        border = COLOR_ALERT if bal["alert"] else (COLOR_WARN if bal["warning"] else COLOR_BORDER)
        icon   = "🔴" if bal["alert"] else ("🟡" if bal["warning"] else "🟢")

        with cols[i]:
            st.markdown(
                f'<div class="region-card" style="border-color:{border};border-width:2px">'
                f'<div class="region-name">{icon} {region}</div>'

                f'<div class="region-stat"><span>Injection</span>'
                f'<strong>{bal["injection"]:,.0f} kWh</strong></div>'

                f'<div class="region-stat"><span>Consommation</span>'
                f'<strong>{bal["total_consumption"]:,.0f} kWh</strong></div>'

                f'<div class="region-stat"><span>Pertes (5%)</span>'
                f'<strong>{bal["losses"]:,.0f} kWh</strong></div>'

                f'<div class="region-stat" style="color:{"#FF4C4C" if bal["alert"] else ("#FFB347" if bal["warning"] else "#8aa8cc")}">'
                f'<span>Écart bilan</span>'
                f'<strong style="color:{"#FF4C4C" if bal["balance_deviation"]>0 else "#00FFA5"}">'
                f'{bal["balance_deviation"]:+,.0f} kWh</strong></div>'

                f'<div class="region-stat"><span>Prob. piquage</span>'
                f'<strong style="color:{"#FF4C4C" if bal["alert"] else ("#FFB347" if bal["warning"] else "#00FFA5")}">'
                f'{bal["piquage_prob"]:.1%}</strong></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Mini sparkline historique
            if not hist_df.empty and "balance_deviation" in hist_df.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(len(hist_df))),
                    y=hist_df["balance_deviation"].tolist(),
                    mode="lines",
                    line=dict(color=COLOR_ALERT if bal["alert"] else COLOR_INFO, width=2),
                    fill="tozeroy",
                    fillcolor=f"rgba(255,76,76,0.1)" if bal["alert"] else f"rgba(0,229,255,0.07)",
                ))
                fig.add_hline(y=0, line_color="#4a7aaa", line_dash="dot", line_width=1)
                fig.update_layout(
                    height=80, margin=dict(l=0, r=0, t=4, b=4),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(visible=False), yaxis=dict(visible=False, showgrid=False),
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── Résumé tableau ─────────────────────────────────────────────────────────
    section_header("SYNTHÈSE BILANS RÉGIONAUX")
    summary_rows = []
    for region in REGIONS:
        if region_injection is not None and not region_injection.empty:
            row = region_injection[
                (region_injection["region"] == region) &
                (region_injection["date"] == str(last_date))
            ]
            inj_kwh = float(row["injection_kwh"].iloc[0]) if not row.empty else _estimate_injection(day_df, region)
        else:
            inj_kwh = _estimate_injection(day_df, region)

        bal = detector.compute_region_balance(region, day_df, inj_kwh)
        n_alerts = sum(1 for cid, e in registry.items()
                       if e.get("region") == region and not e.get("is_confirmed"))
        n_fraud  = sum(1 for cid, e in registry.items()
                       if e.get("region") == region and e.get("is_confirmed"))
        summary_rows.append({
            "Région":          region,
            "Injection kWh":   f"{bal['injection']:,.0f}",
            "Conso kWh":       f"{bal['total_consumption']:,.0f}",
            "Écart kWh":       f"{bal['balance_deviation']:+,.0f}",
            "Prob. piquage":   f"{bal['piquage_prob']:.1%}",
            "Statut":          "🔴 ALERTE" if bal["alert"] else ("🟡 WARNING" if bal["warning"] else "🟢 OK"),
            "Suspects":        n_alerts,
            "Avérés":          n_fraud,
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)


def _estimate_injection(day_df: pd.DataFrame, region: str) -> float:
    """Estimation de l'injection si non fournie : conso + 8% (pertes + fraude potentielle)."""
    conso = day_df[day_df["region"] == region]["consumption_kwh"].sum()
    return conso * 1.08 if conso > 0 else 1000.0
