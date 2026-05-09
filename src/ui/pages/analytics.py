"""
SIPT Pro v4 — Tab : Analytique
Graphiques globaux, profil client individuel, vue topologie segments.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from src.config import (
    WINDOW_SIZE, COLOR_ALERT, COLOR_WARN, COLOR_INFO, COLOR_SUCCESS,
    COLOR_ACCENT, COLOR_PANEL, COLOR_BORDER,
)
from src.ui.components import section_header
from src.detection.topology import TopologyFraudDetector
from src.models.isolation import extract_features


def render(
    df: pd.DataFrame, meta: pd.DataFrame, registry: dict,
    iso, scaler, feature_names,
    feeder_map: pd.DataFrame = None, feeder_injection: pd.DataFrame = None,
):
    tab_global, tab_client, tab_topo = st.tabs([
        "📊 Vue globale", "🔍 Profil client", "🌐 Topologie réseau"
    ])

    # ── Vue globale ────────────────────────────────────────────────────────────
    with tab_global:
        section_header("DISTRIBUTION DES CONSOMMATIONS PAR PROFIL")
        fig_box = px.box(
            df, x="profile", y="consumption_kwh", color="profile",
            color_discrete_sequence=[COLOR_INFO, COLOR_ALERT, COLOR_WARN, COLOR_ACCENT],
            template="plotly_dark",
        )
        fig_box.update_layout(
            paper_bgcolor=COLOR_PANEL, plot_bgcolor="#080d1a",
            font_color="#c8d8f0", height=350, margin=dict(l=0, r=0, t=20, b=0),
            showlegend=False,
        )
        st.plotly_chart(fig_box, use_container_width=True)

        section_header("ÉVOLUTION TEMPORELLE PAR RÉGION")
        daily_region = (
            df.groupby(["timestamp", "region"])["consumption_kwh"].sum().reset_index()
        )
        fig_line = go.Figure()
        colors_reg = [COLOR_INFO, COLOR_ALERT, COLOR_SUCCESS, COLOR_WARN, COLOR_ACCENT]
        for i, region in enumerate(df["region"].unique()):
            rdf = daily_region[daily_region["region"] == region]
            fig_line.add_trace(go.Scatter(
                x=rdf["timestamp"], y=rdf["consumption_kwh"],
                name=region, mode="lines",
                line=dict(color=colors_reg[i % len(colors_reg)], width=1.5),
            ))
        fig_line.update_layout(
            template="plotly_dark", paper_bgcolor=COLOR_PANEL, plot_bgcolor="#080d1a",
            font_color="#c8d8f0", height=320, margin=dict(l=0, r=0, t=20, b=0),
            legend=dict(font=dict(family="Share Tech Mono", size=10)),
        )
        st.plotly_chart(fig_line, use_container_width=True)

        section_header("STATISTIQUES DU REGISTRE")
        reg_summary = []
        for cid, entry in registry.items():
            reg_summary.append({
                "Client": cid,
                "Région": entry.get("region", "?"),
                "Profil": entry.get("profile", "?"),
                "Détections": len(entry.get("detections", [])),
                "Confirmé": "✓ Avéré" if entry.get("is_confirmed") else "Suspect",
                "1ère détection": entry.get("first_seen", "?")[:10],
                "Dernière": entry.get("last_seen", "?")[:10],
            })
        if reg_summary:
            st.dataframe(pd.DataFrame(reg_summary), use_container_width=True, hide_index=True)
        else:
            st.info("Registre vide — lancez une surveillance.")

    # ── Profil client ──────────────────────────────────────────────────────────
    with tab_client:
        section_header("ANALYSE INDIVIDUELLE CLIENT")
        all_clients = sorted(df["client_id"].unique().tolist())
        sel = st.selectbox("Sélectionner un client", all_clients)
        if sel:
            cdf  = df[df["client_id"] == sel].sort_values("timestamp")
            vals = cdf["consumption_kwh"].values
            m    = meta[meta["client_id"] == sel]
            reg  = registry.get(sel, {})

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Profil",    m["profile"].iloc[0] if not m.empty else "?")
            c2.metric("Région",    m["region"].iloc[0]  if not m.empty else "?")
            c3.metric("Conso moy.", f"{vals.mean():.1f} kWh")
            c4.metric("Statut",    "✓ Avéré" if reg.get("is_confirmed") else
                                   ("⚡ Suspect" if reg.get("detections") else "Normal"))

            # Courbe de consommation
            fig_c = go.Figure()
            fig_c.add_trace(go.Scatter(
                x=cdf["timestamp"], y=vals,
                mode="lines", line=dict(color=COLOR_INFO, width=1.5),
                fill="tozeroy", fillcolor="rgba(0,229,255,0.06)",
                name="Consommation",
            ))
            # EMA baseline
            ema = reg.get("ema_baseline")
            if ema:
                fig_c.add_hline(y=ema, line_color=COLOR_SUCCESS, line_dash="dot",
                                annotation_text=f"Baseline EMA: {ema:.1f}")
            fig_c.add_hline(y=0.5, line_color=COLOR_ALERT, line_dash="dot",
                            annotation_text="Seuil faible conso")
            fig_c.update_layout(
                template="plotly_dark", paper_bgcolor=COLOR_PANEL, plot_bgcolor="#080d1a",
                font_color="#c8d8f0", height=300, margin=dict(l=0, r=0, t=30, b=0),
                title=f"Consommation — {sel}",
            )
            st.plotly_chart(fig_c, use_container_width=True)

            # Features IF
            if len(vals) >= WINDOW_SIZE and scaler is not None:
                feat    = extract_features(vals[-WINDOW_SIZE:])
                feat_sc = scaler.transform(feat.reshape(1, -1))
                score   = float(iso.decision_function(feat_sc)[0])
                st.markdown(
                    f'<div style="font-family:Share Tech Mono;font-size:12px;color:#8aa8cc;margin:8px 0">'
                    f'Score IF : <strong style="color:{"#FF4C4C" if score < -0.1 else "#00FFA5"}">{score:.4f}</strong>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                feat_df = pd.DataFrame({
                    "Feature": feature_names,
                    "Valeur brute": feat.round(4),
                    "Valeur scalée": feat_sc[0].round(4),
                })
                st.dataframe(feat_df, use_container_width=True, hide_index=True)

    # ── Topologie ──────────────────────────────────────────────────────────────
    with tab_topo:
        section_header("TOPOLOGIE RÉSEAU — BILANS PAR FEEDER")

        if feeder_map is None or feeder_map.empty:
            st.warning("Fichier client_feeder_map.csv introuvable. Relancez train.py.")
            return

        detector = TopologyFraudDetector()
        feeder_df = detector.evaluate_feeders(feeder_map, feeder_injection, df)

        if feeder_df.empty:
            st.info("Aucune donnée feeder disponible.")
            return

        # Tableau principal
        display_df = feeder_df.copy()
        display_df["Statut"] = display_df["persistent_alert"].map(
            {True: "🔴 ALERTE", False: "🟢 OK"}
        )
        display_df["Écart normalisé"] = display_df["avg_ratio"].map(lambda x: f"{x:.2%}")
        display_df["Écart kWh/j moy"] = display_df["avg_deviation"].map(lambda x: f"{x:+.1f}")
        display_df["Jours suspects"]   = display_df["n_suspicious"]

        st.dataframe(
            display_df[["feeder_id", "region", "n_clients", "Écart normalisé",
                         "Écart kWh/j moy", "Jours suspects", "Statut"]],
            use_container_width=True, hide_index=True,
        )

        # Détail par feeder
        section_header("DÉTAIL FEEDERS")
        for _, row in feeder_df.iterrows():
            border = COLOR_ALERT if row["persistent_alert"] else COLOR_BORDER
            with st.expander(
                f"{'🔴' if row['persistent_alert'] else '🟢'} {row['feeder_id']} "
                f"— {row['region']} — {row['n_clients']} clients"
            ):
                st.markdown(
                    f'<div style="font-family:Share Tech Mono;font-size:11px;color:#8aa8cc">'
                    f'Écart normalisé moy : <strong style="color:{"#FF4C4C" if row["avg_ratio"]>0.12 else "#00FFA5"}">'
                    f'{row["avg_ratio"]:.2%}</strong> · '
                    f'Jours suspects : <strong>{row["n_suspicious"]}</strong>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                # Clients sur ce feeder
                clients = row.get("clients", [])
                if clients:
                    client_rows = []
                    for cid in clients:
                        r_entry = registry.get(cid, {})
                        cdf = df[df["client_id"] == cid]
                        client_rows.append({
                            "Client": cid,
                            "Conso moy.": f"{cdf['consumption_kwh'].mean():.1f} kWh",
                            "Statut": "🔴 Avéré" if r_entry.get("is_confirmed")
                                      else ("🟡 Suspect" if r_entry.get("detections") else "🟢 Normal"),
                        })
                    st.dataframe(pd.DataFrame(client_rows), use_container_width=True, hide_index=True)
