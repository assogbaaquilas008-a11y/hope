"""
SIPT Pro v4 — Composants UI réutilisables (SCADA dark theme v4)
kpi_grid, region_card, fraud_card, ticker_feed, etc.
"""
import streamlit as st
from src.config import (
    COLOR_ALERT, COLOR_WARN, COLOR_INFO, COLOR_SUCCESS,
    COLOR_ACCENT, COLOR_LSTM, COLOR_PANEL, COLOR_BORDER_SOFT, COLOR_BG, COLOR_TEXT,
COLOR_PANEL_2)

# ── CSS global SCADA v4 ───────────────────────────────────────────────────────

SCADA_CSS = f"""
<style>

@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Inter:wght@300;400;500;600;700&display=swap');

/* =========================================================
BASE
========================================================= */

html, body, [class*="css"] {{
    background-color:{COLOR_BG}!important;
    color:{COLOR_TEXT}!important;
    font-family:'Inter',sans-serif;
}}

.stApp {{
    background:
        radial-gradient(
            circle at top right,
            rgba(0,229,255,.05),
            transparent 35%
        ),
        {COLOR_BG};
}}

/* =========================================================
GLOBAL DEPTH
========================================================= */

.glass {{
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
}}

.soft-shadow {{
    box-shadow:
        0 2px 6px rgba(0,0,0,.20),
        0 8px 24px rgba(0,0,0,.18);
}}

.glow-info {{
    box-shadow:0 0 14px rgba(0,229,255,.14);
}}

.glow-alert {{
    box-shadow:0 0 14px rgba(255,76,76,.18);
}}

/* =========================================================
HEADER
========================================================= */

.scada-header {{
    background:
        linear-gradient(
            90deg,
            #060b18 0%,
            #0d1a35 50%,
            #060b18 100%
        );

    border-bottom:1px solid {COLOR_BORDER_SOFT};

    padding:14px 24px;

    display:flex;
    align-items:center;
    justify-content:space-between;

    margin-bottom:24px;

    backdrop-filter: blur(10px);

    box-shadow:
        0 4px 14px rgba(0,0,0,.16);
}}

.scada-title {{
    font-family:'Share Tech Mono',monospace;
    font-size:22px;
    color:{COLOR_INFO};
    letter-spacing:3px;
    text-transform:uppercase;
}}

.scada-subtitle {{
    font-size:11px;
    color:#6e8fb5;
    letter-spacing:2px;
}}

.scada-clock {{
    font-family:'Share Tech Mono',monospace;
    font-size:18px;
    color:{COLOR_SUCCESS};
    letter-spacing:2px;
}}

/* =========================================================
KPI GRID
========================================================= */

.kpi-grid {{
    display:grid;
    grid-template-columns:repeat(6,1fr);
    gap:14px;
    margin-bottom:24px;
}}

.kpi-card {{

    background:
        linear-gradient(
            180deg,
            rgba(255,255,255,.015),
            rgba(255,255,255,.005)
        ),
        {COLOR_PANEL};

    border:1px solid {COLOR_BORDER_SOFT};

    border-radius:12px;

    padding:18px 14px;

    text-align:center;

    position:relative;
    overflow:hidden;

    backdrop-filter: blur(10px);

    box-shadow:
        0 2px 6px rgba(0,0,0,.20),
        0 8px 24px rgba(0,0,0,.18);

    transition:all .2s ease;
}}

.kpi-card:hover {{
    transform:translateY(-2px);
    border-color:rgba(0,229,255,.30);
}}

.kpi-card::before {{
    content:'';
    position:absolute;
    top:0;
    left:0;
    right:0;
    height:3px;
}}

.kpi-ok::before{{background:{COLOR_SUCCESS}}}
.kpi-warn::before{{background:{COLOR_WARN}}}
.kpi-alert::before{{background:{COLOR_ALERT}}}
.kpi-info::before{{background:{COLOR_INFO}}}
.kpi-purple::before{{background:{COLOR_ACCENT}}}
.kpi-lstm::before{{background:{COLOR_LSTM}}}

.kpi-num {{
    font-family:'Share Tech Mono',monospace;
    font-size:32px;
    font-weight:700;
    margin-bottom:6px;
    line-height:1.2;
}}

.kpi-lbl {{
    font-size:10px;
    color:#7d9bc0;
    text-transform:uppercase;
    letter-spacing:1.5px;
}}

.kpi-ok .kpi-num{{color:{COLOR_SUCCESS}}}
.kpi-warn .kpi-num{{color:{COLOR_WARN}}}
.kpi-alert .kpi-num{{color:{COLOR_ALERT}}}
.kpi-info .kpi-num{{color:{COLOR_INFO}}}
.kpi-purple .kpi-num{{color:{COLOR_ACCENT}}}
.kpi-lstm .kpi-num{{color:{COLOR_LSTM}}}

/* =========================================================
REGION CARDS
========================================================= */

.region-card {{

    background:
        linear-gradient(
            180deg,
            rgba(255,255,255,.015),
            rgba(255,255,255,.005)
        ),
        {COLOR_PANEL};

    border:1px solid {COLOR_BORDER_SOFT};

    border-radius:14px;

    padding:18px;

    position:relative;

    backdrop-filter: blur(10px);

    box-shadow:
        0 2px 6px rgba(0,0,0,.20),
        0 8px 24px rgba(0,0,0,.18);
}}

.region-card.has-alert {{
    border-color:rgba(255,76,76,.45);

    box-shadow:
        0 0 12px rgba(255,76,76,.14),
        0 8px 24px rgba(0,0,0,.22);
}}

.region-card.has-warning {{
    border-color:rgba(255,179,71,.35);
}}

.region-name {{
    font-family:'Share Tech Mono',monospace;
    font-size:13px;
    color:{COLOR_INFO};
    letter-spacing:2px;
    margin-bottom:12px;
    text-transform:uppercase;
}}

.region-stat {{
    display:flex;
    justify-content:space-between;

    font-size:11px;

    padding:6px 0;

    border-bottom:1px solid rgba(80,120,200,.12);

    color:#8aa8cc;
}}

.region-stat:last-child {{
    border-bottom:none;
}}

.region-stat strong {{
    color:{COLOR_TEXT};
    font-family:'Share Tech Mono',monospace;
}}

/* =========================================================
ALERTS
========================================================= */

.alert-ticker {{

    background:#0d0812;

    border:1px solid rgba(255,76,76,.20);

    border-radius:8px;

    padding:8px 12px;

    font-family:'Share Tech Mono',monospace;

    font-size:11px;

    color:{COLOR_ALERT};

    margin-bottom:8px;

    display:flex;
    align-items:center;
    gap:8px;

    backdrop-filter: blur(10px);
}}

.ticker-meta {{
    color:#8aa8cc;
    font-size:10px;
}}

.ticker-badge,
.ticker-badge-lstm,
.ticker-badge-both,
.ticker-badge-cusum {{
    color:#fff;
    padding:2px 7px;
    border-radius:5px;
    font-size:9px;
    font-weight:700;
}}

.ticker-badge {{
    background:{COLOR_ALERT};
}}

.ticker-badge-lstm {{
    background:{COLOR_LSTM};
}}

.ticker-badge-both {{
    background:linear-gradient(
        90deg,
        {COLOR_ALERT},
        {COLOR_LSTM}
    );
}}

.ticker-badge-cusum {{
    background:#8b5cf6;
}}

/* =========================================================
CLIENT ROWS
========================================================= */

.client-row {{

    display:flex;
    align-items:center;
    justify-content:space-between;

    padding:8px 12px;

    border-radius:8px;

    margin-bottom:5px;

    font-size:11px;

    font-family:'Share Tech Mono',monospace;

    border:1px solid transparent;

    transition:all .18s ease;
}}

.client-row.normal {{
    background:#0a1628;
    border-color:rgba(80,120,200,.10);
    color:#6e9abf;
}}

.client-row.warning {{
    background:#1a1200;
    border-color:rgba(255,179,71,.18);
    color:{COLOR_WARN};
}}

.client-row.alert {{
    background:#1a0808;
    border-color:rgba(255,76,76,.22);
    color:#ff6b6b;
}}

.client-row.alert_high {{
    background:#1a0510;
    border-color:rgba(255,76,76,.35);
    color:{COLOR_ALERT};

    box-shadow:
        0 0 10px rgba(255,76,76,.12);
}}

/* =========================================================
SECTION HEADERS
========================================================= */

.section-header {{
    font-family:'Share Tech Mono',monospace;

    font-size:12px;

    color:{COLOR_INFO};

    letter-spacing:2px;

    text-transform:uppercase;

    border-bottom:1px solid rgba(80,120,200,.18);

    padding-bottom:6px;

    margin:18px 0 12px 0;
}}

/* =========================================================
FRAUD CARDS (portées depuis v3 — manquantes en v4)
========================================================= */

.fraud-card-confirmed {{
    background:#0f0818;
    border:1px solid #6a1a3e;
    border-left:4px solid {COLOR_ALERT};
    border-radius:10px;
    padding:16px 20px;
    margin-bottom:14px;
    box-shadow:
        0 0 12px rgba(255,76,76,.08),
        0 4px 16px rgba(0,0,0,.22);
    transition:all .2s ease;
}}

.fraud-card-confirmed:hover {{
    border-color:rgba(255,76,76,.60);
    box-shadow:0 0 18px rgba(255,76,76,.14);
}}

.fraud-card-suspect {{
    background:#0f100a;
    border:1px solid rgba(80,120,50,.30);
    border-left:4px solid {COLOR_WARN};
    border-radius:10px;
    padding:14px 18px;
    margin-bottom:10px;
    transition:all .2s ease;
}}

.fraud-card-suspect:hover {{
    border-color:rgba(255,179,71,.45);
}}

.fraud-id {{
    font-family:'Share Tech Mono',monospace;
    font-size:14px;
    color:{COLOR_ALERT};
    font-weight:700;
    margin-bottom:2px;
}}

.suspect-id {{
    font-family:'Share Tech Mono',monospace;
    font-size:13px;
    color:{COLOR_WARN};
    font-weight:700;
    margin-bottom:2px;
}}

.fraud-meta {{
    font-size:11px;
    color:#8aa8cc;
    margin:3px 0 8px 0;
}}

.detection-badge {{
    display:inline-block;
    background:#2a0820;
    border:1px solid rgba(255,76,76,.35);
    border-radius:12px;
    padding:2px 10px;
    font-family:'Share Tech Mono',monospace;
    font-size:11px;
    color:#ff6b6b;
    margin-right:6px;
}}

.detection-badge-lstm {{
    background:#1a0810;
    border-color:rgba(249,115,22,.40);
    color:{COLOR_LSTM};
}}

.detection-badge-cusum {{
    background:#1a0a2e;
    border-color:rgba(139,92,246,.40);
    color:#a78bfa;
}}

.confirmed-badge {{
    display:inline-block;
    background:#3a0a20;
    border:1px solid rgba(200,40,70,.50);
    border-radius:4px;
    padding:2px 8px;
    font-size:10px;
    color:{COLOR_ALERT};
    font-weight:700;
    letter-spacing:1px;
    text-transform:uppercase;
}}

/* =========================================================
SHAP ROWS
========================================================= */

.shap-row {{
    display:flex;
    justify-content:space-between;
    align-items:center;
    padding:4px 0;
    border-bottom:1px solid rgba(60,20,40,.60);
    font-size:11px;
}}

.shap-row:last-child {{
    border-bottom:none;
}}

.shap-feat {{
    color:#c8a0d0;
    min-width:160px;
}}

.shap-val-neg {{
    color:#ff6b6b;
    font-family:'Share Tech Mono',monospace;
    font-weight:700;
    min-width:56px;
    text-align:right;
}}

.shap-val-pos {{
    color:{COLOR_SUCCESS};
    font-family:'Share Tech Mono',monospace;
    font-weight:700;
    min-width:56px;
    text-align:right;
}}

.shap-bar {{
    width:60px;
    height:5px;
    background:#2a1020;
    border-radius:3px;
    overflow:hidden;
    margin:0 8px;
}}

.shap-fill-neg {{
    height:100%;
    background:{COLOR_ALERT};
    border-radius:3px;
}}

.shap-fill-pos {{
    height:100%;
    background:{COLOR_SUCCESS};
    border-radius:3px;
}}

/* =========================================================
DOTS
========================================================= */

.dot {{
    width:7px;
    height:7px;
    border-radius:50%;
    display:inline-block;
    margin-right:5px;
}}

.dot-ok {{
    background:{COLOR_SUCCESS};
    box-shadow:0 0 6px {COLOR_SUCCESS};
}}

.dot-warn {{
    background:{COLOR_WARN};
    box-shadow:0 0 6px {COLOR_WARN};
}}

.dot-alert {{
    background:{COLOR_ALERT};

    box-shadow:
        0 0 10px rgba(255,76,76,.7);

    animation:blink 1s infinite;
}}

@keyframes blink {{
    0%,100%{{opacity:1}}
    50%{{opacity:.35}}
}}

/* =========================================================
TABS
========================================================= */

.stTabs [data-baseweb="tab-list"] {{
    background:{COLOR_PANEL}!important;

    border-bottom:1px solid rgba(80,120,200,.12)!important;

    gap:6px;
}}

.stTabs [data-baseweb="tab"] {{

    background:transparent!important;

    color:#6f8fb5!important;

    font-family:'Share Tech Mono',monospace!important;

    font-size:11px!important;

    letter-spacing:1px!important;

    border-radius:8px 8px 0 0!important;

    padding:8px 16px!important;

    transition:all .2s ease;
}}

.stTabs [aria-selected="true"] {{

    background:rgba(0,229,255,.08)!important;

    color:{COLOR_INFO}!important;

    border-bottom:2px solid {COLOR_INFO}!important;
}}

/* =========================================================
BUTTONS
========================================================= */

.stButton > button {{

    background:#0d2244!important;

    border:1px solid rgba(0,229,255,.30)!important;

    color:{COLOR_INFO}!important;

    font-family:'Share Tech Mono',monospace!important;

    letter-spacing:1px!important;

    border-radius:8px!important;

    font-size:11px!important;

    transition:all .2s ease;

    box-shadow:
        0 2px 6px rgba(0,0,0,.14);
}}

.stButton > button:hover {{

    background:{COLOR_INFO}!important;

    color:{COLOR_BG}!important;

    transform:translateY(-1px);

    box-shadow:
        0 0 14px rgba(0,229,255,.22);
}}

/* =========================================================
SIDEBAR
========================================================= */

[data-testid="stSidebar"] {{

    background:#060b18!important;

    border-right:1px solid rgba(80,120,200,.12)!important;

    backdrop-filter: blur(10px);
}}

[data-testid="stSidebar"] * {{
    color:{COLOR_TEXT}!important;
}}

/* =========================================================
DATAFRAME
========================================================= */

[data-testid="stDataFrame"] {{
    background:{COLOR_PANEL}!important;

    border:1px solid rgba(80,120,200,.14)!important;

    border-radius:10px!important;

    overflow:hidden;
}}

/* =========================================================
PROGRESS
========================================================= */

.stProgress > div > div {{
    background-color:{COLOR_INFO}!important;
}}

/* =========================================================
WEEK BADGE
========================================================= */

.week-badge {{

    display:inline-block;

    background:{COLOR_PANEL};

    border:1px solid rgba(0,229,255,.18);

    border-radius:16px;

    padding:4px 12px;

    font-family:'Share Tech Mono',monospace;

    font-size:11px;

    color:{COLOR_INFO};

    margin-bottom:12px;
}}

/* =========================================================
PRESENTATION MODE
========================================================= */

.pres-mode .kpi-num {{
    font-size:40px!important;
}}

.pres-mode .kpi-lbl {{
    font-size:13px!important;
}}

.pres-mode .region-name {{
    font-size:16px!important;
}}

.pres-mode .region-stat {{
    font-size:13px!important;
}}

</style>
"""
def inject_css():
    st.markdown(SCADA_CSS, unsafe_allow_html=True)


def scada_header(clock_str: str, pres_mode: bool = False):
    size = "26px" if pres_mode else "22px"
    st.markdown(
        f'<div class="scada-header">'
        f'<div><div class="scada-title" style="font-size:{size}">⚡ Holistic Outlier and Pattern Evaluator</div>'
        f'<div class="scada-subtitle">SYSTÈME INTÉGRÉ DE PROTECTION DU RÉSEAU ÉLECTRIQUE</div></div>'
        f'<div class="scada-clock" style="font-size:{"22px" if pres_mode else "18px"}">{clock_str}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def kpi_card(value, label: str, kind: str = "info") -> str:
    return (
        f'<div class="kpi-card kpi-{kind}">'
        f'<div class="kpi-num">{value}</div>'
        f'<div class="kpi-lbl">{label}</div>'
        f'</div>'
    )


def kpi_grid(cards: list[tuple]):
    """cards = [(value, label, kind), ...]"""
    html = '<div class="kpi-grid">'
    for v, l, k in cards:
        html += kpi_card(v, l, k)
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def section_header(text: str):
    st.markdown(
        f'<div class="section-header">◈ {text}</div>',
        unsafe_allow_html=True,
    )


def ticker_item(client_id: str, level: str, risk: float,
                profile: str, region: str, cusum: bool = False) -> str:
    badge = {
        "DOUBLE": '<span class="ticker-badge-both">DOUBLE</span>',
        "IF":     '<span class="ticker-badge">IF</span>',
        "LSTM":   '<span class="ticker-badge-lstm">LSTM</span>',
        "CUSUM":  '<span class="ticker-badge-cusum">CUSUM</span>',
    }.get(level, "")
    cusum_badge = '<span class="ticker-badge-cusum">CUSUM</span>' if cusum and level not in ("CUSUM",) else ""
    return (
        f'<div class="alert-ticker">'
        f'{badge}{cusum_badge} {client_id}'
        f'<span class="ticker-meta"> · {profile} · {region} · risque {risk:.0%}</span>'
        f'</div>'
    )


def fraud_card_html(client_id: str, entry: dict, shap_pairs: list = None) -> str:
    n_det     = len(entry.get("detections", []))
    confirmed = entry.get("is_confirmed", False)
    card_cls  = "fraud-card-confirmed" if confirmed else "fraud-card-suspect"
    id_cls    = "fraud-id" if confirmed else "suspect-id"
    badge_html = '<span class="confirmed-badge">● AVÉRÉ</span>' if confirmed else \
                 f'<span style="color:#FFB347;font-size:10px;font-family:Share Tech Mono">⚡ SUSPECT ({n_det} det.)</span>'

    det_html = ""
    for d in entry.get("detections", [])[-5:]:
        lvl   = d.get("level", "?")
        badge = {
            "DOUBLE": '<span class="detection-badge">IF+LSTM</span>',
            "IF":     '<span class="detection-badge">IF</span>',
            "LSTM":   '<span class="detection-badge detection-badge-lstm">LSTM</span>',
            "CUSUM":  '<span class="detection-badge detection-badge-cusum">CUSUM</span>',
        }.get(lvl, f'<span class="detection-badge">{lvl}</span>')
        det_html += f'{badge}<span style="font-size:10px;color:#8aa8cc">{d["timestamp"][:10]} · {d["risk_pct"]}%</span><br>'

    # SHAP (top 4)
    shap_html = ""
    if shap_pairs:
        for fname, sv in shap_pairs[:4]:
            color = "shap-val-neg" if sv < 0 else "shap-val-pos"
            fill  = "shap-fill-neg" if sv < 0 else "shap-fill-pos"
            pct   = min(100, abs(sv) * 200)
            shap_html += (
                f'<div class="shap-row">'
                f'<span class="shap-feat">{fname[:30]}</span>'
                f'<span class="{color}">{sv:+.3f}</span>'
                f'<div class="shap-bar"><div class="{fill}" style="width:{pct:.0f}%"></div></div>'
                f'</div>'
            )

    max_cusum = max((d.get("cusum_max", 0) for d in entry.get("detections", [])), default=0)
    cusum_info = f'<br><span style="font-size:10px;color:#a78bfa">CUSUM max: {max_cusum:.2f}</span>' if max_cusum > 0 else ""

    return (
        f'<div class="{card_cls}">'
        f'<div class="{id_cls}">{client_id}</div>'
        f'<div class="fraud-meta">{entry.get("profile","?")} · {entry.get("region","?")} · '
        f'1ère det: {entry.get("first_seen","?")[:10]}</div>'
        f'{badge_html}{cusum_info}<br><br>'
        f'{det_html}'
        f'{"<hr style=border-color:#3a1030>" + shap_html if shap_html else ""}'
        f'</div>'
    )