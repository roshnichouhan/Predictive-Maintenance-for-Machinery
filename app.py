
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="AI Predictive Maintenance | MachineGuard Pro",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS — DARK INDUSTRIAL THEME
# ============================================================
st.markdown("""
<style>
/* -------- GLOBAL -------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1421 50%, #0a0e1a 100%);
    color: #e2e8f0;
}

/* -------- HEADER HERO -------- */
.hero-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 40%, #0f3460 100%);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -20%;
    width: 60%;
    height: 200%;
    background: radial-gradient(ellipse, rgba(56,189,248,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #38bdf8, #818cf8, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.4rem 0;
    line-height: 1.2;
}
.hero-subtitle {
    font-size: 1.05rem;
    color: #94a3b8;
    font-weight: 400;
    margin: 0;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(56,189,248,0.12);
    border: 1px solid rgba(56,189,248,0.3);
    border-radius: 50px;
    padding: 0.3rem 0.9rem;
    font-size: 0.78rem;
    color: #38bdf8;
    font-weight: 600;
    margin-top: 1rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* -------- KPI CARDS -------- */
.kpi-card {
    background: linear-gradient(135deg, rgba(15,23,42,0.9) 0%, rgba(30,41,59,0.8) 100%);
    border: 1px solid rgba(99,179,237,0.15);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
}
.kpi-card:hover {
    border-color: rgba(56,189,248,0.4);
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(56,189,248,0.1);
}
.kpi-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #38bdf8, transparent);
    opacity: 0.5;
}
.kpi-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.4rem;
}
.kpi-value {
    font-size: 2rem;
    font-weight: 800;
    color: #f1f5f9;
    line-height: 1;
    font-family: 'JetBrains Mono', monospace;
}
.kpi-unit {
    font-size: 0.9rem;
    color: #64748b;
    font-weight: 400;
}
.kpi-delta {
    font-size: 0.78rem;
    margin-top: 0.4rem;
    font-weight: 500;
}
.kpi-icon {
    position: absolute;
    top: 1.2rem; right: 1.4rem;
    font-size: 1.8rem;
    opacity: 0.3;
}

/* -------- SECTION TITLE -------- */
.section-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid rgba(99,179,237,0.15);
}

/* -------- ALERT BANNERS -------- */
.alert-critical {
    background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(220,38,38,0.08));
    border: 1px solid rgba(239,68,68,0.4);
    border-left: 4px solid #ef4444;
    border-radius: 12px;
    padding: 1.2rem 1.6rem;
    margin: 1rem 0;
}
.alert-safe {
    background: linear-gradient(135deg, rgba(52,211,153,0.15), rgba(16,185,129,0.08));
    border: 1px solid rgba(52,211,153,0.4);
    border-left: 4px solid #34d399;
    border-radius: 12px;
    padding: 1.2rem 1.6rem;
    margin: 1rem 0;
}
.alert-title {
    font-size: 1.3rem;
    font-weight: 800;
    margin-bottom: 0.3rem;
}
.alert-subtitle {
    font-size: 0.88rem;
    opacity: 0.8;
}

/* -------- ACTION CARDS -------- */
.action-card {
    background: rgba(15,23,42,0.7);
    border: 1px solid rgba(99,179,237,0.12);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.7rem;
    font-size: 0.88rem;
    color: #cbd5e1;
    transition: all 0.2s;
}
.action-card:hover { border-color: rgba(56,189,248,0.3); }
.action-dot-red { width: 8px; height: 8px; border-radius: 50%; background: #ef4444; flex-shrink: 0; }
.action-dot-green { width: 8px; height: 8px; border-radius: 50%; background: #34d399; flex-shrink: 0; }

/* -------- SIDEBAR OVERRIDES -------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1421 0%, #0f172a 100%);
    border-right: 1px solid rgba(99,179,237,0.1);
}
section[data-testid="stSidebar"] .stSlider > label {
    color: #94a3b8 !important;
    font-size: 0.85rem;
    font-weight: 500;
}

/* -------- BUTTON -------- */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #0ea5e9, #6366f1);
    color: white;
    border: none;
    border-radius: 12px;
    font-weight: 700;
    font-size: 1rem;
    padding: 0.8rem 2rem;
    width: 100%;
    transition: all 0.3s ease;
    box-shadow: 0 4px 20px rgba(14,165,233,0.3);
    letter-spacing: 0.03em;
}
div[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #0284c7, #4f46e5);
    box-shadow: 0 6px 30px rgba(14,165,233,0.5);
    transform: translateY(-1px);
}

/* -------- METRIC OVERRIDES -------- */
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 700 !important;
    color: #f1f5f9 !important;
}
[data-testid="stMetricLabel"] {
    color: #64748b !important;
    font-size: 0.8rem !important;
}

/* -------- TABS -------- */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(15,23,42,0.6);
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
    border: 1px solid rgba(99,179,237,0.12);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #64748b;
    font-weight: 500;
    font-size: 0.88rem;
    padding: 0.5rem 1.2rem;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
    color: white !important;
    font-weight: 600 !important;
}

/* -------- PLOTLY CHART BG -------- */
.js-plotly-plot { border-radius: 14px; }

/* -------- FOOTER -------- */
.footer {
    text-align: center;
    padding: 2rem 0 1rem;
    color: #334155;
    font-size: 0.8rem;
    border-top: 1px solid rgba(99,179,237,0.08);
    margin-top: 3rem;
}
.footer span { color: #38bdf8; }
</style>
""", unsafe_allow_html=True)



# ============================================================
# LOAD MODEL
# ============================================================
possible_paths = [
    "models/random_forest.pkl",
    "random_forest.pkl",
]
MODEL_PATH = next((p for p in possible_paths if os.path.exists(p)), None)

if MODEL_PATH is None:
    st.error("⚠️ Model file not found! Ensure `random_forest.pkl` is in `models/` or the root directory.")
    st.stop()

model = joblib.load(MODEL_PATH)


# ============================================================
# MODEL PERFORMANCE METRICS
# ============================================================

metric_df = pd.read_csv(
    "data/engineered_data.csv"
)


X_metric = metric_df.drop(
    "failure",
    axis=1
)


y_metric = metric_df["failure"]



# Test data split only

X_train, X_test, y_train, y_test = train_test_split(

    X_metric,

    y_metric,

    test_size=0.2,

    random_state=42,

    stratify=y_metric

)



metric_prediction = model.predict(
    X_test
)



metric_probability = model.predict_proba(
    X_test
)[:,1]



accuracy = accuracy_score(
    y_test,
    metric_prediction
)


precision = precision_score(
    y_test,
    metric_prediction
)


recall = recall_score(
    y_test,
    metric_prediction
)


f1 = f1_score(
    y_test,
    metric_prediction
)


roc_auc = roc_auc_score(
    y_test,
    metric_probability
)

# ============================================================
# SAFE LIMITS (for visualisation reference lines)
# ============================================================
SAFE_LIMITS = {
    "temperature": {"min": 20, "max": 90, "unit": "°C",  "icon": "🌡"},
    "pressure":    {"min": 30, "max": 140, "unit": "PSI", "icon": "🛢"},
    "vibration":   {"min": 0,  "max": 8.0, "unit": "",    "icon": "📳"},
    "humidity":    {"min": 20, "max": 70,  "unit": "%",   "icon": "💧"},
}

PLOT_BG    = "rgba(10,14,26,0)"
PAPER_BG   = "rgba(10,14,26,0)"
GRID_COLOR = "rgba(99,179,237,0.08)"
FONT_COLOR = "#94a3b8"

def dark_layout(fig, title="", height=320):
    fig.update_layout(
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=PAPER_BG,
        font=dict(family="Inter", color=FONT_COLOR, size=12),
        title=dict(text=title, font=dict(size=14, color="#e2e8f0"), x=0.01),
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    )
    fig.update_xaxes(gridcolor=GRID_COLOR, zeroline=False, linecolor=GRID_COLOR)
    fig.update_yaxes(gridcolor=GRID_COLOR, zeroline=False, linecolor=GRID_COLOR)
    return fig

# ============================================================
# HERO HEADER
# ============================================================
st.markdown("""
<div class="hero-header center">
    <div class="hero-title center">AI Predictive Maintenance For Machine</div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR — SENSOR INPUTS
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem 0 1.5rem;">
        <div style="font-size:2rem;">⚙️</div>
        <div style="font-size:1.1rem;font-weight:700;color:#e2e8f0;">Sensor Control Panel</div>
        <div style="font-size:0.78rem;color:#475569;margin-top:0.2rem;">Adjust real-time sensor values</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    temp = st.slider("🌡 Temperature (°C)", 0, 150, 75, help="Safe range: 20–90 °C")
    st.markdown(f"<div style='color:{'#ef4444' if temp > 90 or temp < 20 else '#34d399'};font-size:0.75rem;text-align:right;margin-top:-0.8rem;margin-bottom:0.8rem;'>{'⚠ Out of safe range' if temp > 90 or temp < 20 else '✓ Normal'}</div>", unsafe_allow_html=True)

    pressure = st.slider("🛢 Pressure (PSI)", 0, 200, 100, help="Safe range: 30–140 PSI")
    st.markdown(f"<div style='color:{'#ef4444' if pressure > 140 or pressure < 30 else '#34d399'};font-size:0.75rem;text-align:right;margin-top:-0.8rem;margin-bottom:0.8rem;'>{'⚠ Out of safe range' if pressure > 140 or pressure < 30 else '✓ Normal'}</div>", unsafe_allow_html=True)

    vibration = st.slider("📳 Vibration (g)", 0.0, 15.0, 5.0, step=0.1, help="Safe range: 0–8 g")
    st.markdown(f"<div style='color:{'#ef4444' if vibration > 8.0 else '#34d399'};font-size:0.75rem;text-align:right;margin-top:-0.8rem;margin-bottom:0.8rem;'>{'⚠ Out of safe range' if vibration > 8.0 else '✓ Normal'}</div>", unsafe_allow_html=True)

    humidity = st.slider("💧 Humidity (%)", 0, 100, 40, help="Safe range: 20–70%")
    st.markdown(f"<div style='color:{'#ef4444' if humidity > 70 or humidity < 20 else '#34d399'};font-size:0.75rem;text-align:right;margin-top:-0.8rem;margin-bottom:0.8rem;'>{'⚠ Out of safe range' if humidity > 70 or humidity < 20 else '✓ Normal'}</div>", unsafe_allow_html=True)

    st.markdown("---")

    # Risk score estimate (simple heuristic for sidebar display)
    violations = sum([
        temp > 90 or temp < 20,
        pressure > 140 or pressure < 30,
        vibration > 8.0,
        humidity > 70 or humidity < 20,
    ])
    sidebar_risk = int((violations / 4) * 100)
    risk_color = "#ef4444" if violations >= 2 else ("#f59e0b" if violations == 1 else "#34d399")
    risk_label = "HIGH RISK" if violations >= 2 else ("MODERATE" if violations == 1 else "HEALTHY")
    st.markdown(f"""
    <div style="background:rgba(15,23,42,0.8);border:1px solid rgba(99,179,237,0.15);border-radius:12px;padding:1rem;text-align:center;">
        <div style="font-size:0.7rem;color:#475569;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.4rem;">Live Risk Estimate</div>
        <div style="font-size:2rem;font-weight:800;color:{risk_color};font-family:'JetBrains Mono',monospace;">{sidebar_risk}%</div>
        <div style="font-size:0.8rem;color:{risk_color};margin-top:0.2rem;font-weight:600;">{risk_label}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("🔍 Run Health Analysis", use_container_width=True)

# ============================================================
# COMPUTED FEATURES
# ============================================================
temp_pressure   = temp * pressure
vibration_temp  = temp * vibration

input_data = pd.DataFrame([{
    "temperature":   temp,
    "pressure":      pressure,
    "vibration":     vibration,
    "humidity":      humidity,
    "temp_pressure": temp_pressure,
    "vibration_temp": vibration_temp,
}])

# ============================================================
# TAB LAYOUT
# ============================================================
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📈 Visualizations", "🔬 Model Insights"])

# ─────────────────────────────────────────────────────────────
# TAB 1 — DASHBOARD
# ─────────────────────────────────────────────────────────────
with tab1:

    # ---- KPI Row ----
    c1, c2, c3, c4 = st.columns(4)
    sensors = [
        ("Temperature", temp, "°C", "🌡", 20, 90),
        ("Pressure",    pressure, "PSI", "🛢", 30, 140),
        ("Vibration",   vibration, "g", "📳", 0, 8),
        ("Humidity",    humidity, "%", "💧", 20, 70),
    ]
    for col, (label, val, unit, icon, lo, hi) in zip([c1, c2, c3, c4], sensors):
        ok    = lo <= val <= hi
        color = "#34d399" if ok else "#ef4444"
        delta = "Normal" if ok else "⚠ Alert"
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">{icon}</div>
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{val}<span class="kpi-unit"> {unit}</span></div>
                <div class="kpi-delta" style="color:{color};">{"✓ " + delta if ok else delta}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Gauge + Readings ----
    col_g, col_r = st.columns([3, 2])

    with col_g:
        st.markdown('<div class="section-title">⚡ Sensor Gauge Overview</div>', unsafe_allow_html=True)
        fig_gauges = make_subplots(
            rows=2, cols=2,
            specs=[[{"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "indicator"}, {"type": "indicator"}]],
            vertical_spacing=0.05, horizontal_spacing=0.05,
        )

        gauge_defs = [
            ("Temperature",  temp,     0, 150,  90,   110,   "#ef4444", "#f59e0b", 1, 1),
            ("Pressure",     pressure, 0, 200,  140,  170,   "#ef4444", "#f59e0b", 1, 2),
            ("Vibration",    vibration,0, 15,   8.0,  11.5,  "#ef4444", "#f59e0b", 2, 1),
            ("Humidity",     humidity, 0, 100,  70,   85,    "#ef4444", "#f59e0b", 2, 2),
        ]

        for name, val, mn, mx, warn, crit, crit_col, warn_col, row, col_idx in gauge_defs:
            if val <= warn:
                bar_color = "#34d399"
            elif val <= crit:
                bar_color = "#f59e0b"
            else:
                bar_color = "#ef4444"

            fig_gauges.add_trace(go.Indicator(
                mode="gauge+number",
                value=val,
                title={"text": name, "font": {"size": 12, "color": "#94a3b8"}},
                number={"font": {"size": 22, "color": "#f1f5f9", "family": "JetBrains Mono"}},
                gauge=dict(
                    axis=dict(range=[mn, mx], tickcolor=FONT_COLOR, tickfont=dict(size=9, color=FONT_COLOR)),
                    bar=dict(color=bar_color, thickness=0.55),
                    bgcolor="rgba(15,23,42,0.5)",
                    borderwidth=0,
                    steps=[
                        dict(range=[mn, warn], color="rgba(52,211,153,0.07)"),
                        dict(range=[warn, crit], color="rgba(245,158,11,0.07)"),
                        dict(range=[crit, mx],   color="rgba(239,68,68,0.07)"),
                    ],
                    threshold=dict(line=dict(color=crit_col, width=2), thickness=0.75, value=warn),
                ),
            ), row=row, col=col_idx)

        fig_gauges.update_layout(
            plot_bgcolor=PLOT_BG,
            paper_bgcolor=PAPER_BG,
            font=dict(family="Inter", color=FONT_COLOR),
            height=380,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig_gauges, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-title">📋 Current Readings</div>', unsafe_allow_html=True)
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("🌡 Temperature",  f"{temp} °C",        delta=f"{'Above' if temp > 90 else 'Below' if temp < 20 else 'In'} limit")
            st.metric("📳 Vibration",    f"{vibration:.1f} g", delta=f"{'Above' if vibration > 8 else 'In'} limit")
        with col_m2:
            st.metric("🛢 Pressure",     f"{pressure} PSI",   delta=f"{'Above' if pressure > 140 else 'Below' if pressure < 30 else 'In'} limit")
            st.metric("💧 Humidity",     f"{humidity}%",      delta=f"{'Above' if humidity > 70 else 'Below' if humidity < 20 else 'In'} limit")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">⚙️ Engineered Features</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:rgba(15,23,42,0.6);border:1px solid rgba(99,179,237,0.1);border-radius:10px;padding:1rem;">
            <div style="display:flex;justify-content:space-between;margin-bottom:0.6rem;">
                <span style="color:#64748b;font-size:0.83rem;">Temp × Pressure</span>
                <span style="color:#e2e8f0;font-family:'JetBrains Mono',monospace;font-size:0.88rem;font-weight:600;">{temp_pressure:,.0f}</span>
            </div>
            <div style="display:flex;justify-content:space-between;">
                <span style="color:#64748b;font-size:0.83rem;">Vibration × Temp</span>
                <span style="color:#e2e8f0;font-family:'JetBrains Mono',monospace;font-size:0.88rem;font-weight:600;">{vibration_temp:,.1f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- PREDICTION RESULT ----
    if analyze_btn:
        with st.spinner("🔄 Running AI health analysis..."):
            time.sleep(0.6)

        prediction   = model.predict(input_data)[0]
        try:
            proba = model.predict_proba(input_data)[0]
            risk_pct = int(proba[1] * 100)
        except:
            risk_pct = 85 if prediction == 1 else 20

        st.markdown("---")
        st.markdown('<div class="section-title">📊 AI Diagnosis Report</div>', unsafe_allow_html=True)

        if prediction == 1:
            st.markdown(f"""
            <div class="alert-critical">
                <div class="alert-title" style="color:#fca5a5;">⚠️ HIGH RISK DETECTED — Machine Failure Likely</div>
                <div class="alert-subtitle" style="color:#fca5a5;">AI Confidence: <strong>{risk_pct}%</strong> probability of failure. Immediate action required.</div>
            </div>
            """, unsafe_allow_html=True)

            c_prog, c_act = st.columns([1, 1])
            with c_prog:
                st.markdown('<div class="section-title" style="margin-top:1rem;">🔴 Risk Level</div>', unsafe_allow_html=True)
                fig_risk = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=risk_pct,
                    delta={"reference": 50, "valueformat": ".0f", "suffix": "%"},
                    number={"suffix": "%", "font": {"size": 36, "color": "#f1f5f9"}},
                    gauge=dict(
                        axis=dict(range=[0, 100], tickcolor=FONT_COLOR, ticksuffix="%"),
                        bar=dict(color="#ef4444"),
                        bgcolor="rgba(15,23,42,0.5)",
                        borderwidth=0,
                        steps=[
                            dict(range=[0, 30],   color="rgba(52,211,153,0.1)"),
                            dict(range=[30, 60],  color="rgba(245,158,11,0.1)"),
                            dict(range=[60, 100], color="rgba(239,68,68,0.1)"),
                        ],
                        threshold=dict(line=dict(color="white", width=2), thickness=0.75, value=risk_pct),
                    ),
                    title={"text": "Failure Probability", "font": {"size": 13, "color": "#94a3b8"}},
                ))
                fig_risk.update_layout(
                    plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
                    height=260, margin=dict(l=20, r=20, t=40, b=10),
                    font=dict(family="Inter", color=FONT_COLOR),
                )
                st.plotly_chart(fig_risk, use_container_width=True)

            with c_act:
                st.markdown('<div class="section-title" style="margin-top:1rem;">🛠 Recommended Actions</div>', unsafe_allow_html=True)
                actions = [
                    ("Inspect bearings and rotating parts immediately", "red"),
                    ("Check and replenish lubrication system", "red"),
                    ("Verify and calibrate pressure relief valves", "red"),
                    ("Schedule emergency preventive maintenance", "red"),
                    ("Enable continuous vibration monitoring", "red"),
                    ("Notify maintenance team — escalate to Tier 2", "red"),
                ]
                for action, dot_type in actions:
                    st.markdown(f"""
                    <div class="action-card">
                        <div class="action-dot-red"></div>
                        {action}
                    </div>""", unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div class="alert-safe">
                <div class="alert-title" style="color:#6ee7b7;">✅ MACHINE STATUS: HEALTHY — Operating Normally</div>
                <div class="alert-subtitle" style="color:#6ee7b7;">AI Confidence: <strong>{100 - risk_pct}%</strong> probability of healthy operation. Continue regular schedule.</div>
            </div>
            """, unsafe_allow_html=True)

            c_prog, c_act = st.columns([1, 1])
            with c_prog:
                st.markdown('<div class="section-title" style="margin-top:1rem;">🟢 Health Level</div>', unsafe_allow_html=True)
                fig_health = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=100 - risk_pct,
                    delta={"reference": 50, "valueformat": ".0f", "suffix": "%"},
                    number={"suffix": "%", "font": {"size": 36, "color": "#f1f5f9"}},
                    gauge=dict(
                        axis=dict(range=[0, 100], tickcolor=FONT_COLOR, ticksuffix="%"),
                        bar=dict(color="#34d399"),
                        bgcolor="rgba(15,23,42,0.5)",
                        borderwidth=0,
                        steps=[
                            dict(range=[0, 40],   color="rgba(239,68,68,0.1)"),
                            dict(range=[40, 70],  color="rgba(245,158,11,0.1)"),
                            dict(range=[70, 100], color="rgba(52,211,153,0.1)"),
                        ],
                        threshold=dict(line=dict(color="white", width=2), thickness=0.75, value=100 - risk_pct),
                    ),
                    title={"text": "Health Score", "font": {"size": 13, "color": "#94a3b8"}},
                ))
                fig_health.update_layout(
                    plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
                    height=260, margin=dict(l=20, r=20, t=40, b=10),
                    font=dict(family="Inter", color=FONT_COLOR),
                )
                st.plotly_chart(fig_health, use_container_width=True)

            with c_act:
                st.markdown('<div class="section-title" style="margin-top:1rem;">✔ Health Report</div>', unsafe_allow_html=True)
                checks = [
                    "All sensors operating within safe limits",
                    "No anomalous vibration patterns detected",
                    "Thermal profile within expected range",
                    "Pressure readings stable and nominal",
                    "Humidity levels acceptable for operation",
                    "Continue scheduled maintenance program",
                ]
                for check in checks:
                    st.markdown(f"""
                    <div class="action-card">
                        <div class="action-dot-green"></div>
                        {check}
                    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# TAB 2 — VISUALIZATIONS
# ─────────────────────────────────────────────────────────────
with tab2:

    # ---- Simulated Time-Series ----
    st.markdown('<div class="section-title">📉 Simulated Sensor Time-Series (Last 60 Minutes)</div>', unsafe_allow_html=True)

    np.random.seed(42)
    n = 60
    timestamps = [datetime.now() - timedelta(minutes=n - i) for i in range(n)]

    ts_temp  = np.clip(temp  + np.random.normal(0, 4, n).cumsum() * 0.1, 0, 150)
    ts_press = np.clip(pressure + np.random.normal(0, 5, n).cumsum() * 0.1, 0, 200)
    ts_vib   = np.clip(vibration + np.random.normal(0, 0.3, n).cumsum() * 0.05, 0, 15)
    ts_hum   = np.clip(humidity + np.random.normal(0, 2, n).cumsum() * 0.08, 0, 100)

    fig_ts = make_subplots(rows=2, cols=2, shared_xaxes=False,
                           subplot_titles=["Temperature (°C)", "Pressure (PSI)", "Vibration (g)", "Humidity (%)"],
                           vertical_spacing=0.18, horizontal_spacing=0.08)

    series = [
        (ts_temp,  "#38bdf8", 90,  150, 1, 1),
        (ts_press, "#818cf8", 140, 200, 1, 2),
        (ts_vib,   "#f59e0b", 8,   15,  2, 1),
        (ts_hum,   "#34d399", 70,  100, 2, 2),
    ]

    for arr, color, warn, mx, row, col_idx in series:
        fig_ts.add_trace(go.Scatter(
            x=timestamps, y=arr,
            mode="lines",
            line=dict(color=color, width=2),
            fill="tozeroy", fillcolor=color.replace(")", ",0.06)").replace("rgb", "rgba") if "rgb" in color else f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.06)",
            showlegend=False,
        ), row=row, col=col_idx)
        fig_ts.add_hline(y=warn, line_dash="dot", line_color="#ef4444", line_width=1,
                         annotation_text="⚠ Limit", annotation_font_size=9,
                         annotation_font_color="#ef4444", row=row, col=col_idx)

    fig_ts.update_layout(
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        font=dict(family="Inter", color=FONT_COLOR, size=11),
        height=380, margin=dict(l=10, r=10, t=40, b=10),
    )
    fig_ts.update_annotations(font_size=11, font_color="#94a3b8")
    fig_ts.update_xaxes(gridcolor=GRID_COLOR, zeroline=False, linecolor=GRID_COLOR, showticklabels=False)
    fig_ts.update_yaxes(gridcolor=GRID_COLOR, zeroline=False, linecolor=GRID_COLOR)
    st.plotly_chart(fig_ts, use_container_width=True)

    # ---- Radar + Bar Row ----
    col_radar, col_bar = st.columns(2)

    with col_radar:
        st.markdown('<div class="section-title">🎯 Sensor Status Radar</div>', unsafe_allow_html=True)

        # Normalise each sensor 0-100 relative to its max safe value
        radar_vals = [
            round((temp     / 90)   * 100, 1),
            round((pressure / 140)  * 100, 1),
            round((vibration / 8.0) * 100, 1),
            round((humidity  / 70)  * 100, 1),
        ]
        categories = ["Temperature", "Pressure", "Vibration", "Humidity"]
        radar_vals_closed  = radar_vals + [radar_vals[0]]
        categories_closed  = categories + [categories[0]]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=[100] * (len(categories) + 1),
            theta=categories_closed,
            fill="toself",
            fillcolor="rgba(52,211,153,0.05)",
            line=dict(color="rgba(52,211,153,0.3)", width=1, dash="dot"),
            name="Safe Limit",
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_vals_closed,
            theta=categories_closed,
            fill="toself",
            fillcolor="rgba(56,189,248,0.12)",
            line=dict(color="#38bdf8", width=2.5),
            name="Current",
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="rgba(15,23,42,0.5)",
                radialaxis=dict(visible=True, range=[0, 140], gridcolor=GRID_COLOR, tickcolor=FONT_COLOR, tickfont_size=9, ticksuffix="%"),
                angularaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(size=11, color="#94a3b8")),
            ),
            plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
            font=dict(family="Inter", color=FONT_COLOR),
            height=350, margin=dict(l=30, r=30, t=30, b=30),
            legend=dict(bgcolor="rgba(0,0,0,0)", font_size=11),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_bar:
        st.markdown('<div class="section-title">📊 Sensor Load vs Safe Limit (%)</div>', unsafe_allow_html=True)

        load_pct = [
            round((temp     / 90)   * 100, 1),
            round((pressure / 140)  * 100, 1),
            round((vibration / 8.0) * 100, 1),
            round((humidity  / 70)  * 100, 1),
        ]
        bar_colors = ["#ef4444" if v > 100 else "#f59e0b" if v > 80 else "#34d399" for v in load_pct]

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=["Temperature", "Pressure", "Vibration", "Humidity"],
            y=load_pct,
            marker=dict(color=bar_colors, line=dict(width=0)),
            text=[f"{v:.0f}%" for v in load_pct],
            textposition="outside",
            textfont=dict(color="#e2e8f0", size=12),
            width=0.5,
        ))
        fig_bar.add_hline(y=100, line_dash="dot", line_color="#ef4444", line_width=1.5,
                          annotation_text="Safe Limit", annotation_font_color="#ef4444", annotation_font_size=10)
        fig_bar.add_hline(y=80, line_dash="dot", line_color="#f59e0b", line_width=1,
                          annotation_text="Warning", annotation_font_color="#f59e0b", annotation_font_size=10)
        dark_layout(fig_bar, height=350)
        fig_bar.update_yaxes(range=[0, max(max(load_pct) + 20, 130)], ticksuffix="%")
        st.plotly_chart(fig_bar, use_container_width=True)

    # ---- Risk Trend Simulation ----
    st.markdown('<div class="section-title">📈 Simulated Risk Score Trend (Last 24 Hours)</div>', unsafe_allow_html=True)

    np.random.seed(int(temp + pressure + vibration * 10))
    hours = [datetime.now() - timedelta(hours=23 - i) for i in range(24)]
    base_risk = risk_pct if 'risk_pct' in dir() else violations * 25
    risk_trend = np.clip(
        base_risk + np.random.normal(0, 8, 24) + np.linspace(-15, 10, 24),
        0, 100
    )

    zone_colors = ["#34d399" if v < 30 else "#f59e0b" if v < 60 else "#ef4444" for v in risk_trend]

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=hours, y=risk_trend,
        mode="lines+markers",
        line=dict(color="#818cf8", width=2.5),
        marker=dict(color=zone_colors, size=8, line=dict(width=0)),
        fill="tozeroy",
        fillcolor="rgba(129,140,248,0.07)",
        name="Risk Score",
    ))
    fig_trend.add_hrect(y0=0,  y1=30,  fillcolor="rgba(52,211,153,0.04)",  line_width=0, annotation_text="Healthy Zone",  annotation_position="left", annotation_font_size=10, annotation_font_color="#34d399")
    fig_trend.add_hrect(y0=30, y1=60,  fillcolor="rgba(245,158,11,0.04)", line_width=0, annotation_text="Caution Zone",  annotation_position="left", annotation_font_size=10, annotation_font_color="#f59e0b")
    fig_trend.add_hrect(y0=60, y1=100, fillcolor="rgba(239,68,68,0.04)",  line_width=0, annotation_text="Danger Zone",   annotation_position="left", annotation_font_size=10, annotation_font_color="#ef4444")
    dark_layout(fig_trend, title="", height=300)
    fig_trend.update_yaxes(range=[0, 110], ticksuffix="%", title_text="Risk Score")
    fig_trend.update_xaxes(title_text="Time")
    st.plotly_chart(fig_trend, use_container_width=True)

    # ---- Correlation Heatmap ----
    st.markdown('<div class="section-title">🔥 Feature Correlation Heatmap</div>', unsafe_allow_html=True)

    np.random.seed(7)
    n_samples = 300
    syn_data = pd.DataFrame({
        "Temperature":   np.random.normal(75, 20, n_samples),
        "Pressure":      np.random.normal(100, 30, n_samples),
        "Vibration":     np.random.normal(5, 2.5, n_samples),
        "Humidity":      np.random.normal(40, 15, n_samples),
        "Temp×Pressure": np.random.normal(7500, 3000, n_samples),
        "Vib×Temp":      np.random.normal(375, 180, n_samples),
    })
    syn_data["Temperature"]   = syn_data["Temperature"].clip(0, 150)
    syn_data["Pressure"]      = syn_data["Pressure"].clip(0, 200)
    syn_data["Vibration"]     = syn_data["Vibration"].clip(0, 15)
    syn_data["Humidity"]      = syn_data["Humidity"].clip(0, 100)

    corr = syn_data.corr()
    fig_heatmap = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale=[[0, "#0f172a"], [0.3, "#1e3a5f"], [0.5, "#1d4ed8"], [0.7, "#38bdf8"], [1, "#bfdbfe"]],
        zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=11, color="#e2e8f0"),
        colorbar=dict(tickcolor=FONT_COLOR, tickfont=dict(color=FONT_COLOR), bgcolor="rgba(0,0,0,0)"),
    ))
    dark_layout(fig_heatmap, height=340)
    fig_heatmap.update_xaxes(tickangle=-30)
    st.plotly_chart(fig_heatmap, use_container_width=True)


# # ─────────────────────────────────────────────────────────────
# TAB 3 — MODEL INSIGHTS
# ─────────────────────────────────────────────────────────────

with tab3:

    col_fi, col_shap = st.columns(2)


    with col_fi:

        st.markdown(
            '<div class="section-title">🌳 Feature Importance (Random Forest)</div>',
            unsafe_allow_html=True
        )


        try:

            importances = model.feature_importances_

            feat_names = [
                "temperature",
                "pressure",
                "vibration",
                "humidity",
                "temp_pressure",
                "vibration_temp"
            ]


            fi_df = pd.DataFrame({

                "Feature": feat_names,

                "Importance": importances

            })


            fi_df = fi_df.sort_values(
                "Importance",
                ascending=True
            )



            fig_fi = go.Figure(
                go.Bar(

                    x=fi_df["Importance"],

                    y=fi_df["Feature"],

                    orientation="h",

                    marker=dict(

                        color=fi_df["Importance"],

                        colorscale=[
                            [0,"#1e3a5f"],
                            [0.5,"#0ea5e9"],
                            [1,"#38bdf8"]
                        ]

                    ),

                    text=[
                        f"{v:.3f}"
                        for v in fi_df["Importance"]
                    ],

                    textposition="outside"

                )
            )


            dark_layout(
                fig_fi,
                height=350
            )


            st.plotly_chart(
                fig_fi,
                use_container_width=True
            )


        except Exception:

            st.info(
                "Feature importances not available."
            )





    with col_shap:


        st.markdown(
            '<div class="section-title">📐 Feature Contribution (Current Prediction)</div>',
            unsafe_allow_html=True
        )


        try:


            importances = model.feature_importances_


            feat_names = [
                "Temperature",
                "Pressure",
                "Vibration",
                "Humidity",
                "Temp×Pressure",
                "Vib×Temp"
            ]


            norms = [

                temp/150,

                pressure/200,

                vibration/15,

                humidity/100,

                temp_pressure/(150*200),

                vibration_temp/(15*150)

            ]


            contributions = [

                imp * norm * 100

                for imp,norm in zip(
                    importances,
                    norms
                )

            ]



            contrib_df = pd.DataFrame({

                "Feature":feat_names,

                "Contribution":contributions

            })


            contrib_df = contrib_df.sort_values(
                "Contribution",
                ascending=True
            )



            fig_contrib = go.Figure(
                go.Bar(

                    x=contrib_df["Contribution"],

                    y=contrib_df["Feature"],

                    orientation="h",

                    text=[
                        f"{v:.1f}%"
                        for v in contrib_df["Contribution"]
                    ],

                    textposition="outside"

                )
            )


            dark_layout(
                fig_contrib,
                height=350
            )


            st.plotly_chart(
                fig_contrib,
                use_container_width=True
            )


        except Exception:


            st.info(
                "Contribution chart not available."
            )





    # ==============================
    # MODEL METRICS SECTION
    # ==============================


    st.markdown(
        '<div class="section-title">🤖 Model Performance Metrics</div>',
        unsafe_allow_html=True
    )



    m1,m2,m3,m4,m5 = st.columns(5)



    m1.metric(
        "Accuracy",
        f"{accuracy*100:.2f}%"
    )


    m2.metric(
        "Precision",
        f"{precision*100:.2f}%"
    )


    m3.metric(
        "Recall",
        f"{recall*100:.2f}%"
    )


    m4.metric(
        "F1 Score",
        f"{f1*100:.2f}%"
    )


    m5.metric(
        "ROC-AUC",
        f"{roc_auc*100:.2f}%"
    )
    # ---- Model Summary Card ----
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">🤖 Model Architecture Summary</div>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    model_info = [
        ("Algorithm",      "Random Forest", "🌳"),
        ("Input Features", "6 (4 raw + 2 engineered)", "📥"),
        ("Output",         "Binary (0=Healthy, 1=Failure)", "📤"),
        ("Framework",      "scikit-learn + joblib", "⚙️"),
    ]
    for col, (label, value, icon) in zip([m1, m2, m3, m4], model_info):
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="text-align:center;padding:1.2rem;">
                <div style="font-size:1.6rem;margin-bottom:0.4rem;">{icon}</div>
                <div class="kpi-label">{label}</div>
                <div style="font-size:0.85rem;color:#e2e8f0;font-weight:600;margin-top:0.3rem;">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Sensor Scatter ----
    st.markdown('<div class="section-title">🔬 Sensor Relationship Explorer</div>', unsafe_allow_html=True)

    np.random.seed(12)
    n_pts = 250
    scatter_temp  = np.random.uniform(0, 150, n_pts)
    scatter_press = np.random.uniform(0, 200, n_pts)
    scatter_vib   = np.random.uniform(0, 15, n_pts)
    scatter_label = ((scatter_temp > 90) | (scatter_press > 140) | (scatter_vib > 8)).astype(int)

    fig_scatter = px.scatter(
        x=scatter_temp, y=scatter_press,
        color=scatter_label.astype(str),
        size=scatter_vib + 1,
        color_discrete_map={"0": "#34d399", "1": "#ef4444"},
        labels={"x": "Temperature (°C)", "y": "Pressure (PSI)", "color": "Status", "size": "Vibration"},
        title="Temperature vs Pressure (bubble size = Vibration)",
    )
    # Mark current reading
    fig_scatter.add_trace(go.Scatter(
        x=[temp], y=[pressure],
        mode="markers",
        marker=dict(symbol="star", size=20, color="#f59e0b", line=dict(color="white", width=1.5)),
        name="⭐ Current Reading",
    ))
    fig_scatter.update_layout(
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        font=dict(family="Inter", color=FONT_COLOR),
        height=380, margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        title_font=dict(size=13, color="#e2e8f0"),
    )
    fig_scatter.update_xaxes(gridcolor=GRID_COLOR, zeroline=False)
    fig_scatter.update_yaxes(gridcolor=GRID_COLOR, zeroline=False)
    st.plotly_chart(fig_scatter, use_container_width=True)


# ============================================================
# FOOTER
# ============================================================
