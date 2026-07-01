"""
Smart Grid Load Predictor — Streamlit Deployment
=================================================
Deploys a trained XGBoost model to predict active power (kW) for
a smart electrical grid.  The app enforces the *exact* feature order
the model saw during training so XGBoost never raises a ValueError.
Visual stack: Plotly (gauge + radar + bar), custom CSS glassmorphism,
animated metric cards, dark‑neon theme.
"""
# ──────────────────────────── Imports ────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime, time
# ──────────────────────────── Page Config ────────────────────────
st.set_page_config(
    page_title="Smart Grid Load Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)
# ──────────────────────────── Custom CSS ─────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;900&display=swap');
/* ── Root Variables ── */
:root {
    --bg-primary:    #0a0e1a;
    --bg-card:       rgba(15, 23, 42, 0.65);
    --glass-border:  rgba(99, 179, 237, 0.15);
    --accent-cyan:   #22d3ee;
    --accent-violet: #a78bfa;
    --accent-green:  #34d399;
    --accent-amber:  #fbbf24;
    --text-primary:  #e2e8f0;
    --text-muted:    #94a3b8;
    --glow-cyan:     0 0 25px rgba(34, 211, 238, 0.25);
    --glow-violet:   0 0 25px rgba(167, 139, 250, 0.25);
}
/* ── Global ── */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stSidebar"] {
    background: linear-gradient(195deg, #0f172a 0%, #1e1b4b 100%) !important;
    border-right: 1px solid var(--glass-border) !important;
}
[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}
/* ── Glass Card ── */
.glass-card {
    background: var(--bg-card);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--glass-border);
    border-radius: 18px;
    padding: 28px 24px;
    transition: transform 0.35s cubic-bezier(.22,1,.36,1),
                box-shadow 0.35s cubic-bezier(.22,1,.36,1);
}
.glass-card:hover {
    transform: translateY(-4px);
    box-shadow: var(--glow-cyan);
}
/* ── Metric card ── */
.metric-card {
    text-align: center;
    padding: 22px 16px;
}
.metric-icon { font-size: 2rem; margin-bottom: 6px; }
.metric-label {
    font-size: 0.78rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--text-muted);
    margin-bottom: 4px;
}
.metric-value {
    font-size: 1.65rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-violet));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
/* ── Hero header ── */
.hero {
    text-align: center;
    padding: 36px 0 18px;
}
.hero h1 {
    font-size: 2.6rem;
    font-weight: 900;
    letter-spacing: -1px;
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-violet), var(--accent-green));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 6px;
}
.hero p {
    font-size: 1.05rem;
    color: var(--text-muted);
    max-width: 620px;
    margin: 0 auto;
    line-height: 1.6;
}
/* ── Prediction result ── */
.result-box {
    text-align: center;
    padding: 32px;
    border-radius: 22px;
    background: linear-gradient(135deg,
        rgba(34,211,238,0.08) 0%,
        rgba(167,139,250,0.08) 100%);
    border: 1px solid rgba(34,211,238,0.25);
    animation: resultPulse 2.5s ease-in-out infinite;
}
@keyframes resultPulse {
    0%, 100% { box-shadow: 0 0 20px rgba(34,211,238,0.10); }
    50%      { box-shadow: 0 0 45px rgba(34,211,238,0.22); }
}
.result-label {
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: var(--text-muted);
    margin-bottom: 8px;
}
.result-value {
    font-size: 3.4rem;
    font-weight: 900;
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-green));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.result-unit {
    font-size: 1.3rem;
    font-weight: 600;
    color: var(--accent-cyan);
}
.result-tag {
    display: inline-block;
    margin-top: 14px;
    padding: 6px 18px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 1px;
}
.tag-low    { background: rgba(52,211,153,0.15); color: var(--accent-green); border: 1px solid rgba(52,211,153,0.3); }
.tag-medium { background: rgba(251,191,36,0.15); color: var(--accent-amber); border: 1px solid rgba(251,191,36,0.3); }
.tag-high   { background: rgba(248,113,113,0.15); color: #f87171;            border: 1px solid rgba(248,113,113,0.3); }
/* ── Sidebar section labels ── */
.sidebar-section {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: var(--accent-cyan);
    margin: 22px 0 8px;
    padding-bottom: 4px;
    border-bottom: 1px solid var(--glass-border);
}
/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-violet)) !important;
    color: #0a0e1a !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 0 !important;
    width: 100% !important;
    font-size: 1.05rem !important;
    letter-spacing: 0.5px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(34,211,238,0.25) !important;
}
.stButton > button:hover {
    transform: scale(1.02) !important;
    box-shadow: 0 6px 30px rgba(34,211,238,0.4) !important;
}
/* ── Streamlit overrides ── */
[data-testid="stHeader"]  { background: transparent !important; }
div[data-baseweb="input"] input,
div[data-baseweb="select"] div {
    background: rgba(15,23,42,0.8) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 10px !important;
}
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: var(--accent-cyan) !important;
}
label, .stSelectbox label, .stSlider label, .stNumberInput label,
.stTimeInput label {
    color: var(--text-primary) !important;
    font-weight: 500 !important;
}
/* ── Divider ── */
.neon-divider {
    height: 2px;
    background: linear-gradient(90deg,
        transparent, var(--accent-cyan), var(--accent-violet), transparent);
    margin: 30px 0;
    border: none;
    border-radius: 2px;
}
</style>
""", unsafe_allow_html=True)
# ──────────────────────────── Load Model ─────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(path: str = "xgboost_model.pkl"):
    """Load the trained XGBoost model once and cache it."""
    return joblib.load(path)
model = load_model()
# ──────────────────────────── Constants ──────────────────────────
# Exact feature order used during training in Colab.
# Changing this list or its order will cause a ValueError.
FEATURE_ORDER = [
    "avg_voltage",
    "minute",
    "hour",
    "dayofweek",
    "is_weekend",
    "lag_1_power",
    "lag_2_power",
    "rolling_mean_3",
]
DAY_NAMES = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday",
]
# ──────────────────────────── Helpers ────────────────────────────
def classify_load(kw: float) -> tuple[str, str, str]:
    """Return (label, css‑class, emoji) based on predicted kW."""
    if kw < 1.5:
        return "LOW DEMAND", "tag-low", "🟢"
    elif kw < 3.5:
        return "MODERATE DEMAND", "tag-medium", "🟡"
    return "HIGH DEMAND", "tag-high", "🔴"
def build_gauge(value: float, max_val: float = 6.0) -> go.Figure:
    """Plotly gauge with neon dark theme."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(value, 3),
        number={"font": {"size": 52, "color": "#22d3ee", "family": "Inter"},
                "suffix": " kW"},
        gauge={
            "axis": {
                "range": [0, max_val],
                "tickwidth": 2,
                "tickcolor": "#334155",
                "dtick": max_val / 6,
                "tickfont": {"size": 11, "color": "#64748b"},
            },
            "bar": {"color": "#22d3ee", "thickness": 0.35},
            "bgcolor": "rgba(15,23,42,0.6)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, max_val * 0.25],  "color": "rgba(52,211,153,0.12)"},
                {"range": [max_val * 0.25, max_val * 0.58], "color": "rgba(251,191,36,0.10)"},
                {"range": [max_val * 0.58, max_val], "color": "rgba(248,113,113,0.10)"},
            ],
            "threshold": {
                "line": {"color": "#a78bfa", "width": 3},
                "thickness": 0.8,
                "value": value,
            },
        },
    ))
    fig.update_layout(
        height=280,
        margin=dict(t=30, b=10, l=30, r=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"},
    )
    return fig
def build_feature_radar(data: dict) -> go.Figure:
    """Radar chart showing normalized input features."""
    labels = list(data.keys())
    raw    = list(data.values())
    # Normalize to 0‑1 for visual comparison
    arr   = np.array(raw, dtype=float)
    lo, hi = arr.min(), arr.max()
    norm  = ((arr - lo) / (hi - lo + 1e-9)).tolist()
    norm += [norm[0]]          # close the polygon
    labels_closed = labels + [labels[0]]
    fig = go.Figure()
    # Filled area
    fig.add_trace(go.Scatterpolar(
        r=norm,
        theta=labels_closed,
        fill="toself",
        fillcolor="rgba(34,211,238,0.12)",
        line=dict(color="#22d3ee", width=2),
        hovertemplate="%{theta}: %{customdata:.4f}<extra></extra>",
        customdata=raw + [raw[0]],
        name="Features",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True, range=[0, 1],
                gridcolor="rgba(99,179,237,0.08)",
                tickfont=dict(size=0),     # hide tick labels
            ),
            angularaxis=dict(
                gridcolor="rgba(99,179,237,0.10)",
                tickfont=dict(size=11, color="#94a3b8", family="Inter"),
            ),
        ),
        showlegend=False,
        height=340,
        margin=dict(t=40, b=40, l=60, r=60),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter"),
    )
    return fig
def build_feature_bar(data: dict) -> go.Figure:
    """Horizontal bar chart of raw feature values."""
    labels = list(data.keys())
    values = list(data.values())
    colors = [
        "#22d3ee", "#a78bfa", "#34d399", "#fbbf24",
        "#f87171", "#818cf8", "#fb923c", "#38bdf8",
    ]
    fig = go.Figure(go.Bar(
        y=labels,
        x=values,
        orientation="h",
        marker=dict(
            color=colors[: len(labels)],
            line=dict(width=0),
            cornerradius=6,
        ),
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        height=340,
        margin=dict(t=20, b=20, l=10, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            gridcolor="rgba(99,179,237,0.06)",
            tickfont=dict(color="#64748b", size=11, family="Inter"),
            title=None,
        ),
        yaxis=dict(
            tickfont=dict(color="#94a3b8", size=12, family="Inter"),
            title=None,
        ),
        font=dict(family="Inter"),
    )
    return fig
# ──────────────────────────── Sidebar ────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:10px 0 0;">
        <span style="font-size:2.6rem;">⚡</span>
        <h2 style="margin:4px 0 0; font-weight:800;
            background:linear-gradient(135deg,#22d3ee,#a78bfa);
            -webkit-background-clip:text;
            -webkit-text-fill-color:transparent;">
            Grid Predictor
        </h2>
        <p style="font-size:0.82rem; color:#64748b; margin-top:2px;">
            Real‑time power demand forecasting
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)
    # ── Electrical readings ──
    st.markdown('<p class="sidebar-section">⚙️ Electrical Readings</p>',
                unsafe_allow_html=True)
    avg_voltage = st.slider(
        "Average Voltage (V)",
        min_value=200.0, max_value=260.0, value=230.0, step=0.5,
        help="Mean voltage recorded by the smart meter.",
    )
    # ── Temporal features ──
    st.markdown('<p class="sidebar-section">🕒 Temporal Features</p>',
                unsafe_allow_html=True)
    input_time = st.time_input("Time of Day", value=time(14, 30))
    day_of_week = st.selectbox(
        "Day of the Week",
        options=list(range(7)),
        format_func=lambda d: f"{DAY_NAMES[d]}  {'🏖️' if d >= 5 else '🏢'}",
        index=0,
    )
    is_weekend = 1 if day_of_week >= 5 else 0
    # ── Lag features ──
    st.markdown('<p class="sidebar-section">📈 Lag Features</p>',
                unsafe_allow_html=True)
    lag_1 = st.number_input("Lag‑1 Power (kW)", min_value=0.0,
                            max_value=12.0, value=1.2, step=0.01,
                            format="%.3f")
    lag_2 = st.number_input("Lag‑2 Power (kW)", min_value=0.0,
                            max_value=12.0, value=1.1, step=0.01,
                            format="%.3f")
    rolling_mean = st.number_input("Rolling Mean‑3 (kW)", min_value=0.0,
                                   max_value=12.0, value=1.15, step=0.01,
                                   format="%.3f")
    st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <p style="text-align:center; font-size:0.7rem; color:#475569;
              margin-top:4px;">
        Model: XGBoost · Features: 8 · Target: Active Power (kW)
    </p>
    """, unsafe_allow_html=True)
# ──────────────────────────── Main Area ──────────────────────────
# Hero header
st.markdown("""
<div class="hero">
    <h1>⚡ Smart Grid Load Predictor</h1>
    <p>
        Leverage machine‑learning to forecast real‑time active power demand.
        Configure the features on the left, then hit <strong>Predict</strong>
        to see the estimated load and visual analytics.
    </p>
</div>
<div class="neon-divider"></div>
""", unsafe_allow_html=True)
# ── Input summary cards ──
card_data = [
    ("🔌", "Voltage",      f"{avg_voltage:.1f} V"),
    ("🕐", "Time",         f"{input_time.strftime('%H:%M')}"),
    ("📅", "Day",          DAY_NAMES[day_of_week]),
    ("📊", "Rolling Mean", f"{rolling_mean:.3f} kW"),
]
cols = st.columns(4, gap="medium")
for col, (icon, label, value) in zip(cols, card_data):
    col.markdown(f"""
    <div class="glass-card metric-card">
        <div class="metric-icon">{icon}</div>
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
# ── Predict Button ──
if st.button("🔮  Predict Grid Load", use_container_width=True):
    # 1. Build raw feature dict
    raw_features = {
        "avg_voltage":   avg_voltage,
        "minute":        input_time.minute,
        "hour":          input_time.hour,
        "dayofweek":     day_of_week,
        "is_weekend":    is_weekend,
        "lag_1_power":   lag_1,
        "lag_2_power":   lag_2,
        "rolling_mean_3": rolling_mean,
    }
    # 2. Create DataFrame and enforce the exact training column order
    input_df = pd.DataFrame({k: [v] for k, v in raw_features.items()})
    input_df = input_df[FEATURE_ORDER]
    try:
        # 3. Run prediction
        prediction: float = float(model.predict(input_df)[0])
        # 4. Classify demand level
        demand_label, demand_class, demand_emoji = classify_load(prediction)
        # ── Result card ──
        st.markdown(f"""
        <div class="glass-card result-box">
            <div class="result-label">Predicted Active Power</div>
            <div class="result-value">{prediction:.3f}</div>
            <div class="result-unit">kilowatts</div>
            <div>
                <span class="result-tag {demand_class}">
                    {demand_emoji}  {demand_label}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        # ── Visual analytics row ──
        g_col, r_col = st.columns(2, gap="medium")
        with g_col:
            st.markdown("""
            <div class="glass-card" style="padding:16px 12px 8px;">
                <p style="text-align:center; font-size:0.78rem;
                   text-transform:uppercase; letter-spacing:1.5px;
                   color:#94a3b8; margin-bottom:0;">
                   Power Gauge
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.plotly_chart(build_gauge(prediction),
                           use_container_width=True, config={"displayModeBar": False})
        with r_col:
            st.markdown("""
            <div class="glass-card" style="padding:16px 12px 8px;">
                <p style="text-align:center; font-size:0.78rem;
                   text-transform:uppercase; letter-spacing:1.5px;
                   color:#94a3b8; margin-bottom:0;">
                   Feature Radar
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.plotly_chart(build_feature_radar(raw_features),
                           use_container_width=True, config={"displayModeBar": False})
        # ── Feature bar chart ──
        st.markdown("""
        <div class="glass-card" style="padding:16px 20px 8px; margin-top:8px;">
            <p style="text-align:center; font-size:0.78rem;
               text-transform:uppercase; letter-spacing:1.5px;
               color:#94a3b8; margin-bottom:0;">
               Feature Values Breakdown
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(build_feature_bar(raw_features),
                       use_container_width=True, config={"displayModeBar": False})
        # ── Actionable insight ──
        if prediction < 1.5:
            insight = ("💡 **Low demand detected.** "
                       "The grid can safely reduce generation or store "
                       "surplus energy in batteries.")
        elif prediction < 3.5:
            insight = ("⚙️ **Moderate demand.** "
                       "Current generation capacity is adequate. "
                       "Monitor for potential spikes.")
        else:
            insight = ("🚨 **High demand alert!** "
                       "Consider activating peaking plants or demand‑response "
                       "programs to avoid blackouts.")
        st.markdown(f"""
        <div class="glass-card" style="margin-top:16px; padding:20px 24px;
             border-left: 3px solid var(--accent-cyan);">
            <p style="font-size:0.78rem; text-transform:uppercase;
               letter-spacing:1.5px; color:#94a3b8; margin-bottom:8px;">
               Grid Recommendation
            </p>
            <p style="font-size:0.95rem; color:#e2e8f0; line-height:1.6;">
                {insight}
            </p>
        </div>
        """, unsafe_allow_html=True)
    except Exception as exc:
        st.error(f"**Prediction Error:** {exc}")
# ── Footer ──
st.markdown("""
<div class="neon-divider"></div>
<p style="text-align:center; font-size:0.72rem; color:#334155;
          padding-bottom:16px;">
    Powered by XGBoost · Deployed on Streamlit ·
    Model trained on household power consumption data
</p>
""", unsafe_allow_html=True)
