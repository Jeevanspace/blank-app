"""
AGNISIGHT — Intelligent Flight Anomaly Detection & Post-Flight Analysis System
Agnikul Cosmos | Agnibaan ST | Agnilet SN-07
IsolationForest ML Engine | v3.0
Simulated Telemetry: 10 Hz | T+0 to T+300s | All canonical channels
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AGNISIGHT | Agnikul Cosmos",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
#  THEME  — Clean professional palette
#  Background: near-white (#f8f9fb)
#  Surface:    white (#ffffff)
#  Primary:    deep navy (#0d1f3c)
#  Accent:     engineering blue (#1a56db)
#  Warning:    amber (#d97706)
#  Alert:      red (#dc2626)
#  Success:    green (#16a34a)
#  Border:     #e2e8f0
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: #f0f2f6 !important;
    color: #0d1f3c !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
}
[data-testid="stSidebar"] * { color: #0d1f3c !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stTextInput label { color: #475569 !important; font-size: 12px !important; }

.app-header {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #1a56db;
    border-radius: 6px;
    padding: 20px 28px;
    margin-bottom: 20px;
}
.app-title {
    font-size: 26px; font-weight: 700; color: #0d1f3c;
    letter-spacing: 1px; line-height: 1.2;
}
.app-subtitle { font-size: 12px; color: #64748b; margin-top: 4px; letter-spacing: 0.5px; }

.status-badge {
    display: inline-block; padding: 3px 12px; border-radius: 4px;
    font-size: 11px; font-weight: 600; font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.5px;
}
.badge-ok  { background: #dcfce7; color: #15803d; border: 1px solid #86efac; }
.badge-warn{ background: #fef9c3; color: #92400e; border: 1px solid #fde047; }
.badge-alert{ background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }

.kpi-card {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 6px;
    padding: 16px 18px; position: relative;
}
.kpi-card-alert { border-left: 3px solid #dc2626; }
.kpi-card-warn  { border-left: 3px solid #d97706; }
.kpi-card-ok    { border-left: 3px solid #16a34a; }
.kpi-card-blue  { border-left: 3px solid #1a56db; }
.kpi-label { font-size: 10px; font-weight: 600; color: #64748b;
             text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
.kpi-value { font-size: 24px; font-weight: 700; color: #0d1f3c;
             font-family: 'JetBrains Mono', monospace; line-height: 1; }
.kpi-unit  { font-size: 11px; color: #94a3b8; margin-top: 3px; }

.section-title {
    font-size: 11px; font-weight: 600; color: #475569;
    text-transform: uppercase; letter-spacing: 1.5px;
    padding-bottom: 8px; border-bottom: 1px solid #e2e8f0;
    margin-bottom: 16px;
}

.anomaly-row {
    background: #fff5f5; border: 1px solid #fecaca;
    border-left: 3px solid #dc2626; border-radius: 4px;
    padding: 12px 16px; margin: 6px 0;
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
}
.anomaly-row-ok {
    background: #f0fdf4; border: 1px solid #bbf7d0;
    border-left: 3px solid #16a34a; border-radius: 4px;
    padding: 12px 16px; margin: 6px 0;
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
}
.anom-time  { color: #d97706; font-size: 10px; }
.anom-msg   { color: #991b1b; margin-top: 3px; font-size: 12px; font-weight: 600; }
.anom-ch    { color: #64748b; font-size: 10px; margin-top: 2px; }

.info-box {
    background: #ffffff; border: 1px solid #e2e8f0;
    border-radius: 6px; padding: 16px 18px; margin: 4px 0;
    font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #334155;
    line-height: 1.9;
}
.report-box {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 6px;
    padding: 20px 24px; font-family: 'JetBrains Mono', monospace;
    font-size: 11px; color: #334155; line-height: 1.8;
    white-space: pre-wrap; overflow-x: auto;
}
.nl-answer {
    background: #ffffff; border: 1px solid #bfdbfe;
    border-left: 3px solid #1a56db; border-radius: 6px;
    padding: 16px 20px; font-size: 14px; color: #0d1f3c;
    line-height: 1.8; margin-top: 12px;
}
.nl-answer .hl { color: #1a56db; font-family: 'JetBrains Mono', monospace; font-weight: 600; }

.stTabs [data-baseweb="tab-list"] {
    background: #ffffff !important;
    border-bottom: 1px solid #e2e8f0 !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 12px !important; font-weight: 500 !important;
    color: #64748b !important; background: transparent !important;
    border: none !important; padding: 10px 18px !important;
}
.stTabs [aria-selected="true"] {
    color: #1a56db !important;
    border-bottom: 2px solid #1a56db !important;
    font-weight: 600 !important;
}
.stButton > button {
    background: #1a56db !important; color: #ffffff !important;
    border: none !important; border-radius: 5px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important; font-weight: 600 !important;
    padding: 8px 20px !important; transition: background 0.2s !important;
}
.stButton > button:hover { background: #1e40af !important; }
.stSelectbox > div > div, .stTextInput > div > div {
    background: #ffffff !important; border-color: #e2e8f0 !important;
    color: #0d1f3c !important; font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
}
div[data-testid="metric-container"] {
    background: #ffffff !important; border: 1px solid #e2e8f0 !important;
    border-radius: 6px !important; padding: 14px !important;
}
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #f1f5f9; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
#  PLOT THEME  — clean white charts
# ─────────────────────────────────────────────────────────────────
AXIS = dict(
    gridcolor="#f1f5f9",
    linecolor="#e2e8f0",
    tickfont=dict(color="#94a3b8", size=9, family="JetBrains Mono"),
    zerolinecolor="#e2e8f0",
)
LAYOUT = dict(
    paper_bgcolor="rgba(255,255,255,0)",
    plot_bgcolor="#ffffff",
    font=dict(family="Inter", color="#64748b", size=10),
    legend=dict(
        bgcolor="rgba(255,255,255,0.95)",
        bordercolor="#e2e8f0", borderwidth=1,
        font=dict(color="#334155", size=9),
    ),
    margin=dict(l=48, r=16, t=36, b=32),
)

BLUE  = "#1a56db"
RED   = "#dc2626"
AMBER = "#d97706"
GREEN = "#16a34a"
TEAL  = "#0891b2"
SLATE = "#64748b"

def base_fig(height=None, title="", rows=1, cols=1, **kw):
    fig = make_subplots(rows=rows, cols=cols, **kw) if (rows > 1 or cols > 1) else go.Figure()
    lo = dict(**LAYOUT)
    if height: lo["height"] = height
    if title:  lo["title"] = dict(text=title, font=dict(color="#475569", size=11), x=0.01)
    fig.update_layout(**lo)
    fig.update_xaxes(**AXIS)
    fig.update_yaxes(**AXIS)
    return fig


# ─────────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────────
for k, v in [
    ("flight_data", None), ("anomalies", []), ("model", None),
    ("scaler", None), ("report", None), ("engine_history", None),
    ("flight_mode", "Nominal"), ("mission_id", "AGN-FL-001"),
    ("csv_df", None), ("csv_col_map", {}), ("data_source", "simulate"),
]:
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────
#  CSV → TELEMETRY ADAPTER
#  Maps user CSV columns → canonical telemetry column names.
#  Any missing canonical column is filled with a sensible default.
# ─────────────────────────────────────────────────────────────────
# All canonical columns the dashboard expects
ALL_COLS = [
    "time_s", "phase",
    "chamber_pres", "thrust_kN", "fuel_flow_kgs", "ox_flow_kgs",
    "total_flow_kgs", "OF_ratio", "nozzle_temp_C", "turbo_rpm",
    "lox_tank_pres", "fuel_tank_pres",
    "altitude_km", "altitude_m", "vel_v_ms", "vel_h_ms",
    "pitch_deg", "yaw_deg", "roll_deg",
    "p_rate_degs", "q_rate_degs", "r_rate_degs",
    "accel_x", "accel_y", "accel_z",
    "mach", "dyn_pres_Pa",
    "bus_voltage_V", "cpu_temp_C", "signal_dbm",
]

COL_DEFAULTS = {
    "phase": 2, "chamber_pres": 65.0, "thrust_kN": 98.0,
    "fuel_flow_kgs": 15.0, "ox_flow_kgs": 30.0, "total_flow_kgs": 45.0,
    "OF_ratio": 2.10, "nozzle_temp_C": 1800.0, "turbo_rpm": 31000.0,
    "lox_tank_pres": 34.0, "fuel_tank_pres": 31.0,
    "altitude_km": 0.0, "altitude_m": 0.0, "vel_v_ms": 0.0, "vel_h_ms": 0.0,
    "pitch_deg": 90.0, "yaw_deg": 0.0, "roll_deg": 0.0,
    "p_rate_degs": 0.0, "q_rate_degs": 0.0, "r_rate_degs": 0.0,
    "accel_x": 0.0, "accel_y": 0.0, "accel_z": 0.0,
    "mach": 0.0, "dyn_pres_Pa": 0.0,
    "bus_voltage_V": 28.0, "cpu_temp_C": 48.0, "signal_dbm": -75.0,
}

# Fuzzy alias map — common alternate names → canonical
ALIAS_MAP = {
    # time
    "time": "time_s", "t": "time_s", "timestamp": "time_s", "t_s": "time_s",
    "elapsed": "time_s", "elapsed_s": "time_s",
    # pressure
    "pc": "chamber_pres", "chamber_pressure": "chamber_pres",
    "combustion_pres": "chamber_pres", "chamber_press": "chamber_pres",
    "pc_bar": "chamber_pres",
    # thrust
    "thrust": "thrust_kN", "thrust_kn": "thrust_kN", "force_kn": "thrust_kN",
    # flows
    "fuel_flow": "fuel_flow_kgs", "fuel_mass_flow": "fuel_flow_kgs",
    "ox_flow": "ox_flow_kgs", "lox_flow": "ox_flow_kgs",
    "oxidiser_flow": "ox_flow_kgs", "oxidizer_flow": "ox_flow_kgs",
    "total_flow": "total_flow_kgs", "mass_flow": "total_flow_kgs",
    # OF ratio
    "of": "OF_ratio", "mixture_ratio": "OF_ratio", "of_ratio": "OF_ratio",
    # temperature
    "nozzle_temp": "nozzle_temp_C", "throat_temp": "nozzle_temp_C",
    "nozzle_temperature": "nozzle_temp_C",
    # RPM
    "rpm": "turbo_rpm", "turbopump_rpm": "turbo_rpm", "turbo": "turbo_rpm",
    # tank pressures
    "lox_pres": "lox_tank_pres", "lox_pressure": "lox_tank_pres",
    "fuel_pres": "fuel_tank_pres", "fuel_pressure": "fuel_tank_pres",
    # nav
    "altitude": "altitude_km", "alt_km": "altitude_km", "alt": "altitude_km",
    "alt_m": "altitude_m", "altitude_m": "altitude_m",
    "velocity": "vel_v_ms", "vel": "vel_v_ms", "velocity_ms": "vel_v_ms",
    "mach_number": "mach", "mach_no": "mach",
    "dynamic_pressure": "dyn_pres_Pa", "q": "dyn_pres_Pa",
    # attitude
    "pitch": "pitch_deg", "yaw": "yaw_deg", "roll": "roll_deg",
    # rates
    "pitch_rate": "q_rate_degs", "yaw_rate": "r_rate_degs",
    "roll_rate": "p_rate_degs",
    # accel
    "ax": "accel_x", "ay": "accel_y", "az": "accel_z",
    "accel_axial": "accel_x",
    # avionics
    "bus_voltage": "bus_voltage_V", "voltage": "bus_voltage_V",
    "vbus": "bus_voltage_V",
    "cpu_temp": "cpu_temp_C", "cpu_temperature": "cpu_temp_C",
    "signal": "signal_dbm", "rssi": "signal_dbm", "rssi_dbm": "signal_dbm",
    # phase
    "flight_phase": "phase", "mission_phase": "phase",
}


def auto_map_columns(csv_cols):
    """Return dict: canonical_col → csv_col for best-match columns."""
    mapping = {}
    # Build a lowercase lookup: lower_canonical → canonical (preserves original case)
    all_cols_lower = {c.lower(): c for c in ALL_COLS}
    csv_lower = {c.lower().strip().replace(" ", "_"): c for c in csv_cols}
    for csv_key, csv_original in csv_lower.items():
        # Case-insensitive exact match against canonical names
        if csv_key in all_cols_lower:
            canonical = all_cols_lower[csv_key]
            mapping[canonical] = csv_original
        elif csv_key in ALIAS_MAP:
            mapping[ALIAS_MAP[csv_key]] = csv_original
    return mapping   # canonical → csv_col


def csv_to_telemetry(raw_df: pd.DataFrame, col_map: dict, mission_id: str = "CSV-001") -> pd.DataFrame:
    """
    Convert user CSV to canonical telemetry DataFrame.
    col_map: {canonical_col: csv_col}
    """
    n = len(raw_df)
    out = {}

    for canon in ALL_COLS:
        if canon in col_map:
            out[canon] = pd.to_numeric(raw_df[col_map[canon]], errors="coerce").fillna(COL_DEFAULTS.get(canon, 0)).values
        else:
            out[canon] = np.full(n, COL_DEFAULTS.get(canon, 0), dtype=float)

    # Derive time_s if missing but row index is the only option
    if "time_s" not in col_map:
        out["time_s"] = np.arange(n) / SAMPLE_RATE

    # Auto-derive phase from time if phase not in CSV
    if "phase" not in col_map:
        t = out["time_s"]
        phase = np.zeros(n, dtype=int)
        phase[t >= 2]   = 1
        phase[t >= 8]   = 2
        phase[t >= 130] = 3
        phase[t >= 135] = 4
        phase[t >= 200] = 5
        out["phase"] = phase

    # Derive altitude_m from altitude_km if altitude_m missing
    if "altitude_m" not in col_map:
        out["altitude_m"] = out["altitude_km"] * 1000.0
    if "altitude_km" not in col_map and "altitude_m" in col_map:
        out["altitude_km"] = out["altitude_m"] / 1000.0

    df = pd.DataFrame(out)
    df["phase"] = df["phase"].astype(int)
    return df


# ─────────────────────────────────────────────────────────────────
#  TELEMETRY GENERATOR
#  Physics-accurate Agnibaan ST simulation
#  Channels: all canonical propulsion + GNC + avionics
#  T=0 → T+300s  @ 10 Hz  → 3000 samples per channel
# ─────────────────────────────────────────────────────────────────
SAMPLE_RATE = 10  # Hz

def _smooth(t, t0, t1, v0=0.0, v1=1.0):
    """Smooth-step ramp between t0 and t1."""
    x = np.clip((t - t0) / max(t1 - t0, 1e-9), 0.0, 1.0)
    s = 3 * x**2 - 2 * x**3          # cubic Hermite
    return v0 + (v1 - v0) * s


def generate_telemetry(duration_s: int = 300, mode: str = "Nominal", seed: int = 42) -> pd.DataFrame:
    """
    Simulate rocket telemetry for an Agnibaan ST suborbital profile.

    Flight phases
    ─────────────
    Phase 0  Pre-launch       T <  2 s
    Phase 1  Ignition         2 ≤ T <  8 s
    Phase 2  Powered Ascent   8 ≤ T < 130 s  (Agnilet SN-07 burn)
    Phase 3  MECO coast-out 130 ≤ T < 135 s
    Phase 4  Coast            135 ≤ T < 200 s
    Phase 5  Stage Sep        T ≥ 200 s
    """
    rng = np.random.default_rng(seed)
    n   = duration_s * SAMPLE_RATE
    t   = np.linspace(0, duration_s, n)

    # ── Phase assignment ──────────────────────────────────────────
    phase = np.zeros(n, dtype=int)
    phase[t >= 2]   = 1
    phase[t >= 8]   = 2
    phase[t >= 130] = 3
    phase[t >= 135] = 4
    phase[t >= 200] = 5
    powered = phase == 2

    # ── Chamber pressure (bar)  nominal 60–80 bar ─────────────────
    Pc = (
        _smooth(t, 2, 8, 0, 68)
        + np.where(powered, np.sin(0.05 * t) * 1.4 + rng.normal(0, 0.22, n), 0)
    )
    Pc = np.where(phase < 1, 0.4 + rng.normal(0, 0.03, n), Pc)
    Pc = np.where(phase == 3, _smooth(t, 130, 135, 68, 0.3), Pc)
    Pc = np.where(phase >= 4, rng.normal(0.25, 0.03, n), Pc)
    Pc = np.clip(Pc, 0, None)

    # ── Thrust (kN)  Agnilet design ~100 kN ──────────────────────
    thrust = Pc * 1.47 + rng.normal(0, 0.35, n)
    thrust = np.clip(thrust, 0, None)

    # ── Propellant flows (kg/s)  OF ratio design 2.10 ────────────
    OF_ratio    = np.where(powered, 2.10 + rng.normal(0, 0.013, n), 2.10)
    total_flow  = np.clip(thrust * 1000 / (315 * 9.81), 0, None)   # Isp≈315 s
    fuel_flow   = total_flow / (1 + OF_ratio)
    ox_flow     = total_flow - fuel_flow

    # ── Nozzle throat temperature (°C)  limit 1950 °C ────────────
    nozzle_T = (
        _smooth(t, 2, 10, 22, 1835)
        + np.where(powered, np.sin(0.03 * t) * 22 + rng.normal(0, 6, n), 0)
    )
    nozzle_T = np.where(phase < 1, 22 + rng.normal(0, 0.3, n), nozzle_T)
    nozzle_T = np.where(phase == 3, _smooth(t, 130, 148, 1835, 170), nozzle_T)
    nozzle_T = np.where(phase >= 4, 140 + rng.normal(0, 2.5, n), nozzle_T)

    # ── Turbopump RPM ─────────────────────────────────────────────
    turbo_rpm = np.where(phase >= 1, 31800 * (Pc / 68.0) + rng.normal(0, 110, n), 0.0)
    turbo_rpm = np.clip(turbo_rpm, 0, None)

    # ── Tank pressures (bar) ──────────────────────────────────────
    lox_p  = np.where(phase < 1, 35.0, 35.0 - total_flow * 0.08) + rng.normal(0, 0.11, n)
    fuel_p = np.where(phase < 1, 32.0, 32.0 - total_flow * 0.06) + rng.normal(0, 0.09, n)
    lox_p  = np.clip(lox_p, 0, None)
    fuel_p = np.clip(fuel_p, 0, None)

    # ── Axial acceleration (m/s²) ─────────────────────────────────
    g0        = 9.80665
    mass_init = 560.0      # kg (Agnibaan Stage 1 rough estimate)
    mass      = np.clip(mass_init - total_flow * t, 200, mass_init)
    accel_x   = np.where(powered, thrust * 1000 / mass - g0 + rng.normal(0, 0.25, n), 0.0)
    accel_y   = rng.normal(0, 0.15, n)
    accel_z   = rng.normal(0, 0.15, n)

    # ── Velocity (m/s) via cumulative integration ─────────────────
    vel_v = np.cumsum(np.maximum(accel_x, 0)) / SAMPLE_RATE
    vel_h = np.cumsum(np.abs(accel_y))         / SAMPLE_RATE * 0.55

    # ── Altitude (m) ──────────────────────────────────────────────
    altitude_m  = np.cumsum(vel_v) / SAMPLE_RATE
    altitude_m  = np.clip(altitude_m, 0, None)
    altitude_km = altitude_m / 1000.0

    # ── Mach number ───────────────────────────────────────────────
    sos  = np.maximum(340 - altitude_km * 4, 282)   # rough ISA
    mach = np.clip((vel_v + vel_h) / sos, 0, 10)

    # ── Dynamic pressure (Pa) ─────────────────────────────────────
    rho     = 1.225 * np.exp(-altitude_km / 8.5)    # ISA exponential
    dyn_pres = 0.5 * rho * (vel_v + vel_h) ** 2

    # ── GNC Euler angles (deg) ────────────────────────────────────
    pitch = np.where(powered,
        90 - _smooth(t, 8, 130, 0, 45) + np.sin(0.014 * t) * 1.5 + rng.normal(0, 0.06, n),
        0.0)
    yaw   = np.where(powered, np.cos(0.011 * t) * 1.0 + rng.normal(0, 0.06, n), 0.0)
    roll  = np.where(powered, np.sin(0.008 * t) * 0.35 + rng.normal(0, 0.03, n), 0.0)

    # ── Angular rates (deg/s) ─────────────────────────────────────
    dt    = 1.0 / SAMPLE_RATE
    p_rate = np.gradient(roll,  dt) + rng.normal(0, 0.015, n)
    q_rate = np.gradient(pitch, dt) + rng.normal(0, 0.015, n)
    r_rate = np.gradient(yaw,   dt) + rng.normal(0, 0.015, n)

    # ── Avionics / Power ──────────────────────────────────────────
    bus_V    = 28.0 + rng.normal(0, 0.035, n)
    cpu_T    = 45.0 + np.where(powered, 7 * (Pc / 68) + rng.normal(0, 0.4, n),
                                        rng.normal(0, 0.25, n))
    signal   = -72.0 - altitude_km * 0.16 + rng.normal(0, 0.7, n)

    # ── Build DataFrame ───────────────────────────────────────────
    df = pd.DataFrame({
        "time_s":         t,
        "phase":          phase,
        # Propulsion
        "chamber_pres":   Pc,
        "thrust_kN":      thrust,
        "fuel_flow_kgs":  fuel_flow,
        "ox_flow_kgs":    ox_flow,
        "total_flow_kgs": total_flow,
        "OF_ratio":       OF_ratio,
        "nozzle_temp_C":  nozzle_T,
        "turbo_rpm":      turbo_rpm,
        "lox_tank_pres":  lox_p,
        "fuel_tank_pres": fuel_p,
        # GNC
        "altitude_km":    altitude_km,
        "altitude_m":     altitude_m,
        "vel_v_ms":       vel_v,
        "vel_h_ms":       vel_h,
        "pitch_deg":      pitch,
        "yaw_deg":        yaw,
        "roll_deg":       roll,
        "p_rate_degs":    p_rate,
        "q_rate_degs":    q_rate,
        "r_rate_degs":    r_rate,
        "accel_x":        accel_x,
        "accel_y":        accel_y,
        "accel_z":        accel_z,
        "mach":           mach,
        "dyn_pres_Pa":    dyn_pres,
        # Avionics
        "bus_voltage_V":  bus_V,
        "cpu_temp_C":     cpu_T,
        "signal_dbm":     signal,
    })

    # ── Inject fault modes ────────────────────────────────────────
    if mode == "Pressure Spike":
        idx = (t >= 47) & (t <= 52)
        df.loc[idx, "chamber_pres"]  *= 1.30
        df.loc[idx, "thrust_kN"]     *= 1.24
        df.loc[idx, "nozzle_temp_C"] += 145

    elif mode == "TVC Divergence":
        idx  = (t >= 85) & (t <= 102)
        ramp = np.linspace(0, 7.0, int(idx.sum()))
        df.loc[idx, "pitch_deg"]   += ramp
        df.loc[idx, "yaw_deg"]     += ramp * 0.55
        df.loc[idx, "q_rate_degs"] += np.gradient(ramp, dt)

    elif mode == "Turbopump Cavitation":
        idx = (t >= 62) & (t <= 71)
        ns  = int(idx.sum())
        df.loc[idx, "turbo_rpm"]      *= rng.uniform(0.68, 0.80, ns)
        df.loc[idx, "total_flow_kgs"] *= 0.74
        df.loc[idx, "fuel_flow_kgs"]  *= 0.74
        df.loc[idx, "ox_flow_kgs"]    *= 0.74
        df.loc[idx, "chamber_pres"]   *= 0.78
        df.loc[idx, "thrust_kN"]      *= 0.76

    elif mode == "Thermal Runaway":
        idx  = (t >= 105) & (t <= 116)
        ramp = np.linspace(0, 360, int(idx.sum()))
        df.loc[idx, "nozzle_temp_C"] += ramp

    elif mode == "Power Anomaly":
        idx = (t >= 30) & (t <= 36)
        ns  = int(idx.sum())
        df.loc[idx, "bus_voltage_V"] -= rng.uniform(1.9, 3.4, ns)
        df.loc[idx, "cpu_temp_C"]    += 6.5
        df.loc[idx, "signal_dbm"]    -= 9.0

    elif mode == "OF Ratio Drift":
        idx   = (t >= 75) & (t <= 100)
        drift = np.linspace(0, 0.48, int(idx.sum()))
        df.loc[idx, "OF_ratio"]     += drift
        df.loc[idx, "ox_flow_kgs"]  *= (1 + drift / 2.1)

    return df


# ─────────────────────────────────────────────────────────────────
#  ISOLATION FOREST ENGINE
# ─────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "chamber_pres", "thrust_kN", "total_flow_kgs", "OF_ratio",
    "nozzle_temp_C", "turbo_rpm", "lox_tank_pres", "fuel_tank_pres",
    "bus_voltage_V", "pitch_deg", "yaw_deg", "accel_x", "dyn_pres_Pa",
]


def train_model(nominal_df: pd.DataFrame):
    phase2 = nominal_df[nominal_df["phase"] == 2][FEATURE_COLS].fillna(0)
    scaler = StandardScaler()
    X = scaler.fit_transform(phase2)
    model = IsolationForest(
        n_estimators=300, max_samples="auto",
        contamination=0.015, bootstrap=False,
        n_jobs=-1, random_state=42,
    )
    model.fit(X)
    return model, scaler


def detect_anomalies(df: pd.DataFrame, model, scaler):
    X_all   = scaler.transform(df[FEATURE_COLS].fillna(0))
    scores  = model.score_samples(X_all)
    preds   = model.predict(X_all)
    powered = (df["phase"] == 2).values

    df = df.copy()
    df["if_score"]     = scores
    df["anomaly_flag"] = (preds == -1) & powered

    events, in_evt, t_start, dom_ch = [], False, 0.0, FEATURE_COLS[0]
    for i in range(len(df)):
        flag = bool(df["anomaly_flag"].iloc[i])
        if flag and not in_evt:
            in_evt  = True
            t_start = float(df["time_s"].iloc[i])
            z       = np.abs(X_all[i])
            dom_ch  = FEATURE_COLS[int(np.argmax(z))]
        elif not flag and in_evt:
            in_evt = False
            t_end  = float(df["time_s"].iloc[i])
            score  = float(df["if_score"].iloc[i])
            sev    = "HIGH" if score < -0.25 else "MEDIUM"
            events.append({
                "t_start":  t_start,
                "t_end":    t_end,
                "dur":      t_end - t_start,
                "channel":  dom_ch,
                "severity": sev,
                "score":    round(score, 4),
            })
    return df, events


# ─────────────────────────────────────────────────────────────────
#  ENGINE HEALTH HISTORY
# ─────────────────────────────────────────────────────────────────
def generate_engine_health(n_fires: int = 8) -> pd.DataFrame:
    rng  = np.random.default_rng(7)
    rows = []
    for i in range(1, n_fires + 1):
        deg = (i - 1) * 1.45 + rng.uniform(-0.20, 0.20)
        rows.append({
            "fire_id":        f"SF-{i:02d}" if i <= 6 else f"FL-{i-6:02d}",
            "type":           "Static Fire" if i <= 6 else "Flight",
            "duration_s":     int(120 + rng.integers(-4, 8)) if i <= 6 else int(130 + rng.integers(-2, 5)),
            "peak_Pc_bar":    round(67.8 + float(rng.uniform(-1.0, 1.0)) - (i - 1) * 0.14, 2),
            "avg_thrust_kN":  round(98.2 + float(rng.uniform(-1.2, 1.2)) - (i - 1) * 0.18, 2),
            "turbo_peak_rpm": int(31900 + rng.integers(-180, 180) - i * 28),
            "nozzle_peak_C":  int(1825 + rng.integers(-25, 25) + i * 4),
            "health_score":   round(max(100 - deg, 0), 1),
            "rul_cycles":     max(15 - i + int(rng.integers(0, 2)), 1),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────
#  CHANNEL METADATA
# ─────────────────────────────────────────────────────────────────
CH_DESC = {
    "chamber_pres":   "Combustion Chamber Pressure",
    "thrust_kN":      "Engine Thrust",
    "total_flow_kgs": "Propellant Mass Flow",
    "OF_ratio":       "Oxidiser-to-Fuel Mixture Ratio",
    "nozzle_temp_C":  "Nozzle Throat Temperature",
    "turbo_rpm":      "Turbopump Rotational Speed",
    "lox_tank_pres":  "LOX Tank Pressure",
    "fuel_tank_pres": "Fuel Tank Pressure",
    "bus_voltage_V":  "Avionics Bus Voltage",
    "pitch_deg":      "Vehicle Pitch Angle",
    "yaw_deg":        "Vehicle Yaw Angle",
    "accel_x":        "Axial Acceleration",
    "dyn_pres_Pa":    "Dynamic Pressure",
}

ROOT_CAUSE = {
    "chamber_pres":   "Injector partial blockage or propellant conditioning off-nominal — review PT-07 transducer",
    "thrust_kN":      "Combustion instability or nozzle erosion beyond model prediction",
    "total_flow_kgs": "Turbopump cavitation or pressurant regulator transient",
    "OF_ratio":       "Oxidiser flow regulator drift — mixture ratio moving toward fuel-rich boundary",
    "nozzle_temp_C":  "Ablative liner ablation rate exceeds thermal model — nozzle inspection mandatory",
    "turbo_rpm":      "Inducer cavitation due to NPSHa margin erosion — review LOX inlet conditions",
    "lox_tank_pres":  "LOX pressurant regulator surge — check COPV discharge valve",
    "fuel_tank_pres": "Fuel pressurant system transient",
    "bus_voltage_V":  "Battery sag under peak avionics load — review power budget and BMS",
    "pitch_deg":      "GNC control gain saturation or structural mode excitation at max-Q",
    "yaw_deg":        "Lateral disturbance torque exceeding control authority margin",
    "accel_x":        "Thrust vector misalignment or mass property uncertainty",
    "dyn_pres_Pa":    "Off-nominal trajectory — review GNC guidance law",
}

RECOMMENDATIONS = {
    "Nominal":              "  [1] Data nominal — clear for next static fire commit\n  [2] Archive fingerprint to Agnilet SN-07 health database\n  [3] Update nominal envelope model with this flight data",
    "Pressure Spike":       "  [1] Inspect injector orifice set for partial blockage (mandrel + borescope)\n  [2] Review LOX conditioning temperature at T-2hr\n  [3] High-speed PT-07 transducer data mandatory for review board\n  [4] Hold next flight commit pending board sign-off",
    "TVC Divergence":       "  [1] Actuator LVDT calibration — both pitch and yaw channels\n  [2] GNC loop gain schedule audit for max-Q regime (T+80–100s)\n  [3] Review structural mode coupling at 85s\n  [4] Flight dynamics team to re-evaluate control authority margins",
    "Turbopump Cavitation": "  [1] LOX inlet line inspection — check for vapour pockets and voids\n  [2] NPSHa calculation review at current conditioning protocol\n  [3] Turbopump inducer inspection for erosion marks\n  [4] Reduce LOX conditioning temperature by 3 K before next test",
    "Thermal Runaway":      "  [1] Nozzle throat 3D scan — erosion profile measurement\n  [2] Ablative liner thickness measurement at throat and exit cone\n  [3] Thermocouple TC-12/TC-13 health check\n  [4] Reduce firing duration 12% for next test",
    "Power Anomaly":        "  [1] Battery pack internal resistance measurement (4-wire method)\n  [2] Load profile audit — identify peak-draw subsystem\n  [3] Bus decoupling capacitor bank check\n  [4] Update pre-flight SoC verification procedure",
    "OF Ratio Drift":       "  [1] Oxidiser flow control valve calibration verification\n  [2] Review LOX dome filter for partial clogging\n  [3] Venturi meter calibration — ox flow measurement uncertainty\n  [4] Assess combustion stability margin for off-nominal OF range",
}


# ─────────────────────────────────────────────────────────────────
#  REPORT GENERATOR
# ─────────────────────────────────────────────────────────────────
def generate_report(df: pd.DataFrame, events: list, mode: str, mission_id: str) -> str:
    pwr  = df[df["phase"] == 2]
    t_meco = float(df[df["phase"] == 3]["time_s"].min()) if (df["phase"] == 3).any() else 130.0
    peak_alt  = df["altitude_km"].max()
    peak_mach = df["mach"].max()
    avg_pc    = pwr["chamber_pres"].mean()
    avg_thr   = pwr["thrust_kN"].mean()
    prop_used = pwr["total_flow_kgs"].sum() / SAMPLE_RATE
    Isp_est   = avg_thr * 1000 / (9.80665 * max(pwr["total_flow_kgs"].mean(), 0.01))
    now       = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    sep = "─" * 70
    r   = f"AGNISIGHT POST-FLIGHT ANALYSIS REPORT — AGNIKUL COSMOS\n{sep}\n"
    r  += f"Mission ID    : {mission_id}\n"
    r  += f"Vehicle       : Agnibaan ST — Two-Stage Semi-Cryogenic\n"
    r  += f"Engine        : Agnilet SN-07 (3D Printed, LOX / HTPB)\n"
    r  += f"Launchpad     : Dhanush, Sriharikota Launch Complex\n"
    r  += f"Generated UTC : {now}\n"
    r  += f"ML Engine     : Isolation Forest  n_estimators=300  contamination=1.5%  trained on uploaded CSV\n"
    r  += f"Features      : {len(FEATURE_COLS)} telemetry channels  |  Sample Rate: 10 Hz\n"
    r  += f"Flight Mode   : {mode}\n\n"

    r  += f"{sep}\n1. MISSION SUMMARY\n{sep}\n"
    r  += f"  Peak Altitude         : {peak_alt:.2f} km\n"
    r  += f"  Peak Mach             : {peak_mach:.2f}\n"
    r  += f"  MECO Time             : T+{t_meco:.1f} s\n"
    r  += f"  Total Propellant Used : {prop_used:.1f} kg\n"
    r  += f"  Overall Status        : {'ANOMALOUS — REVIEW REQUIRED' if events else 'NOMINAL — ALL SYSTEMS GREEN'}\n\n"

    r  += f"{sep}\n2. PROPULSION PERFORMANCE — AGNILET SN-07\n{sep}\n"
    r  += f"  Avg Chamber Pressure  : {avg_pc:.2f} bar   (Nominal: 68.0 bar)\n"
    r  += f"  Avg Thrust            : {avg_thr:.2f} kN   (Nominal: 100.0 kN)\n"
    r  += f"  Estimated Isp         : {Isp_est:.0f} s\n"
    r  += f"  Combustion Efficiency : {min(avg_pc / 68.0 * 100, 100):.1f}%\n"
    r  += f"  Peak Turbopump RPM    : {pwr['turbo_rpm'].max():.0f} RPM\n"
    r  += f"  Mean OF Ratio         : {pwr['OF_ratio'].mean():.3f}   (Design: 2.100)\n"
    r  += f"  Peak Nozzle Temp      : {pwr['nozzle_temp_C'].max():.0f} °C   (Limit: 1950 °C)\n\n"

    r  += f"{sep}\n3. GNC PERFORMANCE\n{sep}\n"
    r  += f"  Pitch RMS             : {pwr['pitch_deg'].std():.3f} deg\n"
    r  += f"  Yaw RMS               : {pwr['yaw_deg'].std():.3f} deg\n"
    r  += f"  Roll RMS              : {pwr['roll_deg'].std():.3f} deg\n"
    r  += f"  Peak Pitch Rate       : {df['q_rate_degs'].abs().max():.3f} deg/s\n"
    r  += f"  Peak Lateral Accel    : {max(df['accel_y'].abs().max(), df['accel_z'].abs().max()):.3f} m/s²\n"
    r  += f"  Guidance Status       : {'DEGRADED — TVC divergence detected' if mode == 'TVC Divergence' else 'NOMINAL'}\n\n"

    r  += f"{sep}\n4. ANOMALY TIMELINE\n{sep}\n"
    if not events:
        r += "  [CLEAR] No anomalies detected across all monitored channels.\n"
    else:
        for ev in events:
            rc = ROOT_CAUSE.get(ev["channel"], "unknown — manual review required")
            r += f"\n  [{ev['severity']}]  T+{ev['t_start']:.1f}s → T+{ev['t_end']:.1f}s  ({ev['dur']:.1f}s)\n"
            r += f"  Channel    : {CH_DESC.get(ev['channel'], ev['channel'])}\n"
            r += f"  IF Score   : {ev['score']}  (threshold ≈ −0.20)\n"
            r += f"  Hypothesis : {rc}\n"

    r  += f"\n{sep}\n5. AVIONICS & POWER\n{sep}\n"
    r  += f"  Bus Voltage Mean      : {df['bus_voltage_V'].mean():.3f} V  (Nominal: 28.0 V)\n"
    r  += f"  Bus Voltage Min       : {df['bus_voltage_V'].min():.3f} V\n"
    r  += f"  CPU Temperature Peak  : {df['cpu_temp_C'].max():.1f} °C\n"
    r  += f"  Min Signal Strength   : {df['signal_dbm'].min():.1f} dBm\n"
    r  += f"  Power Status          : {'WARNING — undervoltage event detected' if mode == 'Power Anomaly' else 'NOMINAL'}\n"

    r  += f"\n{sep}\n6. RECOMMENDATIONS\n{sep}\n"
    r  += RECOMMENDATIONS.get(mode, "  [1] Manual review by propulsion team required") + "\n"
    r  += f"\n{sep}\n"
    r  += f"Report generated by AGNISIGHT v3.0  |  Agnikul Cosmos  |  Sriharikota\n"
    r  += f"Model: IsolationForest (n=300)  |  Features: {len(FEATURE_COLS)}  |  Sample rate: 10 Hz\n"
    r  += f"{sep}\n"
    return r


# ─────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:12px 0 6px;">
        <div style="font-size:18px;font-weight:700;color:#0d1f3c;letter-spacing:0.5px;">🚀 AGNISIGHT</div>
        <div style="font-size:10px;color:#64748b;letter-spacing:1px;margin-top:2px;">AGNIKUL COSMOS · ANOMALY DETECTION</div>
    </div>
    <hr style="border:none;border-top:1px solid #e2e8f0;margin:10px 0;"/>
    """, unsafe_allow_html=True)

    # ── DATA SOURCE ───────────────────────────────────────────────
    st.markdown('<p style="font-size:11px;font-weight:600;color:#475569;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">Data Source</p>', unsafe_allow_html=True)
    data_source = st.radio(
        "Input Mode",
        ["Upload CSV", "Simulate Telemetry"],
        index=0 if st.session_state.data_source == "csv" else 1,
        label_visibility="collapsed",
    )
    st.session_state.data_source = "csv" if data_source == "Upload CSV" else "simulate"

    csv_ready = False
    if st.session_state.data_source == "csv":
        uploaded_file = st.file_uploader(
            "Upload your telemetry CSV",
            type=["csv"],
            help="CSV with any column names. AGNISIGHT auto-maps them to the canonical channels.",
        )
        if uploaded_file is not None:
            try:
                raw_csv = pd.read_csv(uploaded_file)
                st.session_state.csv_df = raw_csv
                col_map = auto_map_columns(raw_csv.columns.tolist())
                st.session_state.csv_col_map = col_map
                mapped_count = len(col_map)
                total_canon  = len(ALL_COLS)
                key_mapped  = [c for c in FEATURE_COLS if c in col_map]
                key_missing = [c for c in FEATURE_COLS if c not in col_map]
                all_ok      = mapped_count == total_canon
                key_ok      = len(key_missing) == 0
                border_col  = "#16a34a" if key_ok else "#d97706"
                text_col    = "#166534" if key_ok else "#92400e"
                bg_col      = "#f0fdf4" if key_ok else "#fefce8"
                if all_ok:
                    status_line = "✔ All channels found — using 100% real data"
                elif key_ok:
                    status_line = (f"✔ {len(key_mapped)}/{len(FEATURE_COLS)} ML channels from CSV · "
                                   f"{total_canon - mapped_count} non-essential cols use physics defaults (normal)")
                else:
                    status_line = (f"⚠ {len(key_missing)} ML channel(s) not found in CSV · "
                                   f"using defaults for: {', '.join(key_missing[:4])}"
                                   + ("…" if len(key_missing) > 4 else ""))
                st.markdown(f"""<div style="background:{bg_col};border:1px solid;border-left:3px solid {border_col};
                    border-color:#e2e8f0;border-radius:4px;padding:8px 12px;font-size:11px;color:{text_col};
                    font-family:'JetBrains Mono',monospace;margin-top:6px;line-height:1.9;">
                    ✔ {len(raw_csv):,} rows &nbsp;·&nbsp; {len(raw_csv.columns)} CSV columns<br>
                    ✔ {mapped_count}/{total_canon} total channels matched<br>
                    ✔ {len(key_mapped)}/{len(FEATURE_COLS)} ML feature channels from your data<br>
                    {status_line}
                </div>""", unsafe_allow_html=True)
                with st.expander("Column Mapping Details", expanded=False):
                    st.markdown('<b style="font-size:10px;color:#475569;">Mapped from your CSV:</b>', unsafe_allow_html=True)
                    for canon, csv_col in sorted(col_map.items()):
                        is_key = "🔑 " if canon in FEATURE_COLS else "   "
                        st.markdown(f'<span style="font-size:10px;font-family:monospace;color:#334155;">{is_key}'
                                    f'<b>{canon}</b> ← <span style="color:#1a56db;">{csv_col}</span></span>',
                                    unsafe_allow_html=True)
                    missing = [c for c in ALL_COLS if c not in col_map]
                    if missing:
                        st.markdown(f'<span style="font-size:10px;color:#94a3b8;font-family:monospace;">' +
                                    f'Physics defaults used for: {", ".join(missing[:10])}{"…" if len(missing) > 10 else ""}</span>',
                                    unsafe_allow_html=True)
                csv_ready = True
            except Exception as e:
                st.error(f"CSV parse error: {e}")
        elif st.session_state.csv_df is not None:
            csv_ready = True
            st.markdown('<div style="font-size:11px;color:#64748b;margin-bottom:4px;">Using previously uploaded CSV.</div>',
                        unsafe_allow_html=True)

    st.markdown('<hr style="border:none;border-top:1px solid #e2e8f0;margin:10px 0;"/>', unsafe_allow_html=True)

    # ── MISSION CONFIG ────────────────────────────────────────────
    st.markdown('<p style="font-size:11px;font-weight:600;color:#475569;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">Mission Configuration</p>', unsafe_allow_html=True)
    mission_id  = st.text_input("Mission ID", value="AGN-FL-001")
    st.session_state.mission_id = mission_id

    if st.session_state.data_source == "simulate":
        flight_mode = st.selectbox("Flight Mode / Inject Fault",
            ["Nominal", "Pressure Spike", "TVC Divergence",
             "Turbopump Cavitation", "Thermal Runaway", "Power Anomaly", "OF Ratio Drift"])
        st.session_state.flight_mode = flight_mode
        duration = st.slider("Flight Duration (s)", 200, 400, 300, 10)
        seed_val = st.slider("Telemetry Seed", 1, 99, 42, 1)
    else:
        flight_mode = "Nominal"
        st.session_state.flight_mode = flight_mode
        duration, seed_val = 300, 42

    st.markdown('<hr style="border:none;border-top:1px solid #e2e8f0;margin:10px 0;"/>', unsafe_allow_html=True)
    st.markdown("""<div style="font-size:11px;color:#64748b;line-height:1.9;">
<b>ML Engine</b><br>
Model&nbsp;&nbsp;&nbsp;: Isolation Forest<br>
Trees&nbsp;&nbsp;&nbsp;: 300 estimators<br>
Contam&nbsp;: 1.5 %<br>
Features: 13 channels<br>
Scaler&nbsp;&nbsp;: StandardScaler<br>
Train&nbsp;&nbsp;&nbsp;: Powered ascent phase
</div>""", unsafe_allow_html=True)

    st.markdown('<hr style="border:none;border-top:1px solid #e2e8f0;margin:10px 0;"/>', unsafe_allow_html=True)
    btn_label    = "▶  Analyse CSV" if st.session_state.data_source == "csv" else "▶  Run Simulation"
    btn_disabled = (st.session_state.data_source == "csv" and not csv_ready)
    run_btn = st.button(btn_label, use_container_width=True, disabled=btn_disabled)
    if btn_disabled:
        st.markdown('<div style="font-size:10px;color:#94a3b8;margin-top:4px;">Upload a CSV to enable analysis.</div>',
                    unsafe_allow_html=True)

    st.markdown("""<div style="font-size:10px;color:#94a3b8;line-height:1.9;margin-top:8px;">
Vehicle&nbsp;&nbsp;&nbsp;: Agnibaan ST<br>
Engine&nbsp;&nbsp;&nbsp;: Agnilet SN-07<br>
Propellant: LOX / HTPB<br>
Launchpad : Dhanush, SDSC<br>
Sample Hz : 10 Hz / 3 000 pts
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
#  RUN ANALYSIS
# ─────────────────────────────────────────────────────────────────
if run_btn:
    if st.session_state.data_source == "csv" and st.session_state.csv_df is not None:
        # ── CSV MODE ───────────────────────────────────────────────
        with st.spinner("Parsing CSV · mapping channels · training Isolation Forest on your data…"):
            col_map  = st.session_state.csv_col_map
            flt_df   = csv_to_telemetry(st.session_state.csv_df, col_map, mission_id)

            # ── Train on the uploaded CSV data itself ──────────────
            # Use powered-ascent rows (phase == 2) if enough exist,
            # otherwise fall back to the full dataset so no synthetic
            # data is used at any point.
            pwr_rows = flt_df[flt_df["phase"] == 2]
            if len(pwr_rows) >= 20:
                train_df = flt_df.copy()          # detect_anomalies already filters to phase==2
            else:
                # Not enough phase-tagged rows → treat entire CSV as training data
                train_df = flt_df.copy()
                train_df["phase"] = 2             # mark all rows powered so train_model uses them

            # train_model uses phase==2 rows of the supplied DataFrame
            model, scaler = train_model(train_df)

            # Detect anomalies on the original (unmodified) data
            # If phase==2 rows are too few, run detection over all rows
            if (flt_df["phase"] == 2).sum() < 20:
                flt_df_detect = flt_df.copy()
                flt_df_detect["phase"] = 2
            else:
                flt_df_detect = flt_df

            scored_df, anom_events = detect_anomalies(flt_df_detect, model, scaler)
            report   = generate_report(scored_df, anom_events, "CSV Upload", mission_id)
            eng_hist = generate_engine_health()

            st.session_state.flight_data    = scored_df
            st.session_state.anomalies      = anom_events
            st.session_state.model          = model
            st.session_state.scaler         = scaler
            st.session_state.report         = report
            st.session_state.engine_history = eng_hist
            st.session_state.flight_mode    = "Nominal"
    else:
        # ── SIMULATION MODE ────────────────────────────────────────
        with st.spinner("Generating telemetry & running Isolation Forest…"):
            nom_df   = generate_telemetry(300, "Nominal", seed=99)
            model, scaler = train_model(nom_df)
            flt_df   = generate_telemetry(duration, flight_mode, seed_val)
            scored_df, anom_events = detect_anomalies(flt_df, model, scaler)
            report   = generate_report(scored_df, anom_events, flight_mode, mission_id)
            eng_hist = generate_engine_health()

            st.session_state.flight_data    = scored_df
            st.session_state.anomalies      = anom_events
            st.session_state.model          = model
            st.session_state.scaler         = scaler
            st.session_state.report         = report
            st.session_state.engine_history = eng_hist


# ─────────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────────
n_anom   = len(st.session_state.anomalies)
has_data = st.session_state.flight_data is not None

if not has_data:
    badge_cls, sys_txt = "badge-warn", "AWAITING FLIGHT DATA"
elif n_anom == 0:
    badge_cls, sys_txt = "badge-ok", "ALL SYSTEMS NOMINAL"
else:
    badge_cls, sys_txt = "badge-alert", f"{n_anom} ANOMAL{'Y' if n_anom == 1 else 'IES'} DETECTED"

st.markdown(f"""
<div class="app-header">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;">
    <div>
      <div class="app-title">AGNISIGHT</div>
      <div class="app-subtitle">Intelligent Flight Anomaly Detection &amp; Post-Flight Analysis · Agnikul Cosmos</div>
    </div>
    <div style="text-align:right;">
      <span class="status-badge {badge_cls}">{sys_txt}</span>
      <div style="font-size:10px;color:#94a3b8;margin-top:6px;font-family:'JetBrains Mono',monospace;">
        MISSION: {st.session_state.mission_id}<br>
        {datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S UTC")}
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

if not has_data:
    src_hint = ("Upload a telemetry CSV in the sidebar, then click <b>▶ Analyse CSV</b>."
                if st.session_state.data_source == "csv" else
                "Configure mission parameters in the sidebar and click <b>▶ Run Simulation</b>.")
    st.markdown(f"""
    <div style="text-align:center;padding:80px 20px;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;">
      <div style="font-size:48px;">🚀</div>
      <div style="font-size:16px;font-weight:600;color:#0d1f3c;margin-top:14px;">System Ready</div>
      <div style="font-size:13px;color:#64748b;margin-top:10px;line-height:1.8;">
        {src_hint}<br><br>
        13 channels · Propulsion + GNC + Avionics · IsolationForest ML Engine
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────────────────────────
#  LOAD SESSION DATA
# ─────────────────────────────────────────────────────────────────
df       = st.session_state.flight_data
anomalies = st.session_state.anomalies
report   = st.session_state.report
eng_hist = st.session_state.engine_history
pwr      = df[df["phase"] == 2]
mode     = st.session_state.flight_mode


# ─────────────────────────────────────────────────────────────────
#  KPI ROW
# ─────────────────────────────────────────────────────────────────
cols = st.columns(7)
kpi_def = [
    ("PEAK ALT",   f"{df['altitude_km'].max():.1f}",       "km",           "blue"),
    ("PEAK MACH",  f"{df['mach'].max():.2f}",              "Mach",         "blue"),
    ("AVG THRUST", f"{pwr['thrust_kN'].mean():.1f}",       "kN",           "blue"),
    ("AVG Pc",     f"{pwr['chamber_pres'].mean():.1f}",    "bar",          "blue"),
    ("OF RATIO",   f"{pwr['OF_ratio'].mean():.3f}",        "design 2.10",  "blue"),
    ("ANOMALIES",  str(n_anom),                             "IF events",   "alert" if n_anom > 0 else "ok"),
    ("IF SCORE",   f"{df['if_score'].min():.3f}",          "min score",    "warn" if df['if_score'].min() < -0.2 else "ok"),
]
for col, (label, val, unit, card_cls) in zip(cols, kpi_def):
    with col:
        st.markdown(f"""
        <div class="kpi-card kpi-card-{card_cls}">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value" style="font-size:{'20px' if len(val) > 6 else '24px'};">{val}</div>
          <div class="kpi-unit">{unit}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📡 Telemetry Overview",
    "🔍 Anomaly Detection",
    "🚀 Propulsion",
    "🧭 GNC",
    "⚡ Avionics",
    "🔧 Engine Health",
    "📄 Post-Flight Report",
    "💬 NL Query",
])


# ══════════════════════════════════════════════════════════════════
#  TAB 1 — TELEMETRY OVERVIEW
# ══════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="section-title">All Channels · 10 Hz · T+0 to T+300s</div>', unsafe_allow_html=True)

    ch_grid = [
        ("chamber_pres",  1, 1, BLUE),   ("thrust_kN",      1, 2, GREEN),  ("total_flow_kgs", 1, 3, TEAL),
        ("nozzle_temp_C", 2, 1, RED),    ("turbo_rpm",       2, 2, AMBER),  ("OF_ratio",        2, 3, SLATE),
        ("altitude_km",   3, 1, BLUE),   ("mach",            3, 2, GREEN),  ("dyn_pres_Pa",     3, 3, AMBER),
        ("bus_voltage_V", 4, 1, TEAL),   ("cpu_temp_C",      4, 2, RED),    ("signal_dbm",      4, 3, SLATE),
    ]
    titles = [
        "Chamber Pressure (bar)", "Thrust (kN)", "Total Prop Flow (kg/s)",
        "Nozzle Temp (°C)", "Turbopump RPM", "OF Ratio",
        "Altitude (km)", "Mach", "Dynamic Pressure (Pa)",
        "Bus Voltage (V)", "CPU Temp (°C)", "Signal (dBm)",
    ]

    fig_tl = make_subplots(
        rows=4, cols=3, subplot_titles=titles,
        vertical_spacing=0.07, horizontal_spacing=0.06,
    )
    for (ch, r, c, color) in ch_grid:
        fig_tl.add_trace(go.Scatter(
            x=df["time_s"], y=df[ch], mode="lines",
            line=dict(color=color, width=1.0), name=ch, showlegend=False,
            hovertemplate="T+%{x:.1f}s · %{y:.3f}<extra>" + ch + "</extra>",
        ), row=r, col=c)
        for ev in anomalies:
            if ev["channel"] == ch:
                fig_tl.add_vrect(
                    x0=ev["t_start"], x1=ev["t_end"],
                    fillcolor="rgba(220,38,38,0.10)",
                    line=dict(color="rgba(220,38,38,0.35)", width=1),
                    row=r, col=c,
                )
    fig_tl.update_layout(height=800, **LAYOUT)
    for r in range(1, 5):
        for c in range(1, 4):
            fig_tl.update_xaxes(**AXIS, row=r, col=c)
            fig_tl.update_yaxes(**AXIS, row=r, col=c)
    fig_tl.update_annotations(font=dict(color="#475569", size=9))
    st.plotly_chart(fig_tl, use_container_width=True)

    # Phase timeline
    st.markdown('<div class="section-title" style="margin-top:8px;">Flight Phase Timeline</div>', unsafe_allow_html=True)
    phase_meta = {
        0: ("Pre-Launch", SLATE), 1: ("Ignition", AMBER),
        2: ("Powered Ascent", BLUE), 3: ("MECO", GREEN),
        4: ("Coast", TEAL), 5: ("Stage Sep", RED),
    }
    fig_ph = go.Figure()
    for ph_id, (ph_name, ph_col) in phase_meta.items():
        mask = df["phase"] == ph_id
        if mask.any():
            fig_ph.add_trace(go.Scatter(
                x=df.loc[mask, "time_s"], y=[ph_name] * mask.sum(),
                mode="markers", marker=dict(color=ph_col, size=4, symbol="square"),
                name=ph_name, showlegend=True,
                hovertemplate="T+%{x:.1f}s<extra>" + ph_name + "</extra>",
            ))
    fig_ph.update_layout(height=180, **LAYOUT)
    fig_ph.update_xaxes(**AXIS, title_text="Time (s)")
    fig_ph.update_yaxes(
        categoryorder="array",
        categoryarray=[v[0] for v in phase_meta.values()],
        **AXIS,
    )
    st.plotly_chart(fig_ph, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
#  TAB 2 — ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="section-title">Isolation Forest · Multivariate Anomaly Detection</div>', unsafe_allow_html=True)

    col_l, col_r = st.columns([2, 1])

    with col_l:
        # IF score time series
        fig_sc = base_fig(height=260, title="Isolation Forest Anomaly Score vs Time")
        fig_sc.add_trace(go.Scatter(
            x=df["time_s"], y=df["if_score"], mode="lines",
            line=dict(color=BLUE, width=1.3),
            fill="tozeroy", fillcolor="rgba(26,86,219,0.05)", name="IF Score",
        ))
        fig_sc.add_hline(y=-0.20, line=dict(color=RED, dash="dash", width=1.2),
            annotation_text="Anomaly Threshold",
            annotation_font=dict(color=RED, size=9))
        for ev in anomalies:
            fig_sc.add_vrect(x0=ev["t_start"], x1=ev["t_end"],
                fillcolor="rgba(220,38,38,0.08)",
                line=dict(color="rgba(220,38,38,0.30)", width=1))
        fig_sc.update_xaxes(title_text="Time (s)")
        fig_sc.update_yaxes(title_text="IF Score")
        st.plotly_chart(fig_sc, use_container_width=True)

        # PCA 2D
        st.markdown('<div class="section-title" style="font-size:10px;">Feature Space — PCA 2D Projection</div>', unsafe_allow_html=True)
        X_sc   = st.session_state.scaler.transform(df[FEATURE_COLS].fillna(0))
        pca    = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X_sc)
        var_explained = pca.explained_variance_ratio_.sum() * 100

        fig_pca = base_fig(height=300, title=f"PCA — {var_explained:.1f}% Variance Explained")
        for label, mask, color in [
            ("Nominal (Powered)",  (df["phase"] == 2) & ~df["anomaly_flag"], BLUE),
            ("Anomalous",           df["anomaly_flag"],                       RED),
            ("Non-powered Phase",   df["phase"] != 2,                        "#cbd5e1"),
        ]:
            if mask.any():
                fig_pca.add_trace(go.Scatter(
                    x=coords[mask, 0], y=coords[mask, 1], mode="markers",
                    marker=dict(color=color, size=3, opacity=0.7), name=label,
                ))
        fig_pca.update_xaxes(title_text=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        fig_pca.update_yaxes(title_text=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        st.plotly_chart(fig_pca, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-title" style="font-size:10px;">Detected Events</div>', unsafe_allow_html=True)
        if not anomalies:
            st.markdown("""<div class="anomaly-row-ok">
              <div style="color:#15803d;font-weight:600;">✓ ALL CLEAR</div>
              <div style="color:#64748b;margin-top:3px;">No anomalous events detected across all 13 monitored channels during powered ascent.</div>
            </div>""", unsafe_allow_html=True)
        else:
            for ev in anomalies:
                ch_disp = CH_DESC.get(ev["channel"], ev["channel"])
                st.markdown(f"""<div class="anomaly-row">
                  <div class="anom-time">T+{ev['t_start']:.1f}s → T+{ev['t_end']:.1f}s  ({ev['dur']:.1f}s)</div>
                  <div class="anom-msg">[{ev['severity']}]  {ch_disp}</div>
                  <div class="anom-ch">IF Score: {ev['score']}  |  {ev['channel']}</div>
                </div>""", unsafe_allow_html=True)

        # Channel deviation ranking
        st.markdown('<div class="section-title" style="font-size:10px;margin-top:18px;">Channel Deviation Rank</div>', unsafe_allow_html=True)
        if df["anomaly_flag"].any():
            anom_rows = df[df["anomaly_flag"]][FEATURE_COLS]
            nom_rows  = df[(df["phase"] == 2) & ~df["anomaly_flag"]][FEATURE_COLS]
            if len(nom_rows) > 0 and len(anom_rows) > 0:
                imp = (anom_rows.mean() - nom_rows.mean()).abs().sort_values(ascending=True)
                bar_colors = [RED if v == imp.max() else AMBER if v > imp.median() else GREEN
                              for v in imp.values]
                fig_imp = go.Figure(go.Bar(
                    x=imp.values, y=imp.index, orientation="h",
                    marker_color=bar_colors,
                    text=[f"{v:.3f}" for v in imp.values],
                    textfont=dict(color="#475569", size=8), textposition="outside",
                ))
                fig_imp.update_layout(height=320, **LAYOUT)
                fig_imp.update_xaxes(**AXIS, title_text="Mean Deviation")
                fig_imp.update_yaxes(**AXIS)
                st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.markdown('<div style="font-size:11px;color:#94a3b8;padding:16px 0;">No anomalous samples to rank.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  TAB 3 — PROPULSION
# ══════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-title">Propulsion System — Agnilet SN-07</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        # Pc + Thrust dual axis
        fig_pt = make_subplots(specs=[[{"secondary_y": True}]])
        fig_pt.add_trace(go.Scatter(x=df["time_s"], y=df["chamber_pres"],
            name="Chamber Pressure (bar)", line=dict(color=BLUE, width=1.5)), secondary_y=False)
        fig_pt.add_trace(go.Scatter(x=df["time_s"], y=df["thrust_kN"],
            name="Thrust (kN)", line=dict(color=AMBER, width=1.2, dash="dot")), secondary_y=True)
        if len(pwr) > 0:
            p_mean, p_std = pwr["chamber_pres"].mean(), pwr["chamber_pres"].std()
            fig_pt.add_hrect(y0=p_mean - 2 * p_std, y1=p_mean + 2 * p_std,
                fillcolor="rgba(26,86,219,0.04)",
                line=dict(color="rgba(26,86,219,0.15)", width=0.5))
        fig_pt.update_layout(height=290, title=dict(text="Chamber Pressure & Thrust", font=dict(color="#475569", size=11), x=0.01), **LAYOUT)
        fig_pt.update_yaxes(**AXIS, secondary_y=False, title_text="Pc (bar)")
        fig_pt.update_yaxes(**AXIS, secondary_y=True,  title_text="Thrust (kN)")
        fig_pt.update_xaxes(**AXIS)
        st.plotly_chart(fig_pt, use_container_width=True)

        # OF Ratio
        fig_of = base_fig(height=220, title="Oxidiser-to-Fuel Ratio")
        fig_of.add_trace(go.Scatter(x=df["time_s"], y=df["OF_ratio"],
            name="OF Ratio", line=dict(color=RED, width=1.2)))
        fig_of.add_hline(y=2.10, line=dict(color=GREEN, dash="dash", width=0.9),
            annotation_text="Design 2.10", annotation_font=dict(color=GREEN, size=9))
        fig_of.add_hline(y=2.5, line=dict(color=RED, dash="dash", width=0.7),
            annotation_text="Rich Limit", annotation_font=dict(color=RED, size=9))
        fig_of.add_hline(y=1.7, line=dict(color=AMBER, dash="dash", width=0.7),
            annotation_text="Lean Limit", annotation_font=dict(color=AMBER, size=9))
        st.plotly_chart(fig_of, use_container_width=True)

        # Nozzle temp
        fig_nt = base_fig(height=220, title="Nozzle Throat Temperature (°C)")
        fig_nt.add_trace(go.Scatter(x=df["time_s"], y=df["nozzle_temp_C"],
            name="Nozzle Temp", line=dict(color=RED, width=1.3)))
        fig_nt.add_hline(y=1950, line=dict(color=RED, dash="dash", width=1.0),
            annotation_text="Thermal Limit 1950 °C",
            annotation_font=dict(color=RED, size=9))
        st.plotly_chart(fig_nt, use_container_width=True)

    with c2:
        # Propellant flows
        fig_fl = base_fig(height=220, title="Propellant Mass Flow Rates (kg/s)")
        fig_fl.add_trace(go.Scatter(x=df["time_s"], y=df["ox_flow_kgs"],
            name="Oxidiser (LOX)", line=dict(color=TEAL, width=1.2)))
        fig_fl.add_trace(go.Scatter(x=df["time_s"], y=df["fuel_flow_kgs"],
            name="Fuel (HTPB)", line=dict(color=AMBER, width=1.2)))
        fig_fl.add_trace(go.Scatter(x=df["time_s"], y=df["total_flow_kgs"],
            name="Total", line=dict(color=BLUE, width=1.5, dash="dot")))
        st.plotly_chart(fig_fl, use_container_width=True)

        # Turbopump RPM
        fig_turbo = base_fig(height=220, title="Turbopump RPM")
        fig_turbo.add_trace(go.Scatter(x=df["time_s"], y=df["turbo_rpm"],
            name="RPM", line=dict(color=AMBER, width=1.3)))
        fig_turbo.add_hline(y=32000, line=dict(color=RED, dash="dash", width=0.9),
            annotation_text="Redline 32 000 RPM",
            annotation_font=dict(color=RED, size=9))
        if mode == "Turbopump Cavitation":
            fig_turbo.add_annotation(x=66, y=df["turbo_rpm"].min(),
                text="CAVITATION", showarrow=True,
                arrowcolor=RED, font=dict(color=RED, size=10))
        st.plotly_chart(fig_turbo, use_container_width=True)

        # Tank pressures
        fig_tank = base_fig(height=220, title="Propellant Tank Pressures (bar)")
        fig_tank.add_trace(go.Scatter(x=df["time_s"], y=df["lox_tank_pres"],
            name="LOX Tank", line=dict(color=TEAL, width=1.2)))
        fig_tank.add_trace(go.Scatter(x=df["time_s"], y=df["fuel_tank_pres"],
            name="Fuel Tank", line=dict(color=AMBER, width=1.2)))
        st.plotly_chart(fig_tank, use_container_width=True)

        # Propulsion summary box
        st.markdown(f"""<div class="info-box">
Avg Pc&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: {pwr['chamber_pres'].mean():.2f} bar (nominal 68.0)<br>
Avg Thrust&nbsp;&nbsp;: {pwr['thrust_kN'].mean():.2f} kN (nominal 100.0)<br>
Est. Isp&nbsp;&nbsp;&nbsp;&nbsp;: {pwr['thrust_kN'].mean()*1000/9.80665/max(pwr['total_flow_kgs'].mean(),0.01):.0f} s<br>
Comb. Eff.&nbsp;&nbsp;: {min(pwr['chamber_pres'].mean()/68.0*100, 100):.1f} %<br>
Mean OF Ratio: {pwr['OF_ratio'].mean():.3f} (design 2.100)<br>
Peak Nozzle T: {pwr['nozzle_temp_C'].max():.0f} °C (limit 1950)
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  TAB 4 — GNC
# ══════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="section-title">Guidance, Navigation & Control</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        # Euler angles
        fig_ang = base_fig(height=250, title="Euler Angles — Pitch / Yaw / Roll (deg)")
        for col_name, color, dash in [
            ("pitch_deg", BLUE, "solid"), ("yaw_deg", AMBER, "dot"), ("roll_deg", TEAL, "dash"),
        ]:
            fig_ang.add_trace(go.Scatter(x=df["time_s"], y=df[col_name],
                name=col_name, line=dict(color=color, width=1.2, dash=dash)))
        st.plotly_chart(fig_ang, use_container_width=True)

        # Angular rates
        fig_rate = base_fig(height=220, title="Angular Rates (deg/s)")
        for col_name, color in [("p_rate_degs", BLUE), ("q_rate_degs", AMBER), ("r_rate_degs", TEAL)]:
            fig_rate.add_trace(go.Scatter(x=df["time_s"], y=df[col_name],
                name=col_name, line=dict(color=color, width=1.0)))
        st.plotly_chart(fig_rate, use_container_width=True)

        # Altitude + Mach dual axis
        fig_am = make_subplots(specs=[[{"secondary_y": True}]])
        fig_am.add_trace(go.Scatter(x=df["time_s"], y=df["altitude_km"],
            name="Altitude (km)", line=dict(color=BLUE, width=1.5)), secondary_y=False)
        fig_am.add_trace(go.Scatter(x=df["time_s"], y=df["mach"],
            name="Mach", line=dict(color=TEAL, width=1.2, dash="dot")), secondary_y=True)
        fig_am.update_layout(height=240, title=dict(text="Altitude & Mach vs Time", font=dict(color="#475569", size=11), x=0.01), **LAYOUT)
        fig_am.update_yaxes(**AXIS, secondary_y=False, title_text="Alt (km)")
        fig_am.update_yaxes(**AXIS, secondary_y=True,  title_text="Mach")
        fig_am.update_xaxes(**AXIS)
        st.plotly_chart(fig_am, use_container_width=True)

    with c2:
        # Velocity
        fig_vel = base_fig(height=220, title="Velocity Components (m/s)")
        fig_vel.add_trace(go.Scatter(x=df["time_s"], y=df["vel_v_ms"],
            name="Vertical", line=dict(color=BLUE, width=1.3)))
        fig_vel.add_trace(go.Scatter(x=df["time_s"], y=df["vel_h_ms"],
            name="Horizontal", line=dict(color=AMBER, width=1.3)))
        st.plotly_chart(fig_vel, use_container_width=True)

        # IMU accelerations
        fig_imu = base_fig(height=220, title="IMU Accelerations (m/s²)")
        for col_name, color in [("accel_x", BLUE), ("accel_y", AMBER), ("accel_z", TEAL)]:
            fig_imu.add_trace(go.Scatter(x=df["time_s"], y=df[col_name],
                name=col_name, line=dict(color=color, width=1.0)))
        st.plotly_chart(fig_imu, use_container_width=True)

        # Dynamic pressure
        fig_dq = base_fig(height=220, title="Dynamic Pressure (Pa)")
        fig_dq.add_trace(go.Scatter(x=df["time_s"], y=df["dyn_pres_Pa"],
            name="q_dyn", line=dict(color=AMBER, width=1.3),
            fill="tozeroy", fillcolor="rgba(217,119,6,0.05)"))
        st.plotly_chart(fig_dq, use_container_width=True)

        # Attitude phase space
        fig_ps = base_fig(height=220, title="Attitude Phase Space (Pitch vs Yaw)")
        fig_ps.add_trace(go.Scatter(
            x=pwr["pitch_deg"], y=pwr["yaw_deg"], mode="markers",
            marker=dict(color=pwr["time_s"], colorscale="Blues", size=2.5, opacity=0.6),
            name="Powered Phase",
        ))
        theta = np.linspace(0, 2 * np.pi, 100)
        fig_ps.add_trace(go.Scatter(x=90 + 5 * np.cos(theta), y=5 * np.sin(theta),
            mode="lines", line=dict(color=RED, dash="dash", width=1), name="±5° Limit"))
        st.plotly_chart(fig_ps, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
#  TAB 5 — AVIONICS
# ══════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="section-title">Avionics & Power Systems</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        fig_v = base_fig(height=240, title="Bus Voltage (V)")
        fig_v.add_trace(go.Scatter(x=df["time_s"], y=df["bus_voltage_V"],
            name="28 V Bus", line=dict(color=TEAL, width=1.3)))
        fig_v.add_hline(y=27.0, line=dict(color=RED, dash="dash", width=0.9),
            annotation_text="Under-voltage 27 V", annotation_font=dict(color=RED, size=9))
        fig_v.add_hline(y=29.0, line=dict(color=AMBER, dash="dash", width=0.7),
            annotation_text="Over-voltage 29 V", annotation_font=dict(color=AMBER, size=9))
        if mode == "Power Anomaly":
            fig_v.add_annotation(x=33, y=df["bus_voltage_V"].min(),
                text="UNDERVOLTAGE", showarrow=True,
                arrowcolor=RED, font=dict(color=RED, size=10))
        st.plotly_chart(fig_v, use_container_width=True)
        st.markdown(f"""<div class="info-box">
Mean&nbsp;&nbsp;&nbsp;: {df['bus_voltage_V'].mean():.3f} V<br>
Min&nbsp;&nbsp;&nbsp;&nbsp;: {df['bus_voltage_V'].min():.3f} V<br>
Max&nbsp;&nbsp;&nbsp;&nbsp;: {df['bus_voltage_V'].max():.3f} V<br>
StdDev : {df['bus_voltage_V'].std():.4f} V<br>
Status : {"⚠ ANOMALOUS" if mode == 'Power Anomaly' else "✓ NOMINAL"}
</div>""", unsafe_allow_html=True)

    with c2:
        fig_cpu = base_fig(height=240, title="Flight Computer CPU Temperature (°C)")
        fig_cpu.add_trace(go.Scatter(x=df["time_s"], y=df["cpu_temp_C"],
            name="CPU Temp", line=dict(color=AMBER, width=1.3)))
        fig_cpu.add_hline(y=85, line=dict(color=RED, dash="dash", width=0.9),
            annotation_text="Thermal Limit 85 °C", annotation_font=dict(color=RED, size=9))
        st.plotly_chart(fig_cpu, use_container_width=True)
        st.markdown(f"""<div class="info-box">
Mean&nbsp;&nbsp;&nbsp;: {df['cpu_temp_C'].mean():.1f} °C<br>
Peak&nbsp;&nbsp;&nbsp;: {df['cpu_temp_C'].max():.1f} °C<br>
Min&nbsp;&nbsp;&nbsp;&nbsp;: {df['cpu_temp_C'].min():.1f} °C<br>
Margin : {85 - df['cpu_temp_C'].max():.1f} °C to limit
</div>""", unsafe_allow_html=True)

    with c3:
        fig_sig = base_fig(height=240, title="Telemetry Link Signal (dBm)")
        fig_sig.add_trace(go.Scatter(x=df["time_s"], y=df["signal_dbm"],
            name="Signal", line=dict(color=SLATE, width=1.2)))
        fig_sig.add_hline(y=-90, line=dict(color=RED, dash="dash", width=0.9),
            annotation_text="Loss Threshold −90 dBm", annotation_font=dict(color=RED, size=9))
        st.plotly_chart(fig_sig, use_container_width=True)
        st.markdown(f"""<div class="info-box">
Mean&nbsp;&nbsp;&nbsp;: {df['signal_dbm'].mean():.1f} dBm<br>
Min&nbsp;&nbsp;&nbsp;&nbsp;: {df['signal_dbm'].min():.1f} dBm<br>
Max&nbsp;&nbsp;&nbsp;&nbsp;: {df['signal_dbm'].max():.1f} dBm<br>
Margin : {df['signal_dbm'].min() - (-90):.1f} dB to threshold
</div>""", unsafe_allow_html=True)

    # Correlation heatmap
    st.markdown('<div class="section-title" style="margin-top:10px;">Channel Correlation Matrix (Powered Ascent)</div>', unsafe_allow_html=True)
    corr_cols = ["chamber_pres", "thrust_kN", "OF_ratio", "nozzle_temp_C",
                 "turbo_rpm", "altitude_km", "mach", "bus_voltage_V", "cpu_temp_C", "dyn_pres_Pa"]
    corr_df = pwr[corr_cols].corr()
    fig_corr = go.Figure(go.Heatmap(
        z=corr_df.values, x=corr_cols, y=corr_cols,
        colorscale=[[0, "#fff5f5"], [0.5, "#bfdbfe"], [1, "#1e40af"]],
        zmin=-1, zmax=1,
        text=np.round(corr_df.values, 2),
        texttemplate="%{text}", textfont=dict(size=8, color="#0d1f3c"),
    ))
    fig_corr.update_layout(height=380, title=dict(text="Pearson Correlation — Key Channels", font=dict(color="#475569", size=11), x=0.01), **LAYOUT)
    fig_corr.update_xaxes(tickfont=dict(color="#64748b", size=8))
    fig_corr.update_yaxes(tickfont=dict(color="#64748b", size=8))
    st.plotly_chart(fig_corr, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
#  TAB 6 — ENGINE HEALTH
# ══════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="section-title">Engine Health Fingerprinting — Agnilet SN-07</div>', unsafe_allow_html=True)

    eh = eng_hist
    c1, c2 = st.columns(2)

    with c1:
        bar_cols = [GREEN if s > 90 else AMBER if s > 80 else RED for s in eh["health_score"]]
        fig_hs = base_fig(height=260, title="Engine Health Score per Firing")
        fig_hs.add_trace(go.Bar(x=eh["fire_id"], y=eh["health_score"],
            marker_color=bar_cols,
            text=[f"{v:.1f}%" for v in eh["health_score"]],
            textfont=dict(color="#0d1f3c", size=9), textposition="outside", name="Health"))
        fig_hs.add_trace(go.Scatter(x=eh["fire_id"], y=eh["health_score"],
            mode="lines+markers", line=dict(color=BLUE, width=1.5, dash="dot"),
            marker=dict(color=BLUE, size=5), name="Trend"))
        fig_hs.update_yaxes(range=[60, 108])
        st.plotly_chart(fig_hs, use_container_width=True)

        fig_rul = base_fig(height=220, title="Remaining Useful Life (Cycles)")
        fig_rul.add_trace(go.Scatter(x=eh["fire_id"], y=eh["rul_cycles"],
            mode="lines+markers+text", line=dict(color=AMBER, width=2),
            marker=dict(color=AMBER, size=8),
            text=[str(v) for v in eh["rul_cycles"]],
            textposition="top center", textfont=dict(color=AMBER, size=9), name="RUL"))
        fig_rul.add_hline(y=3, line=dict(color=RED, dash="dash", width=1),
            annotation_text="Retirement Threshold",
            annotation_font=dict(color=RED, size=9))
        st.plotly_chart(fig_rul, use_container_width=True)

    with c2:
        fig_pc_trend = base_fig(height=240, title="Peak Chamber Pressure Degradation")
        fig_pc_trend.add_trace(go.Scatter(x=eh["fire_id"], y=eh["peak_Pc_bar"],
            mode="lines+markers", line=dict(color=BLUE, width=1.5),
            marker=dict(color=BLUE, size=6), name="Peak Pc"))
        fig_pc_trend.add_hline(y=68.0, line=dict(color=SLATE, dash="dash", width=0.8),
            annotation_text="Design Pc 68 bar",
            annotation_font=dict(color=SLATE, size=9))
        fig_pc_trend.update_yaxes(range=[62, 72])
        st.plotly_chart(fig_pc_trend, use_container_width=True)

        fig_nzl = base_fig(height=240, title="Nozzle Peak Temperature Trend (°C)")
        fig_nzl.add_trace(go.Scatter(x=eh["fire_id"], y=eh["nozzle_peak_C"],
            mode="lines+markers", line=dict(color=RED, width=1.5),
            marker=dict(color=RED, size=6), name="Nozzle Peak"))
        fig_nzl.add_hline(y=1950, line=dict(color=RED, dash="dash", width=0.9),
            annotation_text="Thermal Limit 1950 °C",
            annotation_font=dict(color=RED, size=9))
        st.plotly_chart(fig_nzl, use_container_width=True)

    st.markdown('<div class="section-title" style="margin-top:6px;">Agnilet SN-07 — Full Firing Log</div>', unsafe_allow_html=True)
    def _health_color(val):
        """Color health_score cells without requiring matplotlib."""
        try:
            v = float(val)
        except (TypeError, ValueError):
            return ""
        if v >= 90:
            bg, fg = "#dcfce7", "#166534"   # green
        elif v >= 80:
            bg, fg = "#fef9c3", "#854d0e"   # yellow
        else:
            bg, fg = "#fee2e2", "#991b1b"   # red
        return f"background-color:{bg};color:{fg};font-weight:600"

    st.dataframe(
        eh.style.format({
            "peak_Pc_bar":    "{:.2f}",
            "avg_thrust_kN":  "{:.2f}",
            "health_score":   "{:.1f}",
            "turbo_peak_rpm": "{:.0f}",
            "nozzle_peak_C":  "{:.0f}",
        }).map(_health_color, subset=["health_score"]),
        use_container_width=True, height=280,
    )

    # Current state bars
    st.markdown('<div class="section-title" style="margin-top:12px;">Current Engine State</div>', unsafe_allow_html=True)
    latest = eh.iloc[-1]
    h_cols = st.columns(4)
    hm = [
        ("Engine Health",   latest["health_score"],                           BLUE),
        ("Thrust Health",   min(latest["avg_thrust_kN"] / 100 * 100, 100),   GREEN),
        ("Thermal Margin",  max(100 - (latest["nozzle_peak_C"] - 1700) / 2.5, 0), AMBER),
        ("RUL Remaining",   latest["rul_cycles"] / 15 * 100,                  TEAL),
    ]
    for col, (label, pct_raw, color) in zip(h_cols, hm):
        pct     = int(min(max(pct_raw, 0), 100))
        bar_col = color if pct > 60 else AMBER if pct > 30 else RED
        with col:
            st.markdown(f"""
            <div class="info-box" style="padding:14px 16px;">
              <div class="kpi-label">{label}</div>
              <div style="font-size:22px;font-weight:700;color:#0d1f3c;font-family:'JetBrains Mono',monospace;">{pct}%</div>
              <div style="height:6px;background:#f1f5f9;border-radius:3px;margin-top:8px;overflow:hidden;">
                <div style="width:{pct}%;height:100%;background:{bar_col};border-radius:3px;"></div>
              </div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  TAB 7 — POST-FLIGHT REPORT
# ══════════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown('<div class="section-title">Automated Post-Flight Analysis Report</div>', unsafe_allow_html=True)

    mc = st.columns(4)
    for col, (label, value, sub) in zip(mc, [
        ("Generation Latency", "~2.8 s",          "vs 2–7 days manual"),
        ("Channels Monitored", str(len(FEATURE_COLS)), "propulsion + GNC + avionics"),
        ("Data Points",        f"{len(df):,}",      "@ 10 Hz"),
        ("Anomalies",          str(n_anom),          "IF detected events"),
    ]):
        card_cls = "kpi-card-alert" if (label == "Anomalies" and n_anom > 0) else "kpi-card-blue"
        with col:
            st.markdown(f"""
            <div class="kpi-card {card_cls}">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value" style="font-size:22px;">{value}</div>
              <div class="kpi-unit">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.markdown(f'<div class="report-box">{report}</div>', unsafe_allow_html=True)
    st.download_button(
        label="⬇  Export Report (.txt)",
        data=report,
        file_name=f"AGNISIGHT_{mission_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
    )


# ══════════════════════════════════════════════════════════════════
#  TAB 8 — NL QUERY  (Groq LLaMA3 powered — free tier)
# ══════════════════════════════════════════════════════════════════

# ── LLM API keys — fill in at least ONE ─────────────────────────
# Groq  (FREE) : https://console.groq.com  → API Keys
# Anthropic    : https://console.anthropic.com → API Keys (has free tier)
# The app tries Groq first; if it is network-blocked it falls back to Anthropic.
GROQ_API_KEY      = "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"       # ← free from console.groq.com
GROQ_MODEL        = "llama3-70b-8192"
ANTHROPIC_API_KEY = "YOUR_ANTHROPIC_API_KEY_HERE"  # ← fallback key
ANTHROPIC_MODEL   = "claude-haiku-4-5-20251001"


def build_telemetry_context(df, pwr, anomalies, eng_hist, mode, mission_id, n_anom):
    """
    Build a comprehensive plain-text telemetry snapshot to feed to the LLM.
    Covers propulsion, GNC, avionics, anomaly events, engine health history,
    and channel-level statistics so the model can answer any engineering query.
    """
    lines = []
    sep   = "─" * 60

    lines.append("=== AGNISIGHT TELEMETRY CONTEXT ===")
    lines.append(f"Mission ID    : {mission_id}")
    lines.append(f"Vehicle       : Agnibaan ST — Agnilet SN-07 engine")
    lines.append(f"Flight Mode   : {mode}")
    lines.append(f"Total rows    : {len(df)}")
    lines.append(f"Powered-ascent rows (phase=2): {len(pwr)}")
    lines.append(sep)

    # ── Propulsion ─────────────────────────────────────────────────
    lines.append("PROPULSION — AGNILET SN-07")
    lines.append(f"  Chamber pressure (bar)   : mean={pwr['chamber_pres'].mean():.2f}  max={pwr['chamber_pres'].max():.2f}  min={pwr['chamber_pres'].min():.2f}  std={pwr['chamber_pres'].std():.3f}  [nominal 68 bar]")
    lines.append(f"  Thrust (kN)              : mean={pwr['thrust_kN'].mean():.2f}  max={pwr['thrust_kN'].max():.2f}  min={pwr['thrust_kN'].min():.2f}  [nominal 100 kN]")
    lines.append(f"  Total mass flow (kg/s)   : mean={pwr['total_flow_kgs'].mean():.3f}  max={pwr['total_flow_kgs'].max():.3f}")
    lines.append(f"  Oxidiser flow (kg/s)     : mean={pwr['ox_flow_kgs'].mean():.3f}")
    lines.append(f"  Fuel flow (kg/s)         : mean={pwr['fuel_flow_kgs'].mean():.3f}")
    lines.append(f"  OF ratio                 : mean={pwr['OF_ratio'].mean():.4f}  std={pwr['OF_ratio'].std():.4f}  max={pwr['OF_ratio'].max():.4f}  [nominal 2.10]")
    lines.append(f"  Nozzle temp (°C)         : mean={pwr['nozzle_temp_C'].mean():.1f}  max={pwr['nozzle_temp_C'].max():.1f}  [limit 1950 °C]")
    lines.append(f"  Nozzle margin to limit   : {1950 - pwr['nozzle_temp_C'].max():.1f} °C")
    lines.append(f"  Turbopump RPM            : mean={pwr['turbo_rpm'].mean():.0f}  max={pwr['turbo_rpm'].max():.0f}  min={pwr['turbo_rpm'].min():.0f}  [design 32000]")
    lines.append(f"  LOX tank pressure (bar)  : mean={pwr['lox_tank_pres'].mean():.2f}  min={pwr['lox_tank_pres'].min():.2f}")
    lines.append(f"  Fuel tank pressure (bar) : mean={pwr['fuel_tank_pres'].mean():.2f}  min={pwr['fuel_tank_pres'].min():.2f}")
    prop_consumed = pwr["total_flow_kgs"].sum() / SAMPLE_RATE
    Isp_est = pwr["thrust_kN"].mean() * 1000 / 9.80665 / max(pwr["total_flow_kgs"].mean(), 0.01)
    lines.append(f"  Total propellant consumed: {prop_consumed:.1f} kg")
    lines.append(f"  Estimated Isp            : {Isp_est:.0f} s")
    lines.append(sep)

    # ── Trajectory / GNC ──────────────────────────────────────────
    phase3 = df[df["phase"] == 3]
    meco_t = float(phase3["time_s"].min()) if len(phase3) > 0 else float(df["time_s"].max())
    lines.append("TRAJECTORY & GNC")
    lines.append(f"  Peak altitude (km)       : {df['altitude_km'].max():.3f}")
    lines.append(f"  Peak Mach                : {df['mach'].max():.3f}")
    lines.append(f"  Peak vertical velocity   : {df['vel_v_ms'].max():.1f} m/s")
    lines.append(f"  Peak dynamic pressure    : {df['dyn_pres_Pa'].max():.0f} Pa")
    lines.append(f"  MECO time                : T+{meco_t:.1f} s")
    lines.append(f"  Pitch (°) powered        : mean={pwr['pitch_deg'].mean():.2f}  std={pwr['pitch_deg'].std():.3f}  max={pwr['pitch_deg'].max():.2f}")
    lines.append(f"  Yaw   (°) powered        : mean={pwr['yaw_deg'].mean():.2f}  std={pwr['yaw_deg'].std():.3f}  max={pwr['yaw_deg'].max():.2f}")
    lines.append(f"  Roll  (°) powered        : mean={pwr['roll_deg'].mean():.2f}  std={pwr['roll_deg'].std():.3f}")
    lines.append(f"  Peak pitch rate (deg/s)  : {df['q_rate_degs'].abs().max():.3f}")
    lines.append(f"  Axial accel (m/s²) mean  : {pwr['accel_x'].mean():.2f}")
    lines.append(sep)

    # ── Avionics ───────────────────────────────────────────────────
    lines.append("AVIONICS & POWER")
    lines.append(f"  Bus voltage (V)          : mean={df['bus_voltage_V'].mean():.3f}  min={df['bus_voltage_V'].min():.3f}  max={df['bus_voltage_V'].max():.3f}  [nominal 28.0 V]")
    lines.append(f"  CPU temp (°C)            : mean={df['cpu_temp_C'].mean():.1f}  max={df['cpu_temp_C'].max():.1f}")
    lines.append(f"  Signal strength (dBm)    : mean={df['signal_dbm'].mean():.1f}  min={df['signal_dbm'].min():.1f}")
    lines.append(sep)

    # ── Isolation Forest results ───────────────────────────────────
    lines.append("ISOLATION FOREST — ANOMALY DETECTION")
    lines.append(f"  Model                    : IsolationForest (n_estimators=300, contamination=1.5%)")
    lines.append(f"  Features monitored       : {len(FEATURE_COLS)} channels")
    lines.append(f"  Total anomaly events     : {n_anom}")
    lines.append(f"  Min IF score (overall)   : {df['if_score'].min():.4f}  (more negative = more anomalous)")
    if anomalies:
        for idx, ev in enumerate(anomalies, 1):
            ch_human = CH_DESC.get(ev["channel"], ev["channel"])
            root     = ROOT_CAUSE.get(ev["channel"], "unknown")
            lines.append(f"  Event {idx}: T+{ev['t_start']:.1f}s – T+{ev['t_end']:.1f}s  dur={ev['dur']:.1f}s  "
                         f"channel={ch_human}  severity={ev['severity']}  IF_score={ev['score']}  "
                         f"root_cause_hypothesis={root}")
    else:
        lines.append("  No anomaly events — all channels within nominal envelope throughout powered ascent.")
    lines.append(sep)

    # ── Engine health history ──────────────────────────────────────
    lines.append("ENGINE HEALTH HISTORY — AGNILET SN-07")
    for _, row in eng_hist.iterrows():
        lines.append(f"  {row['fire_id']}  {row['type']:12s}  health={row['health_score']:.1f}%  "
                     f"RUL={row['rul_cycles']} cycles  peak_Pc={row['peak_Pc_bar']} bar  "
                     f"avg_thrust={row['avg_thrust_kN']} kN  nozzle_peak={row['nozzle_peak_C']} °C")
    lat = eng_hist.iloc[-1]
    lines.append(f"  Current health score     : {lat['health_score']:.1f}%")
    lines.append(f"  Remaining useful life    : {int(lat['rul_cycles'])} firing cycles")
    lines.append(f"  Degradation rate         : ~1.45% per firing")
    lines.append(sep)

    # ── Channel descriptions ───────────────────────────────────────
    lines.append("MONITORED CHANNEL DESCRIPTIONS")
    for ch, desc in CH_DESC.items():
        lines.append(f"  {ch:20s} : {desc}")
    lines.append(sep)

    return "\n".join(lines)


def _build_system_prompt():
    return (
        "You are AGNISIGHT-AI, an expert aerospace telemetry analyst for Agnikul Cosmos. "
        "You are given a complete snapshot of a rocket flight's telemetry data including "
        "propulsion, GNC, avionics, Isolation Forest anomaly detection results, and engine health history. "
        "Answer the engineer's question accurately, concisely, and with specific numbers from the telemetry data. "
        "If an anomaly was detected, explain it with its timing, channel, severity, and root cause hypothesis. "
        "Use engineering terminology. Do NOT make up numbers — only use values from the provided telemetry context. "
        "Format your answer in plain readable paragraphs — no markdown headers, no bullet symbols."
    )


def _call_groq(user_question: str, telemetry_context: str) -> str:
    """Call Groq API (free tier). Raises exception if network blocked or key invalid."""
    import urllib.request, json as _json
    payload = _json.dumps({
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": _build_system_prompt()},
            {"role": "user",   "content": f"TELEMETRY DATA:\n{telemetry_context}\n\nENGINEER QUESTION: {user_question}"},
        ],
        "temperature": 0.3,
        "max_tokens": 1024,
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {GROQ_API_KEY}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        result = _json.loads(resp.read().decode("utf-8"))
    return result["choices"][0]["message"]["content"].strip()


def _call_anthropic(user_question: str, telemetry_context: str) -> str:
    """Call Anthropic API as fallback. Raises exception if key invalid."""
    import urllib.request, json as _json
    payload = _json.dumps({
        "model": ANTHROPIC_MODEL,
        "max_tokens": 1024,
        "system": _build_system_prompt(),
        "messages": [{"role": "user",
                      "content": f"TELEMETRY DATA:\n{telemetry_context}\n\nENGINEER QUESTION: {user_question}"}],
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={"Content-Type": "application/json",
                 "x-api-key": ANTHROPIC_API_KEY,
                 "anthropic-version": "2023-06-01"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = _json.loads(resp.read().decode("utf-8"))
    return result["content"][0]["text"].strip()


def query_llm(user_question: str, telemetry_context: str) -> tuple[str, str]:
    """
    Smart dual-backend LLM query.
    Tries Groq first (free, fast). If Groq is network-blocked (403) or key missing,
    automatically falls back to Anthropic API.
    Returns (answer_text, backend_used).
    """
    groq_key_set  = GROQ_API_KEY      and GROQ_API_KEY      != "YOUR_GROQ_API_KEY_HERE"
    anth_key_set  = ANTHROPIC_API_KEY and ANTHROPIC_API_KEY != "YOUR_ANTHROPIC_API_KEY_HERE"

    if groq_key_set:
        try:
            return _call_groq(user_question, telemetry_context), "Groq · llama3-70b-8192"
        except Exception as e:
            err = str(e)
            # 403 = network blocked, fall through to Anthropic
            # 401 = bad key, raise immediately
            if "401" in err or "invalid_api_key" in err.lower():
                raise ValueError("Invalid Groq API key — check GROQ_API_KEY") from e
            if "429" in err:
                raise ValueError("Groq rate limit hit — wait a moment") from e
            # 403 / timeout / connection error → fall back silently
            if not anth_key_set:
                raise ValueError(
                    f"Groq is network-blocked (403) in this environment and no "
                    f"ANTHROPIC_API_KEY is set as fallback. "
                    f"Set ANTHROPIC_API_KEY or run the app on your local machine."
                ) from e
            # fall through to Anthropic
    elif not anth_key_set:
        raise ValueError("No API key set. Fill in GROQ_API_KEY or ANTHROPIC_API_KEY in the code.")

    # Anthropic fallback
    return _call_anthropic(user_question, telemetry_context), "Anthropic · claude-haiku"


with tabs[7]:
    st.markdown('<div class="section-title">Natural Language Telemetry Query — AI Powered (Groq / Anthropic)</div>', unsafe_allow_html=True)
    st.markdown("""<div style="font-size:12px;color:#64748b;margin-bottom:14px;line-height:1.9;">
Ask any engineering question in plain English. AGNISIGHT feeds the full telemetry snapshot — propulsion,
GNC, avionics, anomaly events, engine health — to an LLM and returns a data-grounded answer. Groq (free) is tried first; Anthropic is the automatic fallback.<br>
<i>Examples: "What caused the pressure spike?" &nbsp;·&nbsp; "How was the turbopump?" &nbsp;·&nbsp; "Was the OF ratio stable?" &nbsp;·&nbsp; "What is engine remaining life?"</i>
</div>""", unsafe_allow_html=True)

    # API key status indicator
    groq_set  = GROQ_API_KEY      and GROQ_API_KEY      != "YOUR_GROQ_API_KEY_HERE"
    anth_set  = ANTHROPIC_API_KEY and ANTHROPIC_API_KEY != "YOUR_ANTHROPIC_API_KEY_HERE"
    key_ok    = groq_set or anth_set
    if not key_ok:
        st.markdown("""<div style="background:#fef9c3;border:1px solid #fde047;border-left:3px solid #d97706;
            border-radius:4px;padding:10px 14px;font-size:12px;color:#92400e;font-family:'JetBrains Mono',monospace;margin-bottom:12px;">
            ⚠ No API key set. Add <b>GROQ_API_KEY</b> (free: console.groq.com) <i>or</i> <b>ANTHROPIC_API_KEY</b> (console.anthropic.com) in the code.
        </div>""", unsafe_allow_html=True)
    else:
        active = ("Groq · " + GROQ_MODEL) if groq_set else ("Anthropic · " + ANTHROPIC_MODEL)
        st.markdown(f"""<div style="background:#f0fdf4;border:1px solid #86efac;border-left:3px solid #16a34a;
            border-radius:4px;padding:8px 14px;font-size:11px;color:#166534;font-family:'JetBrains Mono',monospace;margin-bottom:12px;">
            ✔ LLM backend ready &nbsp;·&nbsp; Primary: {active} &nbsp;·&nbsp; Auto-fallback enabled
        </div>""", unsafe_allow_html=True)

    # Preset question buttons
    presets = [
        "What was the peak chamber pressure?",
        "Were any anomalies detected?",
        "How did the turbopump perform?",
        "Was the OF ratio stable?",
        "What is the engine remaining life?",
        "How was the GNC performance?",
        "How was the avionics and power system?",
        "Give me a full flight summary.",
        "What are the recommendations for next firing?",
    ]
    p_cols = st.columns(3)
    selected_q = None
    for i, (col, q) in enumerate(zip(p_cols * 3, presets)):
        with col:
            if st.button(q[:38] + ("…" if len(q) > 38 else ""), key=f"pq{i}"):
                selected_q = q

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    query = st.text_input(
        "Enter your engineering query:",
        value=selected_q if selected_q else "",
        placeholder="e.g. What caused the anomaly and what should we inspect?",
        key="nl_query_input",
    )

    if st.button("▶  Ask AGNISIGHT-AI", key="nl_run_btn", use_container_width=False):
        if not key_ok:
            st.error("Set GROQ_API_KEY or ANTHROPIC_API_KEY in the code before running a query.")
        elif not query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Building telemetry context · querying AI backend…"):
                try:
                    telem_ctx = build_telemetry_context(
                        df, pwr, anomalies, eng_hist, mode, mission_id, n_anom
                    )
                    answer, backend_used = query_llm(query.strip(), telem_ctx)
                    st.markdown(
                        f'<div style="font-size:10px;color:#64748b;margin-bottom:4px;'
                        f'font-family:JetBrains Mono,monospace;">answered by {backend_used}</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(f'<div class="nl-answer">{answer}</div>', unsafe_allow_html=True)

                    with st.expander("📋 Telemetry context sent to LLM", expanded=False):
                        st.code(telem_ctx, language="text")

                except ValueError as e:
                    st.error(f"❌ {e}")
                except Exception as e:
                    err_str = str(e)
                    if "timed out" in err_str.lower():
                        st.error("❌ Request timed out. Check your network connection.")
                    else:
                        st.error(f"❌ LLM error: {err_str}")


# ─────────────────────────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="margin-top:36px;border-top:1px solid #e2e8f0;padding:12px 0 4px;
  display:flex;justify-content:space-between;
  font-size:10px;color:#94a3b8;font-family:'JetBrains Mono',monospace;letter-spacing:0.5px;">
  <span>AGNISIGHT v3.0 · Agnikul Cosmos · Sriharikota</span>
  <span>ML: IsolationForest (n=300) · Features: {len(FEATURE_COLS)} · NL: Groq LLaMA3</span>
  <span>Proprietary &amp; Confidential</span>
</div>
""", unsafe_allow_html=True)
