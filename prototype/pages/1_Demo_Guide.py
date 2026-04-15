"""
RelayGuard -- Demo Guide page (loaded via st.navigation in app.py)
"""
import streamlit as st

st.markdown("""<style>
footer { visibility: hidden; }
[data-testid="stAppViewContainer"] { background: #0d1117; }
[data-testid="stSidebar"]          { background: #161b22 !important; }
.block-container { padding: 4.5rem 2rem 1rem; }
p, div, span, li { color: #c9d1d9; }
h1, h2, h3 { color: #e6edf3; }
hr { border-color: rgba(255,255,255,0.1) !important; }
</style>""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;gap:14px;margin-bottom:4px">
  <div style="width:44px;height:44px;background:linear-gradient(135deg,#1f6feb,#0d419d);
    border-radius:10px;display:flex;align-items:center;
    justify-content:center;font-size:22px;flex-shrink:0">&#x1F4D6;</div>
  <div>
    <div style="font-size:26px;font-weight:800;color:#e6edf3;line-height:1">Demo Guide</div>
    <div style="font-size:12px;color:#8b949e">
      RelayGuard Prototype &middot; How to use &middot; Visual legend &middot; Controls reference</div>
  </div>
</div>""", unsafe_allow_html=True)

st.markdown("<hr style='margin:12px 0 20px'>", unsafe_allow_html=True)

# ── Walkthrough ────────────────────────────────────────────────────────────────
st.markdown("## 3-Minute Demo Walkthrough")
st.markdown(
    "<div style='font-size:13px;color:#8b949e;margin-bottom:16px'>"
    "Follow these steps in order when presenting to judges. "
    "Demo target is <b style='color:#c9d1d9'>Relay K3 (HVAC Relay)</b>.</div>",
    unsafe_allow_html=True)

steps = [
    ("1", "#1f6feb", "Open the dashboard",
     "The fleet is live with <b>4 relays at different health states</b>: "
     "K1 (Healthy 93%) &middot; K2 (Warning 55%) &middot; K3 (Healthy 98%) &middot; K4 (Critical 28%). "
     "K4 already has an open <b>Work Order</b> at the top. "
     "Just point out the colour-coded cards to judges."),

    ("2", "#1f6feb", "Click Run Demo in the sidebar",
     "Watch K3 start degrading in real time &mdash; "
     "the big HI% number drops, the chart scrolls, and the trend arrow changes to "
     "<span style='color:#da3633'>Declining</span>. "
     "Use <b>Speed 10x</b> to fast-forward if time is short."),

    ("3", "#f85149", "Watch the LSTM anomaly fire  \u2190 key moment",
     "While K3 is still <b>above 70%</b> (the safe green zone), a "
     "<b style='color:#f85149'>red triangle</b> appears on the chart and a banner "
     "appears. This is the <b>LSTM autoencoder anomaly detection</b> firing "
     "<i>before</i> the rule-based 70% threshold is crossed.<br><br>"
     "<b>What to say:</b> <i>\"Our AI model detected abnormal degradation 10-20 data points "
     "before any rule-based alarm would have fired. That is the predictive advantage.\"</i>"),

    ("4", "#d29922", "Inject an Overload event",
     "Click <b>Overload</b>. A current surge is simulated &mdash; HI drops sharply, "
     "the contact resistance spikes. The K3 card starts <b>pulsing red</b>. "
     "This shows how external stress events accelerate relay degradation."),

    ("5", "#da3633", "Watch the Work Order auto-create",
     "When K3 HI drops below <b>40%</b>, the system automatically creates a "
     "<b>Work Order</b> banner at the top with estimated time to failure. "
     "No human triggered this &mdash; it is fully automated. "
     "Show the judge the <b>Acknowledge</b> button, representing a CMMS integration."),

    ("6", "#2ea043", "Reset and repeat",
     "Click <b>Reset Demo</b> to restore K3 to factory state (HI 98%). "
     "The chart clears, the Work Order disappears, and you can run the scenario again. "
     "Useful if judges want to see it twice."),
]

for num, col, title, body in steps:
    st.markdown(f"""
<div style="display:flex;gap:16px;margin-bottom:14px;
  background:#161b22;border:1px solid rgba(255,255,255,0.08);
  border-radius:12px;padding:16px 20px">
  <div style="font-size:22px;font-weight:900;color:{col};
    font-family:monospace;flex-shrink:0;padding-top:2px;min-width:24px">{num}</div>
  <div>
    <div style="font-size:14px;font-weight:700;color:#e6edf3;margin-bottom:4px">{title}</div>
    <div style="font-size:13px;color:#8b949e;line-height:1.6">{body}</div>
  </div>
</div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Two-column reference ───────────────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("## Alarm State Legend")
    alarms = [
        ("Healthy",           "#2ea043", "HI &gt; 70%",  "Card border green &middot; Normal operation"),
        ("Warning",           "#d29922", "HI 40&ndash;70%", "Card glows orange &middot; Schedule maintenance"),
        ("Critical",          "#da3633", "HI 20&ndash;40%", "Card pulses red &middot; Work order auto-created"),
        ("Failure Imminent",  "#f85149", "HI &lt; 20%",  "Rapid pulse &middot; Remove from service"),
    ]
    icons = ["\u2705", "\u26a0\ufe0f", "\U0001f6a8", "\U0001f480"]
    for (label, col, hi_range, desc), ico in zip(alarms, icons):
        st.markdown(f"""
<div style="display:grid;grid-template-columns:170px 90px 1fr;
  align-items:center;gap:10px;padding:9px 0;
  border-bottom:1px solid rgba(255,255,255,0.06);font-size:13px">
  <div style="font-weight:700;color:{col}">{ico} {label}</div>
  <div style="font-family:monospace;font-size:12px;color:#8b949e">{hi_range}</div>
  <div style="color:#8b949e">{desc}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("## Visual Cues")
    cues = [
        ("Red triangle on chart",  "LSTM anomaly marker &mdash; AI fired before the 70% rule threshold"),
        ("Declining arrow",        "5-tick trend is falling &mdash; relay degrading"),
        ("Stable arrow",           "HI is flat &mdash; relay in steady state"),
        ("Recovering arrow",       "HI is rising &mdash; recent stress event has passed"),
        ("Coloured chart bands",   "Green / Amber / Red / Dark-red zones match alarm thresholds"),
        ("Delta-HI / tick",        "XGBoost model output &mdash; negative = predicts decline"),
        ("Pulsing card glow",      "Alarm state: orange glow = Warning, red pulse = Critical"),
    ]
    for cue, meaning in cues:
        st.markdown(f"""
<div style="display:grid;grid-template-columns:180px 1fr;
  align-items:start;gap:10px;padding:7px 0;
  border-bottom:1px solid rgba(255,255,255,0.06);font-size:13px">
  <div style="font-weight:600;color:#c9d1d9">{cue}</div>
  <div style="color:#8b949e">{meaning}</div>
</div>""", unsafe_allow_html=True)

with col_b:
    st.markdown("## Sidebar Controls")
    controls = [
        ("Run Demo",     "#2ea043", "Starts K3 degrading at normal rate"),
        ("Inject Stress","#d29922", "Increases degradation rate 3.75x &mdash; fast decline"),
        ("Overload",     "#da3633", "Injects a current surge + temperature spike into K3"),
        ("Hot Spell",    "#d29922", "Raises K3 temperature by +25 deg C &mdash; Arrhenius factor"),
        ("Reset Demo",   "#2ea043", "Restores K3 to factory state: HI 98%, clears work order"),
        ("1x / 3x / 10x","#8b949e","Simulation speed &mdash; 10x runs full decay in ~30 s"),
        ("Monitor",      "#8b949e", "Selects which relay to show in the live monitor panel"),
    ]
    for name, col, desc in controls:
        st.markdown(f"""
<div style="display:flex;gap:12px;padding:9px 0;
  border-bottom:1px solid rgba(255,255,255,0.06);font-size:13px">
  <div style="font-weight:700;color:{col};min-width:130px;flex-shrink:0">{name}</div>
  <div style="color:#8b949e">{desc}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("## System Architecture")
    st.markdown("""
<div style="background:#161b22;border:1px solid rgba(255,255,255,0.08);
  border-radius:12px;padding:20px;font-size:13px;line-height:2.2">
  <div>
    <span style="color:#1f6feb;font-weight:700;font-family:monospace">L1</span>
    <span style="color:#8b949e;margin-left:10px">IoT Sensors</span>
    <span style="color:#484f58;margin-left:10px">ACS712 &middot; MLX90614 &middot; INA219 &middot; ZMPT101B</span>
  </div>
  <div style="color:#484f58;padding-left:24px;font-size:11px">&darr; ADC / I2C</div>
  <div>
    <span style="color:#1f6feb;font-weight:700;font-family:monospace">L2</span>
    <span style="color:#8b949e;margin-left:10px">ESP32 FreeRTOS</span>
    <span style="color:#484f58;margin-left:10px">Health Index engine &middot; OLED &middot; LEDs</span>
  </div>
  <div style="color:#484f58;padding-left:24px;font-size:11px">&darr; MODBUS RTU / RS485</div>
  <div>
    <span style="color:#1f6feb;font-weight:700;font-family:monospace">L3</span>
    <span style="color:#8b949e;margin-left:10px">Raspberry Pi Gateway</span>
    <span style="color:#484f58;margin-left:10px">Poll &middot; SQLite &middot; MQTT &middot; FastAPI</span>
  </div>
  <div style="color:#484f58;padding-left:24px;font-size:11px">&darr; MQTT / TLS</div>
  <div>
    <span style="color:#1f6feb;font-weight:700;font-family:monospace">L4</span>
    <span style="color:#8b949e;margin-left:10px">Cloud ML</span>
    <span style="color:#484f58;margin-left:10px">XGBoost Delta-HI &middot; LSTM Anomaly &middot; Weibull RUL</span>
  </div>
  <div style="color:#484f58;padding-left:24px;font-size:11px">&darr; HTTPS / WebSocket</div>
  <div>
    <span style="color:#1f6feb;font-weight:700;font-family:monospace">L5</span>
    <span style="color:#8b949e;margin-left:10px">Dashboard</span>
    <span style="color:#484f58;margin-left:10px">This prototype (Streamlit)</span>
  </div>
</div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="background:rgba(31,111,235,.08);border:1px solid rgba(31,111,235,.25);
  border-radius:10px;padding:14px 18px;font-size:13px;color:#8b949e">
  <b style="color:#c9d1d9">Tip for judges</b> &mdash; All sensor data in this prototype is
  physics-simulated using the Arrhenius thermal aging model and arc-energy accumulation
  from the relay datasheet. The Delta-HI target architecture eliminates covariate shift
  between training and test data, which was the key ML breakthrough in this project.
</div>""", unsafe_allow_html=True)
