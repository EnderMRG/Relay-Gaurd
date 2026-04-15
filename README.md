# ⚡ RelayGuard
### AI-Powered Predictive Maintenance for Electrical Relays

> **Hackathon Prototype** — Real-time relay health monitoring with LSTM anomaly detection, XGBoost Δ-HI prediction, and automated work-order generation.

---

## 🔍 What Is RelayGuard?

RelayGuard is an end-to-end predictive maintenance system for electrical relay fleets. It continuously monitors physical relay parameters — temperature, contact resistance, switching cycles, and arc energy — and computes a **Health Index (HI)** score for each relay. An ML pipeline then predicts degradation *before* rule-based alarms would fire, giving operators a critical early-warning advantage.

The key ML innovation is predicting the **change in Health Index (Δ HI)** rather than absolute values, which eliminates covariate shift between training and deployment data — a common failure mode in predictive maintenance models.

---

## 🏗️ System Architecture

```
L1  IoT Sensors          ACS712 · MLX90614 · INA219 · ZMPT101B
           ↓  ADC / I2C
L2  ESP32 FreeRTOS       Health Index engine · OLED display · Status LEDs
           ↓  MODBUS RTU / RS485
L3  Raspberry Pi Gateway  Poll · SQLite · MQTT broker · FastAPI
           ↓  MQTT / TLS
L4  Cloud ML             XGBoost Δ-HI · LSTM Anomaly Detection · Weibull RUL
           ↓  HTTPS / WebSocket
L5  Dashboard            Streamlit prototype (this repo)
```

---

## ✨ Key Features

| Feature | Description |
|---|---|
| **Health Index (HI)** | Physics-based composite score (0–100%) from temp, resistance, cycles & arc energy |
| **LSTM Anomaly Detection** | Fires *before* the 70% rule threshold — detects subtle degradation patterns early |
| **XGBoost Δ-HI Prediction** | Predicts the tick-by-tick *change* in HI, eliminating covariate shift |
| **Weibull RUL Estimation** | Statistical Remaining Useful Life estimation with confidence intervals |
| **Arrhenius Thermal Model** | Physics-accurate thermal aging simulation from relay datasheet parameters |
| **Auto Work Orders** | Work orders auto-generated at Critical (HI < 40%) with estimated time to failure |
| **Real-Time Fleet View** | 4-relay fleet dashboard with colour-coded status cards and live HI charts |
| **MODBUS / MQTT** | Production gateway stack for real hardware integration |

---

## 📊 Alarm States

| State | HI Range | Visual |
|---|---|---|
| ✅ Healthy | > 70% | Green card border |
| ⚠️ Warning | 40 – 70% | Orange pulsing glow |
| 🚨 Critical | 20 – 40% | Red pulsing glow · Work Order auto-created |
| 💀 Failure Imminent | < 20% | Rapid dark-red pulse · Remove from service |

---

## 🗂️ Repository Structure

```
Relay Guard/
├── prototype/                  # Streamlit dashboard (runnable prototype)
│   ├── app.py                  # Entry point — st.navigation router
│   ├── requirements.txt
│   └── pages/
│       ├── dashboard.py        # Live fleet monitor & simulation engine
│       └── 1_Demo_Guide.py     # Judge-facing walkthrough & visual legend
│
├── extras/                     # ML training & data generation scripts
│   ├── generate_sample_data.py # Physics-based synthetic relay dataset
│   ├── train_model.py          # XGBoost Δ-HI training pipeline
│   ├── RelayGuard_Colab.py     # Google Colab training notebook script
│   ├── relay_data.xlsx         # Generated training dataset
│   ├── predictions.xlsx        # Model prediction outputs
│   └── training_results.png    # Training metric plots
│
├── models/                     # Trained model artefacts
│   ├── relay_guard_model.pkl   # XGBoost Δ-HI model
│   ├── scaler.pkl              # Feature scaler
│   └── RelayGaurd.ipynb        # Training notebook
│
└── docs/                       # Technical documentation (LaTeX + PDF)
    ├── ML_Model.tex / .pdf
    ├── RelayGuard_Complete_System_Guide.tex / .pdf
    └── Relay_Health_System_Dev_Guide.pdf
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- pip

### 1. Clone & install

```bash
git clone <repo-url>
cd "Relay Guard/prototype"
pip install -r requirements.txt
```

### 2. Run the dashboard

```bash
python -m streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🎬 3-Minute Demo Script (for judges)

1. **Open the dashboard** — observe the 4-relay fleet at different health states: K1 (93% Healthy), K2 (55% Warning), K3 (98% Healthy), K4 (28% Critical with open Work Order).
2. **Click "Run Demo"** in the sidebar — watch Relay K3 degrade in real time. Use **10x speed** to fast-forward.
3. **Watch the LSTM anomaly fire** — while K3 is still *above* 70% (green/safe zone), a red triangle appears on the chart. This is the AI catching degradation *before* any rule-based alarm would trigger.
4. **Click "Overload"** — injects a current surge and temperature spike. K3 drops sharply and the card starts pulsing red.
5. **Watch Work Order auto-create** — when K3 HI drops below 40%, a Work Order banner appears automatically with an estimated time to failure. No human triggered this.
6. **Click "Reset Demo"** to restore K3 to factory state (HI 98%) and repeat.

> **Key talking point:** *"Our AI detected abnormal degradation 10–20 data points before any rule-based alarm would have fired. That is the predictive advantage."*

---

## 🧠 ML Details

### Health Index Formula

```
HI = (0.40 × wear_remaining) + (0.25 × thermal_margin) + (0.25 × resistance_margin) + (0.10 × cycle_remaining)
```

All components are normalised to [0, 1] against datasheet limits.

### Δ-HI Architecture

Instead of predicting absolute HI (which suffers from covariate shift), the XGBoost model predicts the **change** in HI per tick (`Δ HI`). This means the model trains on a distribution-invariant target and generalises significantly better to unseen relay states.

### LSTM Anomaly Detection

An LSTM autoencoder is trained on healthy relay sequences. Reconstruction error above a learned threshold triggers an anomaly flag — catching early subtle degradation signatures before the composite HI score crosses any threshold.

### Thermal Aging (Arrhenius)

Arc energy accumulation is accelerated by temperature using the Arrhenius equation:

```
k(T) = exp( Ea/kB × (1/T_ref − 1/T) )
```

Where `Ea = 0.7 eV` (activation energy), matching typical electromechanical relay datasheets.

---

## 📄 Documentation

Full technical documentation is in the `docs/` folder:

- **[Complete System Guide](docs/RelayGuard_Complete_System_Guide.pdf)** — end-to-end architecture, hardware build, firmware, and cloud stack
- **[ML Model Report](docs/ML_Model.pdf)** — feature engineering, Δ-HI rationale, model evaluation
- **[Dev Guide](docs/Relay_Health_System_Dev_Guide.pdf)** — developer setup and contribution guide

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Dashboard | Python · Streamlit · Plotly |
| ML Models | XGBoost · scikit-learn · LSTM (Keras/TF) |
| Data | Pandas · NumPy · OpenPyXL |
| Firmware (planned) | ESP32 · FreeRTOS · MODBUS RTU |
| Gateway (planned) | Raspberry Pi · SQLite · FastAPI · MQTT |
| Docs | LaTeX |

---

## 📜 License

This project was built for a hackathon. All rights reserved by the authors.
