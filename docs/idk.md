# RelayGuard ML System Documentation (Engineer Guide)

---

# 1. System Overview

```
Sensor Data → Physics Model (HI) → ML Model → RUL
```

## Explanation

* The system does **not directly predict failure from raw sensor data**
* Instead:

  1. Sensor data is converted into a **Health Index (HI)** using physics
  2. ML model learns patterns from HI + features
  3. Output is **Remaining Useful Life (RUL)**

---

# 2. Input Features (What We Measure)

| Parameter    | Symbol | Meaning              | Used In               |
| ------------ | ------ | -------------------- | --------------------- |
| Current      | I      | Load through relay   | Arc damage model      |
| Voltage      | V      | Switching voltage    | Arc damage model      |
| Temperature  | T      | Relay heat           | Thermal model         |
| Cycles       | N      | Number of operations | Mechanical wear       |
| Resistance   | R      | Contact degradation  | Direct wear indicator |
| Temp Rate    | dT/dt  | Heating speed        | ML anomaly detection  |
| Peak Current | I_peak | Current spikes       | ML stress feature     |
| Frequency    | f      | Switching rate       | ML usage intensity    |

---

# 3. Physics-Based Modeling (Core Logic)

## 3.1 Arc Damage Model

```
E = I^2 * V * t
```

### Why used:

* Models **contact erosion during switching**
* Damage increases rapidly with current

```
W_total = Σ(E * K_T)
```

### Meaning:

* Total accumulated damage
* Used as long-term degradation indicator

---

## 3.2 Thermal Model (Arrhenius)

```
K_T = exp( Ea/k * (1/T_ref - 1/T) )
```

### Why used:

* Temperature accelerates aging
* High temperature = faster failure

---

## 3.3 Resistance Model

```
R = R0 + k * N^α
```

### Why used:

* Contact resistance increases with usage
* Direct measurable degradation signal

---

## 3.4 Composite Health Index (HI)

```
HI =
0.40 * HI_damage
+ 0.25 * HI_thermal
+ 0.25 * HI_resistance
+ 0.10 * HI_cycles
```

### Why used:

* Combines all degradation mechanisms into one value
* Gives a **single health score (0–100%)**

---

# 4. Where Each Parameter is Used

| Parameter | Used In          | Purpose                 |
| --------- | ---------------- | ----------------------- |
| I         | Damage model     | Arc energy              |
| V         | Damage model     | Arc intensity           |
| T         | Thermal model    | Aging acceleration      |
| N         | Cycle model      | Mechanical wear         |
| R         | Resistance model | Direct degradation      |
| dT/dt     | ML only          | Detect abnormal heating |
| I_peak    | ML only          | Capture spikes          |
| f         | ML only          | Usage rate              |

---

# 5. ML Training Strategy (IMPORTANT)

## Key Concept

```
We DO NOT train on raw sensor data directly
```

Instead:

```
Processed Features → Future HI or RUL
```

### Why:

* Physics model gives **structured, meaningful data**
* ML focuses on **patterns over time**
* Improves accuracy and generalization

---

# 6. Feature Engineering

Final features used for ML:

* Current (I)
* Voltage (V)
* Temperature (T)
* Cycles (N)
* Resistance (R)
* Temperature rate (dT/dt)
* Peak current (I_peak)
* Frequency (f)
* Cumulative damage (W_total)

### Purpose:

* Capture both **instant stress + long-term degradation**

---

# 7. Dataset Structure

```python
features = [
    "current", "voltage", "temp",
    "cycles", "resistance",
    "dT_dt", "peak_current", "frequency",
    "damage"
]

target = "HI_future"
```

### Explanation:

* Input = current state
* Output = future degradation
* Enables prediction before failure

---

# 8. Model Training

## Option 1: XGBoost (Recommended for Round 1)

```python
from xgboost import XGBRegressor

model = XGBRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1
)

model.fit(X_train, y_train)
```

### Why XGBoost:

* Works well with tabular data
* Fast and accurate
* No need for huge dataset

---

## Option 2: LSTM (Advanced)

* Input: time-series window (e.g., last 50 readings)
* Output: future HI or RUL

### Why:

* Captures temporal patterns
* Better long-term prediction

---

# 9. Prediction Pipeline

```
Sensor Data → HI → ML Model → RUL
```

### Steps:

1. Read sensor data
2. Compute HI using physics
3. Generate features
4. Predict future HI/RUL

---

# 10. Alert System

| HI (%) | Status           |
| ------ | ---------------- |
| >70    | Healthy          |
| 40–70  | Warning          |
| 20–40  | Critical         |
| <20    | Failure Imminent |

### Usage:

* Dashboard alerts
* Maintenance scheduling
* Automatic switching logic

---

# 11. Full System Pipeline

```
Sensors
   ↓
Physics Model (HI)
   ↓
Feature Engineering
   ↓
ML Model
   ↓
Prediction (RUL)
   ↓
Alerts
```

---

# 12. Key Insight (MOST IMPORTANT)

```
Physics Model = Ground Truth
ML Model = Pattern Learning
```

### Interpretation:

* Physics ensures correctness
* ML improves prediction accuracy
* Together → reliable predictive maintenance

---

# 13. Final Understanding

* Physics converts raw signals → meaningful degradation
* ML learns how degradation evolves over time
* System predicts failure before it happens

---
