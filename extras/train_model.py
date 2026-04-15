"""
RelayGuard ML Training Pipeline
================================
Based on: ML_Model.pdf (RelayGuard ML System Documentation)

Pipeline:
  Sensor Data (XLSX) → Physics Model (HI) → Feature Engineering → ML Model → RUL Prediction

Required XLSX columns:
  current, voltage, temp, cycles, resistance, peak_current, frequency, time
  (dT_dt will be computed automatically if not present)

Output:
  - Trained XGBoost model saved as relay_guard_model.pkl
  - Feature importance plot
  - Prediction results saved as predictions.xlsx
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from xgboost import XGBRegressor

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  ← Edit these as needed
# ─────────────────────────────────────────────────────────────────────────────
DATA_FILE     = "relay_data.xlsx"     # Your input xlsx file
SHEET_NAME    = 0                     # Sheet index or name
MODEL_OUT     = "relay_guard_model.pkl"
SCALER_OUT    = "scaler.pkl"
PRED_OUT      = "predictions.xlsx"

# Physics constants
Ea            = 0.7       # Activation energy (eV) — typical for relay contacts
k_bolt        = 8.617e-5  # Boltzmann constant (eV/K)
T_ref         = 298.15    # Reference temperature (K) = 25°C
R0            = 0.1       # Initial contact resistance (Ω)
k_wear        = 1e-6      # Resistance wear coefficient
alpha         = 0.5       # Resistance wear exponent

# HI weights (must sum to 1.0)
W_DAMAGE      = 0.40
W_THERMAL     = 0.25
W_RESISTANCE  = 0.25
W_CYCLES      = 0.10

# XGBoost hyperparameters
XGB_PARAMS = {
    "n_estimators":   200,
    "max_depth":      6,
    "learning_rate":  0.05,
    "subsample":      0.8,
    "colsample_bytree": 0.8,
    "random_state":   42,
    "n_jobs":         -1,
}

# How many steps ahead to predict HI (future HI target)
FUTURE_STEPS  = 5

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load Data
# ─────────────────────────────────────────────────────────────────────────────
def load_data(file_path: str, sheet: int | str = 0) -> pd.DataFrame:
    """Load sensor data from an Excel (.xlsx) file."""
    print(f"\n[1] Loading data from: {file_path}")
    df = pd.read_excel(file_path, sheet_name=sheet)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    print(f"    Shape: {df.shape}")
    print(f"    Columns: {list(df.columns)}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Compute Physics-Based Health Index (HI)
# ─────────────────────────────────────────────────────────────────────────────
def arrhenius_factor(T_kelvin: pd.Series) -> pd.Series:
    """
    Arrhenius thermal acceleration factor:
        K_T = exp( Ea/k * (1/T_ref - 1/T) )
    """
    return np.exp((Ea / k_bolt) * (1 / T_ref - 1 / T_kelvin))


def compute_arc_damage(df: pd.DataFrame) -> pd.Series:
    """
    Arc energy per cycle:  E = I^2 * V * t
    Total damage:          W_total = Σ(E * K_T)
    """
    dt = df["time"].diff().fillna(1.0)                # time step (seconds)
    T_kelvin = df["temp"] + 273.15                    # °C → K
    K_T = arrhenius_factor(T_kelvin)
    E = df["current"] ** 2 * df["voltage"] * dt       # arc energy per step
    W = (E * K_T).cumsum()                            # cumulative damage
    return W


def compute_hi_components(df: pd.DataFrame) -> pd.DataFrame:
    """Compute individual HI sub-scores (0–1, where 1 = healthy)."""

    # --- HI_damage (arc erosion) ---
    W = compute_arc_damage(df)
    W_max = W.max() if W.max() > 0 else 1.0
    HI_damage = 1.0 - (W / W_max).clip(0, 1)

    # --- HI_thermal (Arrhenius) ---
    T_kelvin = df["temp"] + 273.15
    K_T = arrhenius_factor(T_kelvin)
    K_T_norm = ((K_T - K_T.min()) / (K_T.max() - K_T.min() + 1e-9)).clip(0, 1)
    HI_thermal = 1.0 - K_T_norm

    # --- HI_resistance (R = R0 + k*N^α) ---
    R_model = R0 + k_wear * (df["cycles"] ** alpha)
    R_norm  = ((R_model - R_model.min()) / (R_model.max() - R_model.min() + 1e-9)).clip(0, 1)
    HI_resistance = 1.0 - R_norm

    # --- HI_cycles (mechanical wear) ---
    N_max = df["cycles"].max() if df["cycles"].max() > 0 else 1.0
    HI_cycles = 1.0 - (df["cycles"] / N_max).clip(0, 1)

    return pd.DataFrame({
        "HI_damage":     HI_damage,
        "HI_thermal":    HI_thermal,
        "HI_resistance": HI_resistance,
        "HI_cycles":     HI_cycles,
    })


def compute_composite_hi(hi_df: pd.DataFrame) -> pd.Series:
    """
    Composite HI:
        HI = 0.40*HI_damage + 0.25*HI_thermal + 0.25*HI_resistance + 0.10*HI_cycles
    """
    HI = (
        W_DAMAGE     * hi_df["HI_damage"]     +
        W_THERMAL    * hi_df["HI_thermal"]    +
        W_RESISTANCE * hi_df["HI_resistance"] +
        W_CYCLES     * hi_df["HI_cycles"]
    )
    return (HI * 100).clip(0, 100)   # expressed as 0–100%


def compute_health_index(df: pd.DataFrame) -> pd.DataFrame:
    """Full HI pipeline."""
    print("\n[2] Computing physics-based Health Index (HI) ...")
    hi_components = compute_hi_components(df)
    df["HI"] = compute_composite_hi(hi_components)
    df = pd.concat([df, hi_components], axis=1)
    print(f"    HI range: {df['HI'].min():.2f}% — {df['HI'].max():.2f}%")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the final feature set for ML.
    Captures both instant stress and long-term degradation.
    """
    print("\n[3] Engineering features ...")

    # Compute dT/dt if not in the file
    if "dt_dt" not in df.columns and "dT_dt" not in df.columns:
        dt = df["time"].diff().replace(0, np.nan)
        df["dT_dt"] = df["temp"].diff() / dt
        df["dT_dt"].fillna(0, inplace=True)
    else:
        col = "dt_dt" if "dt_dt" in df.columns else "dT_dt"
        df.rename(columns={col: "dT_dt"}, inplace=True)

    # Cumulative damage (W_total)
    dt = df["time"].diff().fillna(1.0)
    T_kelvin = df["temp"] + 273.15
    K_T = arrhenius_factor(T_kelvin)
    df["damage"] = (df["current"] ** 2 * df["voltage"] * dt * K_T).cumsum()

    # Rolling statistics (window = 5)
    for col in ["current", "temp", "resistance"]:
        if col in df.columns:
            df[f"{col}_roll_mean"] = df[col].rolling(5, min_periods=1).mean()
            df[f"{col}_roll_std"]  = df[col].rolling(5, min_periods=1).std().fillna(0)

    print(f"    Total features created: {df.shape[1]}")
    return df


def build_feature_matrix(df: pd.DataFrame) -> tuple[list[str], str]:
    """Define which columns are features and which is the target."""
    BASE_FEATURES = [
        "current", "voltage", "temp", "cycles", "resistance",
        "dT_dt", "peak_current", "frequency", "damage",
        "HI", "HI_damage", "HI_thermal", "HI_resistance", "HI_cycles",
    ]
    ROLLING = [c for c in df.columns if "_roll_" in c]
    FEATURE_COLS = [f for f in BASE_FEATURES + ROLLING if f in df.columns]
    TARGET_COL   = "HI_future"
    return FEATURE_COLS, TARGET_COL


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Create Target (Future HI)
# ─────────────────────────────────────────────────────────────────────────────
def create_target(df: pd.DataFrame, steps: int = FUTURE_STEPS) -> pd.DataFrame:
    """
    Shift HI forward by `steps` to create the prediction target:
        HI_future[t] = HI[t + steps]
    """
    print(f"\n[4] Creating target: HI {steps} steps ahead ...")
    df["HI_future"] = df["HI"].shift(-steps)
    df.dropna(subset=["HI_future"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"    Dataset size after target shift: {len(df)} rows")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Train XGBoost Model
# ─────────────────────────────────────────────────────────────────────────────
def train_model(df: pd.DataFrame) -> tuple:
    """Train XGBoost regressor and evaluate on test split."""
    print("\n[5] Training XGBoost model ...")

    feature_cols, target_col = build_feature_matrix(df)
    X = df[feature_cols].values
    y = df[target_col].values

    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Train / test split (80/20, no shuffle to preserve time order)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, shuffle=False
    )

    model = XGBRegressor(**XGB_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    # Evaluate
    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    print(f"\n    ── Evaluation Results ──────────────────")
    print(f"    MAE  : {mae:.4f}%")
    print(f"    RMSE : {rmse:.4f}%")
    print(f"    R²   : {r2:.4f}")
    print(f"    ────────────────────────────────────────")

    return model, scaler, feature_cols, X_test, y_test, y_pred


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Alert Classification
# ─────────────────────────────────────────────────────────────────────────────
def classify_alert(hi_value: float) -> str:
    """Map HI (%) to alert status."""
    if hi_value > 70:
        return "✅ Healthy"
    elif hi_value > 40:
        return "⚠️  Warning"
    elif hi_value > 20:
        return "🔴 Critical"
    else:
        return "🚨 Failure Imminent"


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Save Results & Plots
# ─────────────────────────────────────────────────────────────────────────────
def save_model(model, scaler, model_path: str, scaler_path: str):
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"\n[7] Model saved → {model_path}")
    print(f"    Scaler saved → {scaler_path}")


def save_predictions(df: pd.DataFrame, y_pred: np.ndarray, pred_path: str):
    """Save predictions with alert status to xlsx."""
    n = len(y_pred)
    result = df.tail(n).copy()
    result["HI_predicted"] = y_pred
    result["Alert"]        = result["HI_predicted"].apply(classify_alert)
    result.to_excel(pred_path, index=False)
    print(f"    Predictions saved → {pred_path}")


def plot_results(y_test: np.ndarray, y_pred: np.ndarray, feature_cols: list, model):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("RelayGuard ML — Training Results", fontsize=14, fontweight="bold")

    # Actual vs Predicted
    axes[0].plot(y_test,  label="Actual HI",    color="#2196F3", linewidth=1.5)
    axes[0].plot(y_pred,  label="Predicted HI", color="#FF5722", linewidth=1.5, linestyle="--")
    axes[0].axhline(70, color="green",  linestyle=":",  linewidth=1, label="Healthy threshold (70%)")
    axes[0].axhline(40, color="orange", linestyle=":",  linewidth=1, label="Warning threshold (40%)")
    axes[0].axhline(20, color="red",    linestyle=":",  linewidth=1, label="Critical threshold (20%)")
    axes[0].set_title("Actual vs Predicted HI")
    axes[0].set_xlabel("Sample")
    axes[0].set_ylabel("Health Index (%)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # Feature Importance
    importances = model.feature_importances_
    sorted_idx  = np.argsort(importances)[::-1][:15]   # top 15
    axes[1].barh(
        [feature_cols[i] for i in sorted_idx[::-1]],
        importances[sorted_idx[::-1]],
        color="#7C4DFF"
    )
    axes[1].set_title("Feature Importance (Top 15)")
    axes[1].set_xlabel("Importance Score")
    axes[1].grid(alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig("training_results.png", dpi=150)
    plt.show()
    print("    Plot saved → training_results.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  RelayGuard ML Training Pipeline")
    print("=" * 55)

    # 1. Load
    df = load_data(DATA_FILE, SHEET_NAME)

    # 2. Compute HI
    df = compute_health_index(df)

    # 3. Feature Engineering
    df = engineer_features(df)

    # 4. Create future HI target
    df = create_target(df, steps=FUTURE_STEPS)

    # 5. Train
    model, scaler, feature_cols, X_test, y_test, y_pred = train_model(df)

    # 6. Show some predictions with alerts
    print("\n[6] Sample predictions:")
    print(f"    {'Actual HI':>10} {'Predicted HI':>14} {'Alert':>22}")
    print(f"    {'-'*50}")
    for actual, pred in zip(y_test[:10], y_pred[:10]):
        alert = classify_alert(pred)
        print(f"    {actual:>10.2f}% {pred:>13.2f}%  {alert}")

    # 7. Save
    save_model(model, scaler, MODEL_OUT, SCALER_OUT)
    save_predictions(df, y_pred, PRED_OUT)
    plot_results(y_test, y_pred, feature_cols, model)

    print("\n" + "=" * 55)
    print("  Training complete!")
    print("=" * 55)


if __name__ == "__main__":
    main()
