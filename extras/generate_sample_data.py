"""
generate_sample_data.py  (v2 — physics-based, multi-stage degradation)
=======================================================================
Generates a realistic relay_data.xlsx for training RelayGuard ML.

Key improvements over v1:
  • Three-phase degradation curve (burn-in → normal life → wear-out) instead
    of a naive linear ramp — mirrors the classic bathtub reliability curve.
  • All degradation stages appear THROUGHOUT the dataset, not just at the end,
    so 80/20 train/test splits see balanced HI distributions.
  • Realistic sensor correlations:
      - Temperature rises non-linearly as contacts degrade
      - Contact resistance follows the Holm arc erosion model
      - Current spikes (overloads) increase in frequency as relay ages
      - Switching frequency drops as relay slows down (wear)
  • Thermal events: occasional over-temperature episodes
  • Overload events: random current surges that accelerate degradation
  • Gaussian sensor noise tuned to realistic measurement uncertainty

Run ONCE, then run RelayGuard_Colab.py.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────────────────────
N              = 3000     # total readings
DT             = 10.0     # seconds between readings
CYCLES_PER_READ = 3       # average switching cycles per reading

# Noise levels (standard deviations)
NOISE_CURRENT  = 0.4      # A
NOISE_VOLTAGE  = 3.0      # V
NOISE_TEMP     = 1.5      # °C
NOISE_RESIST   = 0.003    # Ω
NOISE_FREQ     = 0.02     # Hz

# Overload events
N_OVERLOADS    = 25       # number of current-surge events
OVERLOAD_MULT  = (1.8, 3.5)   # overload multiplier range

# Thermal events
N_HOT_SPELLS   = 12       # number of over-temperature episodes
HOT_SPELL_LEN  = 30       # readings long
HOT_SPELL_RISE = 20.0     # extra °C during hot spell

# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Build a three-phase degradation index (0 = new, 1 = failed)
# ─────────────────────────────────────────────────────────────────────────────
# Bathtub-inspired curve:
#   Burn-in   (0  – 10%): quick initial settling, slight early wear
#   Normal    (10 – 75%): slow, steady degradation
#   Wear-out  (75 – 100%): accelerating contact erosion
t  = np.arange(N) / N   # normalised time 0→1

burn_in   = 0.08 * np.exp(-10 * t)                        # rapid early drop
normal    = 0.45 * t                                       # linear mid-life
wear_out  = 0.47 * (np.maximum(0.0, (t - 0.75) / 0.25) ** 2.5)

degradation = (burn_in + normal + wear_out).clip(0, 1)

# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Overload events (random bursts of high current)
# ─────────────────────────────────────────────────────────────────────────────
# Overloads become more frequent in the wear-out phase
overload_prob  = 0.1 + 0.9 * degradation           # higher prob when degraded
overload_prob /= overload_prob.sum()
overload_idx   = np.random.choice(N, size=N_OVERLOADS, replace=False,
                                  p=overload_prob)
overload_mask  = np.zeros(N)
for idx in overload_idx:
    overload_mask[idx] = np.random.uniform(*OVERLOAD_MULT)

# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Thermal events (hot spells)
# ─────────────────────────────────────────────────────────────────────────────
hot_spell_mask = np.zeros(N)
spell_starts   = np.random.choice(N - HOT_SPELL_LEN, size=N_HOT_SPELLS, replace=False)
for s in spell_starts:
    hot_spell_mask[s: s + HOT_SPELL_LEN] += HOT_SPELL_RISE * (
        1 - np.linspace(0, 1, HOT_SPELL_LEN) ** 2   # tapers off
    )

# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Simulate each sensor
# ─────────────────────────────────────────────────────────────────────────────

time = np.arange(N) * DT

# Current: baseline + degradation drift + Gaussian noise + overload spikes
current_base = 5.0 + 3.0 * degradation
current      = current_base + NOISE_CURRENT * np.random.randn(N)
current     += current_base * overload_mask          # spikes during overloads
current      = current.clip(0)

# Voltage: mostly stable, small sag during overloads
voltage = (230.0
           - 8.0 * (overload_mask > 0)              # slight voltage sag on overload
           + NOISE_VOLTAGE * np.random.randn(N))

# Temperature: rises with degradation + hot spells + overload heating
temp_base = (40.0
             + 35.0 * degradation ** 1.4             # non-linear thermal rise
             + 12.0 * overload_mask                  # heating from current surge
             + hot_spell_mask)
temp = temp_base + NOISE_TEMP * np.random.randn(N)

# Cycles: cumulative count, rate slows as relay wears (sticky contacts)
cycle_rate   = CYCLES_PER_READ * (1.0 - 0.4 * degradation)   # slows with wear
cycle_delta  = np.maximum(1, np.round(cycle_rate + np.random.randn(N) * 0.5).astype(int))
cycles       = np.cumsum(cycle_delta)

# Contact resistance: Holm arc erosion model R = R0 + k * N^alpha
# plus extra jump during overloads (micro-welding / pitting)
R0      = 0.10
k_wear  = 8e-7
alpha   = 0.55
resist_model  = R0 + k_wear * (cycles ** alpha)
resist_extra  = 0.05 * (overload_mask > 0) * np.random.rand(N)   # extra pitting
resistance    = resist_model + resist_extra + NOISE_RESIST * np.abs(np.random.randn(N))
resistance    = resistance.clip(0.05)

# Peak current: higher harmonic distortion as contacts degrade
peak_current = current * (1.4 + 0.8 * degradation + 0.3 * np.random.rand(N))
peak_current = peak_current.clip(0)

# Switching frequency: degrades (sticking) as relay wears
freq_base = 0.55 - 0.20 * degradation  # Hz: healthy ~0.55, worn ~0.35
frequency = (freq_base + NOISE_FREQ * np.random.randn(N)).clip(0.05, 1.0)

# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Assemble DataFrame and save
# ─────────────────────────────────────────────────────────────────────────────
df = pd.DataFrame({
    "time":         time,
    "current":      current,
    "voltage":      voltage,
    "temp":         temp,
    "cycles":       cycles,
    "resistance":   resistance,
    "peak_current": peak_current,
    "frequency":    frequency,
})

df.to_excel("relay_data.xlsx", index=False)

# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Summary
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 55)
print("  RelayGuard — Sample Data Generator  v2")
print("=" * 55)
print(f"\n  Rows saved   : {N}")
print(f"  File         : relay_data.xlsx")
print(f"\n  Degradation stages:")
print(f"    Burn-in  (rows   0–{int(N*0.10):4d}): initial settling")
print(f"    Normal   (rows {int(N*0.10):4d}–{int(N*0.75):4d}): steady wear")
print(f"    Wear-out (rows {int(N*0.75):4d}–{N:4d}): accelerating erosion")
print(f"\n  Events injected:")
print(f"    Overload events  : {N_OVERLOADS}")
print(f"    Hot-spell events : {N_HOT_SPELLS} × {HOT_SPELL_LEN} readings")
print(f"\n  Sensor ranges:")
print(df[["current","voltage","temp","resistance","frequency"]].describe().T[
    ["min","mean","max","std"]
].round(3).to_string())

# Degradation spread check
q = [0, 0.25, 0.50, 0.75, 1.0]
print(f"\n  Degradation index quartiles (0=new, 1=failed):")
for qi in q:
    idx = min(int(qi * (N - 1)), N - 1)
    print(f"    t={qi:.0%}  →  deg={degradation[idx]:.3f}  cycles={cycles[idx]}")

print("\n✅  relay_data.xlsx ready — run RelayGuard_Colab.py next.")
