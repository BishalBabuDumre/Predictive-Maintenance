import pandas as pd
import numpy as np
import os
from sklearn.ensemble import IsolationForest
import time

t0 = time.time()

file_path = os.path.join('data/raw/original.csv')
save_path = os.path.join('data/processed/anomalies_0.csv')

df = pd.read_csv(file_path, parse_dates=["DateTime"])
df = df.sort_values("DateTime").reset_index(drop=True)

t1 = time.time()

# Set index for rolling operations
df = df.set_index("DateTime")

# Hour of day
df["hour"] = df.index.hour

# Day of year
df["doy"] = df.index.dayofyear

# Cyclical encoding
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

df["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 365)
df["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 365)

# Rolling statistics
df["roll_mean_24h"] = df["Temperature(F)"].rolling(24).mean()
df["roll_std_24h"] = df["Temperature(F)"].rolling(24).std()

df["roll_mean_7d"] = df["Temperature(F)"].rolling(24*7).mean()

# Deviation from local expectation
df["dev_24h"] = df["Temperature(F)"] - df["roll_mean_24h"]

# Instant change
df["delta_1h"] = df["Temperature(F)"].diff()

# Slope (approx drift)
df["slope_24h"] = (df["Temperature(F)"] - df["Temperature(F)"].shift(24)) / 24
df["slope_7d"] = (df["Temperature(F)"] - df["Temperature(F)"].shift(24*7)) / (24*7)

# Detect flatlining
df["roll_std_6h"] = df["Temperature(F)"].rolling(6).std()

# Count repeated values (simple version)
df["is_same"] = (df["Temperature(F)"].diff() == 0).astype(int)
df["repeat_count"] = df["is_same"].rolling(6).sum()

df = df.dropna()

t2 = time.time()

#Train Isolation Forest
features = [
    "Temperature(F)",
    "hour_sin", "hour_cos",
    "doy_sin", "doy_cos",
    "roll_mean_24h",
    "roll_std_24h",
    "dev_24h",
    "delta_1h",
    "slope_24h",
    "slope_7d",
    "roll_std_6h",
    "repeat_count"
]

X = df[features]

model = IsolationForest(
    n_estimators=100,
    max_samples=8760,
    random_state=42,
    contamination=0.0001
)

df["anomaly"] = model.fit_predict(X)
df["score"] = model.decision_function(X)

t3 = time.time()

# Reset index so DateTime becomes a column again
df_out = df.reset_index()

# Filter anomalies
anomalies = df_out[df_out["anomaly"] == -1]

# Select useful columns
anomalies = anomalies[[
    "DateTime",
    "Temperature(F)",
    "score",
    "delta_1h"
]]

# Save to CSV
anomalies.to_csv(save_path, index=False)

t4 = time.time()

print(f"Load Data: {t1 - t0:.2f}s")
print(f"Feature engineering: {t2 - t1:.2f}s")
print(f"Model training: {t3 - t2:.2f}s")
print(f"Prediction: {t4 - t3:.2f}s")
