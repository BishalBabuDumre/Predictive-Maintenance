import numpy as np
import pandas as pd
import onnxruntime as ort
from sklearn.preprocessing import MinMaxScaler
from data_feature_engineering import prepare_vae_data

def test_single_scenario(historical_df, target_timestamp, onnx_model_path="data/model/vae_model.onnx"):
    """
    Evaluates the VAE model on a specific hour using reconstruction loss.
    
    historical_df: Dataframe containing 'DateTime' and 'Temperature(F)'.
                   Must include enough historical rows prior to the target_timestamp
                   to satisfy the 7-day rolling window requirements.
    target_timestamp: The specific string or pd.Timestamp to evaluate.
    """
    # 1. Initialize ONNX Runtime Session
    session = ort.InferenceSession(onnx_model_path)
    
    # 2. Run the exact same feature engineering pipeline on your evaluation dataframe
    # We use a trick: isolate features by leveraging your existing logic
    df = historical_df.copy().sort_values("DateTime").reset_index(drop=True)
    
    # Re-apply features (mirroring data_feature_engineering.py)
    hour = df["DateTime"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    month = df["DateTime"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    doy = df["DateTime"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    
    df["roll_mean_3h"] = df["Temperature(F)"].rolling(3).mean()
    df["roll_std_3h"] = df["Temperature(F)"].rolling(3).std()
    df["roll_mean_6h"] = df["Temperature(F)"].rolling(6).mean()
    df["roll_std_6h"] = df["Temperature(F)"].rolling(6).std()
    df["roll_mean_24h"] = df["Temperature(F)"].rolling(24).mean()
    df["roll_std_24h"] = df["Temperature(F)"].rolling(24).std()
    df["roll_mean_7d"] = df["Temperature(F)"].rolling(24 * 7).mean()
    df["roll_std_7d"] = df["Temperature(F)"].rolling(24 * 7).std()
    
    df["dev_24h"] = df["Temperature(F)"] - df["roll_mean_24h"]
    df["slope_24h"] = (df["Temperature(F)"] - df["Temperature(F)"].shift(24)) / 24
    df["slope_3h"] = (df["Temperature(F)"] - df["Temperature(F)"].shift(3)) / 3
    df["slope_7d"] = (df["Temperature(F)"] - df["Temperature(F)"].shift(24*7)) / (24*7)
    
    df["accel_3h"]  = df["slope_3h"].diff(1)
    df["accel_24h"] = df["slope_24h"].diff(1)
    df["accel_7d"]  = df["slope_7d"].diff(1)
    df["repeat_count"] = (df["Temperature(F)"].diff().fillna(1) == 0).astype(int).rolling(6).sum()

    features = [
        "hour_sin", "hour_cos", "doy_sin", "doy_cos", "month_sin", "month_cos",
        "dev_24h", "repeat_count",
        "roll_mean_3h", "roll_std_3h",
        "roll_mean_6h", "roll_std_6h",
        "roll_mean_24h", "roll_std_24h",
        "roll_mean_7d", "roll_std_7d",
        "slope_3h", "slope_24h", "slope_7d",
        "accel_3h", "accel_24h", "accel_7d"
    ]
    
    # Filter for our specific evaluation row
    target_row = df[df["DateTime"] == pd.to_datetime(target_timestamp)]
    
    if target_row.empty or target_row[features].isnull().values.any():
        raise ValueError(f"Insufficient history or timestamp {target_timestamp} missing to compute features.")
        
    X_raw = target_row[features].values.astype(np.float32)
    
    # 3. Fit-Scale mirroring (Ideally, fit scaler on train set, here we approximate for isolated bounds)
    scaler_x = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = scaler_x.fit_transform(X_raw) # Switch to scaler_x.transform(X_raw) if loading an external pickle scaler.
    
    # 4. Bind ONNX Input and Run Inference
    onnx_inputs = {session.get_inputs()[0].name: X_scaled}
    # ONNX export outputs: ['output', 'mu', 'logvar']
    recon_x, mu, logvar = session.run(None, onnx_inputs)
    
    # 5. Calculate Reconstruction Error (MSE)
    reconstruction_loss = np.mean((recon_x - X_scaled) ** 2)
    
    print(f"--- Evaluation for {target_timestamp} ---")
    print(f"Raw Temperature: {target_row['Temperature(F)'].values[0]} °F")
    print(f"Reconstruction Loss (MSE): {reconstruction_loss:.6f}")
    
    return reconstruction_loss

# Create a base context timeline of 8 days (192 hours) at a baseline of ~65°F
timestamps = pd.date_range(start="2026-05-01", periods=192, freq="H")
base_temp = 65.0 + 5.0 * np.sin(2 * np.pi * timestamps.hour / 24) # standard daily oscillation

# --- Scenario A: Normal Execution ---
df_normal = pd.DataFrame({"DateTime": timestamps, "Temperature(F)": base_temp})
target_hour = "2026-05-08 23:00:00"

print("Simulating Scenario A (Normal)...")
loss_normal = test_single_scenario(df_normal, target_hour)


# --- Scenario B: Massive Spike Execution ---
df_spike = df_normal.copy()
# Inject an explicit hardware error or sudden jump at our target hour: 130°F instead of ~65°F
df_spike.loc[df_spike["DateTime"] == target_hour, "Temperature(F)"] = 130.0

print("\nSimulating Scenario B (Massive Spike)...")
loss_spike = test_single_scenario(df_spike, target_hour)


# --- Comparison Interpretation ---
print("\n--- Diagnostic Verdict ---")
if loss_spike > (loss_normal * 5): 
    print(f"Success! The spike triggered a significantly higher reconstruction error.")
    print(f"Normal Error: {loss_normal:.5f} vs. Spike Error: {loss_spike:.5f}")
else:
    print("Warning: The model's loss deviation wasn't prominent. Check feature scaling bounds.")
