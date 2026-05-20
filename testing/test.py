import os
import numpy as np
import pandas as pd
import onnxruntime as ort
import joblib
from sklearn.preprocessing import MinMaxScaler
from training.feature_engineering import prepare_data_frame

def test_single_scenario(historical_file, target_timestamp, onnx_model_path="data/model/vae_model.onnx"):
    """
    Evaluates the VAE model on a specific hour using reconstruction loss.
    
    historical_file: File containing 'DateTime' and 'Temperature(F)' columns.
                   Must include enough historical rows prior to the target_timestamp
                   to satisfy the 7-day rolling window requirements.
    target_timestamp: The specific string or pd.Timestamp to evaluate.
    """
    # 1. Initialize ONNX Runtime Session
    session = ort.InferenceSession(onnx_model_path)
    
    # 2. Run the exact same feature engineering pipeline on your evaluation dataframe
    df, features, target = prepare_data_frame(historical_file)
    df["DateTime"] = df["DateTime"].dt.tz_localize(None)
    
    # Filter for our specific evaluation row
    target_row = df[df["DateTime"] == pd.to_datetime(target_timestamp)]
    
    if target_row.empty or target_row[features].isnull().values.any():
        raise ValueError(f"Insufficient history or timestamp {target_timestamp} missing to compute features.")
        
    X_raw = target_row[features].values.astype(np.float32)
    
    # 3. Load external scaler and TRANSFORM only
    scaler_path = "data/model/scaler_x.pkl"
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found at {scaler_path}. Run save_scaler.py first.")
        
    scaler_x = joblib.load(scaler_path)
    X_scaled = scaler_x.transform(X_raw) # <-- Crucial fix: use transform(), NOT fit_transform()
    
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

file_path = "testing/8_rows.csv"
target_hour = "2006-06-17 06:00:00"

print("Simulating Scenario A (Normal)...")
loss_normal = test_single_scenario(file_path, target_hour)

df_normal = pd.read_csv(file_path)
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
