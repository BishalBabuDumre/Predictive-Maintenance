import os
import numpy as np
import pandas as pd
import onnxruntime as ort
import joblib
from testing.edge_cases import inject_edge_cases
from training.feature_engineering import prepare_data_frame
from testing.fill_missing import production_impute_temperature

def evaluate_pipeline(data_source, onnx_model_path="data/model/vae_model.onnx", scaler_path="data/model/scaler_x.pkl"):
    """
    Evaluates every single valid timestamp in the dataset sequentially.
    Accepts either a file path string or a pre-loaded pandas DataFrame.
    """
    # 1. Initialize Runtime
    session = ort.InferenceSession(onnx_model_path)

    #Calling imputation function to fill NaNs
    data_source = production_impute_temperature(data_source)
    
    # 2. Extract and Process Features (Handles internal 7-day rolling calculations)
    df_valid, features, target = prepare_data_frame(data_source)
    
    if df_valid["DateTime"].dt.tz is not None:
        df_valid["DateTime"] = df_valid["DateTime"].dt.tz_localize(None)
    
    if df_valid.empty:
        raise ValueError("No valid rows left to evaluate after feature engineering. Check imputation logic.")

    X_raw = df_valid[features].values.astype(np.float32)
    
    # 3. Scale Data using Production Scaler (Transform Only)
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at {scaler_path}.")
    scaler_x = joblib.load(scaler_path)
    X_scaled = scaler_x.transform(X_raw)
    
    # 4. Batch Inference over all Timestamps
    onnx_inputs = {session.get_inputs()[0].name: X_scaled}
    recon_x, mu, logvar = session.run(None, onnx_inputs)
    print(len(mu), len(X_raw), len(df_valid[target]))
    # 5. Calculate Vectorized Reconstruction Loss per row
    # Vectorized calculation: Mean Square Error along axis 1 (features)
    losses = np.mean((recon_x - X_scaled) ** 2, axis=1)
    
    # Attach tracking metrics back to the valid dataframe
    df_valid['Reconstruction_Loss'] = losses
    
    return df_valid


if __name__ == "__main__":
    # Define paths
    clean_file_path = "data/raw/testing_data.csv"  # Swap out for your half-year file path
    onnx_path = "data/model/vae_model.onnx"
    scaler_path = "data/model/scaler_X.pkl"
    
    print("Step 1: Running baseline evaluation on every timestamp (Clean Data)...")
    df_clean_results = evaluate_pipeline(clean_file_path, onnx_path, scaler_path)
    
    # Print clean benchmark
    print(f"Evaluated {len(df_clean_results)} baseline timestamps.")
    clean_losses = df_clean_results['Reconstruction_Loss'].values
    med = np.median(clean_losses)
    mad = np.median(np.abs(clean_losses - med))
    print(f"Production Configuration -> Median: {med:.6f}, MAD: {mad:.6f}")
    
    # Load raw data into memory to perform automated edge injections
    raw_df = pd.read_csv(clean_file_path)
    df_perturbed_input = inject_edge_cases(raw_df)
    
    print("\nStep 2: Running pipeline evaluation across corrupted dataset...")
    df_perturbed_results = evaluate_pipeline(df_perturbed_input, onnx_path, scaler_path)
    
    # --- Step 3: Analysis & Verification ---
    print("\n--- Edge Case Testing Diagnostic Report ---")
    
    # Merge on DateTime to cross-reference clean vs corrupted reconstruction anomalies
    diagnostic_df = pd.merge(
        df_clean_results[['DateTime', 'Temperature(F)', 'Reconstruction_Loss']],
        df_perturbed_results[['DateTime', 'Temperature(F)', 'Reconstruction_Loss']],
        on='DateTime',
        suffixes=('_Clean', '_Perturbed')
    )
    
    # Look at the final 10 rows to see the immediate effect of the final spike
    print("\nChecking tail observations (Targeting Immediate Spike):")
    print(diagnostic_df[['DateTime', 'Temperature(F)_Clean', 'Temperature(F)_Perturbed', 'Reconstruction_Loss_Clean', 'Reconstruction_Loss_Perturbed']].tail(5).to_string(index=False))
    
    # Check intermediate segments where Flatline occurred
    # Locate where the perturbed data was forced to 72.0 while the clean data differed
    flatline_mask = (diagnostic_df['Temperature(F)_Perturbed'] == 72.0) & (diagnostic_df['Temperature(F)_Clean'] != 72.0)
    if flatline_mask.any():
        print("\nChecking segment during active Flatline Sensor Failure:")
        print(diagnostic_df[flatline_mask][['DateTime', 'Temperature(F)_Clean', 'Reconstruction_Loss_Clean', 'Reconstruction_Loss_Perturbed']].head(5).to_string(index=False))
        
    # Validation Rule assertion
    max_perturbed_loss = diagnostic_df['Reconstruction_Loss_Perturbed'].max()
    median_clean_loss = diagnostic_df['Reconstruction_Loss_Clean'].median()
    
    print("\n--- Final Framework Verdict ---")
    if max_perturbed_loss > (median_clean_loss * 10):
        print(f"PASSED: The pipeline caught the injected anomalies successfully.")
        print(f"Peak Perturbed Error: {max_perturbed_loss:.5f} vs. Normal Median Error: {median_clean_loss:.5f}")
    else:
        print("FAILED/WARNING: Injected anomalies did not show distinct elevation profiles in loss space. Check your scaling bounds or feature engineering imputation defaults.")
