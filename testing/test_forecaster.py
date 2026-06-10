import os
import numpy as np
import pandas as pd
import onnxruntime as ort
import joblib
from testing.edge_cases import inject_edge_cases
from testing.metric_utils import get_error_summary
from training.feature_engineering import prepare_data_frame
from testing.fill_missing import production_impute_temperature
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_pipeline(data_source, onnx_vae_model_path="data/model/vae_model.onnx",
                      scaler_X_path="data/model/scaler_X.pkl",
                      onnx_forecast_model_path="data/model/forecast_model.onnx",
                      scaler_y_path="data/model/scaler_target.pkl",
                      scaler_mu_path="data/model/scaler_mu.pkl"
                     ):
    """
    Evaluates every single valid timestamp in the dataset sequentially.
    Accepts either a file path string or a pre-loaded pandas DataFrame.
    """
    # 1. Initialize Runtime
    session_vae = ort.InferenceSession(onnx_vae_model_path)
    session_forecast = ort.InferenceSession(onnx_forecast_model_path)

    #Calling imputation function to fill NaNs
    data_source = production_impute_temperature(data_source)
    
    # 2. Extract and Process Features (Handles internal 7-day rolling calculations)
    df_valid, features, target = prepare_data_frame(data_source)
    
    if df_valid["DateTime"].dt.tz is not None:
        df_valid["DateTime"] = df_valid["DateTime"].dt.tz_localize(None)
    
    if df_valid.empty:
        raise ValueError("No valid rows left to evaluate after feature engineering. Check imputation logic.")

    # --- NEW: Calculate Naïve Persistence Baseline ---
    # Shift target values by 1 step (1 hour) to represent "t-1"
    # Use the raw data_source or df_valid depending on where the continuous timeline is intact
    df_valid['Persistence_Pred'] = df_valid[target].shift(1)
    
    # Drop the very first row only for baseline comparison since it won't have a t-1 value
    df_metrics = df_valid.dropna(subset=['Persistence_Pred']).copy()

    X_raw = df_metrics[features].values.astype(np.float32)
    Y_raw = df_metrics[target].values.astype(np.float32)
    Y_persistence = df_metrics['Persistence_Pred'].values.astype(np.float32)
  
    # 3. Scale X Data using Production Scaler (Transform Only)
    if not os.path.exists(scaler_X_path):
        raise FileNotFoundError(f"Scaler not found at {scaler_X_path}.")
    scaler_X = joblib.load(scaler_X_path)
    X_scaled = scaler_X.transform(X_raw)    
    
    # 4. Batch Inference over all Timestamps
    onnx_inputs_vae = {session_vae.get_inputs()[0].name: X_scaled}
    _, mu, _ = session_vae.run(None, onnx_inputs_vae)

    # 5. Scale mu in order to feed it into the ONNX model
    scaler_mu = joblib.load(scaler_mu_path)
    mu_scaled = scaler_mu.transform(mu)    
                          
    # 6. Batch Inference over all Timestamps
    onnx_inputs_forecast = {session_forecast.get_inputs()[0].name: mu_scaled}
    _, predictions_scaled, _ = session_forecast.run(None, onnx_inputs_forecast)

    # 7. Scale Y Data using Production Scaler (Transform Only)
    if not os.path.exists(scaler_y_path):
        raise FileNotFoundError(f"Scaler not found at {scaler_y_path}.")
    scaler_y = joblib.load(scaler_y_path)
    predictions = scaler_y.inverse_transform(predictions_scaled)

    # 8. Ensure arrays are strictly 1D using .flatten() to prevent broadcasting issues
    Y_raw = Y_raw.flatten()
    predictions = predictions.flatten()
    Y_persistence = Y_persistence.flatten()
                         
    # 8. Vectorized calculation: Mean Square Error along axis 1 (features)
    df_metrics['AE'] = np.abs(Y_raw - predictions) #Absolute Error
    df_metrics['RMSE'] = np.sqrt(((Y_raw - predictions)**2).mean())
    df_metrics['R2'] = r2_score(Y_raw, predictions)

    # Persistence Baseline Errors
    df_metrics['Persistence_AE'] = np.abs(Y_raw - Y_persistence)
    df_metrics['Persistence_RMSE'] = np.sqrt(((Y_raw - Y_persistence)**2).mean())
    df_metrics['Persistence_R2'] = r2_score(Y_raw, Y_persistence)
    
    return df_metrics


if __name__ == "__main__":
    # Define paths
    clean_file_path = "data/raw/testing_data.csv"  # Swap out for your half-year file path
    onnx_forecast_path = "data/model/forecast_model.onnx"
    onnx_vae_path = "data/model/vae_model.onnx"
    scaler_X_path = "data/model/scaler_X.pkl"
    scaler_y_path = "data/model/scaler_target.pkl"
    scaler_mu_path="data/model/scaler_mu.pkl"
    
    print("Step 1: Running baseline evaluation on every timestamp (Clean Data)...")
    df_clean_results = evaluate_pipeline(clean_file_path, onnx_vae_path, scaler_X_path, onnx_forecast_path, scaler_y_path, scaler_mu_path)
    
    # Printing Errors
    print(f"Evaluated {len(df_clean_results)} baseline timestamps.")
    print(f"""Errors (Baseline/Persistence):
    RMSE:
    {df_clean_results['RMSE'].mean():.6f}, / {df_clean_results['Persistence_RMSE'].mean():.6f}
    R²:
    {df_clean_results['R2'].mean():.6f}, / {df_clean_results['Persistence_R2'].mean():.6f}""")

    final_df = get_error_summary(df_clean_results)
    print(final_df)
    
    # Plotting Distribution of Errors:
    print("Plotting Distribution of Errors for Clean Data:")
    data = df_clean_results['AE']
    hist, bins = np.histogram(data, bins=20)
    max_count = hist.max()
    for count, left, right in zip(hist, bins[:-1], bins[1:]):
        bar = "█" * int(50 * count / max_count)
        print(f"{left:8.4f} - {right:8.4f} | {bar} ({count})")
    
    # Load raw data into memory to perform automated edge injections
    raw_df = pd.read_csv(clean_file_path)
    df_perturbed_input = inject_edge_cases(raw_df)
    
    print("\nStep 2: Running pipeline evaluation across corrupted dataset...")
    df_perturbed_results = evaluate_pipeline(df_perturbed_input, onnx_vae_path, scaler_X_path, onnx_forecast_path, scaler_y_path, scaler_mu_path)
    
    # --- Step 3: Analysis & Verification ---
    print("\n--- Edge Case Testing Diagnostic Report ---")
    
    # Merge on DateTime to cross-reference clean vs corrupted reconstruction anomalies
    diagnostic_df = pd.merge(
        df_clean_results[['DateTime', 'Temperature(F)', 'AE', 'RMSE', 'R2']],
        df_perturbed_results[['DateTime', 'Temperature(F)', 'AE', 'RMSE', 'R2']],
        on='DateTime',
        suffixes=('_Clean', '_Perturbed')
    )
    
    # Look at the final 10 rows to see the immediate effect of the final spike
    print("\nChecking tail observations (Targeting Immediate Spike):")
    print(diagnostic_df[['DateTime', 'Temperature(F)_Clean', 'Temperature(F)_Perturbed', 'AE_Clean', 'RMSE_Clean', 'R2_Clean', 'AE_Perturbed', 'RMSE_Perturbed', 'R2_Perturbed']].tail(5).to_string(index=False))
    
    # Check intermediate segments where Flatline occurred
    # Locate where the perturbed data was forced to 72.0 while the clean data differed
    flatline_mask = (diagnostic_df['Temperature(F)_Perturbed'] == 72.0) & (diagnostic_df['Temperature(F)_Clean'] != 72.0)
    if flatline_mask.any():
        print("\nChecking segment during active Flatline Sensor Failure:")
        print(diagnostic_df[flatline_mask][['DateTime', 'Temperature(F)_Clean', 'AE_Clean', 'RMSE_Clean', 'R2_Clean', 'AE_Perturbed', 'RMSE_Perturbed', 'R2_Perturbed']].head(5).to_string(index=False))
        
    # Validation Rule assertion
    max_perturbed_mae_loss = diagnostic_df['AE_Perturbed'].max()
    median_clean_mae_loss = diagnostic_df['AE_Clean'].median()
    
    print("\n--- Final Framework Verdict ---")
    if max_perturbed_mae_loss > (median_clean_mae_loss * 10):
        print(f"PASSED: The pipeline caught the injected anomalies successfully.")
        print(f"Peak Perturbed Error: {max_perturbed_mae_loss:.5f} vs. Normal Median Error: {median_clean_mae_loss:.5f}")
    else:
        print("FAILED/WARNING: Injected anomalies did not show distinct elevation profiles in loss space. Check your scaling bounds or feature engineering imputation defaults.")
