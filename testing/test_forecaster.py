import os
import numpy as np
import pandas as pd
import onnxruntime as ort
import joblib
from testing.metric_utils import get_error_summary
from training.feature_engineering import prepare_data_frame
from testing.fill_missing import production_impute_temperature
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def inject_edge_cases(df_clean):
    """
    Automated injector that introduces meaningful edge cases starting from 
    the end of the dataset and working backwards.
    """
    df_perturbed = df_clean.copy()
    
    # Ensure datetime sorting to cleanly target the end of the dataset
    df_perturbed['DateTime'] = pd.to_datetime(df_perturbed['DateTime'])
    df_perturbed = df_perturbed.sort_values('DateTime').reset_index(drop=True)
    
    total_rows = len(df_perturbed)
    if total_rows < 1000:
        raise ValueError("Dataset too small to reliably inject cascading multi-day edge cases.")

    print("\n--- Injecting Automated Edge Cases (From End of Dataset Backwards) ---")

    # --- Case 1: Extreme Sudden Spike (The "Impossible Jump") ---
    # Injected right at the very last row
    idx_spike = total_rows - 1
    ts_spike = df_perturbed.loc[idx_spike, 'DateTime']
    original_val = df_perturbed.loc[idx_spike, 'Temperature(F)']
    df_perturbed.loc[idx_spike, 'Temperature(F)'] = 145.0  # Physical out-of-bounds extreme
    print(f"[Injected Spike] Timestamp: {ts_spike} | Original: {original_val}°F -> Injected: 145.0°F")

    # --- Case 2: The "Flatline" / Dead Sensor (Persistent Value) ---
    # Injected a few days before the end; lasts for 48 consecutive steps (hours)
    # Target: roughly 3 to 5 days before the end
    start_flatline = total_rows - 120 
    end_flatline = start_flatline + 48
    ts_flatline_start = df_perturbed.loc[start_flatline, 'DateTime']
    
    # Lock the sensor reading to a single constant value
    flatline_value = 72.0
    df_perturbed.loc[start_flatline:end_flatline, 'Temperature(F)'] = flatline_value
    print(f"[Injected Flatline] Timestamps: {ts_flatline_start} to {df_perturbed.loc[end_flatline, 'DateTime']} (48 steps fixed at {flatline_value}°F)")

    # --- Case 3: Missing Data Gaps / NaN Imputation Strain ---
    # Injected further back (e.g., ~10 days before the end); creates a 12-hour dropout
    start_gap = total_rows - 300
    end_gap = start_gap + 12
    ts_gap_start = df_perturbed.loc[start_gap, 'DateTime']
    
    # Set to NaN to see how your `prepare_data_frame` handling behaves under stress
    df_perturbed.loc[start_gap:end_gap, 'Temperature(F)'] = np.nan
    print(f"[Injected Missing Data Gap] Timestamps: {ts_gap_start} to {df_perturbed.loc[end_gap, 'DateTime']} (12 steps set to NaN)")

    return df_perturbed


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
    
    # Mean of Errors
    print(f"Evaluated {len(df_clean_results)} baseline timestamps.")
    print(f"""Mean of Errors (Baseline/Persistence):
    Absolute Error (AE):
    {df_clean_results['AE'].mean():.6f}, / {df_clean_results['Persistence_AE'].mean():.6f}
    
    RMSE:
    {df_clean_results['RMSE'].mean():.6f}, / {df_clean_results['Persistence_RMSE'].mean():.6f}
    
    R²:
    {df_clean_results['R2'].mean():.6f}, / {df_clean_results['Persistence_R2'].mean():.6f}""")

    # Maximum of Errors
    print(f"""Maximum of Errors (Baseline/Persistence):
    AE:
    {df_clean_results['AE'].max():.6f}, / {df_clean_results['Persistence_AE'].max():.6f}
    
    RMSE:
    {df_clean_results['RMSE'].max():.6f}, / {df_clean_results['Persistence_RMSE'].max():.6f}
    
    R²:
    {df_clean_results['R2'].max():.6f}, / {df_clean_results['Persistence_R2'].max():.6f}""")

    # Minimum of Errors
    print(f"""Minimum of Errors (Baseline/Persistence):
    AE:
    {df_clean_results['AE'].min():.6f}, / {df_clean_results['Persistence_AE'].min():.6f}
    
    RMSE:
    {df_clean_results['RMSE'].min():.6f}, / {df_clean_results['Persistence_RMSE'].min():.6f}
    
    R²:
    {df_clean_results['R2'].min():.6f}, / {df_clean_results['Persistence_R2'].min():.6f}""")
    
    # Standard Deviation of Errors
    print(f"""Standard Deviation of Errors (Baseline/Persistence):
    AE:
    {df_clean_results['AE'].std():.6f}, / {df_clean_results['Persistence_AE'].std():.6f}
    
    RMSE:
    {df_clean_results['RMSE'].std():.6f}, / {df_clean_results['Persistence_RMSE'].std():.6f}
    
    R²:
    {df_clean_results['R2'].std():.6f}, / {df_clean_results['Persistence_R2'].std():.6f}""")

    # 95% Quantile of Errors
    print(f"""95% Quantile of Errors (Baseline/Persistence):
    AE:
    {df_clean_results['AE'].quantile(0.95):.6f}, / {df_clean_results['Persistence_AE'].quantile(0.95):.6f}
    
    RMSE:
    {df_clean_results['RMSE'].quantile(0.95):.6f}, / {df_clean_results['Persistence_RMSE'].quantile(0.95):.6f}
    
    R²:
    {df_clean_results['R2'].quantile(0.95):.6f}, / {df_clean_results['Persistence_R2'].quantile(0.95):.6f}""")

    # 99% Quantile of Errors
    print(f"""99% Quantile of Errors (Baseline/Persistence):
    AE:
    {df_clean_results['AE'].quantile(0.99):.6f}, / {df_clean_results['Persistence_AE'].quantile(0.99):.6f}
    
    RMSE:
    {df_clean_results['RMSE'].quantile(0.99):.6f}, / {df_clean_results['Persistence_RMSE'].quantile(0.99):.6f}
    
    R²:
    {df_clean_results['R2'].quantile(0.99):.6f}, / {df_clean_results['Persistence_R2'].quantile(0.99):.6f}""")

    # 80% Quantile of Errors
    print(f"""80% Quantile of Errors (Baseline/Persistence):
    AE:
    {df_clean_results['AE'].quantile(0.80):.6f}, / {df_clean_results['Persistence_AE'].quantile(0.80):.6f}
    
    RMSE:
    {df_clean_results['RMSE'].quantile(0.80):.6f}, / {df_clean_results['Persistence_RMSE'].quantile(0.80):.6f}
    
    R²:
    {df_clean_results['R2'].quantile(0.80):.6f}, / {df_clean_results['Persistence_R2'].quantile(0.80):.6f}""")

    # 85% Quantile of Errors
    print(f"""85% Quantile of Errors (Baseline/Persistence):
    AE:
    {df_clean_results['AE'].quantile(0.85):.6f}, / {df_clean_results['Persistence_AE'].quantile(0.85):.6f}
    
    RMSE:
    {df_clean_results['RMSE'].quantile(0.85):.6f}, / {df_clean_results['Persistence_RMSE'].quantile(0.85):.6f}
    
    R²:
    {df_clean_results['R2'].quantile(0.85):.6f}, / {df_clean_results['Persistence_R2'].quantile(0.85):.6f}""")

    # 90% Quantile of Errors
    print(f"""90% Quantile of Errors (Baseline/Persistence):
    AE:
    {df_clean_results['AE'].quantile(0.90):.6f}, / {df_clean_results['Persistence_AE'].quantile(0.90):.6f}
    
    RMSE:
    {df_clean_results['RMSE'].quantile(0.90):.6f}, / {df_clean_results['Persistence_RMSE'].quantile(0.90):.6f}
    
    R²:
    {df_clean_results['R2'].quantile(0.90):.6f}, / {df_clean_results['Persistence_R2'].quantile(0.90):.6f}""")

    # 80% Quantile of Errors
    print(f"""92% Quantile of Errors (Baseline/Persistence):
    AE:
    {df_clean_results['AE'].quantile(0.92):.6f}, / {df_clean_results['Persistence_AE'].quantile(0.92):.6f}
    
    RMSE:
    {df_clean_results['RMSE'].quantile(0.92):.6f}, / {df_clean_results['Persistence_RMSE'].quantile(0.92):.6f}
    
    R²:
    {df_clean_results['R2'].quantile(0.92):.6f}, / {df_clean_results['Persistence_R2'].quantile(0.92):.6f}""")
    
    # Plotting Distribution of Errors:
    print("Plotting Distribution of Errors for Clean Data:")
    data = df_clean_results['AE']
    hist, bins = np.histogram(data, bins=20)
    max_count = hist.max()
    for count, left, right in zip(hist, bins[:-1], bins[1:]):
        bar = "█" * int(50 * count / max_count)
        print(f"{left:8.4f} - {right:8.4f} | {bar} ({count})")

    final = get_error_summary(df_clean_results)
    print(final)
    for f in final:
        print(f)
    
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
