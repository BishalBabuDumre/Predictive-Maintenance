import pandas as pd
import numpy as np

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
