import numpy as np
import pandas as pd

def production_impute_temperature(file_input, target_col='Temperature(F)', max_forward_fill=3):
    """
    Production-grade imputation pipeline for real-time streaming data frames.
    Stresses resilience: handles short sensor drops gracefully without breaking 
    historical rolling lookback alignments.
    """

    if isinstance(file_input, str):
        df = pd.read_csv(file_input, parse_dates=["DateTime"])
    else:
        df = file_input.copy() # Keeps original dataframe pristine
        df["DateTime"] = pd.to_datetime(df["DateTime"]) # Ensure correct type
        
    df_imputed = df.copy()
    
    # Ensure dataframe is sorted by time before doing any directional imputation
    df_imputed['DateTime'] = pd.to_datetime(df_imputed['DateTime'])
    df_imputed = df_imputed.sort_values('DateTime').reset_index(drop=True)
    
    # Check if we even have NaNs to process
    nan_count = df_imputed[target_col].isnull().sum()
    if nan_count == 0:
        return df_imputed
        
    print(f"[NaN Handler] Found {nan_count} missing values. Executing imputation layers...")

    # --- Tier 1: Short-term Dropout (Forward Fill) ---
    # If a sensor drops out for 1 to 3 hours, we assume the physical space 
    # hasn't drastically changed. Limit ensures we don't carry values forward indefinitely.
    df_imputed[target_col] = df_imputed[target_col].ffill(limit=max_forward_fill)
    
    # --- Tier 2: Mid-term Dropout (Linear Interpolation) ---
    # If there are remaining gaps (e.g., a 4-hour network drop), use time-based 
    # linear interpolation to bridge the start and end of the gap smoothly.
    df_imputed = df_imputed.set_index('DateTime')
    df_imputed[target_col] = df_imputed[target_col].interpolate(method='time')
    df_imputed = df_imputed.reset_index()

    # --- Tier 3: Hard Fallback (Diurnal/Historical Default) ---
    # If the sensor dies on system startup and we have NO historical lookback 
    # to forward fill from, fallback to a safe static seasonal historical default.
    if df_imputed[target_col].isnull().any():
        fallback_default_temp = 68.0  # Calibrate based on your global training median
        df_imputed[target_col] = df_imputed[target_col].fillna(fallback_default_temp)
        print(f"[NaN Handler] Warning: Severe gap encountered. Applied static fallback default: {fallback_default_temp}°F")

    return df_imputed
