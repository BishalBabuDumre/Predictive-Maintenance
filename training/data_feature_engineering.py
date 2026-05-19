import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

def prepare_vae_data(file_path, batch_size=64):
    # 1. Load data
    df = pd.read_csv(file_path, parse_dates=["DateTime"])
    df = df.sort_values("DateTime").reset_index(drop=True)
    
    # 2. Cyclical Time Features
    # Hourly cycle (24h)
    hour = df["DateTime"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    # Monthly cycle (12 months)
    month = df["DateTime"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    
    # Seasonal cycle (365.25 days)
    doy = df["DateTime"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    
    # 3. Multi-Scale Rolling Windows (Assuming Hourly Data)
    # 3-Hour Window (Micro)
    df["roll_mean_3h"] = df["Temperature(F)"].rolling(3).mean()
    df["roll_std_3h"] = df["Temperature(F)"].rolling(3).std()

    # 6-Hour Window (Meso)
    df["roll_mean_6h"] = df["Temperature(F)"].rolling(6).mean()
    df["roll_std_6h"] = df["Temperature(F)"].rolling(6).std()
    
    # 24-Hour Window (Diurnal)
    df["roll_mean_24h"] = df["Temperature(F)"].rolling(24).mean()
    df["roll_std_24h"] = df["Temperature(F)"].rolling(24).std()
    
    # 7-Day Window (Macro)
    df["roll_mean_7d"] = df["Temperature(F)"].rolling(24 * 7).mean()
    df["roll_std_7d"] = df["Temperature(F)"].rolling(24 * 7).std()

    # Deviation from local expectation
    df["dev_24h"] = df["Temperature(F)"] - df["roll_mean_24h"]
    
    # Slope (approx drift)
    df["slope_24h"] = (df["Temperature(F)"] - df["Temperature(F)"].shift(24)) / 24
    df["slope_3h"] = (df["Temperature(F)"] - df["Temperature(F)"].shift(3)) / 3
    df["slope_7d"] = (df["Temperature(F)"] - df["Temperature(F)"].shift(24*7)) / (24*7)

    # --- 2nd Derivatives (Accelerations) ---
    # 1. Micro-Acceleration: Catches sudden atmospheric shocks and instant sensor jumps
    df["accel_3h"]  = df["slope_3h"].diff(1)
    
    # 2. Diurnal-Acceleration: Catches day-over-day heating/cooling cycle compounding
    df["accel_24h"] = df["slope_24h"].diff(1)
    
    # 3. Macro-Acceleration: Catches exponential, runaway hardware component degradation
    df["accel_7d"]  = df["slope_7d"].diff(1)

    # Count repeated values
    df["repeat_count"] = (df["Temperature(F)"].diff().fillna(1) == 0).astype(int).rolling(6).sum()
    
    # 4. Target Variable: Next Step Delta (Keeps VAE honest)
    df["target_delta"] = df["Temperature(F)"] - df["temp_lag_1h"]
    
    # Drop NaNs from the large 7-day lag window
    df = df.dropna().reset_index(drop=True)
    
    # 5. Feature Selection
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
    
    X = df[features].values
    y = df[["target_delta"]].values
    
    # 4. Scaling
    # VAEs perform best when inputs are bounded (0,1) or (-1,1)
    scaler_x = MinMaxScaler(feature_range=(-1, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    
    X_scaled = scaler_x.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    # 5. Create Tensors
    X_tensor = torch.FloatTensor(X_scaled)
    y_tensor = torch.FloatTensor(y_scaled)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return loader
