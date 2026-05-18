def prepare_vae_data(file_path, batch_size=64):
    # 1. Load data
    df = pd.read_csv(file_path, parse_dates=["DateTime"])
    df = df.sort_values("DateTime").reset_index(drop=True)
    
    # 2. Cyclical Time Features
    # Hourly cycle (24h)
    hour = df["DateTime"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    month = df["DateTime"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * month / 24)
    df["month_cos"] = np.cos(2 * np.pi * month / 24)
    
    # Seasonal cycle (365.25 days)
    doy = df["DateTime"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    
    # 2. Immediate Autoregressive Input
    df["temp_lag_1h"] = df["Temperature(F)"].shift(1)
    
    # 3. Multi-Scale Rolling Windows (Assuming Hourly Data)
    # 3-Hour Window (Micro)
    df["roll_mean_3h"] = df["Temperature(F)"].rolling(3).mean()
    df["roll_std_3h"] = df["Temperature(F)"].rolling(3).std()
    
    # 24-Hour Window (Meso)
    df["roll_mean_24h"] = df["Temperature(F)"].rolling(24).mean()
    df["roll_std_24h"] = df["Temperature(F)"].rolling(24).std()
    
    # 7-Day Window (Macro)
    df["roll_mean_7d"] = df["Temperature(F)"].rolling(24 * 7).mean()
    df["roll_std_7d"] = df["Temperature(F)"].rolling(24 * 7).std()

    # Slope (approx drift)
    df["slope_24h"] = (df["Temperature(F)"] - df["Temperature(F)"].shift(24)) / 24
    df["slope_3h"] = (df["Temperature(F)"] - df["Temperature(F)"].shift(3)) / 3
    df["slope_7d"] = (df["Temperature(F)"] - df["Temperature(F)"].shift(24*7)) / (24*7)
    
    # 4. Target Variable: Next Step Delta (Keeps VAE honest)
    df["target_delta"] = df["Temperature(F)"] - df["temp_lag_1h"]
    
    # Drop NaNs from the large 7-day lag window
    df = df.dropna().reset_index(drop=True)
    
    # 5. Feature Selection
    features = [
        "hour_sin", "hour_cos", "doy_sin", "doy_cos", "month_sin", "month_cos",
        "temp_lag_1h",
        "roll_mean_3h", "roll_std_3h",
        "roll_mean_24h", "roll_std_24h",
        "roll_mean_7d", "roll_std_7d",
        "slope_24h", "slope_3h", "slope_7d"
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
