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
    
    # 3. Define X and y
    # Input X: Cyclical time features
    # Output y: Temperature
    features = ["hour_sin", "hour_cos", "doy_sin", "doy_cos", "month_sin", "month_cos"]
    X = df[features].values
    y = df[["Temperature(F)"]].values
    
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
