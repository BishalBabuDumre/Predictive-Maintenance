import torch
import joblib
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from training.save_file import save_filename

def prepare_vae_data(df, features, target, scaler_x_name, scaler_y_name, batch_size=64):
    
    X = df[features].to_numpy()
    y = df[target].to_numpy()
    
    # 4. Scaling
    # VAEs perform best when inputs are bounded (0,1) or (-1,1)
    scaler_x = MinMaxScaler(feature_range=(-1, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))

    if scaler_x_name:
        save_filename(scaler_x, scaler_x_name)

    if scaler_y_name:
        save_filename(scaler_y, scaler_y_name)
    
    X_scaled = scaler_x.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    # 5. Create Tensors
    X_tensor = torch.FloatTensor(X_scaled)
    y_tensor = torch.FloatTensor(y_scaled)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return loader
