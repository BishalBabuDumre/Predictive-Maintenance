import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

file_path = os.path.join('data/raw/original.csv')
save_path = os.path.join('data/model/vae_model.onnx')

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=4):
        super(VAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(16, latent_dim)
        self.fc_logvar = nn.Linear(16, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

def vae_loss_function(recon_x, x, mu, logvar):
    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD

def prepare_vae_data(file_path, batch_size=64):
    # 1. Load data
    df = pd.read_csv(file_path, parse_dates=["DateTime"])
    df = df.sort_values("DateTime").reset_index(drop=True)
    
    # 2. Cyclical Time Features
    # Hourly cycle (24h)
    hour = df["DateTime"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    
    # Seasonal cycle (365.25 days)
    doy = df["DateTime"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    
    # 3. Define X and y
    # Input X: Cyclical time features
    # Output y: Temperature
    features = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]
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
    
    return loader, scaler_x, scaler_y

# Usage
# train_loader, sx, sy = prepare_vae_data('data/processed/clean_weather.csv')

# Training Loop
model = VAE(input_dim=len(features))
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(50):
    for batch in train_loader:
        data = batch[0]
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = vae_loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        optimizer.step()

# 4. Export to ONNX
model.eval()
# Create a dummy input that matches your feature dimensions
dummy_input = torch.randn(1, input_dim) 

onnx_path = "model.onnx"
torch.onnx.export(
    model, 
    dummy_input, 
    onnx_path,
    export_params=True,
    opset_version=12,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output', 'mu', 'logvar'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

print(f"Model successfully saved to {onnx_path}")
