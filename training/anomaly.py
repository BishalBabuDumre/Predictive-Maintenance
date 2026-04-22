import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle

# Hyperparameters
SEQ_LEN = 60  # 1 hour window
FEATURES = 4  # Voltage, Current, Irradiation, Temperature
BATCH_SIZE = 64
EPOCHS = 50
LR = 0.001

# Dataset Preparation
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len):
        self.seq_len = seq_len
        self.data = data
    def __len__(self):
        return len(self.data) - self.seq_len
    def __getitem__(self, idx):
        seq = self.data[idx:idx+self.seq_len]
        return seq

# Model Architecture
class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              padding=(kernel_size-1)*dilation, dilation=dilation)
        self.relu = nn.ReLU()
        self.norm = nn.BatchNorm1d(out_channels)
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.norm(x)
        return x

class TCN_Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            TCNBlock(input_dim, 16, dilation=1),
            TCNBlock(16, 32, dilation=2),
            TCNBlock(32, 64, dilation=4),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=3, padding=2, dilation=4),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.ConvTranspose1d(16, input_dim, kernel_size=3, padding=1, dilation=1),
        )
    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, features, sequence)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = decoded[:, :, :SEQ_LEN]
        return decoded.permute(0, 2, 1)

# Load your real normal data here as numpy array (N, 4)
# For demo:
normal_data = np.random.rand(5000, FEATURES)  # Replace with real data
dataset = TimeSeriesDataset(torch.FloatTensor(normal_data), SEQ_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = TCN_Autoencoder(input_dim=FEATURES)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    total_loss = 0
    for batch in dataloader:
        output = model(batch)
        loss = criterion(output, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

# Save model for deployment
torch.save(model.state_dict(), 'tcn_model.pth')




# Threshold calibration

model.eval()
recon_errors = []

with torch.no_grad():
    for batch in dataloader:
        output = model(batch)
        loss = nn.functional.mse_loss(output, batch, reduction='none')
        batch_errors = loss.mean(dim=[1, 2]).cpu().numpy()  # error per sequence
        recon_errors.extend(batch_errors)

recon_errors = np.array(recon_errors)
threshold = np.mean(recon_errors) + 3 * np.std(recon_errors)
print(f"Calculated Threshold: {threshold:.6f}")

# Save threshold for deployment
with open("threshold.pkl", "wb") as f:
    pickle.dump(threshold, f)




# Threshold calibration

model.eval()
recon_errors = []

with torch.no_grad():
    for batch in dataloader:
        output = model(batch)
        loss = nn.functional.mse_loss(output, batch, reduction='none')
        batch_errors = loss.mean(dim=[1, 2]).cpu().numpy()  # error per sequence
        recon_errors.extend(batch_errors)

recon_errors = np.array(recon_errors)
threshold = np.mean(recon_errors) + 3 * np.std(recon_errors)
print(f"Calculated Threshold: {threshold:.6f}")

# Save threshold for deployment
with open("threshold.pkl", "wb") as f:
    pickle.dump(threshold, f)
