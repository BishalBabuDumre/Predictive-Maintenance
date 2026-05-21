import os
import torch
import torch.nn as nn
import torch.optim as optim
from data_preparation import prepare_vae_data
from feature_engineering import prepare_data_frame

file_path = os.path.join('data/raw/original.csv')

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

# Usage
df, features, target = prepare_data_frame(file_path)
train_loader = prepare_vae_data(df, features, target)
input_dim = len(features)

# Training Loop
model = VAE(input_dim)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(50):
    model.train()
    total_train_loss = 0
    for batch in train_loader:
        data = batch[0]
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = vae_loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    avg_train_loss = total_train_loss / len(train_loader.dataset)
    
    # 1. Log EVERY epoch to wandb for rich visualization dashboards
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": avg_train_loss
    })

    # 2. Only print to terminal every 5 epochs to keep the console clean
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] | Avg Train Loss: {avg_train_loss:.4f}")
        
    # --- STEP OUT of the batch loop to log once per epoch ---
    if (epoch + 1) % 10 == 0:
        avg_train_loss = total_train_loss / len(train_loader.dataset)
        print(f'Epoch: {epoch+1}/{epochs}, Avg Training Loss: {avg_train_loss:.4f}')
        
        # Validation Pass
        model.eval() 
        total_test_loss = 0
        
        with torch.no_grad():
            # Assuming 'test' is a DataLoader yielding (inputs, targets)
            for inputs, _ in test: 
                # 1. Unpack all 3 returned values from VAE
                val_recon, val_mu, val_logvar = model(inputs)
                
                # 2. Evaluate using the VAE loss function, not classification accuracy
                val_loss = vae_loss_function(val_recon, inputs, val_mu, val_logvar)
                total_test_loss += val_loss.item()
                
        avg_test_loss = total_test_loss / len(test.dataset)
        print(f'---> Test Loss: {avg_test_loss:.4f}')(f'Accuracy:{accuracy:.6f}')

# 4. Export to ONNX
model.eval()
# Create a dummy input that matches your feature dimensions
dummy_input = torch.randn(1, input_dim) 

# Creating folder inside the repo
folder_path = "data/model"
os.makedirs(folder_path, exist_ok=True)

save_path = os.path.join(folder_path, "vae_model.onnx")

torch.onnx.export(
    model, 
    dummy_input, 
    save_path,
    export_params=True,
    opset_version=12,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output', 'mu', 'logvar'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

print(f"Model successfully saved to {save_path}")
