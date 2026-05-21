import os
import sys
import torch
import torch.optim as optim
from training.model import VAE
from training.model import vae_loss_function
from training.data_preparation import prepare_vae_data
from training.feature_engineering import prepare_data_frame
from training.early_stopping import EarlyStopping
from training.onnx_export import export_and_verify_onnx

file_path = os.path.join('data/raw/original.csv')

# Usage
df, features, target = prepare_data_frame(file_path)
train_loader = prepare_vae_data(df, features, target)
input_dim = len(features)

# Training Loop
model = VAE(input_dim)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Initialize your tracking tool before training starts
early_stopper = EarlyStopping(patience=5, min_delta=0.001)
epochs = 1000 # Set a high limit, early stopping will save you!

for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    for batch in train_loader:
        data = batch[0]
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = vae_loss_function(recon_batch, data, mu, logvar)
        if math.isnan(loss.item()) or math.isinf(loss.item()):
            print(f"!!! CRITICAL ERROR: Gradient exploded at Epoch {epoch}, crashing to NaN. Halting training.")
            raise ValueError("Validation loss exploded to NaN/Inf. Halting program.")
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
        
    # Validation Pass
    model.eval() 
    total_val_loss = 0
    with torch.no_grad():
        # Assuming 'test' is a DataLoader yielding (inputs, targets)
        for inputs, _ in test: 
            # 1. Unpack all 3 returned values from VAE
            val_recon, val_mu, val_logvar = model(inputs)
            
            # 2. Evaluate using the VAE loss function, not classification accuracy
            val_loss = vae_loss_function(val_recon, inputs, val_mu, val_logvar)
            total_val_loss += val_loss.item()
            
    avg_val_loss = total_val_loss / len(test.dataset)
    print(f'---> Validation Loss: {avg_val_loss:.4f}')

    # Check early stopping thresholds
    early_stopper(avg_val_loss, model)
    
    if early_stopper.early_stop:
        print("Early stopping triggered. Training halted.")
        break
        
#export_and_verify_onnx(model, input_dim)
