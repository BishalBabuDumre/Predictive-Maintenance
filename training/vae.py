import os
import wand
import torch
import torch.optim as optim
from training.model import VAE
from training.model import vae_loss_function
from training.vae_utils import batch_loss
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
epochs = 1000 

for epoch in range(epochs):
    # ==================== TRAINING PHASE ====================
    model.train()
    total_train_loss = 0
    for batch in train_loader:
        data = batch[0]
        optimizer.zero_grad()
        
        # Replaced with the clean utility function
        loss = batch_loss(
            model, data, vae_loss_function, stage_name="Training", epoch_idx=epoch
        )
        
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
        
    avg_train_loss = total_train_loss / len(train_loader.dataset)
    
    # 1. Log EVERY epoch to wandb
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": avg_train_loss
    })

    # 2. Only print to terminal every 5 epochs
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] | Avg Train Loss: {avg_train_loss:.4f}")
        
    # ==================== VALIDATION PHASE ====================
    model.eval() 
    total_val_loss = 0
    with torch.no_grad():
        for inputs, _ in test: 
            
            # Replaced with the exact same utility function!
            val_loss = batch_loss(
                model, inputs, vae_loss_function, stage_name="Validation", epoch_idx=epoch
            )
            
            total_val_loss += val_loss.item()
            
    avg_val_loss = total_val_loss / len(test.dataset)
    print(f'---> Validation Loss: {avg_val_loss:.4f}')

    # Check early stopping thresholds
    early_stopper(avg_val_loss, model)
    
    if early_stopper.early_stop:
        print("Early stopping triggered. Training halted.")
        break
        
#export_and_verify_onnx(model, input_dim)
