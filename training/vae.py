import os
import wandb
import torch
import torch.optim as optim
from training.model import VAE
from training.model import vae_loss_function
from training.vae_utils import batch_loss
from training.data_preparation import prepare_vae_data
from training.feature_engineering import prepare_data_frame
from training.early_stopping import EarlyStopping
from training.onnx_export import export_and_verify_onnx

# 1. Setup WandB Configuration
config = {
    "learning_rate": 1e-3,
    "epochs": 1000,
    "patience": 5,
    "min_delta": 0.001,
    "latent_dim": 2,            # Bottleneck Sweep
    "hidden_layers": [32, 16],  # Layer depth & width combined
    "activation": "LeakyReLU",  # Layer activation function
    "dropout": None,             # Dropout Regularization
    "beta": 0.5                 # Loss scaling factor
}

# 2. Available Raw Data
train_path = os.path.join('data/raw/training_data.csv')
valid_path = os.path.join('data/raw/validation_data.csv')

#Loading training data
df_train, features_train, target_train = prepare_data_frame(train_path)
train_loader = prepare_vae_data(df_train, features_train, target_train)
input_dim = len(features_train)

#Loading validation data
df_val, features_val, target_val = prepare_data_frame(valid_path)
val_loader = prepare_vae_data(df_val, features_val, target_val)

# Initialize the run
wandb.init(
        project="VAE-Anomaly-Detection",
        job_type="Stage_1-Bottleneck-Sweep",
        name=f"Stage-1_Latent_Dimension-{config['latent_dim']}",
        config=config
    )

# Instantiate model & optimizer training loop
model = VAE(input_dim)
optimizer = optim.Adam(model.parameters(), wandb.config.learning_rate)

# TOOL 1: Hook into the model layers to watch gradients/weights
wandb.watch(model, log="all", log_freq=10)

# Initializing early stopping before training starts
early_stopper = EarlyStopping(wandb.config.patience, wandb.config.min_delta)

for epoch in range(wandb.config.epochs):
    # ==================== TRAINING PHASE ====================
    model.train()
    total_train_loss = 0
    for train_data, _ in train_loader:
        optimizer.zero_grad()
        total_train_loss += batch_loss(model, train_data, vae_loss_function, stage_name="Training", epoch_idx=epoch)
        loss.backward()
        optimizer.step()
    avg_train_loss = total_train_loss / len(train_loader.dataset)
        
    # ==================== VALIDATION PHASE ====================
    model.eval() 
    total_val_loss = 0
    with torch.no_grad():
        for valid_data, _ in val_loader: 
            total_val_loss += batch_loss(model, valid_data, vae_loss_function, stage_name="Validation", epoch_idx=epoch)   
    avg_val_loss = total_val_loss / len(val_loader.dataset)

    # TOOL 2: Manually log global metrics every epoch
    wandb.log({
        "epoch": current_epoch,
        "losses/train_loss": avg_train_loss,
        "losses/val_loss": avg_val_loss
    })

    # 2. Printing logs of Train and Valid losses.
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch [{current_epoch:03d}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Check early stopping thresholds
    early_stopper(avg_val_loss)
    if early_stopper.early_stop:
        print("Early stopping triggered. Training halted.")
        break
        
#export_and_verify_onnx(model, input_dim)
