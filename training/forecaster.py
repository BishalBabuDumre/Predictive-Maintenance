import os
import wandb
import torch
import optuna
import numpy as np
import onnxruntime as ort
import torch.optim as optim
from training.model import VAE
from training.data_preparation import prepare_vae_data
from training.feature_engineering import prepare_data_frame
from training.early_stopping import EarlyStopping
from training.onnx_export import export_and_verify_onnx

final_input_dim = None
study = None  # Will be assigned in the main block to allow global context checks

def objective(trial):
    global best_global_wts, final_input_dim, study

    #Variables being optimized!!!
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    activation = trial.suggest_categorical("activation", ["ReLU", "LeakyReLU", "ELU", "Tanh"])
    dropout = trial.suggest_categorical("dropout", [None, 0.1, 0.2])
    hidden_layers = trial.suggest_categorical("hidden_layers", [
        [16],              # 1 Hidden Layer (Very shallow/customary)
        [32],              # 1 Hidden Layer (Slightly wider)
        [32, 16],          # 2 Hidden Layers (Medium depth)
        [64, 32],          # 2 Hidden Layers (Wide)
        [64, 32, 16]       # 3 Hidden Layers (Deep predictive network)
    ])
    
    config = {
        "learning_rate": learning_rate,
        "epochs": 1000,
        "patience": 5,
        "min_delta": 0.001,
        "latent_dim": 1,
        "hidden_layers": hidden_layers,  
        "activation": activation,  
        "dropout": dropout             
    }
    
    train_path = os.path.join('data/raw/training_data.csv')
    valid_path = os.path.join('data/raw/validation_data.csv')
    onnx_model_path = "data/model/vae_model.onnx"

    
    train_loader, input_dim = extract_latent_dataset(train_path, onnx_model_path)
    val_loader, _ = prepare_vae_data(valid_path, onnx_model_path)
    
    final_input_dim = input_dim  # Update global reference
    
    # FIXED: Fixed double string assignment bug
    run = wandb.init(
        project="Temperature_Forecasting",
        job_type="Hyperparameters-Tuning",
        name=f"Trial-{trial.number}",
        config=config,
        reinit=True 
    )
    
    # FIXED: Added missing parameters to model configuration initialization
    model = VAE(
        input_dim=input_dim,
        latent_dim=1,
        hidden_layers=wandb.config.hidden_layers,
        activation=wandb.config.activation,
        dropout=wandb.config.dropout
    )
    optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
    criterion = nn.MSELoss()
    wandb.watch(model, log="all", log_freq=10)
    early_stopper = EarlyStopping(wandb.config.patience, wandb.config.min_delta)
    
    # FIXED: Initialized variable before using it in comparison evaluations
    best_trial_val_loss = float('inf')
    
    for epoch in range(wandb.config.epochs):
        # ==================== TRAINING PHASE ====================
        model.train()
        total_train_loss = 0
        for train_data, train_target in train_loader:
            optimizer.zero_grad()
            recon_batch, predictions, logvar = model(train_data)
            t_loss = criterion(predictions, train_target, reduction='sum')
            total_train_loss += t_loss.item()
            t_loss.backward()
            optimizer.step()
            
        avg_train_loss = total_train_loss / len(train_loader.dataset)
            
        # ==================== VALIDATION PHASE ====================
        model.eval() 
        total_val_loss = 0
        with torch.no_grad():
            for valid_data, valid_target in val_loader: 
                recon_batch, predictions, logvar = model(valid_data)
                v_loss = criterion(predictions, valid_target, reduction='sum')
                total_val_loss += v_loss.item()
        avg_val_loss = total_val_loss / len(val_loader.dataset)
    
        wandb.log({
            "epoch": epoch,
            "losses/train_loss": avg_train_loss,
            "losses/val_loss": avg_val_loss
        })
    
        # Save the trial's best weights in memory
        if avg_val_loss < best_trial_val_loss:
            best_trial_val_loss = avg_val_loss
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/trial_{trial.number}.pt")
    
        early_stopper(avg_val_loss)
        if early_stopper.early_stop:
            print("Early stopping triggered. Training halted.")
            break

        trial.report(avg_val_loss, epoch)
        if trial.should_prune():
            run.finish()
            raise optuna.TrialPruned()

    run.finish()
    return best_trial_val_loss

# ==================== MAIN EXECUTION SECTION ====================
if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)

    print(f"\n" + "═"*50)
    print("OPTUNA SWEEP COMPLETE")
    print("Best Trial Hyperparameters:")
    for param_name, param_val in study.best_params.items():
        print(f"  - {param_name}: {param_val}")
    print(f"Best Validation Loss: {study.best_value:.4f}")
    print(f"Exporting the champion model to ONNX...")
    print("═"*50 + "\n")

    best_model = VAE(
        input_dim=final_input_dim,
        latent_dim=1,
        hidden_layers=study.best_params["hidden_layers"],
        activation=study.best_params["activation"],
        dropout=study.best_params["dropout"]
    )
    
    # Inject the optimal weights we cached during our best objective run
    best_trial_num = study.best_trial.number
    best_model.load_state_dict(torch.load(f"checkpoints/trial_{best_trial_num}.pt"))
    best_model.eval()
    
    # Optional Production Practice: Log the final winning learning rate alongside the exported asset
    final_lr = study.best_params["learning_rate"]
    print(f"Note: Champion model weights were optimized using a learning rate of: {final_lr:.6f}")
    
    # Export only the single ultimate champion containing all correct shapes and properties
    export_and_verify_onnx(best_model, final_input_dim, file_name="forecast_model.onnx")

    # Optional cleanup: Clean up the checkpoint files if you don't want them cluttering your workspace
    for t in study.trials:
        cp_path = f"checkpoints/trial_{t.number}.pt"
        if os.path.exists(cp_path) and t.number != best_trial_num:
            os.remove(cp_path)
