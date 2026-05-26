import os
import copy
import wandb
import torch
import optuna
import torch.optim as optim
from training.model import VAE
from training.model import vae_loss_function
from training.vae_utils import batch_loss
from training.data_preparation import prepare_vae_data
from training.feature_engineering import prepare_data_frame
from training.early_stopping import EarlyStopping
from training.onnx_export import export_and_verify_onnx

best_global_wts = None
final_input_dim = None

def objective(trial):
    global best_global_wts, final_input_dim
    
    latent_dimension = trial.suggest_categorical("latent_dim", [2, 4, 8, 16])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    beta = trial.suggest_float("beta", 0.1, 1.0, step=0.1)
    activation = trial.suggest_categorical("activation", ["ReLU", "LeakyReLU", "ELU"])
    dropout = trial.suggest_categorical("dropout", [None, 0.1, 0.2])
    
    # Suggesting structural architectures (tuples/lists)
    hidden_layers = trial.suggest_categorical("hidden_layers", [
        [32, 16],
        [64, 32],
        [64, 32, 16]
    ])
    
    config = {
        "learning_rate": learning_rate,
        "epochs": 1000,
        "patience": 5,
        "min_delta": 0.001,
        "latent_dim": latent_dimension,
        "hidden_layers": hidden_layers,  
        "activation": activation,  
        "dropout": dropout,             
        "beta": beta                 
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
    run = wandb.init(
            project="VAE-Anomaly-Detection",
            job_type="Stage_1-Bottleneck-Sweep",
            name=f"name=f"Trial-{trial.number}",
            config=config,
            reinit=True # CRITICAL: Allows launching multiple runs in one script
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
            t_loss = batch_loss(model, train_data, vae_loss_function, stage_name="Training", epoch_idx=epoch)
            total_train_loss += t_loss
            loss.backward()
            optimizer.step()
        avg_train_loss = total_train_loss / len(train_loader.dataset)
            
        # ==================== VALIDATION PHASE ====================
        model.eval() 
        total_val_loss = 0
        with torch.no_grad():
            for valid_data, _ in val_loader: 
                v_loss = batch_loss(model, valid_data, vae_loss_function, stage_name="Validation", epoch_idx=epoch)
                total_val_loss += v_loss
        avg_val_loss = total_val_loss / len(val_loader.dataset)
    
        # TOOL 2: Manually log global metrics every epoch
        wandb.log({
            "epoch": epoch,
            "losses/train_loss": avg_train_loss,
            "losses/val_loss": avg_val_loss
        })
    
        # Save the trial's best weights in memory
        if avg_val_loss < best_trial_val_loss:
            best_trial_val_loss = avg_val_loss
            # If this is the best trial overall so far, cache its weights globally
            if best_global_wts is None or avg_val_loss < study.best_value:
                best_global_wts = copy.deepcopy(model.state_dict())
    
        # Check early stopping thresholds
        early_stopper(avg_val_loss)
        if early_stopper.early_stop:
            print("Early stopping triggered. Training halted.")
            break

        # OPTUNA BONUS: Report intermediate values to allow trial pruning if it looks bad
        trial.report(avg_val_loss, epoch)
        if trial.should_prune():
            run.finish()
            raise optuna.TrialPruned()

    # Close out the current WandB run cleanly before starting the next loop iteration
    run.finish()

    # Optuna minimizes whatever value is returned here
    return best_trial_val_loss

# ==================== MAIN EXECUTION SECTION ====================
if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)

    print(f"\n" + "═"*50)
    print(f"🏆 OPTUNA SWEEP COMPLETE 🏆")
    print(f"Best Trial Hyperparameters: {study.best_params}")
    print(f"Best Validation Loss: {study.best_value:.4f}")
    print(f"Exporting the champion model to ONNX...")
    print("═"*50 + "\n")
    
    # Recreate the winning architecture using Optuna's tracked champion settings
    # This completely resolves the problem of parameters getting mixed up across trials!
    best_model = VAE(
        input_dim=final_input_dim,
        latent_dim=study.best_params["latent_dim"],
        hidden_layers=study.best_params["hidden_layers"],
        activation=study.best_params["activation"]
        # Add any other initialization arguments your VAE class takes here
    )
    
    # Inject the optimal weights we cached during our best objective run
    best_model.load_state_dict(best_global_wts)
    best_model.eval()
    
    # Export only the single ultimate champion
    export_and_verify_onnx(best_model, final_input_dim)
