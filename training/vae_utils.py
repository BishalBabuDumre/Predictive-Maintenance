import math
from training.model import vae_loss_function

def batch_loss(model, data, beta, stage_name="Training", epoch_idx=None):
    """
    Processes a single batch for a VAE (Forward pass + Loss evaluation + Stability check).
    
    Args:
        model: The VAE model instance.
        data: The input tensor for this batch.
        beta: The factor that weighs KLD VS MSE.
        stage_name (str): "Training" or "Validation" (used for error reporting).
        epoch_idx (int): Optional current epoch index for better logs.
    
    Returns:
        loss: The PyTorch scalar loss tensor for backward propagation or tracking.
    """
    # 1. Forward pass (Works for both train and eval modes)
    recon_batch, mu, logvar = model(data)
    
    # 2. Compute VAE loss
    loss = vae_loss_function(recon_batch, data, mu, logvar, beta)
    
    # 3. Aggressive stability check (Triggers for both training and validation)
    loss_val = loss.item()
    if math.isnan(loss_val) or math.isinf(loss_val):
        epoch_str = f" at Epoch {epoch_idx}" if epoch_idx is not None else ""
        error_msg = (
            f"\n!!! CRITICAL ERROR: Gradient exploded during {stage_name}{epoch_str}!"
            f"\n-> Loss evaluated to: {loss_val}"
            f"\n-> Halting execution immediately to save compute."
        )
        print(error_msg)
        raise ValueError(f"{stage_name} loss exploded to NaN/Inf.")
        
    return loss
