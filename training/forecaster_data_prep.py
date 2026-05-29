import numpy as np
from training.feature_engineering import prepare_data_frame

def extract_latent_dataset(csv_path, onnx_model_path, batch_size=64):
    """
    Passes data through the static frozen ONNX VAE to generate a clean, 
    unsupervised tensor dataset ready for downstream forecasting.
    """
    # Load raw frames using your preprocessing steps
    df, features, target = prepare_data_frame(csv_path)
    
    # Run the ONNX Inference Session
    ort_session = ort.InferenceSession(onnx_model_path)
    ort_inputs = {'input': features.values.astype(np.float32)}
    
    # Match the output names specified in your original torch.onnx.export step
    _, mu, _ = ort_session.run(None, ort_inputs)
    df_train["mu"] = list(mu_train)
    features_train = ["mu"]
    train_loader = prepare_vae_data(df_train, features_train, target_train)

    
    return loader, mu.shape[1] # Returns the data loader and the VAE's latent dimension size
