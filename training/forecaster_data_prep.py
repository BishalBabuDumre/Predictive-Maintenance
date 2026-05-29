import numpy as np
import pandas as pd
import onnxruntime as ort
from training.feature_engineering import prepare_data_frame

def extract_latent_dataset(csv_path, onnx_model_path):
    """
    Passes data through the static frozen ONNX VAE to generate a clean, 
    unsupervised tensor dataset ready for downstream forecasting.
    """
    # Load raw frames using your preprocessing steps
    df, features, target = prepare_data_frame(csv_path)
    
    # Run the ONNX Inference Session
    ort_session = ort.InferenceSession(onnx_model_path)
    ort_inputs = {'input': df[features].values.astype(np.float32)}
    
    # Match the output names specified in your original torch.onnx.export step
    _, mu, _ = ort_session.run(None, ort_inputs)
    mu_df = pd.DataFrame(mu, columns=[f'feature_{i}' for i in range(8)])
    tar_df = df[target]
    df_new = pd.concat([tar_df, mu_df], axis=1)
    features_new = [f'feature_{i}' for i in range(8)]
    loader = prepare_vae_data(df_new, features_new, target, scaler_y_name = "scaler_y_forecaster.pkl")
    
    return loader, mu.shape[1]
