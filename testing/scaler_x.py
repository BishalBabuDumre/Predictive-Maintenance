import os
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from training.feature_engineering import prepare_data_frame

def generate_and_save_scalers(train_csv_path, output_dir="data/model"):
    print("Loading data and engineering features...")
    # 1. Run the exact same pipeline used for training
    df, features, target = prepare_data_frame(train_csv_path)
    
    X = df[features].to_numpy()
    
    print("Fitting MinMaxScaler...")
    # 2. Fit the scaler on the full training data matrix
    scaler_x = MinMaxScaler(feature_range=(-1, 1))
    scaler_x.fit(X)
    
    # 3. Save the scaler artifact
    os.makedirs(output_dir, exist_ok=True)
    scaler_path = os.path.join(output_dir, "scaler_x.pkl")
    
    joblib.dump(scaler_x, scaler_path)
    print(f"Success! Scaler saved to: {scaler_path}")

if __name__ == "__main__":
    TRAIN_DATA_PATH = "data/raw/training_data.csv" 
    generate_and_save_scalers(TRAIN_DATA_PATH)
