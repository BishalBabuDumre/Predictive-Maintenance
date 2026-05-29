import os
import joblib

def save_filename(scaler, target_filename, directory_path="data/model", ):
  
  full_path = os.path.join(directory_path, target_filename)
  
  os.makedirs(directory_path, exist_ok=True)
  
  joblib.dump(scaler, full_path)
  
  print(f"Success! Scaler saved to: {full_path}")
