import numpy as np
import pandas as pd
import optuna
import wandb
import datetime
import time

from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest

run_name = f"run_{datetime.datetime.now().strftime('%m%d_%H%M')}"

# 🚀 Simulated IoT data function (replace with your real solar IoT data source)
def get_new_data():
    # Load CSV file
    file_path = 'data/raw/original.csv'
    df = pd.read_csv(file_path, parse_dates=["DateTime"])
    df = df.sort_values("DateTime").reset_index(drop=True)
    
    # Set index for rolling operations
    df = df.set_index("DateTime")
    
    # Hour of day
    df["hour"] = df.index.hour
    
    # Day of year
    df["doy"] = df.index.dayofyear
    
    # Cyclical encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    
    df["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 365)
    
    # Rolling statistics
    df["roll_mean_24h"] = df["Temperature(F)"].rolling(24).mean()
    df["roll_std_24h"] = df["Temperature(F)"].rolling(24).std()
    
    df["roll_mean_7d"] = df["Temperature(F)"].rolling(24*7).mean()
    
    # Deviation from local expectation
    df["dev_24h"] = df["Temperature(F)"] - df["roll_mean_24h"]
    
    # Instant change
    df["delta_1h"] = df["Temperature(F)"].diff()
    
    # Slope (approx drift)
    df["slope_24h"] = (df["Temperature(F)"] - df["Temperature(F)"].shift(24)) / 24
    df["slope_7d"] = (df["Temperature(F)"] - df["Temperature(F)"].shift(24*7)) / (24*7)
    
    # Detect flatlining
    df["roll_std_6h"] = df["Temperature(F)"].rolling(6).std()
    
    # Count repeated values (simple version)
    df["is_same"] = (df["Temperature(F)"].diff() == 0).astype(int)
    df["repeat_count"] = df["is_same"].rolling(6).sum()
    
    df = df.dropna()
    
    # Select desired columns
    columns_to_extract = [
        "Temperature(F)",
        "hour_sin", "hour_cos",
        "doy_sin", "doy_cos",
        "roll_mean_24h",
        "roll_std_24h",
        "dev_24h",
        "delta_1h",
        "slope_24h",
        "slope_7d",
        "roll_std_6h",
        "repeat_count"
    ]
    selected_df = df[columns_to_extract]

    # Convert to NumPy array
    data_array = selected_df.to_numpy()

    return data_array

# 🚀 Create full pipeline (scaler + model)
def create_pipeline(n_estimators, max_samples, max_features, contamination):
    return Pipeline([
        ('model', IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            contamination=contamination,
            random_state=42
        ))
    ])

# 🚀 Single retraining function
def retrain_pipeline():
    print(f"\n🚀 Retraining triggered at {datetime.datetime.now()}")

    X = get_new_data()

    wandb.init(
        project="IsolationForest-Sensor",
        job_type="hyperparameter-optimization-without-scaler",
        name=run_name,
        config={}
    )

    # Objective function for Optuna
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 70, 100),
            'max_samples': trial.suggest_float('max_samples', 0.6, 0.95),
            'max_features': trial.suggest_int('max_features', 8, X.shape[1]),
            'contamination': trial.suggest_float('contamination', 0.001, 0.1, log = True)
        }

        pipeline = create_pipeline(**params)
        pipeline.fit(X)

        # Using mean decision_function as objective score
        scores = pipeline.named_steps['model'].decision_function(X)
        mean_score = np.mean(scores)

        # Log parameters & score to wandb
        wandb.log(params)
        wandb.log({'mean_decision_function': mean_score})

        return mean_score

    # Run Optuna study
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    wandb.summary['best_score'] = study.best_value
    wandb.summary['best_params'] = study.best_params

    print("✅ Best hyperparameters:", study.best_params)

    wandb.finish()

# 🚀 Auto retraining loop (run every X seconds)
def automatic_retraining(interval_seconds=3600):
    while True:
        retrain_pipeline()
        print(f"Waiting {interval_seconds} seconds for next retraining...")
        time.sleep(interval_seconds)

# ✅ Manual run
if __name__ == "__main__":
    retrain_pipeline()
