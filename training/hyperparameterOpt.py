import numpy as np
import pandas as pd
import optuna
import wandb
import datetime
import time

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# 🚀 Simulated IoT data function (replace with your real solar IoT data source)
def get_new_data():
    # Load CSV file
    filename = '2025-06-15.csv'
    df = pd.read_csv(filename)

    # Select desired columns
    columns_to_extract = ['Vpeak', 'Ipeak', 'Voc', 'Isc', 'Temp1', 'Temp2', 'Temperature_Ambient', 'Temperature_Top of MT',
                       'auxInput_converted_value_Humidity (relative)', 'auxInput_converted_value_Wind Speed', 
                       'Irradiance_before_Irrad Si-Ref 058', 'Irradiance_before_Irrad Si-Ref 061']
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
        project="IsolationForest-Anomaly-Detection",
        job_type="hyperparameter-optimization-without-scaler",
        config={}
    )

    # Objective function for Optuna
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 120),
            'max_samples': trial.suggest_float('max_samples', 0.1, 1.0),
            'max_features': trial.suggest_int('max_features', 1, X.shape[1]),
            'contamination': trial.suggest_float('contamination', 0.01, 0.3)
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
    study.optimize(objective, n_trials=20)

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

    # Optional: run automatic retraining (disable for now)
    # automatic_retraining(interval_seconds=3600)
