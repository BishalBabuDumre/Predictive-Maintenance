# Predictive Maintenance for Sensors
## Introduction
Our motto is develop a model that can help us in the early detection of sensor failure and temperature forecasting. Temperature data (main/data/raw/training_data.csv) taken hourly using multiple sensors is available for 20 years in a single geographical location. The idea of multiple sensors is actually about the sensors replaced in either usual maintenance cycle, or hardware failure. Neither, it is just a single data value of temperature for that single location. First, any anomalies are hunted during the exploratory phase. After that, a baseline weather data is trained in Deep Learning (DL) algorithms, ultimately providing us the model to be deployed in the edge. Model development is in Python for faster iteration, whereas edge deployment is carried out in C++ to handle latency.
## Isolation Forest
Isolation Forest (IsoFor) is one of the best classical Machine Learning (ML) algorithm in order to hunt for anomalies. A multi-stage Bayesian Optimization (BayOpt) process (main/training/hyperparameterOpt.py) is conducted to move from broad initial exploration to a converged "Search Space" as given in the table below:
| Phase        | Trials | Strategy               | Result                                                         |
|--------------|--------|------------------------|----------------------------------------------------------------|
| Initial Exploration  | 20     | Random/Bayesian Search | Identified high sensitivity in contamination.                  |
| Refinement   | 100+   | Narrowed Bounds        | Converged on max_samples > 0.9 and contamination < 0.005.      |

Please find the graphs below that explain the initial exploration and the last refinement processes. In Figure 1, contamination is seen as the most sensitive parameter which is causing the decision function to be unresponsive as observed in their compatibly mirroring behavior. max_features and max_samples seem to be moving in a trend. n-estimators is wobbling in the search space. Hence, we focus on our attention to the contamination parameter and proceeded with iterative refining. As in Figure 2, contamination always hugged the bottom floor indicating that there is little to no anomaly here. This is expected for this dataset because modern sensors can sample data in sub-second timings. We have the data presented at every hour, which is already averaged beforehand. Hence, the expectation of little to no anomaly is real. However, if the data is available for sub-second timings coming from a sensor, we could potentially catch some spurious data as anomalies. We implemented a final tuned configuration as given below in IsoFor using main/training/isolationForest.py. After observing each anomaly manually, 264 in number present at main/data/processed/anomalies_0.csv, it is found that they present themselves as a smooth data point continuously moving from one hour to the other in the spatio-temporal space. This IsoFor step provides a sanity check to detect anomalies that might otherwise go unnoticed. It has been proved that the data is clean, and can be utilized as the ground truth baseline weather in the next DL step.


![Plot 1](images/Initial_Exploration.png)
**Figure 1:** Initial Exploration.


![Plot 2](images/Final_Refinement.png)
**Figure 2:** Final Refinement Step.


#### Final Tuned Configuration in Isolation Forest

```python
optimized_params = {
    "n_estimators": 88,
    "max_samples": 0.96,
    "max_features": 12,
    "contamination": 0.0015,  # Validated against smoothed hourly logs
    "random_state": 42
}
```

## Variational Auto-Encoder (VAE) with Prognostic Axon (VAPA)
### Why VAE?
Classical ML models such as Support Vector Machine (SVM) and IsoFor, implemented above, are only able to detect anomaly but can't forecast. They calculate on the basis of how far a value is in the feature landscape and treat each event as a separate incident completely ignoring the chronology. Similarly, Classical Time-Series Forecasting algorithms such as AutoRegressive Integrated Moving Average (ARIMA) treats daily, weekly, and seasonal cycles as a linear function and totally ignore the multi-variate complex non-linear interplay of weather observations. Likewise, DL algorithms such as Long Short-Term Memory Network (LSTM) and Gated Recurrent Unit (GRU) are highly susceptible to the noise in the data and focus only on raw prediction. Hence, we have chosen VAE for accomplishing our tasks because it weeds out the noise in the data and compresses all the complex weather interplay in a robust latent space closely reconstructing the original input and calculating the downstream task of temperature prediction.
### Pipeline Architecture

```text
       [ Raw Input Data ]  (Features, Lagged Variables, Rolling Windows)
               │
               ▼
   ┌───────────────────────┐
   │  VAE BACKBONE (Soma)  │  --> Extracts deep spatial-temporal representations
   └───────────┬───────────┘
               │
               ▼
    [ Latent Space (μ) ]      --> Compressed probabilistic normal distribution
               │
               ▼
   ┌───────────────────────┐
   │    PROGNOSTIC AXON    │  --> Decodes representation down downstream tasks
   └─────┬───────────┬─────┘
         │           │
         ▼           ▼
   ┌───────────┐┌───────────┐
   │ Forecast  ││Diagnostic │
   │  Branch   ││  Branch   │
   └─────┬─────┘└─────┬─────┘
         │           │
         ▼           ▼
     [ 72.4°F ]  [ 98% Flatline Risk ]
