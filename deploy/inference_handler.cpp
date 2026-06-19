#include "inference_handler.h"

InferenceHandler::InferenceHandler() {}

InferenceResult InferenceHandler::parseOutputs(float anomaly_metric) {
    InferenceResult result;

    // 1. 🔴 Critical Threshold
    if (anomaly_metric > 0.15770f) {
        result.alertTier = "🔴 Critical";
        result.status    = "CRITICAL";
        result.meaning   = "ALERT TRIGGERED: Catastrophic system shock or immediate hardware failure.";
    }
    // 2. 🟡 Warning Threshold
    else if (anomaly_metric > 0.04810f) {
        result.alertTier = "🟡 Warning";
        result.status    = "WARNING";
        result.meaning   = "ALERT TRIGGERED: Pattern disruption confirmed. Evaluates for Sensor Flatlines or Drift.";
    }
    // 3. ⚪ System Buffer Threshold
    else if (anomaly_metric > 0.04125f) {
        result.alertTier = "⚪ System Buffer";
        result.status    = "HEALTHY (High Volatility)";
        result.meaning   = "Normal environmental noise, weather transitions, or minor sensor fluctuations. No alert triggered.";
    }
    // 4. 🟢 System Normal Baseline
    else {
        result.alertTier = "🟢 System Normal";
        result.status    = "HEALTHY";
        result.meaning   = "HEALTHY: Ideal, expected cyclic operational patterns.";
    }

    return result;
}
