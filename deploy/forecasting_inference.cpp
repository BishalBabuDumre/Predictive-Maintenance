#include "forecasting_inference.h"

ForecastingInference::ForecastingInference(const std::string& model_path)
    : model_engine_(model_path), decision_handler_() {}

InferenceResult ForecastingInference::analyzeSensorState(const std::vector<float>& feature_vector) {
    // Dynamic execution shape identification matching python pipeline expectations
    std::vector<int64_t> input_shape = {1, static_cast<int64_t>(feature_vector.size())};

    // Forward execution pass
    std::vector<float> inference_output = model_engine_.runInference(feature_vector, input_shape);

    // Fallback security checks if outputs emerge blank
    float anomaly_score = inference_output.empty() ? 0.0f : inference_output[0];

    // Evaluate against structural operational rules
    return decision_handler_.parseOutputs(anomaly_score);
}
