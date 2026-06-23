#pragma once

#include <string>
#include <vector>
#include "onnx_engine.h"
#include "inference_handler.h"

class ForecastingInference {
public:
    ForecastingInference(const std::string& model_path);
    ~ForecastingInference() = default;

    // Evaluates an incoming vector and returns structured diagnostic logs
    InferenceResult analyzeSensorState(const std::vector<float>& feature_vector);

private:
    OnnxEngine model_engine_;
    InferenceHandler decision_handler_;
};
