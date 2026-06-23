#pragma once

#include <string>

struct InferenceResult {
    std::string alertTier;     // 🟢 System Normal, ⚪ System Buffer, etc.
    std::string status;        // HEALTHY, WARNING, CRITICAL
    std::string meaning;       // Description of the current operational state
};

class InferenceHandler {
public:
    InferenceHandler();

    // Evaluates the target metric against the statistical concrete limits
    InferenceResult parseOutputs(float anomaly_metric);
};
