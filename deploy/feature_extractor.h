#pragma once

#include <vector>
#include "sensor_hardware.h"

class FeatureExtractor {
public:
    FeatureExtractor();

    // Builds the exact 22-element feature vector matching your Python pipeline execution
    std::vector<float> extractFeatures(float current_temp, const SensorHardware& sensor);

private:
    // Helper to get a past element relative to the newest write position
    float getPastReading(const float* buffer, int index_ptr, int steps_back) const;

    // Rolling windows engine calculation
    void computeRollingStats(const float* buffer, int index_ptr, int window_size, float& mean, float& std) const;
};
