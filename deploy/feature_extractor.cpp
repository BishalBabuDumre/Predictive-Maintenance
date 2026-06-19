#include "feature_extractor.h"
#include <cmath>

FeatureExtractor::FeatureExtractor() {}

float FeatureExtractor::computeMean(const float* buffer) const {
    float sum = 0.0f;
    for (int i = 0; i < SensorHardware::WINDOW; i++) sum += buffer[i];
    return sum / SensorHardware::WINDOW;
}

float FeatureExtractor::computeStd(const float* buffer, float mean) const {
    float sum = 0.0f;
    for (int i = 0; i < SensorHardware::WINDOW; i++) {
        float d = buffer[i] - mean;
        sum += d * d;
    }
    return std::sqrt(sum / SensorHardware::WINDOW);
}

float FeatureExtractor::computeSlope(const float* buffer, int index_ptr) const {
    int oldest = index_ptr;
    int newest = (index_ptr + SensorHardware::WINDOW - 1) % SensorHardware::WINDOW;
    return (buffer[newest] - buffer[oldest]) / SensorHardware::WINDOW;
}

float FeatureExtractor::computeBias(const float* buffer, int index_ptr, float mean) const {
    int newest = (index_ptr + SensorHardware::WINDOW - 1) % SensorHardware::WINDOW;
    return buffer[newest] - mean;
}

std::vector<float> FeatureExtractor::extractFeatures(float current_temp, const SensorHardware& sensor) {
    const float* data = sensor.getBufferData();
    int current_ptr = sensor.getIndexPtr();
    
    float mean = computeMean(data);
    
    return {
        current_temp,
        mean,
        computeStd(data, mean),
        computeSlope(data, current_ptr),
        computeBias(data, current_ptr, mean)
    };
}
