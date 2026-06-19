#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H

#include <vector>
#include "sensor_hardware.h" // We need this to look inside the sensor's window state

class FeatureExtractor {
public:
    FeatureExtractor();

    // High-level method that builds the 5-element model input vector
    std::vector<float> extractFeatures(float current_temp, const SensorHardware& sensor);

private:
    // Pure mathematical engines
    float computeMean(const float* buffer) const;
    float computeStd(const float* buffer, float mean) const;
    float computeSlope(const float* buffer, int index_ptr) const;
    float computeBias(const float* buffer, int index_ptr, float mean) const;
};

#endif // FEATURE_EXTRACTOR_H
