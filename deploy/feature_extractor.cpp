#include "feature_extractor.h"
#include <cmath>
#include <ctime>

FeatureExtractor::FeatureExtractor() {}

// Helper function to safely traverse backward into the ring buffer
float FeatureExtractor::getPastReading(const float* buffer, int index_ptr, int steps_back) const {
    // sensor.addReading advances index_ptr, so the newest element is at index_ptr - 1
    int target_idx = (index_ptr - 1 - steps_back + SensorHardware::WINDOW) % SensorHardware::WINDOW;
    return buffer[target_idx];
}

void FeatureExtractor::computeRollingStats(const float* buffer, int index_ptr, int window_size, float& mean, float& std) const {
    float sum = 0.0f;
    for (int i = 0; i < window_size; ++i) {
        sum += getPastReading(buffer, index_ptr, i);
    }
    mean = sum / window_size;

    float variance_sum = 0.0f;
    for (int i = 0; i < window_size; ++i) {
        float diff = getPastReading(buffer, index_ptr, i) - mean;
        variance_sum += diff * diff;
    }
    std = (window_size > 1) ? std::sqrt(variance_sum / (window_size - 1)) : 0.0f;
}

std::array<float, 22> FeatureExtractor::extractFeatures(float current_temp, const SensorHardware& sensor) {
    const float* data = sensor.getBufferData();
    int ptr = sensor.getIndexPtr();

    // 1. Cyclical Time Features using system time
    std::time_t now = std::time(nullptr);
    std::tm* time_info = std::localtime(&now);

    int hour = time_info->tm_hour;
    int month = time_info->tm_mon + 1; // tm_mon is 0-11
    int doy = time_info->tm_yday + 1;  // tm_yday is 0-364

    float hour_sin = std::sin(2.0f * M_PI * hour / 24.0f);
    float hour_cos = std::cos(2.0f * M_PI * hour / 24.0f);
    float month_sin = std::sin(2.0f * M_PI * month / 12.0f);
    float month_cos = std::cos(2.0f * M_PI * month / 12.0f);
    float doy_sin = std::sin(2.0f * M_PI * doy / 365.25f);
    float doy_cos = std::cos(2.0f * M_PI * doy / 365.25f);

    // 2. Rolling Window Extractions
    float mean_3h = 0.0f,  std_3h = 0.0f;
    float mean_6h = 0.0f,  std_6h = 0.0f;
    float mean_24h = 0.0f, std_24h = 0.0f;
    float mean_7d = 0.0f,  std_7d = 0.0f;

    computeRollingStats(data, ptr, 3, mean_3h, std_3h);
    computeRollingStats(data, ptr, 6, mean_6h, std_6h);
    computeRollingStats(data, ptr, 24, mean_24h, std_24h);
    computeRollingStats(data, ptr, 168, mean_7d, std_7d);

    // 3. Metric Deviations and Slopes
    float dev_24h = current_temp - mean_24h;

    float slope_3h = (current_temp - getPastReading(data, ptr, 3)) / 3.0f;
    float slope_24h = (current_temp - getPastReading(data, ptr, 24)) / 24.0f;
    float slope_7d = (current_temp - getPastReading(data, ptr, 168)) / 168.0f;

    // 4. Past Slopes for 2nd Derivatives (Accelerations)
    float prev_temp = getPastReading(data, ptr, 1);
    float prev_slope_3h = (prev_temp - getPastReading(data, ptr, 4)) / 3.0f;
    float prev_slope_24h = (prev_temp - getPastReading(data, ptr, 25)) / 24.0f;
    float prev_slope_7d = (prev_temp - getPastReading(data, ptr, 169)) / 168.0f;

    float accel_3h = slope_3h - prev_slope_3h;
    float accel_24h = slope_24h - prev_slope_24h;
    float accel_7d = slope_7d - prev_slope_7d;

    // 5. Repeat Count (consecutive flatline readings over a rolling 6h span)
    float repeat_count = 0.0f;
    for (int i = 0; i < 6; ++i) {
        if (getPastReading(data, ptr, i) == getPastReading(data, ptr, i + 1)) {
            repeat_count += 1.0f;
        }
    }

    // Return final sequence perfectly matching training order expectances
    return {
        hour_sin, hour_cos, doy_sin, doy_cos, month_sin, month_cos,
        dev_24h, repeat_count,
        mean_3h, std_3h,
        mean_6h, std_6h,
        mean_24h, std_24h,
        mean_7d, std_7d,
        slope_3h, slope_24h, slope_7d,
        accel_3h, accel_24h, accel_7d
    };
}
