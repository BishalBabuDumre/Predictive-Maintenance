#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <numeric>

// Struct representing a single log/row of your streaming data
struct WindowRow {
    std::string date_time;
    double temperature_f;
    double reconstruction_loss;
};

// Struct to represent the returned status dictionary
struct ClassificationResult {
    std::string status;
    std::string type;
    double loss;
};

class ProductionAnomalyClassifier {
private:
    double median;
    double mad;
    double warning_threshold;
    double critical_threshold;

    // Helper helper to calculate standard deviation of temperature
    double calculate_temp_std(const std::vector<WindowRow>& window, size_t count) const {
        size_t n = window.size();
        double sum = 0.0;
        
        // Calculate mean of the last 'count' elements
        for (size_t i = n - count; i < n; ++i) {
            sum += window[i].temperature_f;
        }
        double mean = sum / count;

        // Calculate variance
        double sq_sum = 0.0;
        for (size_t i = n - count; i < n; ++i) {
            double diff = window[i].temperature_f - mean;
            sq_sum += diff * diff;
        }
        
        // Using sample standard deviation (N-1) matching Pandas default ddof=1
        return std::sqrt(sq_sum / (count - 1));
    }

public:
    // Constructor matching your default parameters
    ProductionAnomalyClassifier(double baseline_median = 0.021036, 
                                double baseline_mad = 0.005, 
                                double k_warning = 4.0, 
                                double k_critical = 50.0) 
        : median(baseline_median), mad(baseline_mad) 
    {
        warning_threshold = median + (k_warning * mad);    // ~0.041
        critical_threshold = median + (k_critical * mad);  // ~0.271
    }

    ClassificationResult process_latest_window(const std::vector<WindowRow>& window_df) {
        // Fallback for safety if an empty window is passed
        if (window_df.empty()) {
            return {"UNKNOWN", "NO_DATA", 0.0};
        }

        // Grab the immediate current state (the tail of your stream)
        const auto& current_row = window_df.back();
        double current_loss = current_row.reconstruction_loss;
        
        // Scenario 1: Everything behaves within normal distributions
        if (current_loss <= warning_threshold) {
            return {"HEALTHY", "NONE", current_loss};
        }

        // Scenario 2: It passes the warning threshold, check for massive hardware spike
        if (current_loss >= critical_threshold) {
            if (window_df.size() >= 2) {
                // window_df.iloc[-2]
                double prev_loss = window_df[window_df.size() - 2].reconstruction_loss;
                
                // If loss accelerated instantly up into critical territory
                if ((current_loss - prev_loss) > (warning_threshold * 5.0)) {
                    return {"CRITICAL", "SUDDEN_SPIKE", current_loss};
                }
            }
            return {"CRITICAL", "EXTREME_OUT_OF_BOUNDS", current_loss};
        }

        // Scenario 3: Loss is elevated but sits in warning territory (Flatline vs Drift)
        if (window_df.size() >= 6) {
            // Pandas .std() defaults to ddof=1 (sample standard deviation)
            if (calculate_temp_std(window_df, 6) == 0.0) {
                return {"WARNING", "SENSOR_FLATLINE", current_loss};
            }
        }

        // Persistent warning loss without zero temperature variance indicates drift
        return {"WARNING", "ENVIRONMENTAL_DRIFT", current_loss};
    }
};
