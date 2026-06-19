#include <iostream>
#include <string>
#include <vector>

#include "sensor_hardware.h"
#include "feature_extractor.h"
#include "forecasting_inference.h"

const std::string BUFFER_FILE = "/home/pi/project_edge/sensor_history.dat";
const std::string MODEL_PATH  = "/home/pi/project_edge/model.onnx";

int main() {
    // 1. Initialize components
    SensorHardware sensor(BUFFER_FILE);
    FeatureExtractor extractor;
    ForecastingInference forecaster(MODEL_PATH);

    sensor.loadBuffer();

    // 2. Hardware reading and storage tracking
    float new_temp = sensor.readSensorHardware("28-xxxxxxxxxxxx");
    if (new_temp == SensorHardware::ERROR_CODE) {
        std::cerr << "Hardware Read Error" << std::endl;
        return 1; 
    }
    sensor.addReading(new_temp);
    sensor.saveBuffer();

    // Prevent execution flow until rolling history windows have completely filled up
    if (!sensor.isBufferFull()) {
        std::cout << "Buffer filling: " << sensor.getIndexPtr() 
                  << "/" << SensorHardware::WINDOW << std::endl;
        return 0; 
    }

    // 3. Extract the 22-element target vector
    std::vector<float> features = extractor.extractFeatures(new_temp, sensor);

    // 4. Abstracted execution pipeline calls
    InferenceResult prediction = forecaster.analyzeSensorState(features);
    
    // 5. Output Presentation
    std::cout << "Tier: "    << prediction.alertTier << "\n"
              << "Status: "  << prediction.status    << "\n"
              << "Details: " << prediction.meaning   << "\n";

    return 0; 
}
