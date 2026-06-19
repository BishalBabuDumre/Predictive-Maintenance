#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>

#include "sensor_hardware.h"
#include "feature_extractor.h"
#include "inference_handler.h"

const std::string BUFFER_FILE = "/home/pi/project_edge/sensor_history.dat";
const std::string MODEL_PATH = "/home/pi/project_edge/model.onnx";

int main() {
    // 1. Initialize components
    SensorHardware sensor(BUFFER_FILE);
    FeatureExtractor extractor;
    InferenceHandler handler; // Handles post-processing parsing

    sensor.loadBuffer();

    // 2. Hardware reading and storage tracking
    float new_temp = sensor.readSensorHardware("28-xxxxxxxxxxxx");
    if (new_temp == SensorHardware::ERROR_CODE) {
        std::cerr << "Hardware Read Error" << std::endl;
        return 1; 
    }
    sensor.addReading(new_temp);
    sensor.saveBuffer();

    if (!sensor.isBufferFull()) {
        std::cout << "Buffer filling: " << sensor.getIndexPtr() 
                  << "/" << SensorHardware::WINDOW << std::endl;
        return 0; 
    }

    // 3. Mathematical feature generation
    std::vector<float> features = extractor.extractFeatures(new_temp, sensor);

    // ===== ONNX RUNTIME DIRECT EXECUTION IN MAIN =====
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "edge");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options.EnableMemPattern();
    session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    
    Ort::Session session(env, MODEL_PATH.c_str(), session_options);
    Ort::AllocatorWithDefaultOptions allocator;

    const char* input_names[] = {"input"};
    const char* output_names[] = {"output"};
    std::vector<int64_t> input_shape = {1, static_cast<int64_t>(features.size())};

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        allocator.GetInfo(),
        features.data(),
        features.size(),
        input_shape.data(),
        input_shape.size()
    );

    // Run the model directly inside main
    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr}, 
        input_names, 
        &input_tensor, 1, 
        output_names, 1
    );

    // Get a pointer to the raw float data array
    float* raw_output = output_tensors[0].GetTensorMutableData<float>();
    // =================================================

    // Assuming index 0 of your output tensor holds your target model evaluation score
    float current_anomaly_score = raw_output[0]; 
    
    // Run the new tier filtering logic
    InferenceResult prediction = handler.parseOutputs(current_anomaly_score);
    
    // Present outputs using the clean string fields
    std::cout << "Tier: " << prediction.alertTier << "\n"
              << "Status: " << prediction.status << "\n"
              << "Details: " << prediction.meaning << "\n";

    return 0; 
}
