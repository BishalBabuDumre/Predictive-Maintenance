#include <onnxruntime_cxx_api.h>
#include <vector>
#include <iostream>
#include <string>
#include <cstdlib>

// Import our two micro-modules
#include "sensor_hardware.h"
#include "feature_extractor.h"

const std::string BUFFER_FILE = "/home/pi/project_edge/sensor_history.dat";
const std::string MODEL_PATH = "/home/pi/project_edge/model.onnx";

int main() {
    // 1. Instantiate our decoupled components
    SensorHardware sensor(BUFFER_FILE);
    FeatureExtractor extractor;

    sensor.loadBuffer();

    // 2. Fetch hardware reading
    float new_temp = sensor.readSensorHardware("28-xxxxxxxxxxxx");
    if (new_temp == SensorHardware::ERROR_CODE) {
        std::cerr << "Hardware Read Error" << std::endl;
        return 1; 
    }

    // 3. Store the reading
    sensor.addReading(new_temp);
    sensor.saveBuffer();

    // 4. Terminate early if our window baseline is not complete
    if (!sensor.isBufferFull()) {
        std::cout << "Buffer filling: " << sensor.getIndexPtr() 
                  << "/" << SensorHardware::WINDOW << std::endl;
        return 0; 
    }

    // ===== ONNX SETUP =====
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

    // ===== CALL EXTRACTOR WITH DATA COMPONENT =====
    std::vector<float> input_tensor_values = extractor.extractFeatures(new_temp, sensor);
    std::vector<int64_t> input_shape = {1, 5};

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        allocator.GetInfo(),
        input_tensor_values.data(),
        input_tensor_values.size(),
        input_shape.data(),
        input_shape.size()
    );

    // ===== RUN INFERENCE =====
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
    float* output = output_tensors[0].GetTensorMutableData<float>();

    int class_id = 0;
    float max_prob = output[0];
    for (int i = 1; i < 4; i++) {
        if (output[i] > max_prob) {
            max_prob = output[i];
            class_id = i;
        }
    }

    const char* labels[] = {"NORMAL", "SPIKE", "DRIFT", "BIAS"};
    std::cout << "Temp: " << new_temp << " -> Detected Class: " << labels[class_id] << "\n";

    // ===== TELEMETRY DISPATCH =====
    std::string cmd = "curl -X POST http://your-server/api -d \"temp=" + std::to_string(new_temp) + 
                      "&class=" + labels[class_id] + 
                      "&spike=" + std::to_string(output[4]) + 
                      "&drift=" + std::to_string(output[5]) + 
                      "&bias=" + std::to_string(output[6]) + "\"";
    std::system(cmd.c_str());

    return 0; 
}
