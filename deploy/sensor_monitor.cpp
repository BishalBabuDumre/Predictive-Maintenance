#include <onnxruntime_cxx_api.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <string>
#include <cstdlib>
#include <fstream>
#include <fcntl.h>
#include <linux/i2c-dev.h>
#include <sys/ioctl.h>
#include <unistd.h>

// ===== CONFIG =====
#define WINDOW 168  // 7 days hourly

// Filename in SD Card
const std::string BUFFER_FILE = "/home/pi/project_edge/sensor_history.dat";

// ===== GLOBAL BUFFER =====
float temp_buffer[WINDOW] = {0};
int index_ptr = 0;
bool buffer_full = false;

// Loads the buffer from the SD card
void load_buffer() {
    std::ifstream is(BUFFER_FILE, std::ios::binary);
    if (is) {
        is.read(reinterpret_cast<char*>(temp_buffer), sizeof(temp_buffer));
        is.read(reinterpret_cast<char*>(&index_ptr), sizeof(index_ptr));
        is.read(reinterpret_cast<char*>(&buffer_full), sizeof(buffer_full));
    }
}

// Saves the buffer back to the SD card
void save_buffer() {
    std::ofstream os(BUFFER_FILE, std::ios::binary);
    if (os) {
        os.write(reinterpret_cast<char*>(temp_buffer), sizeof(temp_buffer));
        os.write(reinterpret_cast<char*>(&index_ptr), sizeof(index_ptr));
        os.write(reinterpret_cast<char*>(&buffer_full), sizeof(buffer_full));
    }
}

float read_sensor_hardware() {
    // Replace with your actual sensor ID found via 'ls /sys/bus/w1/devices/'
    std::string device_id = "28-xxxxxxxxxxxx"; 
    std::string path = "/sys/bus/w1/devices/" + device_id + "/w1_slave";
    
    std::ifstream file(path);
    if (!file.is_open()) return -999.0;

    std::string line;
    float celsius = 0.0;
    bool valid = false;

    // The DS18B20 output has two lines. 
    // Line 1 ends in "YES" if the checksum passed.
    // Line 2 contains "t=23500" (which is 23.500°C)
    while (std::getline(file, line)) {
        if (line.find("YES") != std::string::npos) {
            valid = true;
        } else if (valid && line.find("t=") != std::string::npos) {
            size_t pos = line.find("t=");
            int raw = std::stoi(line.substr(pos + 2));
            celsius = raw / 1000.0;
        }
    }
    file.close();

    if (!valid) return -999.0;

    // Convert to Fahrenheit for your "10s of fahrenheit" scale
    return (celsius * 9.0 / 5.0) + 32.0;
}

// ===== ADD NEW DATA =====
void add_reading(float temp) {
    temp_buffer[index_ptr] = temp;
    index_ptr = (index_ptr + 1) % WINDOW;

    if (index_ptr == 0) buffer_full = true;
}

// ===== FEATURE COMPUTATION =====
float compute_mean() {
    float sum = 0;
    for (int i = 0; i < WINDOW; i++) sum += temp_buffer[i];
    return sum / WINDOW;
}

float compute_std(float mean) {
    float sum = 0;
    for (int i = 0; i < WINDOW; i++) {
        float d = temp_buffer[i] - mean;
        sum += d * d;
    }
    return sqrt(sum / WINDOW);
}

float compute_slope() {
    int oldest = index_ptr;
    int newest = (index_ptr + WINDOW - 1) % WINDOW;
    return (temp_buffer[newest] - temp_buffer[oldest]) / WINDOW;
}

float compute_bias(float mean) {
    return temp_buffer[(index_ptr + WINDOW - 1) % WINDOW] - mean;
}

// ===== MAIN =====
int main() {

    // 1. Initialize data from SD Card
    load_buffer();

    // 2. Get the new sensor reading
    float new_temp= read_sensor_hardware();

    // Check for hardware failure (using our custom return codes!)
    if (new_temp == -999.0) {
        std::cerr << "Hardware Read Error" << std::endl;
        return 1; 
    }

    // 3. Update the buffer and save immediately
    add_reading(new_temp);
    save_buffer();

    // 4. Check if we have enough data (7 days / 168 hours)
    if (!buffer_full) {
        std::cout << "Buffer filling: " << index_ptr << "/" << WINDOW << std::endl;
        return 0; // Exit early, not enough data for inference yet
    }

    // ===== ONNX INIT =====
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "edge");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options.EnableMemPattern();
    session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    
    Ort::Session session(env, "/home/pi/project_edge/model.onnx", session_options);
    Ort::AllocatorWithDefaultOptions allocator;

    const char* input_names[] = {"input"};
    const char* output_names[] = {"output"};

    // ===== FEATURE VECTOR =====
    float mean = compute_mean();
    float std = compute_std(mean);
    float slope = compute_slope();
    float bias = compute_bias(mean);

    std::vector<float> input_tensor_values = {
        new_temp, mean, std, slope, bias
    };

    std::vector<int64_t> input_shape = {1, 5};

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        allocator.GetInfo(),
        input_tensor_values.data(),
        input_tensor_values.size(),
        input_shape.data(),
        input_shape.size()
    );

    // ===== RUN INFERENCE =====
    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        input_names,
        &input_tensor,
        1,
        output_names,
        1
    );

    float* output = output_tensors[0].GetTensorMutableData<float>();

    // ===== OUTPUT PARSING =====
    int class_id = 0;
    float max_prob = output[0];

    for (int i = 1; i < 4; i++) {
        if (output[i] > max_prob) {
            max_prob = output[i];
            class_id = i;
        }
    }

    const char* labels[] = {"NORMAL", "SPIKE", "DRIFT", "BIAS"};

    float spike_mag = output[4];
    float drift_mag = output[5];
    float bias_mag  = output[6];

    std::cout << "Temp: " << new_temp << "\n";
    std::cout << "Class: " << labels[class_id] << "\n";
    std::cout << "Spike: " << spike_mag
              << " Drift: " << drift_mag
              << " Bias: " << bias_mag << "\n";

    // ===== WIFI SEND (simple curl-style placeholder) =====
    // Replace with real HTTP client (libcurl)
    std::string cmd = "curl -X POST http://your-server/api "
                      "-d \"temp=" + std::to_string(new_temp) +
                      "&class=" + labels[class_id] +
                      "&spike=" + std::to_string(spike_mag) +
                      "&drift=" + std::to_string(drift_mag) +
                      "&bias=" + std::to_string(bias_mag) + "\"";

    system(cmd.c_str());

    std::cout << "Inference complete for hour." << std::endl;
    return 0; // Program ends, freeing all RAM
}
