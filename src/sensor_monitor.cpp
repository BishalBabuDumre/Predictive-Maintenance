#include <onnxruntime_cxx_api.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <cstring>

// ===== CONFIG =====
#define WINDOW 168  // 7 days hourly

// ===== GLOBAL BUFFER =====
float temp_buffer[WINDOW] = {0};
int index_ptr = 0;
bool buffer_full = false;

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

    // ===== ONNX INIT =====
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "edge");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    Ort::Session session(env, "model.onnx", session_options);
    Ort::AllocatorWithDefaultOptions allocator;

    const char* input_names[] = {"input"};
    const char* output_names[] = {"output"};

    // ===== LOOP (simulate hourly) =====
    while (true) {

        float new_temp;

        // TODO: Replace with sensor read
        std::cin >> new_temp;

        add_reading(new_temp);

        if (!buffer_full) continue;

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
    }

    return 0;
}
