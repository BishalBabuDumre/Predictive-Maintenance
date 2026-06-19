#ifndef ONNX_ENGINE_H
#define ONNX_ENGINE_H

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>

class OnnxEngine {
public:
    OnnxEngine(const std::string& model_path);
    ~OnnxEngine() = default;

    // Executes a forward pass for a single batch instance
    std::vector<float> runInference(const std::vector<float>& input_features, const std::vector<int64_t>& input_shape);

private:
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    Ort::Session session_;
    Ort::AllocatorWithDefaultOptions allocator_;

    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
};

#endif // ONNX_ENGINE_H
