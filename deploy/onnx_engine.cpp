#include "onnx_engine.h"
#include <stdexcept>

OnnxEngine::OnnxEngine(const std::string& model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "edge_engine"),
      session_options_(),
      session_(env_, model_path.c_str(), session_options_),
      allocator_() {
    
    // Optimize performance constraints for edge runtime deployment
    session_options_.SetIntraOpNumThreads(1);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options_.EnableMemPattern();
    session_options_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

    // Default target IO layer names mapping to training definitions
    input_names_ = {"input"};
    output_names_ = {"output"};
}

std::vector<float> OnnxEngine::runInference(const std::vector<float>& input_features, const std::vector<int64_t>& input_shape) {
    // Create an un-managed wrapper tensor around raw vector allocations
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        allocator_.GetInfo(),
        const_cast<float*>(input_features.data()),
        input_features.size(),
        input_shape.data(),
        input_shape.size()
    );

    auto output_tensors = session_.Run(
        Ort::RunOptions{nullptr},
        input_names_.data(),
        &input_tensor, 1,
        output_names_.data(), 1
    );

    // Extract element attributes safely from output buffers
    float* raw_output = output_tensors[0].GetTensorMutableData<float>();
    auto tensor_info = output_tensors[0].GetTensorTypeAndShapeInfo();
    size_t total_elements = tensor_info.GetElementCount();

    return std::vector<float>(raw_output, raw_output + total_elements);
}
