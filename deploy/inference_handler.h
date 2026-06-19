#ifndef INFERENCE_HANDLER_H
#define INFERENCE_HANDLER_H

#include <string>
#include <vector>

// Struct to package the parsed results cleanly
struct InferenceResult {
    std::string className;
    float spikeMag;
    float driftMag;
    float biasMag;
};

class InferenceHandler {
public:
    InferenceHandler();

    // Takes the raw flat float array from ONNX and extracts classes/magnitudes
    InferenceResult parseOutputs(const float* raw_output);
};

#endif // INFERENCE_HANDLER_H
