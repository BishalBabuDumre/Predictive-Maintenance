#include "inference_handler.h"

InferenceHandler::InferenceHandler() {}

InferenceResult InferenceHandler::parseOutputs(const float* raw_output) {
    // 1. Determine maximum probability index for classification (first 4 elements)
    int class_id = 0;
    float max_prob = raw_output[0];
    for (int i = 1; i < 4; i++) {
        if (raw_output[i] > max_prob) {
            max_prob = raw_output[i];
            class_id = i;
        }
    }

    const char* labels[] = {"NORMAL", "SPIKE", "DRIFT", "BIAS"};

    // 2. Package the labels alongside the trailing regression magnitudes
    InferenceResult result;
    result.className = labels[class_id];
    result.spikeMag  = raw_output[4];
    result.driftMag  = raw_output[5];
    result.biasMag   = raw_output[6];

    return result;
}
