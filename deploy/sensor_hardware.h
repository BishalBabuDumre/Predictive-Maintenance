#ifndef SENSOR_HARDWARE_H
#define SENSOR_HARDWARE_H

#include <string>

class SensorHardware {
public:
    static constexpr int WINDOW = 168; // 7 days of hourly data
    static constexpr float ERROR_CODE = -999.0f;

    SensorHardware(const std::string& buffer_filepath);

    // Hardware and Storage API
    bool loadBuffer();
    bool saveBuffer();
    float readSensorHardware(const std::string& device_id = "28-xxxxxxxxxxxx");
    void addReading(float temp);

    // Getters for status tracking
    bool isBufferFull() const;
    int getIndexPtr() const;
    const float* getBufferData() const; // Allows the feature extractor to read the array safely

private:
    std::string buffer_file_;
    float temp_buffer_[WINDOW] = {0.0f};
    int index_ptr_ = 0;
    bool buffer_full_ = false;
};

#endif // SENSOR_HARDWARE_H
