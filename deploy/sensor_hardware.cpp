#include "sensor_hardware.h"
#include <fstream>

SensorHardware::SensorHardware(const std::string& buffer_filepath) 
    : buffer_file_(buffer_filepath) {}

bool SensorHardware::loadBuffer() {
    std::ifstream is(buffer_file_, std::ios::binary);
    if (!is) return false;
    is.read(reinterpret_cast<char*>(temp_buffer_), sizeof(temp_buffer_));
    is.read(reinterpret_cast<char*>(&index_ptr_), sizeof(index_ptr_));
    is.read(reinterpret_cast<char*>(&buffer_full_), sizeof(buffer_full_));
    return true;
}

bool SensorHardware::saveBuffer() {
    std::ofstream os(buffer_file_, std::ios::binary);
    if (!os) return false;
    os.write(reinterpret_cast<char*>(temp_buffer_), sizeof(temp_buffer_));
    os.write(reinterpret_cast<char*>(&index_ptr_), sizeof(index_ptr_));
    os.write(reinterpret_cast<char*>(&buffer_full_), sizeof(buffer_full_));
    return true;
}

float SensorHardware::readSensorHardware(const std::string& device_id) {
    std::string path = "/sys/bus/w1/devices/" + device_id + "/w1_slave";
    std::ifstream file(path);
    if (!file.is_open()) return ERROR_CODE;

    std::string line;
    float celsius = 0.0f;
    bool valid = false;

    while (std::getline(file, line)) {
        if (line.find("YES") != std::string::npos) {
            valid = true;
        } else if (valid && line.find("t=") != std::string::npos) {
            size_t pos = line.find("t=");
            int raw = std::stoi(line.substr(pos + 2));
            celsius = raw / 1000.0f;
        }
    }
    file.close();

    if (!valid) return ERROR_CODE;
    return (celsius * 9.0f / 5.0f) + 32.0f; // Fahrenheit
}

void SensorHardware::addReading(float temp) {
    temp_buffer_[index_ptr_] = temp;
    index_ptr_ = (index_ptr_ + 1) % WINDOW;
    if (index_ptr_ == 0) buffer_full_ = true;
}

bool SensorHardware::isBufferFull() const { return buffer_full_; }
int SensorHardware::getIndexPtr() const { return index_ptr_; }
const float* SensorHardware::getBufferData() const { return temp_buffer_; }
