#include "TimeUtils.h"

#define INFO_STREAM( stream ) \
std::cout << stream << std::endl

#define WARN_STREAM( stream ) \
std::cout << "Warning: " << stream << std::endl

#define ERROR_STREAM( stream ) \
std::cout << "Error: " << stream << std::endl


void Utils::timeStart() {

    this->start = cv::getTickCount();
}

void Utils::timeEnd(std::string& str) {

    INFO_STREAM(str << (double) (cv::getTickCount() - this->start ) / cv::getTickFrequency());
}

void Utils::timeUpdate(std::string& str) {

    int64 now = cv::getTickCount();
    INFO_STREAM(str << (double) (now - this->start ) / cv::getTickFrequency());
    this->start = now;
}
