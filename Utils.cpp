#include "Utils.h"

Utils::Utils()
{

}

void Utils::timeStart() {
    this->_start = cv::getTickCount();
}

void Utils::timeEnd(std::string& str) {
    std::cout<< str << (double) (cv::getTickCount() - this->_start ) / cv::getTickFrequency()
                    <<std::endl;
}

void Utils::timeUpdate(std::string& str) {
    int64 _now = cv::getTickCount();
    std::cout<< str << (double) (_now - this->_start ) / cv::getTickFrequency()
                    <<std::endl;
    this->_start = _now;
}
