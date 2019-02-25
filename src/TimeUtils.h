#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>

class Utils
{
public:
    Utils() {};
    void timeStart();
    void timeEnd(std::string& str);
    void timeUpdate(std::string& str);

private:
    int64 start;

};

#endif // UTILS_H
