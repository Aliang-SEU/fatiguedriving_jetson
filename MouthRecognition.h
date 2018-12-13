#ifndef MOUTHRECOGNITION_H
#define MOUTHRECOGNITION_H
#include <opencv2/core.hpp>
using namespace cv;

/**
 * 双阈值法哈欠检测
 */
class MouthRecognition{
public:
    MouthRecognition();
    void calculate();
    void correctPoint(float theta, std::vector<cv::Point>& points);

private:
    std::vector<cv::Point> innerMouth;  //内嘴角
    float H;
    float L;
    float alpha; //张口度 alpha = H / L
                 // H = MAX(F1(X) - F2(X))
                 // L = |X1 - X2|

    int flagCount;  //张和度超过alpha的连续帧
    int stage = 0;  //不同的阶段
    float threshAlpha = 0.65;
    const float threshAlpha1 = 0.65;
    const float threshAlpha2 = 0.5;
    const int warningCount1 = 25;//连续帧警戒值1
    const int warningCount2 = 75;//连续帧警戒值1
    const int warningCount3 = 125;//连续帧警戒值2

};

#endif // MOUTHRECOGNITION_H
