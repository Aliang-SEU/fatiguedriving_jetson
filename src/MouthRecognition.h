#ifndef MOUTHRECOGNITION_H
#define MOUTHRECOGNITION_H

#include <opencv2/core.hpp>
#include <deque>
#include <fstream>
/**
 * 双阈值法哈欠检测
 */
class MouthRecognition{

public:
    MouthRecognition(float fps=25.0);
    ~MouthRecognition();
    void calculate(std::vector<cv::Point2f>& points);

private:
    void correctPoint(float theta, std::vector<cv::Point2f>& points);

private:
    std::ofstream outFile;

    std::vector<cv::Point2f> innerMouth;  //内嘴角
    float H;
    float L;
    float alpha; //张口度 alpha = H / L
                 // H = MAX(F1(X) - F2(X))
                 // L = |X1 - X2|

    int flagCount;  //张和度超过alpha的连续帧
    int stage = 0;  //不同的阶段
    float threshAlpha;
    const float threshAlpha1 = 0.5;
    const float threshAlpha2 = 0.4;
    int warningCount1;//连续帧警戒值1
    int warningCount2;//连续帧警戒值1
    int warningCount3;//总帧数警戒值
    int totalCount;
    std::deque<float> alphaQueue; //
    float fps;

};

#endif // MOUTHRECOGNITION_H
