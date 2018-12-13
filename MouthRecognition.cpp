#include "MouthRecognition.h"
#include "Message.h"

MouthRecognition::MouthRecognition(){}

//void checkFlag() {
//    if(flagCount > warningCount) {

//    }
//}

//void clearFlag() {
//    //清除已经累计的数值
//    flagCount = 0;
//}

float getDistance(cv::Point p1, cv::Point p2) {
    float res = std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2));
    return res;
}
/*
 * 人脸倾斜矫正2D(需要结合姿态估计计算3维情况下的人脸)
 */
void MouthRecognition::correctPoint(float theta, std::vector<cv::Point>& points) {
    cv::Point& p0 = points[0];
    for(size_t i = 1; i < points.size(); i++) {
        cv::Point p(points[i].x, points[i].y);
        points[i].x = (p.x - p0.x) * std::cos(theta) - (p.y - p0.y) * std::sin(theta);
        points[i].y = (p.x - p0.x) * std::sin(theta) + (p.y - p0.y) * std::cos(theta);
    }

}
void MouthRecognition::calculate() {
    // 此处需要进行矫正
    float theta = atan2(innerMouth[4].y - innerMouth[0].y, innerMouth[4].x - innerMouth[0].x);

    correctPoint(theta, innerMouth);

    float maxDis = 0;
    for(int i = 0; i < 3; i++) {
        float temp = getDistance(innerMouth[1 + 1], innerMouth[7 - i]);
        if(temp > maxDis) {
            maxDis = temp;
        }
    }
    H = maxDis;
    L = std::abs(innerMouth[0].y - innerMouth[4].y); //0 和 4 点分别为左右嘴角点
    alpha = H / (L + 0.00001);      //张合度

//    if(alpha >= threshAlpha) {
//        //嘴角过大需要进行判决
//        flagCount++;
//        checkFlag();
//    }else {
//        clearFlag();
//    }
}
