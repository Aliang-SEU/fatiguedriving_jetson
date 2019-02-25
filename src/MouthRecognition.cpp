#include "MouthRecognition.h"
#include <iostream>


MouthRecognition::MouthRecognition(float fps){
    outFile.open("mouthState.txt");
    this->fps = fps;
    this->flagCount = 0;
    this->threshAlpha = threshAlpha1;
    this->warningCount1 = (int) (this->fps * 1);
    this->warningCount2 = (int) (this->fps * 2);
    this->warningCount3 = (int) (this->fps * 5);
    this->totalCount = (int) (this->fps * 20);
}

MouthRecognition::~MouthRecognition() {
    outFile.close();
}

/**
 * @brief getDistance
 * @param p1
 * @param p2
 * @return
 */
float getDistance(cv::Point2f p1, cv::Point2f p2) {
    float res = std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2));
    return res;
}

/** 人脸倾斜矫正2D(需要结合姿态估计计算3维情况下的人脸)
 * @brief MouthRecognition::correctPoint
 * @param theta
 * @param points
 */
void MouthRecognition::correctPoint(float theta, std::vector<cv::Point2f>& points) {
    cv::Point2f& p0 = points[0];
    for(size_t i = 1; i < points.size(); i++) {
        cv::Point2f p(points[i].x, points[i].y);
        points[i].x = (p.x - p0.x) * std::cos(theta) - (p.y - p0.y) * std::sin(theta);
        points[i].y = (p.x - p0.x) * std::sin(theta) + (p.y - p0.y) * std::cos(theta);
    }
}

/** 计算人脸对应的张合度
 * @brief MouthRecognition::calculate
 * @param points
 */
void MouthRecognition::calculate(std::vector<cv::Point2f>& points) {

    //get innerMouth Points 60-67
    innerMouth.clear();

    for(int i = 0; i < 8; i++) {
        innerMouth.push_back(points[60 + i]);
    }

    //此处需要进行矫正
    //float theta = atan2(innerMouth[4].y - innerMouth[0].y, innerMouth[4].x - innerMouth[0].x);
    //correctPoint(theta, innerMouth);

    float averageDistance = 0;
    for(int i = 0; i < 3; i++) {
        float temp = getDistance(innerMouth[1 + i], innerMouth[7 - i]);
        averageDistance += temp;
    }

    averageDistance /= 2.0;
    H = averageDistance;
    L = std::abs(innerMouth[0].x - innerMouth[4].x); //0 和 4 点分别为左右嘴角点
    alpha = H / (L + 0.00001);      //张合度

    if(alpha >= threshAlpha) {
        flagCount++;
    }else{
        flagCount = 0;
    }
    if(flagCount > warningCount2) {
        std::cout << "检测到轻度哈欠疲劳" << std::endl;
        flagCount = 0;
    }
    if(flagCount > warningCount1) {
        this->threshAlpha = threshAlpha2;
    }

    alphaQueue.push_back(alpha);
    if(alphaQueue.size() > totalCount){
        alphaQueue.pop_front();
        int count = 0;
        for(size_t i = 0 ; i < alphaQueue.size(); i++) {
            if(alphaQueue.at(i) > threshAlpha)
                count++;
        }
        if(count > warningCount3) {
            std::cout << "检测到重度哈欠疲劳!" << std::endl;
            alphaQueue.clear();
        }
    }

    /*for(int i = 0; i < 8; i++) {
        outFile << innerMouth[i].x << " " << innerMouth[i].y << std::endl;
    }
    outFile << alpha << std::endl;
    outFile << std::endl;
    std::cout<<"张合度:" << alpha << std::endl;
    */
}

void detectYawn() {


}
