#ifndef LANDMARK_H
#define LANDMARK_H

#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "Utils.h"

using namespace caffe;

// 鼻尖 30
// 鼻根 27
// 下巴 8
// 左眼外角 36
// 左眼内角 39
// 右眼外角 45
// 右眼内角 42
// 嘴中心   66
// 嘴左角   48
// 嘴右角   54
// 左脸最外 0
// 右脸最外 16

typedef struct LandmarkPoints {
    std::vector<cv::Point> faceLandmark; //0-67
    std::vector<cv::Point> faceEdge;    //0-16
    std::vector<cv::Point> leftEyebrow; //17-21
    std::vector<cv::Point> rightEybrow; //22-26
    std::vector<cv::Point> nose;        //27-35
    std::vector<cv::Point> leftEye;     //36-41 顺时针
    std::vector<cv::Point> rightEye;    //42-47 顺时针
    std::vector<cv::Point> outerMouth;  //48-59 顺时针
    std::vector<cv::Point> innerMouth;  //60-67 顺时针
} LandmarkPoints;

class Landmark{
public:
    Landmark();
    LandmarkPoints detectLandmark(cv::Mat& img);
private:
    std::string network = "../model/landmark.prototxt";
    std::string weights = "../model/landmark.caffemodel";

    std::shared_ptr<Net<float>> net;
    cv::Size  input_size;
    int num_channels_;

};

#endif // LANDMARK_H
