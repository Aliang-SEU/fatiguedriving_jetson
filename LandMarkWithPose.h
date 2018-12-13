#ifndef LANDMARKWITHPOSE_H
#define LANDMARKWITHPOSE_H

#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace caffe;

typedef struct LandmarkAndPose {
    std::vector<float> landmark;
    std::vector<float> pose;
} LandmarkAndPose;

class LandmarkWithPose{

public:
    LandmarkWithPose();
    void setMean(const std::string& meanFile);
    void preprocess(cv::Mat& img, std::vector<cv::Mat>* input_channels);
    void wrapInputLayer(std::vector<cv::Mat>* input_channels);
    LandmarkAndPose getPredict(cv::Mat& image);

private:
    std::string network = "model/deploy.prototxt";
    std::string param = "model/68point_dlib_with_pose.caffemodel";
    std::string meanfile = "model/VGG_mean.binaryproto";

    std::shared_ptr<Net<float>> net;
    cv::Size inputSize;
    cv::Mat mean;
    int numChannels;
};

#endif // LANDMARKWITHPOSE_H
