#ifndef EYERECOGNITION_H
#define EYERECOGNITION_H
#include <opencv2/opencv.hpp>
#include <caffe/caffe.hpp>
#include <iostream>
using namespace caffe;

enum EyeState{
    CLOSED,
    OPEN
};


class EyeRecognition {

public:
    EyeRecognition(float fps=25.0);
    ~EyeRecognition();

	bool predict(const cv::Mat& img);
	void WrapInputLayer(std::vector<cv::Mat>* input_channels);
	void Preprocess(const cv::Mat& img,std::vector<cv::Mat>* input_channels);
    void recordEyeState(enum EyeState leftEye, enum EyeState rightEye);
    bool decideFatigue(int length, float ratio);
    void setFps(float fps) ;

private:
	std::shared_ptr<Net<float>> net;
	cv::Size input_size;
	cv::Mat mean_;
	int num_channels_;
    Blob<float>* input_layer;
    const std::string modelPath = "../model/eye/detect_eye.prototxt";
    const std::string paramPath = "../model/eye/detect_eye.caffemodel";

private:
    std::deque<enum EyeState> eyeStateQueue;
    float fps;

    int minLength;
    int halfLength;
    int maxLength;

    float ratio = 0.5;
};
#endif
