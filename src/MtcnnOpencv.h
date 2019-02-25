#ifndef MTCNNOPENCV_H
#define MTCNNOPENCV_H

#include <fstream>
#include <iostream>

#include <omp.h>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

const int threads_num = 4;
const float pnet_stride = 2;
const float pnet_cell_size = 12;
const int pnet_max_detect_num = 5000;
//mean & std
const float mean_val = 127.5f;
const float std_val = 0.0078125f;
//minibatch size
const int step_size = 128;

typedef struct FaceBox {
    float xmin;
    float ymin;
    float xmax;
    float ymax;
    float score;
} FaceBox;
typedef struct FaceInfo {
    float bbox_reg[4];
    float landmark_reg[10];
    float landmark[10];
    FaceBox bbox;
} FaceInfo;


class MtcnnOpencv{

public:
    MtcnnOpencv(const std::string& proto_model_dir);
    std::vector<FaceInfo> Detect(const cv::Mat& img, const int stage = 3);
    void drawResult(std::vector<FaceInfo>& faceInfo, cv::Mat& image);
//protected:
    std::vector<FaceInfo> ProposalNet(const cv::Mat& img, int min_size, float threshold, float factor);
    std::vector<FaceInfo> NextStage(const cv::Mat& image, std::vector<FaceInfo> &pre_stage_res, int input_w, int input_h, int stage_num, const float threshold);
    void BBoxRegression(std::vector<FaceInfo>& bboxes);
    void BBoxPadSquare(std::vector<FaceInfo>& bboxes, int width, int height);
    void BBoxPad(std::vector<FaceInfo>& bboxes, int width, int height);
    void GenerateBBox(cv::Mat* confidence, cv::Mat* reg_box, float scale, float thresh);
    std::vector<FaceInfo> NMS(std::vector<FaceInfo>& bboxes, float thresh, char methodType);
    float IoU(float xmin, float ymin, float xmax, float ymax, float xmin_, float ymin_, float xmax_, float ymax_, bool is_iom = false);

public:
    cv::dnn::Net PNet_;
    cv::dnn::Net RNet_;
    cv::dnn::Net ONet_;

    std::vector<FaceInfo> candidate_boxes_;
    std::vector<FaceInfo> total_boxes_;

    const float factor = 0.5f;
    const float threshold[3] = { 0.7f, 0.7f, 0.7f };
    const int minSize = 40;
};

#endif // MTCNNOPENCV_H
