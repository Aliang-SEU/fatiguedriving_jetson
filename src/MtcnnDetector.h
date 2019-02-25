#ifndef MTCNNDETECTOR_H
#define MTCNNDETECTOR_H

#include <fstream>
#include <iostream>

#include <omp.h>
#include <boost/shared_ptr.hpp>

#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>

using namespace caffe;

//人脸边界框
typedef struct FaceBox {
    float xmin;
    float ymin;
    float xmax;
    float ymax;
    float score;
} FaceBox;

//人脸信息
typedef struct FaceInfo {
    float bbox_reg[4];
    float landmark_reg[10];
    float landmark[10];
    FaceBox bbox;
} FaceInfo;

namespace MtcnnNamespace {
    struct FaceInfoComparator{
        bool operator()(const FaceInfo& f1, const FaceInfo& f2) {
            return f1.bbox.score < f2.bbox.score;
        }
    };
}

class MtcnnDetector {
public:

    MtcnnDetector(const std::string& proto_model_dir);
    std::vector<FaceInfo> Detect(const cv::Mat& img, const int stage = 3);

    void drawResult(std::vector<FaceInfo>& faceInfo, cv::Mat& image);

protected:

    void autoSet(std::vector<FaceInfo>& faceInfo, const cv::Mat& image);
    std::vector<FaceInfo> ProposalNet(const cv::Mat& img, int min_size, float threshold, float factor);
    std::vector<FaceInfo> NextStage(const cv::Mat& image, std::vector<FaceInfo> &pre_stage_res, int input_w, int input_h, int stage_num, const float threshold);
    void BBoxRegression(std::vector<FaceInfo>& bboxes);
    void BBoxPadSquare(std::vector<FaceInfo>& bboxes, int width, int height);
    void BBoxPad(std::vector<FaceInfo>& bboxes, int width, int height);
    void GenerateBBox(Blob<float>* confidence, Blob<float>* reg_box, float scale, float thresh);
    std::vector<FaceInfo> NMS(std::vector<FaceInfo>& bboxes, float thresh, char methodType);
    float IoU(float xmin, float ymin, float xmax, float ymax, float xmin_, float ymin_, float xmax_, float ymax_, bool is_iom = false);

    static inline bool CompareBBox(const FaceInfo & a, const FaceInfo & b) {
        return a.bbox.score > b.bbox.score;
    }

private:
    boost::shared_ptr<Net<float>> PNet_;
    boost::shared_ptr<Net<float>> RNet_;
    boost::shared_ptr<Net<float>> ONet_;

    std::vector<FaceInfo> candidate_boxes_;
    std::vector<FaceInfo> total_boxes_;
    std::vector<FaceInfo> faceInfo;

    const float factor = 0.5f;
    const float threshold[3] = { 0.9f, 0.9f, 0.9f };
    const int minSize = 60;

    //omp
    const int threads_num = 6;
    //pnet config
    const float pnet_stride = 2;
    const float pnet_cell_size = 12;
    const int pnet_max_detect_num = 1000;
    //mean & std
    const float mean_val = 127.5f;
    const float std_val = 0.0078125f;
    //minibatch size
    const int step_size = 128;

};

#endif // MTCNNDETECTOR_H
