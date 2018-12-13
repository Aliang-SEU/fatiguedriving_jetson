#include <opencv2/opencv.hpp>
#include <caffe/caffe.hpp>

using namespace caffe;

class EyeRecognition {

public:
    EyeRecognition();
    EyeRecognition(const std::string model_file, const std::string train_model);
    ~EyeRecognition();

	bool predict(const cv::Mat& img);
	void WrapInputLayer(std::vector<cv::Mat>* input_channels);
	void Preprocess(const cv::Mat& img,std::vector<cv::Mat>* input_channels);
private:
	std::shared_ptr<Net<float>> net;
	cv::Size input_size;
	cv::Mat mean_;
	int num_channels_;

    const std::string modelPath="./model/eye/detect_eye.prototxt";
    const std::string paramPath="./model/eye/detect_eye.caffemodel";
};

