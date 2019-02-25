#include <EyeRecognition.h>

/** 构造函数
 * @brief EyeRecognition::EyeRecognition
 */
EyeRecognition::EyeRecognition(float fps) {

    Caffe::set_mode(Caffe::GPU);
    net = std::make_shared<Net<float>>(modelPath, TEST);
    net->CopyTrainedLayersFrom(paramPath);
    input_layer = net->input_blobs()[0];
    input_size = cv::Size(input_layer->width(), input_layer->height());
    num_channels_ = input_layer->channels();
    CHECK(num_channels_ == 3 )
        << "Input layer should have 3 channels.";

    this->fps = fps;

    this->minLength = (int) fps * 2;
    this->halfLength = (int) fps * 10;
    this->maxLength = (int) fps * 30; //这里设置30s为一个基准长度
}

void EyeRecognition::setFps(float fps) {

    this->fps = fps;

    this->minLength = (int) fps * 3;

    this->halfLength = (int) fps * 15;

    this->maxLength = (int) fps * 30; //这里设置30s为一个基准长度
}


EyeRecognition::~EyeRecognition() {}

/**
 * @brief EyeRecognition::WrapInputLayer
 * @param input_channels
 */
void EyeRecognition::WrapInputLayer(std::vector<cv::Mat>* input_channels){
    Blob<float>* input_layer = net->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i) {
      cv::Mat channel(height, width, CV_32FC1, input_data);
      input_channels->push_back(channel);
      input_data += width * height;
    }
}

/**
 * @brief EyeRecognition::Preprocess
 * @param img
 * @param input_channels
 */
void EyeRecognition::Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels){

	cv::Mat sample;

    if (img.channels() == 3 && num_channels_ == 1)
		cv::cvtColor(img, sample, CV_BGR2GRAY);
	else if (img.channels() == 4 && num_channels_ == 1)
		cv::cvtColor(img, sample, CV_BGRA2GRAY);
	else if (img.channels() == 4 && num_channels_ == 3)
		cv::cvtColor(img, sample, CV_BGRA2BGR);
	else if (img.channels() == 1 && num_channels_ == 3)
		cv::cvtColor(img, sample, CV_GRAY2BGR);
	else
		sample = img;

	cv::Mat sample_resized;
	if (sample.size() != input_size)
		cv::resize(sample, sample_resized, input_size);
	else
		sample_resized = sample;

	cv::Mat sample_float;
	if (num_channels_ == 3)
		sample_resized.convertTo(sample_float, CV_32FC3);
	else
		sample_resized.convertTo(sample_float, CV_32FC1);

    cv::Mat sample_normalized;
	//cv::subtract(sample_float, mean_, sample_normalized);
    sample_normalized = sample_float;// * 0.00390625;

	cv::split(sample_normalized, *input_channels);
    CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
          == net->input_blobs()[0]->cpu_data())
      << "Input channels are not wrapping the input layer of the network.";
}

/** 判断眼睛的状态
 * @brief EyeRecognition::predict
 * @param img
 * @return
 */
bool EyeRecognition::predict(const cv::Mat& img){
	std::vector<cv::Mat> input_channels;
	WrapInputLayer(&input_channels);

	Preprocess(img, &input_channels);

	net->Forward();

	//confidence
	Blob<float>* output_layer = net->output_blobs()[0];
	int count = output_layer->count(); //the channel of confidence is two
	const float* confidence_begin = output_layer->cpu_data();

	if (confidence_begin[0] > confidence_begin[1])
		return true;
	else
		return false;
}


bool EyeRecognition::decideFatigue(int length, float ratio) {

    int closeCount = 0;

    for(size_t i = 0; i < length; i++) {
        if(eyeStateQueue.at(length - i - 1) == CLOSED) {
            closeCount++;
        }
    }
    int thresh = (int) (ratio * length);
    //std::cout<< closeCount << "/" <<  thresh << std::endl;

    if(closeCount >= thresh) {
        return true;
    }
    else
        return false;

}
/**
 * @brief recordEyeState
 * @param leftEye
 * @param rightEye
 */
void EyeRecognition::recordEyeState(enum EyeState leftEye, enum EyeState rightEye){

    enum EyeState curState = OPEN;
    //仅仅当双眼都是闭合的时候才判定为一次闭眼
    if(leftEye == CLOSED && rightEye == CLOSED) {
        curState = CLOSED;
    }
    eyeStateQueue.push_back(curState);

//    //满区间检测
//    if(eyeStateQueue.size() > maxLength) {

//        eyeStateQueue.pop_front();
//        //当队列满 进行判断
//        bool flag = decideFatigue(maxLength, 0.3);
//        if(flag == true) {
//            std::cout << "检测到眼睛深度疲劳，请进行休息！";
//        }
//    }
//    //半区间检测
//    if(eyeStateQueue.size() > halfLength) {

//        bool flag = decideFatigue(halfLength, 0.4);
//        if(flag == true)  {
//            eyeStateQueue.clear();
//            std::cout << "检测到眼睛中度疲劳，请进行休息!" ;
//        }
//    }

    //短时间检测
    if(eyeStateQueue.size() > minLength) {
        bool flag = decideFatigue(minLength, 0.8);
        if(flag == true) {
            eyeStateQueue.clear();
            std::cout << "检测到眼睛轻度疲劳，请注意休息!" << std::endl;
        }
        eyeStateQueue.pop_front();
    }

}
