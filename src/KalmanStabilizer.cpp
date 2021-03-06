#include "KalmanStabilizer.h"
#include <assert.h>

KalmanStabilizer::KalmanStabilizer(int state_num, int measure_num, float cov_process, float cov_measure):
      stateNum(state_num), measureNum(measure_num),covProcess(cov_process), covMeasure(cov_measure) {

    assert(state_num == 2 || state_num == 4);

    filter = cv::KalmanFilter(state_num, measure_num, 0);
    state = cv::Mat::zeros(state_num, 1, CV_32F);
    measurement = cv::Mat::ones(measure_num, 1, CV_32F);
    prediction = cv::Mat::zeros(state_num, 1, CV_32F);

    if(measure_num == 1) {
        cv::Mat transitionMat = (cv::Mat_<float>(2, 2) << 1.0, 1.0, 0.0, 1.0);
        cv::Mat measurementMat =(cv::Mat_<float>(1, 2) << 1.0, 1.0);
        cv::Mat processNoiseCov = (cv::Mat_<float>(2, 2) << 1.0, 0.0, 0.0, 1.0);
        processNoiseCov *= covProcess;
        cv::Mat measurementNoiseCov = (cv::Mat_<float>(1, 1) << 1.0);
        measurementNoiseCov *= covMeasure;

        filter.transitionMatrix = transitionMat;
        filter.measurementMatrix = measurementMat;
        filter.processNoiseCov = processNoiseCov;
        filter.measurementNoiseCov = measurementNoiseCov;

    }else if(measure_num == 2) {
    }
//    cv::waitKey(0);

//   std::cout<< "filter.transitionMatrix:" << filter.transitionMatrix << std::endl \
//            << "filter.measurementMatrix:" << filter.measurementMatrix << std::endl \
//            << "filter.processNoiseCov:" << filter.processNoiseCov << std::endl \
//            << "filter.measurementNoiseCov:" << filter.measurementNoiseCov << std::endl << std::endl;
}

void KalmanStabilizer::update(float _measurement) {
    //进行一次预测
    prediction = filter.predict();
    //获取新的预测值
    if(measureNum == 1) {
       measurement.at<float>(0) = _measurement;
    }else {

    }

    //使用预测值进行修正
    filter.correct(measurement);
    //更新状态值
    state = filter.statePost;
}

/**
 * @brief KalmanStabilizer::setQR  重新设置测量矩阵
 * @param cov_process
 * @param cov_measure
 */
void KalmanStabilizer::setQR(float cov_process, float cov_measure){

    if(measureNum == 1) {
        float temp1[] = {1, 0, 0, 1};
        filter.processNoiseCov = cv::Mat(2, 2, CV_32F, temp1);
        filter.processNoiseCov *= cov_process;
        float temp2[] = {1};
        filter.measurementNoiseCov = cv::Mat(1, 1, CV_32F, temp2);
        filter.measurementNoiseCov *= cov_measure;
    }else {

    }
}

float KalmanStabilizer::getState(){
    return state.at<float>(0);
}
