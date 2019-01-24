#ifndef KALMANSTABILIZER_H
#define KALMANSTABILIZER_H

#include <opencv2/opencv.hpp>

class KalmanStabilizer{
public:
    KalmanStabilizer(int state_num, int measure_num, float cov_process, float cov_measure);
    void setQR(float cov_process = 0.1, float cov_measure = 0.001);
    void update(float _measurement);
    float getState();

private:
    int stateNum; //状态数量
    int measureNum; //测量数量
    float covProcess; //
    float covMeasure;

    cv::KalmanFilter filter;
    cv::Mat state;
    cv::Mat measurement;
    cv::Mat prediction;

};

#endif // KALMANSTABILIZER_H
