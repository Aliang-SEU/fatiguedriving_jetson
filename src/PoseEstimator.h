#ifndef POSEESTIMATOR_H
#define POSEESTIMATOR_H

#include <opencv2/opencv.hpp>

class PoseEstimator
{
public:
//    PoseEstimator();
    PoseEstimator(cv::Size imageSize);
    void getFullModelPoints();
    std::vector<cv::Mat_<float>> solvePoseBy68Points(std::vector<cv::Point2f> imagePoints);
    cv::Mat convertToEularAngle(cv::Mat rotation_vec, cv::Mat translation_vec);
    void drawAnnotationBox(cv::Mat image, cv::Mat rVec, cv::Mat tVec);

private:

    //float modelPoints[][] = new float[][]{{0.0, 0.0, 0.0},{0.0, -330.0, -65.0},{-225.0, 170.0, -135.0},{225.0, 170.0,- 135.0},{-150.0, -150.0, -125.0},{150.0, -150.0, -125.0}};
    std::vector<cv::Point3f> modelPoints68;
    //相机内坐标参数
    float focalLength;
    float cameraCenter[2];
    cv::Mat cameraMatrix;
    std::vector<float> distCoeefs;
    cv::Mat rVec;
    cv::Mat tVec;

    std::vector<cv::Point3f> point3d;

    const int rearSize = 75;
    const int rearDepth = 0;
    const int frontSize = 100;
    const int frontDepth = 100;
    // RANSAC parameters
    const int iterationsCount = 300;      // number of Ransac iterations.
    const float reprojectionError = 5.991;  // maximum allowed distance to consider it an inlier.
    const double confidence = 0.95;        // ransac successful confidence.


};


#endif // POSEESTIMATOR_H
