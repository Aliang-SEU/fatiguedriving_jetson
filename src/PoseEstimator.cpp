#include "PoseEstimator.h"

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <vector>

/**
 * @brief PoseEstimator::PoseEstimator 构造函数
 * @param imageSize
 */
PoseEstimator::PoseEstimator(cv::Size imageSize){

    getFullModelPoints();

    focalLength = imageSize.width;
    cameraCenter[0] = imageSize.width / 2;
    cameraCenter[1] = imageSize.height / 2;

    cameraMatrix = cv::Mat::zeros(3,3, CV_32F);
    float tempMatrix[3][3] = { { focalLength,0 ,cameraCenter[0] }, { 0, focalLength, cameraCenter[1]}, { 0, 0 ,1} };
    for (int i = 0; i < 3;i++){
        for (int j = 0; j < 3;j++){
             cameraMatrix.at<float>(i, j) = tempMatrix[i][j];
         }
    }

    float rvec[] = {0.01891013, 0.08560084, -3.14392813};
    float tvec[] = {-14.97821226, -10.62040383, -2053.03596872};

    rVec = cv::Mat(3, 1, CV_32F, rvec);
    tVec = cv::Mat(3, 1, CV_32F, tvec);

    point3d.push_back(cv::Point3f(-rearSize, -rearSize, rearDepth));
    point3d.push_back(cv::Point3f(-rearSize, rearSize, rearDepth));
    point3d.push_back(cv::Point3f(rearSize, rearSize, rearDepth));
    point3d.push_back(cv::Point3f(rearSize, -rearSize, rearDepth));
    point3d.push_back(cv::Point3f(-rearSize, -rearSize, rearDepth));

    point3d.push_back(cv::Point3f(-frontSize, -frontSize, frontDepth));
    point3d.push_back(cv::Point3f(-frontSize, frontSize, frontDepth));
    point3d.push_back(cv::Point3f(frontSize, frontSize, frontDepth));
    point3d.push_back(cv::Point3f(frontSize, -frontSize, frontDepth));
    point3d.push_back(cv::Point3f(-frontSize, -frontSize, frontDepth));

}

/**
 * @brief PoseEstimator::solvePoseBy68Points 求解对应的姿态
 * @param imagePoints
 * @return
 */
std::vector<cv::Mat_<float>> PoseEstimator::solvePoseBy68Points(std::vector<cv::Point2f> imagePoints){

    std::vector<cv::Mat_<float>> res;
    //cv::Mat inliers;
    //cv::solvePnPRansac(modelPoints68, imagePoints, cameraMatrix, cv::noArray(), rVec, tVec, false);
    cv::solvePnP(modelPoints68, imagePoints, cameraMatrix, cv::noArray(), rVec, tVec, false, cv::SOLVEPNP_EPNP);

    res.push_back(rVec);
    res.push_back(tVec);
    return res;
}

/**
 * @brief PoseEstimator::getFullModelPoints 载入标定文件
 */
void PoseEstimator::getFullModelPoints(){

    std::ifstream inFile;
    inFile.open("../model/pointsModel.txt");
    if(!inFile.is_open()) {
        std::cerr<< "cannot find pointsModel file" << std::endl;
        exit(0);
    }
    float num;
    std::vector<float> rawValue;
    while(!inFile.eof()) {
        inFile >> num;
        rawValue.push_back(num);
    }

    for(int i = 0; i < 68; i++) {
        cv::Point3f point = cv::Point3f(rawValue[i], rawValue[i + 68], rawValue[i + 136] * -1);
        modelPoints68.push_back(point);
    }
}

/**
 * @brief calEular 计算对应的
 */
cv::Mat PoseEstimator::convertToEularAngle(cv::Mat rotation_vec, cv::Mat translation_vec) {

    cv::Mat rotation_matrix;
    cv::Mat pose_mat = cv::Mat(3, 4, CV_32F);
    cv::Mat euler_angle = cv::Mat(3, 1, CV_32F);
    cv::Mat out_intrinsics = cv::Mat(3, 3, CV_32F);
    cv::Mat out_rotation = cv::Mat(3, 3, CV_32F);
    cv::Mat out_translation = cv::Mat(4, 1, CV_32F);

    cv::Rodrigues(rotation_vec, rotation_matrix);
    cv::hconcat(rotation_matrix, translation_vec, pose_mat);
    cv::decomposeProjectionMatrix(pose_mat, out_intrinsics, out_rotation, out_translation, cv::noArray(), cv::noArray(), cv::noArray(), euler_angle);
    return euler_angle;
}


std::vector<cv::Point> convert2I(std::vector<cv::Point2f> inPoints){

    std::vector<cv::Point> res;
    for(size_t i = 0; i < inPoints.size(); i++){
        cv::Point temp = inPoints[i];
        res.push_back(temp);
    }

    return res;
}
/**
 * @brief PoseEstimator::drawAnnotationBox 可视化
 * @param image
 * @param rVec
 * @param tVec
 */
void PoseEstimator::drawAnnotationBox(cv::Mat image, cv::Mat r_vec, cv::Mat t_vec) {

    std::vector<cv::Point2f> imagePoints;

    //投影函数
    cv::projectPoints(point3d, r_vec, t_vec, cameraMatrix, cv::noArray(), imagePoints);

    std::vector<cv::Point> outPoints = convert2I(imagePoints);

    //绘图
    cv::polylines(image, outPoints, true, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
    cv::line(image, outPoints[1], outPoints[6], cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
    cv::line(image, outPoints[2], outPoints[7], cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
    cv::line(image, outPoints[3], outPoints[8], cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
}



