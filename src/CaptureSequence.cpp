#include "CaptureSequence.h"

#include <iostream>

#include <filesystem.hpp>
#include <filesystem/fstream.hpp>
#include <boost/algorithm/string.hpp>

#include <opencv2/imgproc.hpp>

#include <chrono>
#include <ctime>

using namespace Utilities;

#define INFO_STREAM( stream ) \
std::cout << stream << std::endl

#define WARN_STREAM( stream ) \
std::cout << "Warning: " << stream << std::endl

#define ERROR_STREAM( stream ) \
std::cout << "Error: " << stream << std::endl


void CaptureSequence::close(){
    // Close the capturing threads
    isCapturing = false;

    // If the queue is full it will be blocked, so need to empty it
    while (!captureQueue.empty())
    {
        captureQueue.pop();
    }

    if (captureThread.joinable())
        captureThread.join();

    // Release the capture objects
    if (capture.isOpened())
        capture.release();
}

//析构函数
CaptureSequence::~CaptureSequence() {
    this->close();
}


const std::string currentDateTime() {

    time_t rawTime;
    struct tm * timeInfo;

    char buffer[80];
    time(&rawTime);
    timeInfo = localtime(&rawTime);

    strftime(buffer, sizeof(buffer), "%Y-%m-%d-%H-%M", timeInfo);

    return buffer;
}

//指定VideoCapture，确保设备已经打开才能初始化
bool CaptureSequence::readWebcam(cv::VideoCapture cap) {

    if(cap.isOpened()) {
        frameNum = 0;
        timeStamp = 0;
        capture = cap;
        frameWidth = (int)capture.get(cv::CAP_PROP_FRAME_WIDTH);
        frameHeight = (int)capture.get(cv::CAP_PROP_FRAME_HEIGHT);
        isWebCam = true;
        isImageSeq = false;
        vidLength = 0;
        fps = capture.get(cv::CAP_PROP_FPS);

        // Check if fps is nan or less than 0
        if (fps != fps || fps <= 0)        {
            INFO_STREAM("FPS of the webcam cannot be determined, assuming 30");
            fps = 30;
        }

        std::string time = currentDateTime();
        name = "webcam_" + time;

        startTime = cv::getTickCount();
        isCapturing = true;

        return true;

    }else{

        return false;
    }
}

bool CaptureSequence::openVideoFile(std::string videoFile){

    INFO_STREAM("Attempting to read from file: " << videoFile);

    frameNum = 0;
    timeStamp = 0;
    latestFrame = cv::Mat();
    latestGrayFrame = cv::Mat();

    capture.open(videoFile);

    if(!capture.isOpened()) {
        ERROR_STREAM("Failed to open the video file at location: " << videoFile );
        return false;
    }

    fps = capture.get(cv::CAP_PROP_FPS);
    std::cout<< "fps:" << fps;

    // Check if fps is nan or less than 0
    if (fps != fps || fps <= 0){
        WARN_STREAM("FPS of the video file cannot be determined, assuming 30");
        fps = 30;
    }

    isWebCam = false;
    isImageSeq = false;

    frameWidth = (int)capture.get(cv::CAP_PROP_FRAME_WIDTH);
    frameHeight = (int)capture.get(cv::CAP_PROP_FRAME_HEIGHT);

    vidLength = (int)capture.get(cv::CAP_PROP_FRAME_COUNT);

    this->name = videoFile;

    isCapturing = true;

    //开启一个独立的线程获取视频中的视频帧
    captureThread = std::thread(&CaptureSequence::captureThreadFunc,this);

    return true;

}


//读取某个路径下的所有图片序列
bool CaptureSequence::openImageSequence(std::string directory) {

    INFO_STREAM("Attempting to read from directory: " << directory);

    frameNum = 0;
    timeStamp = 0;

    //先清除原先的图片容器
    imageFiles.clear();

    boost::filesystem::path imageDirectory(directory);

    if (!boost::filesystem::exists(imageDirectory)) {

        INFO_STREAM("Provided directory does not exist: " << directory);
        return false;
    }

    std::vector<boost::filesystem::path> fileInDirectory;

    std::copy(boost::filesystem::directory_iterator(imageDirectory), boost::filesystem::directory_iterator(), std::back_inserter(fileInDirectory));

    //对原先的图片文件进行排序
    std::sort(fileInDirectory.begin(), fileInDirectory.end());

    std::vector<std::string> currDirFiles;

    //遍历文件夹 查找所有图片后缀的图片文件
    for (std::vector<boost::filesystem::path>::const_iterator fileIterator(fileInDirectory.begin()); fileIterator != fileInDirectory.end(); ++fileIterator) {
        // Possible image extension .jpg and .png
        if (fileIterator->extension().string().compare(".jpg") == 0
                || fileIterator->extension().string().compare(".jpeg") == 0
                || fileIterator->extension().string().compare(".png") == 0
                || fileIterator->extension().string().compare(".bmp") == 0) {

            currDirFiles.push_back(fileIterator->string());
        }
    }

    //保存之后的结果
    imageFiles = currDirFiles;

    if(imageFiles.empty()) {

        ERROR_STREAM("No images found in the directory:" << directory);
        return false;
    }

    //预先读取一张图片来获得图片的大小，一般情况下需要保证所有的图片拥有相同的大小
    cv::Mat tmp = cv::imread(imageFiles[0], cv::IMREAD_COLOR);
    frameHeight = tmp.size().height;
    frameWidth = tmp.size().width;

    fps = 0;
    name = directory;

    isWebCam = false;
    isImageSeq = true;

    vidLength = imageFiles.size();
    isCapturing = true;

    captureThread = std::thread(&CaptureSequence::captureThreadFunc, this);

    return true;
}

//将原图转换为8bit的灰度图像
void ConvertToGrayscale_8bit(const cv::Mat& in, cv::Mat& out) {

    if (in.channels() == 3){
        // Make sure it's in a correct format
        if (in.depth() == CV_16U){
            cv::Mat tmp = in / 256;
            tmp.convertTo(out, CV_8U);
            cv::cvtColor(out, out, cv::COLOR_BGR2GRAY);
        }else{
            cv::cvtColor(in, out, cv::COLOR_BGR2GRAY);
        }
    }else if (in.channels() == 4){
        if (in.depth() == CV_16U){
            cv::Mat tmp = in / 256;
            tmp.convertTo(out, CV_8U);
            cv::cvtColor(out, out, cv::COLOR_BGRA2GRAY);
        }else{
            cv::cvtColor(in, out, cv::COLOR_BGRA2GRAY);
        }
    }else{
        if (in.depth() == CV_16U){
            cv::Mat tmp = in / 256;
            tmp.convertTo(out, CV_8U);
        }else if (in.depth() == CV_8U){
            out = in.clone();
        }
    }
}

//图片捕捉线程 用于视频和图片序列 摄像头不用这个函数
void CaptureSequence::captureThreadFunc() {
    //首先设置缓存的容量大小
    int capacity = (CAPTURE_CAPACITY * 1024 * 1024) / (4 * frameWidth * frameHeight);
    captureQueue.set_capacity(capacity);

    int frameNumInt = 0;

    while(isCapturing) {

        double timeStampCurr = 0;
        cv::Mat tmpFrame;
        cv::Mat_<uchar> tmpGrayFrame;

        if(!isImageSeq) {

            bool success = capture.read(tmpFrame);
            //读取失败 表明图像没了或者出错
            if(!success) {
                tmpFrame = cv::Mat();
                isCapturing = false;
            }

            timeStampCurr = frameNumInt * (1.0 / fps);
        }else if(isImageSeq) {

            if(imageFiles.empty() || frameNumInt >= (int)imageFiles.size()) {
                tmpFrame = cv::Mat();
                isCapturing = false;
            }else {
                tmpFrame = cv::imread(imageFiles[frameNumInt], cv::IMREAD_COLOR);
            }
            timeStampCurr = 0;
        }

        frameNumInt++;

        ConvertToGrayscale_8bit(tmpFrame, tmpGrayFrame);

        captureQueue.push(std::make_tuple(timeStampCurr, tmpFrame, tmpGrayFrame));

    }

}

cv::Mat CaptureSequence::getNextFrame() {

    if(!isWebCam) {
        std::tuple<double, cv::Mat, cv::Mat_<uchar> > data;
        data = captureQueue.pop();
        timeStamp = std::get<0>(data);
        latestFrame = std::get<1>(data);
        latestGrayFrame = std::get<2>(data);
    }else {
        bool success = capture.read(latestFrame);
        //记录时间
        timeStamp = (cv::getTickCount() - startTime) / cv::getTickFrequency();

        if(!success) {
            latestFrame = cv::Mat();
        }

        //ConvertToGrayscale_8bit(latestFrame, latestGrayFrame);
    }

    frameNum++;

    return latestFrame;
}

//获取当前的处理进度
double CaptureSequence::getProgress() {
    if(isWebcam()) {
        return -1.0;
    }else {
        return (double)frameNum / (double)vidLength;
    }
}

bool CaptureSequence::isOpened() {
    if(isWebcam() || !isImageSeq) {
        return capture.isOpened();
    }else {
        return (imageFiles.size() > 0 && frameNum < imageFiles.size());
    }
}

cv::Mat_<uchar> CaptureSequence::getGrayFrame() {
    return latestGrayFrame;
}
