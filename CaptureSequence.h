#ifndef CAPTURESEQUENCE_H
#define CAPTURESEQUENCE_H

#include <fstream>
#include <sstream>
#include <vector>

#include <thread>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <ConcurrentQueue.h>

namespace Utilities {

    class CaptureSequence {

    public:
        CaptureSequence() {};

        ~CaptureSequence();

        bool readWebcam(cv::VideoCapture cap);

        // Webcam
        bool openWebcam(int deviceId, int imageWidth = 640, int imageHeight = 480, float fx = -1, float fy = -1, float cx = -1, float cy = -1);

        // Image sequence in the directory
        bool openImageSequence(std::string directory);

        // Video file
        bool openVideoFile(std::string videoFile);

        bool isWebcam() { return isWebCam; }

        // Getting the next frame
        cv::Mat getNextFrame();

        // Getting the most recent grayscale frame (need to call GetNextFrame first)
        cv::Mat_<uchar> getGrayFrame();

        // Parameters describing the sequence and it's progress
        double getProgress();

        size_t getFrameNumber() { return frameNum; }

        bool isOpened();

        void close();

        int frameWidth;
        int frameHeight;

        double fps;

        double timeStamp;

        // Name of the video file, image directory, or the webcam
        std::string name;

        //当前存储的图片数量
        static const int CAPTURE_CAPACITY = 200; // 200 MB

    private:
        //用来跟踪当前线程是否还在写入
        bool isCapturing;

        //图片捕捉线程
        std::thread captureThread;

        //图片捕捉函数
        void captureThreadFunc();

        // Blocking copy and move, as it doesn't make sense to have several readers pointed at the same source, and this would cause issues, especially with webcams
        CaptureSequence & operator= (const CaptureSequence& other);
        CaptureSequence & operator= (const CaptureSequence&& other);
        CaptureSequence(const CaptureSequence&& other);
        CaptureSequence(const CaptureSequence& other);

        //视频
        cv::VideoCapture capture;

        cv::Mat latestFrame;   //最新一帧的图像
        cv::Mat_<uchar> latestGrayFrame;   //最新一帧的灰度图像

        //线程安全队列，用来保证在并发情况下捕捉的图片有序
        ConcurrentQueue<std::tuple<double, cv::Mat, cv::Mat_<uchar>>> captureQueue;

        //捕捉到的帧数
        size_t frameNum;
        std::vector<std::string> imageFiles;

        //存储视频的长度以便观察处理进度
        size_t vidLength;

        //记录摄像头打开的时间
        int64 startTime;

        //标记处理的类型
        bool isWebCam;
        bool isImageSeq;

    };
}

#endif // CAPTURESEQUENCE_H
