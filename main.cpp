
#include <MtcnnDetector.h>
#include <Landmark.h>
#include <LandMarkWithPose.h>
#include <MouthRecognition.h>
#include <VideoCaptureSource.h>
#include <CaptureSequence.h>

#include <opencv2/viz.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>

/*
 * 矫正人脸检测框 以便获得更好的效果
 */
cv::Rect refineBBox(cv::Mat& image, std::vector<FaceInfo>& faceInfo) {

    int h = (int) (faceInfo[0].bbox.ymax - faceInfo[0].bbox.ymin + 1);
    int w = (int) (faceInfo[0].bbox.xmax - faceInfo[0].bbox.xmin + 1);
    int x = (int) faceInfo[0].bbox.xmin  + (w - h) / 2;
    if(x <= 0) x = 0;
    w = h;
    int y = (int) faceInfo[0].bbox.ymin + h * 0.2;

    if(y + h > image.rows)
        y = image.rows - h;

    cv::Rect bbox(x, y, w, h);

    return bbox;
}

cv::Rect backScale(cv::Rect inputRect, int Scale) {
    return cv::Rect(inputRect.x * Scale, inputRect.y * Scale, inputRect.width * Scale, inputRect.height * Scale);
}

int main(int argc, char **argv){

    ::google::InitGoogleLogging(argv[0]);

    //视频处理
    Utilities::CaptureSequence captureSequence;
    //captureSequence.readWebcam(VideoCaptureSource::openUsbCam());
    captureSequence.openVideoFile("/home/nvidia/Videos/5.mp4");

    //人脸检测
    MtcnnDetector MTCNNdetector("../model");

    //目标跟踪
    cv::Ptr<cv::Tracker> tracker = cv::TrackerMedianFlow::create();

    //特征点标定
    LandmarkWithPose landmarkWithPose;
    Landmark landmark;
    //第一帧做初始化
    cv::Mat image, resizedImage;
    image = captureSequence.getNextFrame();

    int resizedScale = 3;
    cv::Size resizedSize(image.cols / resizedScale, image.rows / resizedScale);
    cv::resize(image, resizedImage, resizedSize);

    std::vector<FaceInfo> faceInfo = MTCNNdetector.Detect(resizedImage);

    cv::Rect2d resizedBBox = refineBBox(resizedImage, faceInfo);

    cv::Rect2d bbox = backScale(resizedBBox, resizedScale);

    cv::viz::Viz3d window("mywindow");
    window.showWidget("image", viz::WCoordinateSystem());

    //tracker->init(resizedImage, resizedBBox);

    while(true){


        image = captureSequence.getNextFrame();
        cv::resize(image, resizedImage, resizedSize);

        bool trackFlag = false; //tracker->update(resizedImage, resizedBBox);

        if(trackFlag == true) {

            bbox = backScale(resizedBBox, resizedScale);
        }else {
            std::cout<< "本次使用检测算法" << std::endl;
            faceInfo = MTCNNdetector.Detect(resizedImage);
            resizedBBox = refineBBox(resizedImage, faceInfo);
            bbox = backScale(resizedBBox, resizedScale);
            //tracker->init(resizedImage, resizedBBox);
        }

        double t = (double) cv::getTickCount();
        cv::Mat faceRegion(image, bbox);

        LandmarkAndPose res = landmarkWithPose.getPredict(faceRegion);

        std::cout << " time:" << (double) (cv::getTickCount() - t) * 1000 / cv::getTickFrequency() << "ms"
                  << std::endl;

        for(size_t k = 0; k < res.landmark.size(); k += 2){
            cv::Point point = cv::Point(int(res.landmark[k] * (bbox.height / 2) + bbox.width / 2) + bbox.x, int(res.landmark[k + 1] * (bbox.height / 2) + bbox.width / 2) + bbox.y);
            cv::circle(image, point, 0.1, cv::Scalar(0, 255, 0), 2, 2, 0);
        }

        cv::Mat rvec = cv::Mat::zeros(1, 3, CV_32F);
        rvec.at<float>(0, 0) = res.pose[0] / 180 * CV_PI;
        rvec.at<float>(0, 1) = res.pose[1] / 180 * CV_PI;
        rvec.at<float>(0, 2) = res.pose[2] / 180 * CV_PI;

        cv::Mat rmat;
        cv::Rodrigues(rvec, rmat);
        std::cout<< res.pose[0] << " " << res.pose[1] << " " << res.pose[2] << std::endl;
        Affine3f pose(rvec, Vec3f(0, 0, 0)); //这一句就是整个可视化窗口能够动起来的核心语句了，
        //说白了就是利用循环来不断调整上面plane部件的位姿，达到动画的效果
        //另外这里就利用到了平面的ID，来表征调整的是平面的位姿
        window.setWidgetPose("image", pose); //控制单帧暂留时间，调整time参数的效果就是平面转的快慢，本质上就是每一个位姿的停留显示时间。
        window.spinOnce(1, false);

        //LandmarkPoints res1 = landmark.detectLandmark(faceRegion);


        //mouthDetector.calculate(res.innerMouth);
        rectangle(image, bbox, Scalar( 255, 0, 0 ), 2, 1 );
        cv::imshow("image", image);
        cv::waitKey(1);
    }
    return 1;
}

