#include <MtcnnDetector.h>
#include <Landmark.h>

cv::VideoCapture openUsbCam() {
    return cv::VideoCapture("v4l2src device=/dev/video1 ! video/x-raw, width=(int)640, height=(int)480, format=(string)RGB ! videoconvert ! appsink", cv::CAP_GSTREAMER);
}
cv::VideoCapture openOnboardCam() {
    return cv::VideoCapture("nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)2592, height=(int)1458, format=(string)I420, frmaerate=(fraction)30/1 ! nvvidconv ! "
                            " video/x-raw, width=(int)640, height=(int)480, format=(string)BGRx ! videoconvert ! appsink", cv::CAP_GSTREAMER);
}

int main(int argc, char **argv)
{
    ::google::InitGoogleLogging(argv[0]);
    MtcnnDetector detector("../model");

    cv::VideoCapture cap = openUsbCam();
    cv::Mat image;

    Landmark landmark;
     while(true){
           cap>>image;

            double t = (double) cv::getTickCount();
            std::vector<FaceInfo> faceInfo = detector.Detect(image);
            std::cout << " time:" << (double) (cv::getTickCount() - t) / cv::getTickFrequency() << "s"
                      << std::endl;
            for(size_t i = 0; i < faceInfo.size(); i++) {
                int x = (int) faceInfo[i].bbox.xmin;
                int y = (int) faceInfo[i].bbox.ymin;
                int w = (int) (faceInfo[i].bbox.xmax - faceInfo[i].bbox.xmin + 1);
                int h = (int) (faceInfo[i].bbox.ymax - faceInfo[i].bbox.ymin + 1);
                cv::Mat srcROI(image, cv::Rect(x, y, w, h));
                std::vector<float> res = landmark.detectLandmark(srcROI);
                size_t feat_dim = res.size();
                for(size_t k = 0;k < feat_dim/2;k++){
                    cv::Point point = cv::Point(int(res[2*k]*w) + x,int(res[2*k + 1]*h) + y);
                    cv::circle(image, point, 0.1, cv::Scalar(255, 0, 0), 2, 2, 0);
                }
            }
            std::cout << " time:" << (double) (cv::getTickCount() - t) / cv::getTickFrequency() << "s"
                      << std::endl;

            detector.drawResult(faceInfo, image);

            cv::imshow("image", image);
            cv::waitKey(1);
    }
    return 1;
}

