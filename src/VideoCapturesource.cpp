#include <VideoCaptureSource.h>


namespace VideoCaptureSource {

    /**
     * @brief openUsbCam
     * @param width
     * @param deviceId
     * @return
     */
    cv::VideoCapture openUsbCam(int height, int width, int deviceId){
        boost::format f = boost::format("v4l2src device=/dev/video%d ! video/x-raw, width=(int)%d, height=(int)%d, format=(string)RGB ! videoconvert ! appsink") % deviceId % width % height;
        return cv::VideoCapture(f.str(), cv::CAP_GSTREAMER);
    }

    cv::VideoCapture openOnboardCam(int height, int width) {
        boost::format f = boost::format("nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)2592, height=(int)1458, format=(string)I420, frmaerate=(fraction)30/1 ! nvvidconv ! "
                                    " video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! videoconvert ! appsink") % width % height;
        return cv::VideoCapture(f.str(), cv::CAP_GSTREAMER);
    }

}
