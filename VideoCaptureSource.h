#ifndef VIDEOCAPTURESOURCE_H
#define VIDEOCAPTURESOURCE_H

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <boost/format.hpp>

//JetSon上打开摄像头需要注意使用GStream，否则会有较大的延迟
namespace VideoCaptureSource {

    //打开usb摄像头 默认的设备号为1
    cv::VideoCapture openUsbCam(int height=480, int width=640, int deviceId=1 ){
        boost::format f = boost::format("v4l2src device=/dev/video%d ! video/x-raw, width=(int)%d, height=(int)%d, format=(string)RGB ! videoconvert ! appsink") % deviceId % width % height;
        return cv::VideoCapture(f.str(), cv::CAP_GSTREAMER);
    }

    //打开板载CSI摄像头
    cv::VideoCapture openOnboardCam(int height=480, int width=640) {
        boost::format f = boost::format("nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)2592, height=(int)1458, format=(string)I420, frmaerate=(fraction)30/1 ! nvvidconv ! "
                                    " video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! videoconvert ! appsink") % width % height;
        return cv::VideoCapture(f.str(), cv::CAP_GSTREAMER);
    }

}

#endif // VIDEOCAPTURESOURCE_H
