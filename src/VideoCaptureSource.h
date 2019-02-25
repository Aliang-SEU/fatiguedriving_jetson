#ifndef VIDEOCAPTURESOURCE_H
#define VIDEOCAPTURESOURCE_H

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <boost/format.hpp>

//JetSon上打开摄像头需要注意使用GStream，否则会有较大的延迟
namespace VideoCaptureSource {

    //打开usb摄像头 默认的设备号为1
    cv::VideoCapture openUsbCam(int height=480, int width=640, int deviceId=1);

    //打开板载CSI摄像头
    cv::VideoCapture openOnboardCam(int height=480, int width=640);

}

#endif // VIDEOCAPTURESOURCE_H
