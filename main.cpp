#include <Processor.h>
#include "DriverDatabase.h"

int main(int argc, char **argv){

    //关闭caffe自带的日志系统
    ::google::InitGoogleLogging(argv[0]);

    Processor processor;
    //processor.readVideoCapture(0);
    processor.openVideoFile("/home/hzl/Videos/5.mp4");

    while(processor.getNextFrame()){

        processor.detectFace();
        processor.detectLandmark();
        processor.estimatorPose();
        processor.detectEyeState();
        processor.detectMouthState();
        cv::Mat image = processor.getProcessedImage();
        cv::imshow("image", image);
        cv::waitKey(1);
    }
    return 1;
}

