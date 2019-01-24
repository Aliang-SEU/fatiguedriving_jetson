#include <Processor.h>


int main(int argc, char **argv){

    //关闭caffe自带的日志系统
    //::google::InitGoogleLogging(argv[0]);

    Processor processor;

    //processor.openWebcam();
    processor.openVideoFile("/home/nvidia/Videos/5.mp4");

    while(processor.getNextFrame()){

        processor.detectFace();
        processor.detectLandmark();
        processor.estimatorPose();
        processor.detectEyeState();
        cv::Mat image = processor.getProcessedImage();

        cv::imshow("image", image);
        cv::waitKey(1);
    }
    return 1;
}

