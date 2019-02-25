#include <Processor.h>
#include <fstream>
std::string num2str(const char *s, long i){
        char ss[200];
        sprintf(ss, s, i);
        return ss;
}

//#define CAPTURE
int main(int argc, char **argv) {
    //关闭caffe自带的日志系统
#ifdef CAPTURE
    ::google::InitGoogleLogging(argv[0]);

    Processor processor;
\
    //processor.openVideoFile("/home/hzl/Videos/1.wmv");
    processor.readVideoCapture(0);
    long idx = 1656;

    while(processor.getNextFrame()){

        if(processor.detectFace()) {
          processor.detectLandmark();
          processor.detectEyeState();
          FrameInfo frameInfo = processor.getFrameInfo();
          cv::Mat leftEye = frameInfo.image(frameInfo.leftEyeBBox);
          cv::imshow("1", leftEye);
          cv::imwrite(num2str("/home/hzl/Videos/example/eye%08d.jpg", idx++), leftEye);
          cv::Mat rightEye = frameInfo.image(frameInfo.rightEyeBBox);
          cv::imshow("2", rightEye);
          cv::imwrite(num2str("/home/hzl/Videos/example/eye%08d.jpg", idx++), rightEye);

        }
        cv::waitKey(1);
    }
#else
    std::ofstream file("/home/hzl/Videos/eye.txt");

    int idx[] = {0, 276, 371, 379, 382, 433, 434, 460, \
                464, 524, 646, 690, 828, 908, 910, \
                930, 934, 950, 954, 1536, 1542, 1870, 2070};
    int len = sizeof(idx) / sizeof(idx[0]);

    int label = 1;

    for(int i = 0; i < len - 1; i++) {
        for(int j = idx[i]; j < idx[i + 1]; j++) {
            file << num2str("/home/hzl/Videos/example/eye%08d.jpg",j) << " " << label << std::endl;
        }
        label = 1 - label;
    }

    std::ifstream train("/home/hzl/Videos/eye.txt");
    std::string imageFile;

    std::ofstream out("/home/hzl/Videos/train.txt");
    long cnt = 0;
    while(!train.eof()) {
        train >> imageFile >> label;
        cv::Mat image = cv::imread(imageFile, cv::IMREAD_COLOR);
        cv::resize(image, image, cv::Size(32, 32));
        cv::imwrite(num2str("/home/hzl/Videos/train/eye%08d.jpg", cnt), image);
        out << num2str("eye%08d.jpg", cnt) << " " << label << std::endl;
        cnt++;
    }
    file.close();
    train.close();
    out.close();
#endif
    return 1;
}
