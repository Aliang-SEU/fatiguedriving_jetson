
#include <MtcnnDetector.h>
#include <Landmark.h>
#include <MouthRecognition.h>
#include <VideoCaptureSource.h>
#include <CaptureSequence.h>

//#include "LandmarkCoreIncludes.h"
//#include <SequenceCapture.h>
//#include <Visualizer.h>
//#include <VisualizationUtils.h>

#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>

#include <LandMarkWithPose.h>

cv::Rect correctBBox(cv::Mat& image, std::vector<FaceInfo>& faceInfo) {

    int h = (int) (faceInfo[0].bbox.ymax - faceInfo[0].bbox.ymin + 1);
    int w = (int) (faceInfo[0].bbox.xmax - faceInfo[0].bbox.xmin + 1);
    int x = (int) faceInfo[0].bbox.xmin  + (w - h) / 2;
    w = h;
    int y = (int) faceInfo[0].bbox.ymin + 20;

    cv::Rect bbox(x, y, w, h);

    return bbox;
}

int main(int argc, char **argv){

    ::google::InitGoogleLogging(argv[0]);

    MtcnnDetector MTCNNdetector("../model");

    Utilities::CaptureSequence captureSequence;

    //captureSequence.readWebcam(VideoCaptureSource::openUsbCam());
    captureSequence.openVideoFile("/home/nvidia/Videos/3.mp4");

    cv::Mat image;

    LandmarkWithPose landmarkWithPose;

    image = captureSequence.getNextFrame();

    // Now tell the face detector to give us a list of bounding boxes
    // around all the faces in the image.
    cv::Mat resizedImage;

    std::vector<FaceInfo> faceInfo = MTCNNdetector.Detect(image);

    cv::Rect2d bbox = correctBBox(image, faceInfo);

//    MouthRecognition mouthDetector;
    cv::Ptr<cv::Tracker> tracker = cv::TrackerMedianFlow::create();
    tracker->init(image, bbox);

    while(true){
        double t = (double) cv::getTickCount();
        image = captureSequence.getNextFrame();

        std::cout<< "是否跟踪到" << tracker->update(image, bbox) << std::endl;

        std::cout << " time:" << (double) (cv::getTickCount() - t) * 1000 / cv::getTickFrequency() << "ms"
                  << std::endl;
        cv::Mat srcROI(image, bbox);

        LandmarkAndPose res = landmarkWithPose.getPredict(srcROI);


//        for(size_t k = 0;k < res.faceLandmark.size();k++){
//            cv::Point point = cv::Point(int(res.faceLandmark[k].x) + bbox.x, int(res.faceLandmark[k].y) + bbox.y);
//            cv::circle(image, point, 0.1, cv::Scalar(255, 0, 0), 2, 2, 0);
//        }

        for(size_t k = 0; k < res.landmark.size(); k += 2){
            cv::Point point = cv::Point(int(res.landmark[k] * (bbox.height / 2) + bbox.width / 2) + bbox.x, int(res.landmark[k + 1] * (bbox.height / 2) + bbox.width / 2) + bbox.y);
            cv::circle(image, point, 0.1, cv::Scalar(0, 255, 0), 2, 2, 0);
        }


        //mouthDetector.calculate(res.innerMouth);

        //cv::rectangle(image, Point(tmp.left(), tmp.top()), Point(tmp.right(), tmp.bottom()), Scalar(0,0,255), 1,4,0);
        //cv::rectangle(image, cv::Rect(x, y, w, h), cv::Scalar(0,0,255), 1, 4, 0);
        //MTCNNdetector.drawResult(faceInfo, image);
        rectangle(image, bbox, Scalar( 255, 0, 0 ), 2, 1 );
        cv::imshow("image", image);

        cv::waitKey(1);
    }
    return 1;
}

