#include <MtcnnDetector.h>
#include <Landmark.h>
#include <MouthRecognition.h>
#include <VideoCaptureSource.h>
#include <CaptureSequence.h>

//#include "LandmarkCoreIncludes.h"
//#include <SequenceCapture.h>
//#include <Visualizer.h>
//#include <VisualizationUtils.h>

#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/opencv.h>
#include <LandMarkWithPose.h>

#define INFO_STREAM( stream ) \
std::cout << stream << std::endl

#define WARN_STREAM( stream ) \
std::cout << "Warning: " << stream << std::endl

#define ERROR_STREAM( stream ) \
std::cout << "Error: " << stream << std::endl


cv::Rect correctBBox(cv::Mat& image, std::vector<FaceInfo>& faceInfo) {

    int h = (int) (faceInfo[0].bbox.ymax - faceInfo[0].bbox.ymin + 1);
    int w = (int) (faceInfo[0].bbox.xmax - faceInfo[0].bbox.xmin + 1);
    int x = (int) faceInfo[0].bbox.xmin  + (w - h) / 2;
    w = h;
    int y = (int) faceInfo[0].bbox.ymin;

    cv::Rect bbox(x, y, w, h);

    return bbox;
}


int main(int argc, char **argv){

    ::google::InitGoogleLogging(argv[0]);

    MtcnnDetector MTCNNdetector("../model");

    Utilities::CaptureSequence captureSequence;

    //captureSequence.readWebcam(VideoCaptureSource::openUsbCam());
    captureSequence.openVideoFile("/home/nvidia/Videos/4.mp4");

    cv::Mat image;

    Landmark landmark;
    LandmarkWithPose landmarkWithPose;
//    MouthRecognition mouthDetector;

    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
    while(true){

        image = captureSequence.getNextFrame();

        // Now tell the face detector to give us a list of bounding boxes
        // around all the faces in the image.
        image = cv::imread("/home/nvidia/Documents/face-landmark/result/7.jpg");
        std::vector<FaceInfo> faceInfo = MTCNNdetector.Detect(image);

        cv::Rect bbox = correctBBox(image, faceInfo);

        cv::Mat srcROI(image, bbox);
        double t = (double) cv::getTickCount();

        LandmarkPoints res = landmark.detectLandmark(srcROI);
        std::cout << " time:" << (double) (cv::getTickCount() - t) * 1000/ cv::getTickFrequency() << "ms"
                  << std::endl;

        t = (double) cv::getTickCount();
        LandmarkAndPose res1 = landmarkWithPose.getPredict(srcROI);
        std::cout << " time:" << (double) (cv::getTickCount() - t) * 1000 / cv::getTickFrequency() << "ms"
                  << std::endl;

        for(size_t k = 0;k < res.faceLandmark.size();k++){
            cv::Point point = cv::Point(int(res.faceLandmark[k].x) + bbox.x, int(res.faceLandmark[k].y) + bbox.y);
            cv::circle(image, point, 0.1, cv::Scalar(255, 0, 0), 2, 2, 0);
        }

        for(size_t k = 0; k < res1.landmark.size(); k += 2){
            cv::Point point = cv::Point(int(res1.landmark[k] * (bbox.height / 2) + bbox.width / 2) + bbox.x, int(res1.landmark[k + 1] * (bbox.height / 2) + bbox.width / 2) + bbox.y);
            cv::circle(image, point, 0.1, cv::Scalar(0, 255, 0), 2, 2, 0);
        }
//        for(size_t k = 0;k < res1.faceLandmark.size();k++){
//            cv::Point point = cv::Point(int(res1.faceLandmark[k].x) + tmp.left(),int(res1.faceLandmark[k].y) + tmp.top());
//            cv::circle(image, point, 0.1, cv::Scalar(0, 255, 0), 2, 2, 0);
//        }

        //mouthDetector.calculate(res.innerMouth);

        //cv::rectangle(image, Point(tmp.left(), tmp.top()), Point(tmp.right(), tmp.bottom()), Scalar(0,0,255), 1,4,0);
        //cv::rectangle(image, cv::Rect(x, y, w, h), cv::Scalar(0,0,255), 1, 4, 0);
        MTCNNdetector.drawResult(faceInfo, image);
        cv::imshow("image", image);

        cv::waitKey(1);
    }
    return 1;
}

