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

//        dlib::array2d<dlib::rgb_pixel> img;

//        dlib::assign_image(img, dlib::cv_image<dlib::bgr_pixel>(image));

//        //dlib::pyramid_up(img);
//        double t = (double) cv::getTickCount();
//        std::vector<dlib::rectangle> dets = detector(img);
//        std::cout << " time:" << (double) (cv::getTickCount() - t) / cv::getTsickFrequency() << "ms"
//                  << std::endl;

//        std::cout<< dets.size() << std::endl;

//        dlib::rectangle tmp = dets[0];
//        //cv::rectangle(image, Point(tmp.y1, tmp.x1), Point(tmp.y2, tmp.x2), Scalar(0,0,255), 1,4,0);
//        //Mat srcROI(image, Rect(tmp.y1,tmp.x1,tmp.y2 - tmp.y1,tmp.x2-tmp.x1));


//        std::cout << tmp.left() << " " << tmp.top() <<" "<< tmp.left() << " " << tmp.right() << std::endl;
//        Mat srcROI1(image, Rect(tmp.left(),tmp.top(),tmp.right()-tmp.left(),tmp.bottom() - tmp.top()));
//        LandmarkPoints res1 = landmark.detectLandmark(srcROI1);

        // Now tell the face detector to give us a list of bounding boxes
        // around all the faces in the image.


        std::vector<FaceInfo> faceInfo = MTCNNdetector.Detect(image);
//            std::cout << " time:" << (double) (cv::getTickCount() - t) / cv::getTickFrequency() << "s"
//                      << std::endl;

        int h = (int) (faceInfo[0].bbox.ymax - faceInfo[0].bbox.ymin + 1);
        int w = (int) (faceInfo[0].bbox.xmax - faceInfo[0].bbox.xmin + 1);
        int x = (int) faceInfo[0].bbox.xmin  + (w - h) / 2;
        w = h;
        int y = (int) faceInfo[0].bbox.ymin;

        cv::Rect_<double> bounding_box(x, y, w, h);

        cv::Mat srcROI(image, bounding_box);
        double t = (double) cv::getTickCount();

        LandmarkPoints res = landmark.detectLandmark(srcROI);
        std::cout << " time:" << (double) (cv::getTickCount() - t) * 1000/ cv::getTickFrequency() << "ms"
                  << std::endl;

        t = (double) cv::getTickCount();
        LandmarkAndPose res1 = landmarkWithPose.getPredict(srcROI);
        std::cout << " time:" << (double) (cv::getTickCount() - t) * 1000 / cv::getTickFrequency() << "ms"
                  << std::endl;

        for(size_t k = 0;k < res.faceLandmark.size();k++){
            cv::Point point = cv::Point(int(res.faceLandmark[k].x) + x,int(res.faceLandmark[k].y) + y);
            cv::circle(image, point, 0.1, cv::Scalar(255, 0, 0), 2, 2, 0);
        }

        for(size_t k = 0; k < res1.landmark.size(); k += 2){
            cv::Point point = cv::Point(int(res1.landmark[k] * (h / 2) + w / 2) + x,int(res1.landmark[k + 1] * (h / 2) + w / 2) + y);
            cv::circle(image, point, 0.1, cv::Scalar(0, 255, 0), 2, 2, 0);
        }
//        for(size_t k = 0;k < res1.faceLandmark.size();k++){
//            cv::Point point = cv::Point(int(res1.faceLandmark[k].x) + tmp.left(),int(res1.faceLandmark[k].y) + tmp.top());
//            cv::circle(image, point, 0.1, cv::Scalar(0, 255, 0), 2, 2, 0);
//        }

        //mouthDetector.calculate(res.innerMouth);

        //cv::rectangle(image, Point(tmp.left(), tmp.top()), Point(tmp.right(), tmp.bottom()), Scalar(0,0,255), 1,4,0);
        cv::rectangle(image, cv::Rect(x, y, w, h), cv::Scalar(0,0,255), 1, 4, 0);
        cv::imshow("image", image);
        cv::waitKey(1);
    }
    return 1;
}

