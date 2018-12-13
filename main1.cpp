#include <MtcnnDetector.h>
#include <Landmark.h>
#include <MouthRecognition.h>
#include <VideoCaptureSource.h>
#include <CaptureSequence.h>

//#include "LandmarkCoreIncludes.h"
//#include <SequenceCapture.h>
//#include <Visualizer.h>
//#include <VisualizationUtils.h>

#define INFO_STREAM( stream ) \
std::cout << stream << std::endl

#define WARN_STREAM( stream ) \
std::cout << "Warning: " << stream << std::endl

#define ERROR_STREAM( stream ) \
std::cout << "Error: " << stream << std::endl


int main(int argc, char **argv)
{
    ::google::InitGoogleLogging(argv[0]);

    MtcnnDetector detector("../model");

    Utilities::CaptureSequence captureSequence;

    //captureSequence.readWebcam(VideoCaptureSource::openUsbCam());
    captureSequence.openVideoFile("/home/nvidia/Desktop/5.mp4");

    LandmarkDetector::FaceModelParameters det_parameters;
    //det_parameters.model_location = "/home/nvidia/fatiguedriving/lib/local/LandmarkDetector/model/main_ceclm_general.txt";
    LandmarkDetector::CLNF face_model(det_parameters.model_location);

    cv::Mat image;
    MouthRecognition mouthDetector;
    Landmark landmark;
    // Open a sequence
    Utilities::SequenceCapture sequence_reader;

    // A utility for visualizing the results (show just the tracks)
    Utilities::Visualizer visualizer(true, false, false, false);

    // Tracking :: for visualization
    Utilities::FpsTracker fps_tracker;
    fps_tracker.AddFrame();

    //dlib::frontal_face_detector face_detector_hog = dlib::get_frontal_face_detector();
     while(true){
           cap>>image;
           cv::resize(image, image, cv::Size(640,480));
//            cv::cvtColor(image, image, CV_BGR2RGB);
//            cv::Mat grayscale_image;
//            cv::cvtColor(image, grayscale_image, CV_RGB2GRAY);

            double confidence;
            //cv::Rect_<double> bounding_box;
           // bool face_detection_success = LandmarkDetector::DetectSingleFaceHOG(bounding_box, grayscale_image, face_model.face_detector_HOG, confidence);

            std::vector<FaceInfo> faceInfo = detector.Detect(image);
//            std::cout << " time:" << (double) (cv::getTickCount() - t) / cv::getTickFrequency() << "s"
//                      << std::endl;
            for(size_t i = 0; i < faceInfo.size(); i++) {
                int x = (int) faceInfo[0].bbox.xmin;
                int y = (int) faceInfo[0].bbox.ymin;
                int w = (int) (faceInfo[0].bbox.xmax - faceInfo[0].bbox.xmin + 1);
                int h = (int) (faceInfo[0].bbox.ymax - faceInfo[0].bbox.ymin + 1);
                cv::Rect_<double> bounding_box(x, y, w, h);
                double t = (double) cv::getTickCount();
                //bool detection_success = LandmarkDetector::DetectLandmarksInImage(image, bounding_box, face_model, det_parameters, grayscale_image);
                //bool detection_success = LandmarkDetector::DetectLandmarksInVideo(image, face_model, det_parameters, grayscale_image);
//                std::cout << " time:" << (double) (cv::getTickCount() - t) / cv::getTickFrequency() << "s"
//                          << std::endl;
//                if(detection_success) {
//                    for(size_t k = 0;k < 68;k++){
//                       cv::Point point = cv::Point((int)face_model.detected_landmarks.at<uchar>(k,0), (int)face_model.detected_landmarks.at<uchar>(k+68,0));
//                       cv::circle(image, point, 0.1, cv::Scalar(255, 0, 0), 2, 2, 0);
//                    }

//                }
                cv::Mat srcROI(image, cv::Rect(x, y, w, h));
                LandmarkPoints res = landmark.detectLandmark(srcROI);
                for(size_t k = 0;k < res.faceLandmark.size();k++){
                    cv::Point point = cv::Point(int(res.faceLandmark[k].x) + x,int(res.faceLandmark[k].y) + y);
                    cv::circle(image, point, 0.1, cv::Scalar(255, 0, 0), 2, 2, 0);
                }
                mouthDetector.calculate(res.innerMouth);
            }

//            std::cout << " time:" << (double) (cv::getTickCount() - t) / cv::getTickFrequency() << "s"
//                      << std::endl;

            detector.drawResult(faceInfo, image);
            cv::imshow("image", image);
            cv::waitKey(1);
    }
    return 1;
}

