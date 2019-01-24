QT += core
QT -= gui

TARGET = fatiguedriving
CONFIG += console
CONFIG -= app_bundle
QMAKE_CXX = g++
QMAKE_CXXFLAGS += -std=c++11 -fopenmp -o3
TEMPLATE = app

INCLUDEPATH +=  /home/nvidia/Documents/MTCNN_Caffe/include \
                /home/nvidia/Documents/MTCNN_Caffe/build/src \
                /usr/local/include \
                /usr/local/include/opencv \
                /usr/local/include/opencv2 \
                /usr/local/cuda/include \
                /usr/include/boost/ \
                #/home/nvidia/Desktop/fatiguedriving/dsst
                #/home/nvidia/fatiguedriving/lib/local/LandmarkDetector/include \
                #/home/nvidia/fatiguedriving/lib/local/Utilities/include \

HEADERS += MtcnnDetector.h \
    CaffeLayerHeadear.h \
    EyeRecognition.h \
    Landmark.h \
    MouthRecognition.h \
    CTaskQueue.h \
    Message.h \
    VideoCaptureSource.h \
    ConcurrentQueue.h \
    CaptureSequence.h \
    TimeUtils.h \
#    lib/local/LandmarkDetector/include/CCNF_patch_expert.h \
#    lib/local/LandmarkDetector/include/CEN_patch_expert.h \
#    lib/local/LandmarkDetector/include/CNN_utils.h \
#    lib/local/LandmarkDetector/include/FaceDetectorMTCNN.h \
#    lib/local/LandmarkDetector/include/LandmarkCoreIncludes.h \
#    lib/local/LandmarkDetector/include/LandmarkDetectionValidator.h \
#    lib/local/LandmarkDetector/include/LandmarkDetectorFunc.h \
#    lib/local/LandmarkDetector/include/LandmarkDetectorModel.h \
#    lib/local/LandmarkDetector/include/LandmarkDetectorParameters.h \
#    lib/local/LandmarkDetector/include/LandmarkDetectorUtils.h \
#    lib/local/LandmarkDetector/include/Patch_experts.h \
#    lib/local/LandmarkDetector/include/PAW.h \
#    lib/local/LandmarkDetector/include/PDM.h \
#    lib/local/LandmarkDetector/include/stdafx.h \
#    lib/local/LandmarkDetector/include/SVR_patch_expert.h \
#    lib/local/Utilities/include/ConcurrentQueue.h \
#    lib/local/Utilities/include/ImageCapture.h \
#    lib/local/Utilities/include/ImageManipulationHelpers.h \
#    lib/local/Utilities/include/RecorderCSV.h \
#    lib/local/Utilities/include/RecorderHOG.h \
#    lib/local/Utilities/include/RecorderOpenFace.h \
#    lib/local/Utilities/include/RecorderOpenFaceParameters.h \
#    lib/local/Utilities/include/RotationHelpers.h \
#    lib/local/Utilities/include/SequenceCapture.h \
#    lib/local/Utilities/include/VisualizationUtils.h \
#    lib/local/Utilities/include/Visualizer.h \
    LandMarkWithPose.h \
    MtcnnOpencv.h \
    PoseEstimator.h \
    KalmanStabilizer.h \
    Processor.h


SOURCES += \
    main.cpp \
    MtcnnDetector.cpp \
    EyeRecognition.cpp \
    Landmark.cpp \
    MouthRecognition.cpp \
    CaptureSequence.cpp \
    TimeUtils.cpp \
#    lib/local/LandmarkDetector/src/CCNF_patch_expert.cpp \
#    lib/local/LandmarkDetector/src/CEN_patch_expert.cpp \
#    lib/local/LandmarkDetector/src/CNN_utils.cpp \
#    lib/local/LandmarkDetector/src/FaceDetectorMTCNN.cpp \
#    lib/local/LandmarkDetector/src/LandmarkDetectionValidator.cpp \
#    lib/local/LandmarkDetector/src/LandmarkDetectorFunc.cpp \
#    lib/local/LandmarkDetector/src/LandmarkDetectorModel.cpp \
#    lib/local/LandmarkDetector/src/LandmarkDetectorParameters.cpp \
#    lib/local/LandmarkDetector/src/LandmarkDetectorUtils.cpp \
#    lib/local/LandmarkDetector/src/Patch_experts.cpp \
#    lib/local/LandmarkDetector/src/PAW.cpp \
#    lib/local/LandmarkDetector/src/PDM.cpp \
#    lib/local/LandmarkDetector/src/stdafx.cpp \
#    lib/local/LandmarkDetector/src/SVR_patch_expert.cpp \
#    lib/local/Utilities/src/ImageCapture.cpp \
#    lib/local/Utilities/src/RecorderCSV.cpp \
#    lib/local/Utilities/src/RecorderHOG.cpp \
#    lib/local/Utilities/src/RecorderOpenFace.cpp \
#    lib/local/Utilities/src/RecorderOpenFaceParameters.cpp \
#    lib/local/Utilities/src/SequenceCapture.cpp \
#    lib/local/Utilities/src/VisualizationUtils.cpp \
#    lib/local/Utilities/src/Visualizer.cpp \
    LandMarkWithPose.cpp \
    MtcnnOpencv.cpp \
    PoseEstimator.cpp \
    KalmanStabilizer.cpp \
    Processor.cpp \
    VideoCapturesource.cpp
#    DrivingSystem.cpp

#  main1.cpp


LIBS += -L/usr/local/lib/ \
        -lopencv_highgui \
        -lopencv_core \
        -lopencv_imgcodecs \
        -lopencv_imgproc \
        -lopencv_dnn \
        -lopencv_videoio \
        -lopencv_objdetect \
        -lopencv_calib3d \
        -lopencv_video \
        -L/home/nvidia/Documents/MTCNN_Caffe/build/lib \
        -lopenblas \
        -lcaffe \
        -lglog \
        -lprotobuf \
        -lboost_system \
        -lboost_filesystem \
        -ldlib \
        -lgomp \
        -lopencv_tracking \
        -lopencv_viz
