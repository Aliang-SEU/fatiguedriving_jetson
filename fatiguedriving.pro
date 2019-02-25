QT += core sql
QT -= gui

TARGET = fatiguedriving
CONFIG += console
CONFIG -= app_bundle
QMAKE_CXX = g++
QMAKE_CXXFLAGS += -std=c++11 -fopenmp -o3
QMAKE_CXXFLAGS += -Wno-unused-parameter -Wno-unused-variable

TEMPLATE = app

INCLUDEPATH +=  /home/hzl/Documents/caffe-master/include \
                /usr/local/include \
                /usr/local/include/opencv \
                /usr/local/include/opencv2 \
                /usr/local/cuda/include \
                /usr/include/boost/ \
                src
                #/home/nvidia/Downloads/cvplot-0.0.3/include
                #/home/nvidia/Desktop/fatiguedriving/dsst

HEADERS += src/MtcnnDetector.h \
    src/CaffeLayerHeadear.h \
    src/EyeRecognition.h \
    src/Landmark.h \
    src/MouthRecognition.h \
    src/CTaskQueue.h \
    src/Message.h \
    src/VideoCaptureSource.h \
    src/ConcurrentQueue.h \
    src/CaptureSequence.h \
    src/TimeUtils.h \
    src/LandMarkWithPose.h \
    src/MtcnnOpencv.h \
    src/PoseEstimator.h \
    src/KalmanStabilizer.h \
    src/Processor.h \
    src/DriverDatabase.h


SOURCES += src/MtcnnDetector.cpp \
    src/EyeRecognition.cpp \
    src/Landmark.cpp \
    src/MouthRecognition.cpp \
    src/CaptureSequence.cpp \
    src/TimeUtils.cpp \
    src/LandMarkWithPose.cpp \
    src/MtcnnOpencv.cpp \
    src/PoseEstimator.cpp \
    src/KalmanStabilizer.cpp \
    src/Processor.cpp \
    src/VideoCapturesource.cpp \
#    makeExampleForEyeRec.cpp
#    DrivingSystem.cpp
    main.cpp \
    src/DriverDatabase.cpp
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
        -lopencv_tracking \
        -L/home/hzl/Documents/caffe-master/build/lib \
        -lcaffe \
        -lopenblas \
        -lglog \
        -lprotobuf \
        -lboost_system \
        -lboost_filesystem \
        -lgomp
