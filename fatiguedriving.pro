QT += core
QT -= gui

TARGET = fatiguedriving
CONFIG += console
CONFIG -= app_bundle

QMAKE_CXXFLAGS += -std=c++0x -o3
TEMPLATE = app

INCLUDEPATH +=  /home/nvidia/Documents/MTCNN_Caffe/include \
                /home/nvidia/Documents/MTCNN_Caffe/build/src \
                /usr/local/include \
                /usr/local/include/opencv \
                /usr/local/include/opencv2 \
                /usr/local/cuda/include
HEADERS += \
    mouthdetector.h \
    MtcnnDetector.h \
    CaffeLayerHeadear.h \
    EyeRecognition.h \
    Landmark.h \
    Utils.h \
    MouthRecognition.h \
    CTaskQueue.h \
    Message.h

SOURCES += \
    main.cpp \
    MtcnnDetector.cpp \
    EyeRecognition.cpp \
    Landmark.cpp \
    Utils.cpp \
    MouthRecognition.cpp

LIBS += -L/usr/local/lib/ \
        -lopencv_highgui \
        -lopencv_core \
        -lopencv_imgcodecs \
        -lopencv_imgproc \
        -lopencv_dnn \
        -lopencv_videoio \
        -L/home/nvidia/Documents/MTCNN_Caffe/build/lib \
        -lcaffe \
        -lglog \
        -lprotobuf \
        -lboost_system
