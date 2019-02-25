#include <Processor.h>
#include <algorithm>

/** 矫正人脸检测框 以便获得更好的效果
 * @brief Processor::refineBBox
 * @param image
 * @param faceInfo
 * @return
 */
cv::Rect Processor::refineBBox(cv::Mat& image, std::vector<FaceInfo>& faceInfo) {

    int h = (int) (faceInfo[0].bbox.ymax - faceInfo[0].bbox.ymin + 1);
    int w = (int) (faceInfo[0].bbox.xmax - faceInfo[0].bbox.xmin + 1);
    int x = (int) faceInfo[0].bbox.xmin  + (w - h) / 2;
    if(x <= 0) x = 0;
    w = h;
    int y = (int) faceInfo[0].bbox.ymin + h * 0.1;

    if(y + h > image.rows)
        y = image.rows - h;

    cv::Rect bbox(x, y, w, h);

    return bbox;
}

/** 恢复检测框
 * @brief backScale
 * @param inputRect
 * @param Scale
 * @return
 */
cv::Rect Processor::backScale(cv::Rect inputRect, int Scale) {
    return cv::Rect(inputRect.x * Scale, inputRect.y * Scale, inputRect.width * Scale, inputRect.height * Scale);
}

/** 构造函数 加载相应的网络模型参数
 * @brief Processor::Processor
 */
Processor::Processor() {
   captureSequence = std::make_shared<Utilities::CaptureSequence>();
   faceDetector = std::make_shared<MtcnnDetector>("../model");
   landmarkWithPose = std::make_shared<LandmarkWithPose>();
   for(int i = 0; i < 3; i++) {
       poseStabilizers.push_back(std::make_shared<KalmanStabilizer>(2, 1, 0.1f, 0.1f));
   }
   for(int i = 0; i < 6; i++) {
       poseEstimatorStabilizers.push_back(std::make_shared<KalmanStabilizer>(2, 1, 0.001f, 0.1f));
   }
}

/**
 * @brief Processor::openVideoFile
 * @param videoFile
 * @return
 */
bool Processor::openVideoFile(const std::string videoFile) {
    return this->captureSequence->openVideoFile(videoFile);
}

/**
 * @brief Processor::readVideoCapture
 * @param deviceId
 * @return
 */
bool Processor::readVideoCapture(int deviceId){
    cv::VideoCapture capture(deviceId);
    return this->captureSequence->readWebcam(capture);
}

/**
 * @brief Processor::openWebcam
 * @param height
 * @param width
 * @param deviceId
 * @return
 */
bool Processor::openWebcam(int height, int width, int deviceId) {
    return this->captureSequence->readWebcam(VideoCaptureSource::openUsbCam(height, width, deviceId));
}

/**
 * @brief Processor::getNextFrame
 * @return
 */
bool Processor::getNextFrame(){
    if(this->captureSequence->isOpened()) {
        cv::Mat image = captureSequence->getNextFrame();

        if(!image.data)
            return false;

        cv::Mat resizedImage;

        if(!poseEstimator) //检测当前的姿态估计器是否已经初始化
            poseEstimator = std::make_shared<PoseEstimator>(image.size());
        if(!eyeRecognition) //检测眼睛检测器是否已经初始化
            eyeRecognition = std::make_shared<EyeRecognition>(captureSequence->fps);
        if(!mouthRecognition) //检测嘴巴检测器是否已经初始化
            mouthRecognition = std::make_shared<MouthRecognition>(captureSequence->fps);

        frameInfo = FrameInfo();
        //保存当前帧
        frameInfo.image = image;
        frameInfo.processedImage = image.clone();
        if(resizedSize.empty())
            resizedSize = cv::Size(image.cols / resizedScale, image.rows / resizedScale);

        cv::resize(image, resizedImage, resizedSize);
        frameInfo.resizedImage = resizedImage;

        return true;
    }else {
        std::cerr<< "输入设备打开错误!";
        return false;
    }
}

/**
 * @brief Processor::detectFace
 * @return
 */
bool Processor::detectFace(){
    cv::Rect2f resizedBBox,bbox;

    if(!frameInfo.resizedImage.empty()) {
        frameInfo.faceInfo = faceDetector->Detect(frameInfo.resizedImage);
        if(frameInfo.faceInfo.size() == 0) {
            return false;
        }

        int h = (int) ((frameInfo.faceInfo[0].bbox.ymax - frameInfo.faceInfo[0].bbox.ymin + 1) * resizedScale);
        int w = (int) ((frameInfo.faceInfo[0].bbox.xmax - frameInfo.faceInfo[0].bbox.xmin + 1) * resizedScale);
        int x = (int) (frameInfo.faceInfo[0].bbox.xmin * resizedScale);
        int y = (int) (frameInfo.faceInfo[0].bbox.ymin * resizedScale);
        frameInfo.originFacebbox = cv::Rect(x, y, w, h);

        resizedBBox = refineBBox(frameInfo.resizedImage, frameInfo.faceInfo);
        frameInfo.facebbox = backScale(resizedBBox, resizedScale);

        for(int i = 0; i < 5; i++) {
            int x = int(frameInfo.faceInfo[0].landmark[2 * i] * resizedScale);
            int y = int(frameInfo.faceInfo[0].landmark[2 * i + 1] *  resizedScale);
            cv::Point point(x, y);
            frameInfo.facePoints.push_back(point);
        }

        return true;
    }else
        return false;
}

/**
 * @brief Processor::detectLandmark
 * @return
 */
bool Processor::detectLandmark(){
    if(!frameInfo.facebbox.empty()){
        cv::Mat faceRegion(frameInfo.image, frameInfo.facebbox);
        std::vector<cv::Point2f> imagePoints;
        LandmarkAndPose res = landmarkWithPose->getPredict(faceRegion);

        for(size_t k = 0; k < res.landmark.size(); k += 2){
            cv::Point point = cv::Point(int(res.landmark[k] * (frameInfo.facebbox.height / 2) + frameInfo.facebbox.width / 2) + frameInfo.facebbox.x, int(res.landmark[k + 1] * (frameInfo.facebbox.height / 2) + frameInfo.facebbox.width / 2) + frameInfo.facebbox.y);
            imagePoints.push_back(point);
        }
        for(int i = 0; i < 3; i++) {
            poseStabilizers[i]->update(res.pose[i]);
            res.pose[i] = poseStabilizers[i]->getState();
        }
        frameInfo.landmarkAndPose = res;


        frameInfo.landmarkPoints = imagePoints;
        return true;
    }else{
        return false;
    }
}

/**
 * @brief Processor::estimatorPose
 * @return
 */
bool Processor::estimatorPose(){
    if(!frameInfo.landmarkPoints.empty()){

        std::vector<cv::Mat_<float>> vec = poseEstimator->solvePoseBy68Points(frameInfo.landmarkPoints);
        for(int i = 0; i < 3; i++) {
            poseEstimatorStabilizers[i]->update(vec[0].at<float>(i));
            vec[0].at<float>(i) = poseEstimatorStabilizers[i]->getState();
        }
        for(int i = 3; i < 6; i++) {
            poseEstimatorStabilizers[i]->update(vec[1].at<float>(i - 3));
            vec[1].at<float>(i - 3) = poseEstimatorStabilizers[i]->getState();
        }
        frameInfo.vec = vec;
        return true;
    }else{
        return false;
    }
}

/**
 * @brief Processor::detectEyeState
 * @return
 */
bool Processor::detectEyeState(){
    if(!frameInfo.facePoints.empty() && !frameInfo.landmarkPoints.empty()) {

        //leftEye
        float width = (frameInfo.landmarkPoints[39].x - frameInfo.landmarkPoints[36].x);
        float height = (frameInfo.landmarkPoints[41].y - frameInfo.landmarkPoints[19].y);
        frameInfo.leftEyePoint = (frameInfo.landmarkPoints[36] + frameInfo.landmarkPoints[39]) / 2;
        cv::Rect2f leftEyeBBox(std::max(frameInfo.leftEyePoint.x - width, 0.0f), std::max(frameInfo.leftEyePoint.y - height / 2, 0.0f), 2 * width, height);
        frameInfo.leftEyeBBox = leftEyeBBox; //record bbox

        cv::Mat leftEye = frameInfo.image(frameInfo.leftEyeBBox);   //left eye image

        frameInfo.leftEye = (eyeRecognition->predict(leftEye) == true) ? CLOSED : OPEN;

        //rightEye
        width = (frameInfo.landmarkPoints[45].x - frameInfo.landmarkPoints[42].x);
        height = (frameInfo.landmarkPoints[47].y - frameInfo.landmarkPoints[24].y);
        frameInfo.rightEyePoint = (frameInfo.landmarkPoints[45] + frameInfo.landmarkPoints[42]) / 2;
        cv::Rect2f rightEyeBBox(std::min(frameInfo.rightEyePoint.x - width, frameInfo.image.cols - 2 * width),
                std::min(frameInfo.rightEyePoint.y - height / 2, frameInfo.image.rows - 2 * height), 2 * width, height);
        frameInfo.rightEyeBBox = rightEyeBBox;

        cv::Mat rightEye = frameInfo.image(frameInfo.rightEyeBBox); //right eye image

        frameInfo.rightEye = (eyeRecognition->predict(rightEye) == true) ? CLOSED : OPEN;

        eyeRecognition->recordEyeState(frameInfo.leftEye, frameInfo.rightEye);

        return true;
    }else{
        return false;
    }
}

/**
 * @brief Processor::detectMouthState
 * @return
 */
bool Processor::detectMouthState(){
    if(!frameInfo.landmarkPoints.empty()){
        mouthRecognition->calculate(frameInfo.landmarkPoints);
        return true;
    }else{
        return false;
    }
}


FrameInfo Processor::getFrameInfo(){
    return this->frameInfo;
}


void Processor::drawEyeState(){
    if(!frameInfo.leftEyeBBox.empty()){
        cv::rectangle(frameInfo.processedImage, frameInfo.leftEyeBBox, cv::Scalar(0, 0, 255), 2, 1);
        cv::putText(frameInfo.processedImage, frameInfo.leftEye == OPEN ? "OPEN" : "CLOSED", frameInfo.leftEyePoint, cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,0,255));
    }
    if(!frameInfo.rightEyeBBox.empty()){
        cv::rectangle(frameInfo.processedImage, frameInfo.rightEyeBBox, cv::Scalar(0, 0, 255), 2, 1);
        cv::putText(frameInfo.processedImage, frameInfo.rightEye == OPEN ? "OPEN" : "CLOSED", frameInfo.rightEyePoint, cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,0,255));
    }
}

void Processor::drawFace() {
    if(!frameInfo.originFacebbox.empty()){
        cv::rectangle(frameInfo.processedImage, frameInfo.facebbox, cv::Scalar(255, 0, 0), 2, 1);
    }
}


void Processor::drawOriginFace() {
    if(!frameInfo.originFacebbox.empty()){
        cv::rectangle(frameInfo.processedImage, frameInfo.originFacebbox, cv::Scalar(255, 0, 0), 2, 1);
    }
}

void Processor::drawFacePoints() {
    if(!frameInfo.facePoints.empty()){
        for(size_t i = 0; i < frameInfo.facePoints.size(); i++){
            cv::circle(frameInfo.processedImage, frameInfo.facePoints[i], 0.1, cv::Scalar(0, 0, 255), 2, 2, 0);
        }
    }
}

void Processor::drawLandmark() {
    if(!frameInfo.landmarkPoints.empty()){
        for(size_t i = 0; i < frameInfo.landmarkPoints.size(); i++) {
            cv::circle(frameInfo.processedImage, frameInfo.landmarkPoints[i], 0.1, cv::Scalar(0, 255, 0), 2, 2, 0);
        }
    }
}

void Processor::drawAnnotationBox(){
    if(frameInfo.vec.size() != 0){
        poseEstimator->drawAnnotationBox(frameInfo.processedImage, frameInfo.vec[0], frameInfo.vec[1]);
    }
}

void Processor::drawPose(){
    if(!frameInfo.landmarkAndPose.pose.empty()){
        landmarkWithPose->drawPose(frameInfo.processedImage, frameInfo.landmarkAndPose, 80);
    }
}

cv::Mat Processor::getProcessedImage(){
    //this->drawFace();
    //this->drawOriginFace();
    this->drawLandmark();
    //this->drawFacePoints();
    this->drawPose();
    this->drawAnnotationBox();
    this->drawEyeState();
    return frameInfo.processedImage;
}
