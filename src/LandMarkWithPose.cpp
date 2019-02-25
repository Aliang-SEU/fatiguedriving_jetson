/*  cite
 *  git:
 *  repo1: https://github.com/guozhongluo/head-pose-estimation-and-face-landmark
 *  repo2: https://github.com/qiexing/face-landmark-localization
 *  目前的速度大概在30ms 一帧
 */

#include "LandMarkWithPose.h"
#include <iostream>
LandmarkWithPose::LandmarkWithPose(){

    Caffe::set_mode(Caffe::GPU);
    net.reset(new Net<float>(network, TEST));
    net->CopyTrainedLayersFrom(param);
    Blob<float>* input_layer = net->input_blobs()[0];
    this->inputSize = cv::Size(input_layer->width(), input_layer->height());
    this->numChannels = input_layer->channels();
    CHECK(numChannels == 3 || numChannels == 1)
      << "Input layer should have 1 or 3 channels.";
    setMean(meanfile);
}

void LandmarkWithPose::setMean(const std::string& meanFile){

    cv::Scalar channelMean;

    CHECK(!meanFile.empty()) <<
     "Cannot specify mean_file and mean_value at the same time";

    BlobProto blobProto;
    ReadProtoFromBinaryFileOrDie(meanFile.c_str(), &blobProto);

    Blob<float> meanBlob;
    meanBlob.FromProto(blobProto);

    CHECK_EQ(meanBlob.channels(), numChannels)
        << "Number of channels of mean file doesn't match input layer.";

    std::vector<cv::Mat> channels;
    float* data = meanBlob.mutable_cpu_data();
    for(int i = 0; i < numChannels; ++i) {
        cv::Mat channel(meanBlob.height(), meanBlob.width(), CV_32FC1, data);
        channels.push_back(channel);
        data += meanBlob.height() * meanBlob.width();
    }

    cv::Mat _mean;
    cv::merge(channels, _mean);

    /* Compute the global mean pixel value and create a mean image
     * filled with this value. */

    // channelMean = cv::mean(_mean);
    // mean = cv::Mat(inputSize, _mean.type(), channelMean);
    mean = _mean;
}

void LandmarkWithPose::wrapInputLayer(std::vector<cv::Mat>* input_channels) {

  Blob<float>* input_layer = net->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void LandmarkWithPose::preprocess(cv::Mat& img, std::vector<cv::Mat>* input_channels) {

    cv::Mat sample;

    if (img.channels() == 3 && numChannels == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4 && numChannels == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    else if (img.channels() == 4 && numChannels == 3)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    else if (img.channels() == 1 && numChannels == 3)
        cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    else
        sample = img;

    cv::Mat sampleResized;
    if(sample.size() != inputSize)
        cv::resize(sample, sampleResized, inputSize, 0, 0, cv::INTER_AREA);
    else
        sampleResized = sample;

    cv::Mat sampleFloat;
    if(numChannels == 3)
        sampleResized.convertTo(sampleFloat, CV_32FC3);
    else
        sampleResized.convertTo(sampleFloat, CV_32FC1);

    cv::Mat sampleNormalized;
    cv::subtract(sampleFloat, mean, sampleNormalized);

    cv::split(sampleNormalized, *input_channels);

    CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
          == net->input_blobs()[0]->cpu_data())
      << "Input channels are not wrapping the input layer of the network.";

}

LandmarkAndPose LandmarkWithPose::getPredict(cv::Mat& img){

    LandmarkAndPose landmarkAndPose;

    std::vector<cv::Mat> input_channels;
    wrapInputLayer(&input_channels);

    preprocess(img, &input_channels);

    net->Forward();

    const boost::shared_ptr<Blob<float> >  predictPoints =  net->blob_by_name("68point");
    const boost::shared_ptr<Blob<float> >  predictPose =  net->blob_by_name("poselayer");

    const float* result = predictPoints->cpu_data();

    for(int i = 0; i < 136; i++) {
        landmarkAndPose.landmark.push_back(*(result+i));
    }

    result = predictPose->cpu_data();
    for(int i = 0; i < 3; i++){
        landmarkAndPose.pose.push_back(*(result + i) ); //50
    }

    return landmarkAndPose;
}


void LandmarkWithPose::drawPose(cv::Mat& img, LandmarkAndPose& landmarkAndPose, float lineL){

    //convert angle to vector
    cv::Mat rotAngle(landmarkAndPose.pose);
    cv::Mat_<float> rot(3,3);
    cv::Rodrigues(rotAngle, rot);

    //roll, yaw, pitch
    int loc[2] = { img.size().width / 2, img.size().height / 2 };
    int thickness = 2;
    int lineType = 8;

    cv::Mat P = (cv::Mat_<float>(3, 4) <<
        0, lineL, 0, 0,
        0, 0, -lineL, 0,
        0, 0, 0, -lineL);
    P = rot.rowRange(0, 2)*P;
    P.row(0) += loc[0];
    P.row(1) += loc[1];
    cv::Point p0(P.at<float>(0, 0), P.at<float>(1, 0));

    cv::line(img, p0, cv::Point(P.at<float>(0, 1), P.at<float>(1, 1)), cv::Scalar(255, 0, 0), thickness, lineType);
    cv::line(img, p0, cv::Point(P.at<float>(0, 2), P.at<float>(1, 2)), cv::Scalar(0, 255, 0), thickness, lineType);
    cv::line(img, p0, cv::Point(P.at<float>(0, 3), P.at<float>(1, 3)), cv::Scalar(0, 0, 255), thickness, lineType);

    //printf("%f %f %f\n", rot.at<float>(0, 0), rot.at<float>(0, 1), rot.at<float>(0, 2));
    //printf("%f %f %f\n", rot.at<float>(1, 0), rot.at<float>(1, 1), rot.at<float>(1, 2));
    //printf("%f %f %f\n", rot.at<float>(2, 0), rot.at<float>(2, 1), rot.at<float>(2, 2));

    cv::Vec3d eav;
    cv::Mat tmp, tmp1, tmp2, tmp3, tmp4, tmp5;
    double _pm[12] = { rot.at<float>(0, 0), rot.at<float>(0, 1),rot.at<float>(0, 2), 0,
        rot.at<float>(1, 0), rot.at<float>(1, 1),rot.at<float>(1, 2),0,
        rot.at<float>(2, 0),rot.at<float>(2, 1),rot.at<float>(2, 2),0 };
    cv::decomposeProjectionMatrix(cv::Mat(3, 4, CV_64FC1, _pm), tmp, tmp1, tmp2, tmp3, tmp4, tmp5, eav);
    /*stringstream ss;
    ss << eav[0];
    string txt = "Pitch: " + ss.str();
    putText(img, txt, Point(60, 20), 0.5, 0.5, Scalar(255, 255, 255));
    stringstream ss1;
    ss1 << eav[1];
    string txt1 = "Yaw: " + ss1.str();
    putText(img, txt1, Point(60, 40), 0.5, 0.5, Scalar(255, 255, 255));
    stringstream ss2;
    ss2 << eav[2];
    string txt2 = "Roll: " + ss2.str();
    putText(img, txt2, Point(60, 60), 0.5, 0.5, Scalar(255, 255, 255));*/
}
