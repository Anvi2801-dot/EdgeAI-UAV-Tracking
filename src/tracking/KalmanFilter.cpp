#include "KalmanFilter.hpp"

BBoxKalmanFilter::BBoxKalmanFilter() : kf(8, 4, 0) {
    cv::Mat trans = cv::Mat::eye(8, 8, CV_32F);
    trans.at<float>(0, 4) = 1;
    trans.at<float>(1, 5) = 1;
    trans.at<float>(2, 6) = 1;
    trans.at<float>(3, 7) = 1;
    kf.transitionMatrix = trans;

    kf.measurementMatrix = cv::Mat::zeros(4, 8, CV_32F);
    kf.measurementMatrix.at<float>(0, 0) = 1;
    kf.measurementMatrix.at<float>(1, 1) = 1;
    kf.measurementMatrix.at<float>(2, 2) = 1;
    kf.measurementMatrix.at<float>(3, 3) = 1;

    cv::setIdentity(kf.processNoiseCov, cv::Scalar::all(1e-3));
    cv::setIdentity(kf.measurementNoiseCov, cv::Scalar::all(1.0));
    cv::setIdentity(kf.errorCovPost, cv::Scalar::all(1.0));
}

void BBoxKalmanFilter::init(const cv::Rect& bbox) {
    kf.statePost = cv::Mat::zeros(8, 1, CV_32F);
    kf.statePost.at<float>(0) = (float)bbox.x;
    kf.statePost.at<float>(1) = (float)bbox.y;
    kf.statePost.at<float>(2) = (float)bbox.width;
    kf.statePost.at<float>(3) = (float)bbox.height;
    initialized = true;
}

cv::Rect BBoxKalmanFilter::predict() {
    if (!initialized) return cv::Rect();
    cv::Mat pred = kf.predict();
    return cv::Rect(
        (int)pred.at<float>(0),
        (int)pred.at<float>(1),
        (int)pred.at<float>(2),
        (int)pred.at<float>(3)
    );
}

cv::Rect BBoxKalmanFilter::update(const cv::Rect& bbox) {
    cv::Mat measurement = cv::Mat::zeros(4, 1, CV_32F);
    measurement.at<float>(0) = (float)bbox.x;
    measurement.at<float>(1) = (float)bbox.y;
    measurement.at<float>(2) = (float)bbox.width;
    measurement.at<float>(3) = (float)bbox.height;
    cv::Mat corrected = kf.correct(measurement);
    return cv::Rect(
        (int)corrected.at<float>(0),
        (int)corrected.at<float>(1),
        (int)corrected.at<float>(2),
        (int)corrected.at<float>(3)
    );
}