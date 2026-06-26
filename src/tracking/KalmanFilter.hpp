#pragma once
#include <opencv2/video/tracking.hpp>

// State: [x, y, w, h, vx, vy, vw, vh]
// Measurement: [x, y, w, h]
class BBoxKalmanFilter {
public:
    BBoxKalmanFilter();
    void     init(const cv::Rect& bbox);
    cv::Rect predict();
    cv::Rect update(const cv::Rect& bbox);

private:
    cv::KalmanFilter kf;
    bool initialized = false;
};