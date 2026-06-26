#ifndef DETECTOR_HPP
#define DETECTOR_HPP

#include <opencv2/opencv.hpp>
#include <vector>

struct Detection {
    int class_id;
    float confidence;
    cv::Rect box;

    cv::Point getCenter() const {
        return cv::Point(box.x + box.width / 2, box.y + box.height / 2);
    }
    
    float getArea() const {
        return (float)box.width * box.height;
    }
};

class Detector {
public:
    Detector(const std::string& modelPath);
    std::vector<Detection> runInference(cv::Mat& frame);

private:
    cv::dnn::Net net;
    const cv::Size inputSize = cv::Size(640, 640);
    const float scoreThreshold = 0.6;
    const float nmsThreshold = 0.4;
};

#endif
