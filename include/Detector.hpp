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
    Detector(const std::string& modelPath, const std::string& classNamesPath, int numModelClasses = 0);
    std::vector<Detection> runInference(cv::Mat& frame);

    const std::vector<std::string>& getClassNames() const { return _classNames; }

private:
    cv::dnn::Net net;
    std::vector<std::string> _classNames;
    const cv::Size inputSize = cv::Size(640, 640);
    const float scoreThreshold = 0.25;
    const float nmsThreshold = 0.45;
    int _numModelClasses = 0;
};

#endif
