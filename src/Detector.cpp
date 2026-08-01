#include "Detector.hpp"
#include <fstream>

Detector::Detector(const std::string& modelPath,
                   const std::string& classNamesPath,
                   int numModelClasses) {
    net = cv::dnn::readNetFromONNX(modelPath);

#ifdef __linux__
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
#else
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
#endif

    std::ifstream classFile(classNamesPath);
    std::string line;
    while (std::getline(classFile, line))
        if (!line.empty()) _classNames.push_back(line);

    // If numModelClasses not specified, use names file size
    _numModelClasses = (numModelClasses > 0) ? numModelClasses : (int)_classNames.size();
}


std::vector<Detection> Detector::runInference(cv::Mat& frame) {
    cv::Mat blob;
    cv::dnn::blobFromImage(frame, blob, 1 / 255.0, inputSize, cv::Scalar(0, 0, 0), true, false);
    net.setInput(blob);

    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    cv::Mat output = outputs[0];
    output = output.reshape(1, output.size[1]);
    cv::transpose(output, output);

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    float* data = (float*)output.data;
    for (int i = 0; i < output.rows; ++i) {
        float* classes_scores = data + 4;

        // Use _numModelClasses for inference (actual model output size)
        cv::Mat scores(1, _numModelClasses, CV_32FC1, classes_scores);
        cv::Point class_id_point;
        double max_class_score;
        cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id_point);

        if (max_class_score > scoreThreshold) {
            float x = data[0], y = data[1], w = data[2], h = data[3];
            int left = int((x - 0.5 * w) * (frame.cols / 640.0));
            int top = int((y - 0.5 * h) * (frame.rows / 640.0));
            int width = int(w * (frame.cols / 640.0));
            int height = int(h * (frame.rows / 640.0));

            class_ids.push_back(class_id_point.x);
            confidences.push_back(max_class_score);
            boxes.push_back(cv::Rect(left, top, width, height));
        }
        data += 4 + _numModelClasses;
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, scoreThreshold, nmsThreshold, indices);

    std::vector<Detection> final_detections;
    for (int idx : indices) {
        Detection det;
        det.class_id = class_ids[idx];
        det.confidence = confidences[idx];
        det.box = boxes[idx];
        final_detections.push_back(det);
    }

    // Remap only if model has more classes than display names
    // i.e. pretrained 10-class model remapped to 2-class display
    if (_numModelClasses > (int)_classNames.size()) {
        for (auto& det : final_detections) {
            // pedestrian=0, people=1 → person=0
            if (det.class_id == 0 || det.class_id == 1) {
                det.class_id = 0;
            }
            // all vehicles → car=1
            else {
                det.class_id = 1;
            }
        }
    }

    return final_detections;
}