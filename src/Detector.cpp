#include "Detector.hpp"
#include <fstream>

Detector::Detector(const std::string& modelPath,
                   const std::string& classNamesPath) {
    net = cv::dnn::readNetFromONNX(modelPath);

#ifdef __linux__
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
#else
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
#endif

    // Load COCO class names
    std::ifstream classFile(classNamesPath);
    std::string line;
    while (std::getline(classFile, line))
        if (!line.empty()) _classNames.push_back(line);
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
        cv::Mat scores(1, 80, CV_32FC1, classes_scores);
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
        data += 84;
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
    return final_detections;
}
