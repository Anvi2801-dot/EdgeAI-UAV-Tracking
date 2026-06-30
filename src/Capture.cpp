#include "Capture.hpp"
#include <iostream>

Capture::Capture(int deviceID) : isVideoFile(false) {
#ifdef _WIN32
	cap.open(deviceID, cv::CAP_DSHOW);
#elif __APPLE__
	cap.open(deviceID, cv::CAP_AVFOUNDATION);
#else
	cap.open(deviceID, cv::CAP_V4L2);
#endif
}

Capture::Capture(const std::string& videoPath) : isVideoFile(true) {
    cap.open(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "[Capture Error] Unable to open video file path at: " << videoPath << std::endl;
    }
}

Capture::~Capture() {
    if (cap.isOpened()) {
        cap.release();
    }
}

bool Capture::isReady() {
    return cap.isOpened();
}

cv::Mat Capture::getFrame() {
    cv::Mat frame;
    if (cap.isOpened()) {
        cap.read(frame);
    }
    return frame;
}

void Capture::resetPlayback() {
    if (isVideoFile && cap.isOpened()) {
        cap.set(cv::CAP_PROP_POS_FRAMES, 0);
    }
}
