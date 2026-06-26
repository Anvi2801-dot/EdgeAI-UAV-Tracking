#include "Capture.hpp"
#include <iostream>

// Hardware Webcam Constructor (Preserving your AVFoundation Mac framework configuration)
Capture::Capture(int deviceID) : isVideoFile(false) {
    cap.open(deviceID, cv::CAP_DSHOW);
}

// 🎯 NEW: Video File Playback Constructor Implementation
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

// 🎯 NEW: Resets video track index back to the very first frame
void Capture::resetPlayback() {
    if (isVideoFile && cap.isOpened()) {
        cap.set(cv::CAP_PROP_POS_FRAMES, 0);
    }
}