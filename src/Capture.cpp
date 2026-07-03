#include "Capture.hpp"
#include <iostream>
#include <algorithm>
#include <filesystem>

namespace fs = std::filesystem;

Capture::Capture(int deviceID) : isVideoFile(false) {
    // Try RTSP stream first
    const std::string pipeline = 
        "rtspsrc location=rtsp://192.168.144.25:8554/main.264 latency=100 protocols=tcp ! "
        "rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! "
        "video/x-raw,format=BGRx ! videoconvert ! "
        "video/x-raw,format=BGR ! appsink drop=1 sync=0";
cap.open(pipeline, cv::CAP_GSTREAMER);
    
    if (!cap.isOpened()) {
        std::cerr << "[Capture] RTSP stream failed, falling back to device " << deviceID << std::endl;
#if defined(__APPLE__)
        cap.open(deviceID, cv::CAP_AVFOUNDATION);
#elif defined(_WIN32)
        cap.open(deviceID, cv::CAP_MSMF);
#else
        cap.open(deviceID, cv::CAP_V4L2);
#endif
    } else {
        std::cout << "[Capture] RTSP stream opened successfully." << std::endl;
    }
}

Capture::Capture(const std::string& videoPath) : isVideoFile(true) {
    cap.open(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "[Capture Error] Unable to open video file path at: " << videoPath << std::endl;
    }
}

// NEW: image sequence constructor
Capture::Capture(const std::string& folderPath, bool isImageSequence) : isImageSeq(isImageSequence) {
    if (!isImageSeq) return; // safety, shouldn't happen given how it's called

    for (const auto& entry : fs::directory_iterator(folderPath)) {
        if (!entry.is_regular_file()) continue;
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") {
            framePaths.push_back(entry.path().string());
        }
    }

    std::sort(framePaths.begin(), framePaths.end());

    if (framePaths.empty()) {
        std::cerr << "[Capture Error] No image frames found in: " << folderPath << std::endl;
    } else {
        std::cout << "[Capture] Loaded " << framePaths.size() << " frames from: " << folderPath << std::endl;
    }
}

Capture::~Capture() {
    if (cap.isOpened()) {
        cap.release();
    }
}

bool Capture::isReady() {
    if (isImageSeq) {
        return !framePaths.empty();
    }
    return cap.isOpened();
}

cv::Mat Capture::getFrame() {
    if (isImageSeq) {
        if (frameIdx >= framePaths.size()) {
            return cv::Mat(); // empty signals end of sequence to main loop
        }
        cv::Mat frame = cv::imread(framePaths[frameIdx]);
        frameIdx++;
        return frame;
    }

    cv::Mat frame;
    if (cap.isOpened()) {
        cap.read(frame);
    }
    return frame;
}

void Capture::resetPlayback() {
    if (isImageSeq) {
        frameIdx = 0;
        return;
    }
    if (isVideoFile && cap.isOpened()) {
        cap.set(cv::CAP_PROP_POS_FRAMES, 0);
    }
}