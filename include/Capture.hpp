#ifndef CAPTURE_HPP
#define CAPTURE_HPP

#include <opencv2/opencv.hpp>
#include <string>

class Capture {
public:
    // Existing webcam constructor (default to 0)
    Capture(int deviceID = 0); 
    
    // 🎯 NEW: Overloaded constructor for prerecorded testing video files
    Capture(const std::string& videoPath);
    
    ~Capture();
    
    bool isReady();
    cv::Mat getFrame();
    
    // 🎯 NEW: Loop back helper to rewind video assets
    void resetPlayback();

private:
    cv::VideoCapture cap;
    bool isVideoFile = false; // Internal track flag to separate file from hardware logic
};

#endif