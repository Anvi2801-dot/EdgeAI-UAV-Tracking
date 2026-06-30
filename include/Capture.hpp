#ifndef CAPTURE_HPP
#define CAPTURE_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class Capture {
public:
    // Existing webcam constructor (default to 0)
    Capture(int deviceID = 0); 
    
    // Overloaded constructor for prerecorded testing video files
    Capture(const std::string& videoPath);

    // NEW: Overloaded constructor for an image-sequence folder (e.g. VisDrone frames)
    // isImageSequence is just a tag to disambiguate from the video-file constructor above
    Capture(const std::string& folderPath, bool isImageSequence);
    
    ~Capture();
    
    bool isReady();
    cv::Mat getFrame();
    
    // Loop back helper to rewind video assets / image sequences
    void resetPlayback();

private:
    cv::VideoCapture cap;
    bool isVideoFile = false; // Internal track flag to separate file from hardware logic

    // NEW: image sequence state
    bool isImageSeq = false;
    std::vector<std::string> framePaths;
    size_t frameIdx = 0;
};

#endif