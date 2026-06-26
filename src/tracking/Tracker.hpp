#pragma once
#include <vector>
#include "Track.hpp"
#include "Detector.hpp"

// Abstract base — swap ByteTracker for BotSortTracker by changing
// one line in main.cpp. No other files need to change.
class Tracker {
public:
    virtual ~Tracker() = default;

    // Feed detections, get back active tracks
    virtual std::vector<Track> update(
        const std::vector<Detection>& detections,
        const cv::Mat& frame
    ) = 0;

    // Reset all tracks (e.g. on scene change)
    virtual void reset() = 0;
};