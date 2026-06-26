#pragma once
#include <opencv2/core.hpp>

enum class TrackState {
    New,        // just appeared
    Tracked,    // actively tracked
    Lost,       // temporarily missing
    Removed     // dead, remove from list
};

struct Track {
    int         id;
    int         class_id;
    float       confidence;
    cv::Rect    box;
    TrackState  state;
    int         frames_since_update;
    int         hit_streak;          // consecutive detections
    cv::Point   getCenter() const {
        return cv::Point(box.x + box.width / 2, box.y + box.height / 2);
    }
    float getArea() const {
        return static_cast<float>(box.width * box.height);
    }
};