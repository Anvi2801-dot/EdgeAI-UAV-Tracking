#pragma once
#include "Tracker.hpp"
#include "KalmanFilter.hpp"
#include <map>

struct InternalTrack {
    Track              track;
    BBoxKalmanFilter   kalman;
};

class ByteTracker : public Tracker {
public:
    ByteTracker(
        float high_thresh = 0.5f,   // high confidence threshold
        float low_thresh = 0.1f,   // low confidence threshold
        float match_thresh = 0.8f,   // IOU match threshold
        int   max_lost = 60,      // frames before track removed
        int _min_hits = 3
    );

    std::vector<Track> update(
        const std::vector<Detection>& detections,
        const cv::Mat& frame
    ) override;

    void reset() override;

private:
    float _high_thresh;
    float _low_thresh;
    float _match_thresh;
    int   _max_lost;
    int   _next_id = 1;

    std::vector<InternalTrack> _tracked;
    std::vector<InternalTrack> _lost;

    float iou(const cv::Rect& a, const cv::Rect& b);
    std::vector<std::vector<float>> iouMatrix(
        const std::vector<cv::Rect>& boxes_a,
        const std::vector<cv::Rect>& boxes_b
    );
    // Hungarian algorithm for optimal assignment
    std::vector<int> hungarianMatch(
        const std::vector<std::vector<float>>& cost_matrix,
        float thresh
    );
};