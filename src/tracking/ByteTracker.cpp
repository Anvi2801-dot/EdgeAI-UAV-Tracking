#include "ByteTracker.hpp"
#include <algorithm>
#include <limits>

ByteTracker::ByteTracker(float high_thresh, float low_thresh,
    float match_thresh, int max_lost)
    : _high_thresh(high_thresh), _low_thresh(low_thresh),
    _match_thresh(match_thresh), _max_lost(max_lost) {
}

void ByteTracker::reset() {
    _tracked.clear();
    _lost.clear();
    _next_id = 1;
}

float ByteTracker::iou(const cv::Rect& a, const cv::Rect& b) {
    cv::Rect inter = a & b;
    if (inter.area() <= 0) return 0.0f;
    float uni = a.area() + b.area() - inter.area();
    return inter.area() / uni;
}

std::vector<std::vector<float>> ByteTracker::iouMatrix(
    const std::vector<cv::Rect>& boxes_a,
    const std::vector<cv::Rect>& boxes_b)
{
    std::vector<std::vector<float>> matrix(
        boxes_a.size(), std::vector<float>(boxes_b.size(), 0.0f));
    for (size_t i = 0; i < boxes_a.size(); i++)
        for (size_t j = 0; j < boxes_b.size(); j++)
            matrix[i][j] = iou(boxes_a[i], boxes_b[j]);
    return matrix;
}

std::vector<int> ByteTracker::hungarianMatch(
    const std::vector<std::vector<float>>& cost_matrix,
    float thresh)
{
    std::vector<int> assignment(cost_matrix.size(), -1);
    std::vector<bool> used(cost_matrix.empty() ? 0 : cost_matrix[0].size(), false);

    for (size_t i = 0; i < cost_matrix.size(); i++) {
        float best = thresh;
        int   best_j = -1;
        for (size_t j = 0; j < cost_matrix[i].size(); j++) {
            if (!used[j] && cost_matrix[i][j] > best) {
                best = cost_matrix[i][j];
                best_j = (int)j;
            }
        }
        if (best_j >= 0) {
            assignment[i] = best_j;
            used[best_j] = true;
        }
    }
    return assignment;
}

std::vector<Track> ByteTracker::update(
    const std::vector<Detection>& detections,
    const cv::Mat& /*frame*/)
{
    // Split detections into high and low confidence
    std::vector<Detection> high_dets, low_dets;
    for (auto& d : detections) {
        if (d.confidence >= _high_thresh)      high_dets.push_back(d);
        else if (d.confidence >= _low_thresh)  low_dets.push_back(d);
    }

    // --- Step 1: Predict all tracked positions ---
    std::vector<cv::Rect> predicted_boxes;
    for (auto& t : _tracked) {
        predicted_boxes.push_back(t.kalman.predict());
    }

    // --- Step 2: Match high confidence detections to tracked ---
    std::vector<cv::Rect> high_boxes;
    for (auto& d : high_dets) high_boxes.push_back(d.box);

    std::vector<bool> det_matched(high_dets.size(), false);
    std::vector<bool> trk_matched(_tracked.size(), false);

    if (!_tracked.empty() && !high_dets.empty()) {
        auto iou_mat = iouMatrix(predicted_boxes, high_boxes);
        std::vector<std::vector<float>> cost(_tracked.size(),
            std::vector<float>(high_dets.size()));
        for (size_t i = 0; i < _tracked.size(); i++)
            for (size_t j = 0; j < high_dets.size(); j++)
                cost[i][j] = iou_mat[i][j];

        auto assignment = hungarianMatch(cost, _match_thresh);
        for (size_t i = 0; i < assignment.size(); i++) {
            if (assignment[i] >= 0) {
                int j = assignment[i];
                _tracked[i].track.box =
                    _tracked[i].kalman.update(high_dets[j].box);
                _tracked[i].track.confidence = high_dets[j].confidence;
                _tracked[i].track.state = TrackState::Tracked;
                _tracked[i].track.frames_since_update = 0;
                _tracked[i].track.hit_streak++;
                trk_matched[i] = true;
                det_matched[j] = true;
            }
        }
    }

    // --- Step 3: Match low confidence detections to unmatched tracks ---
    std::vector<cv::Rect> low_boxes;
    for (auto& d : low_dets) low_boxes.push_back(d.box);

    if (!low_dets.empty()) {
        std::vector<cv::Rect> unmatched_pred;
        std::vector<int>      unmatched_idx;
        for (size_t i = 0; i < _tracked.size(); i++) {
            if (!trk_matched[i]) {
                unmatched_pred.push_back(predicted_boxes[i]);
                unmatched_idx.push_back((int)i);
            }
        }
        if (!unmatched_pred.empty()) {
            auto iou_mat = iouMatrix(unmatched_pred, low_boxes);
            auto assignment = hungarianMatch(iou_mat, _match_thresh);
            for (size_t k = 0; k < assignment.size(); k++) {
                if (assignment[k] >= 0) {
                    int i = unmatched_idx[k];
                    int j = assignment[k];
                    _tracked[i].track.box =
                        _tracked[i].kalman.update(low_dets[j].box);
                    _tracked[i].track.state = TrackState::Tracked;
                    _tracked[i].track.frames_since_update = 0;
                    _tracked[i].track.hit_streak++;
                    trk_matched[i] = true;
                }
            }
        }
    }

    // --- Step 4: Handle unmatched tracks → move to lost ---
    std::vector<InternalTrack> still_tracked;
    for (size_t i = 0; i < _tracked.size(); i++) {
        if (!trk_matched[i]) {
            _tracked[i].track.frames_since_update++;
            _tracked[i].track.hit_streak = 0;
            _tracked[i].track.state = TrackState::Lost;
            if (_tracked[i].track.frames_since_update <= _max_lost) {
                _lost.push_back(_tracked[i]);
            }
        } else {
            still_tracked.push_back(_tracked[i]);
        }
    }
    _tracked = still_tracked;

    // --- Step 5: Init new tracks from unmatched high-conf detections ---
    for (size_t j = 0; j < high_dets.size(); j++) {
        if (!det_matched[j]) {
            InternalTrack nt;
            nt.track.id         = _next_id++;
            nt.track.class_id   = high_dets[j].class_id;
            nt.track.confidence = high_dets[j].confidence;
            nt.track.box        = high_dets[j].box;
            nt.track.state      = TrackState::New;
            nt.track.frames_since_update = 0;
            nt.track.hit_streak = 1;
            nt.kalman.init(high_dets[j].box);
            _tracked.push_back(nt);
        }
    }

    // --- Step 6: Re-match lost tracks with unmatched high-conf detections ---
    std::vector<InternalTrack> still_lost;
    for (auto& lt : _lost) {
        lt.track.frames_since_update++;
        cv::Rect pred = lt.kalman.predict();
        bool rematched = false;
        for (size_t j = 0; j < high_dets.size(); j++) {
            if (!det_matched[j] && iou(pred, high_dets[j].box) > 0.2f) {
                lt.track.box = lt.kalman.update(high_dets[j].box);
                lt.track.state = TrackState::Tracked;
                lt.track.frames_since_update = 0;
                lt.track.hit_streak++;
                det_matched[j] = true;
                _tracked.push_back(lt);
                rematched = true;
                break;
            }
        }
        if (!rematched) still_lost.push_back(lt);
    }
    _lost = still_lost;

    // --- Step 7: Clean up expired lost tracks ---
    _lost.erase(std::remove_if(_lost.begin(), _lost.end(),
        [this](const InternalTrack& t) {
            return t.track.frames_since_update > _max_lost;
        }), _lost.end());

    // Return all active and lost tracks
    std::vector<Track> result;
    for (auto& t : _tracked) result.push_back(t.track);
    for (auto& t : _lost)    result.push_back(t.track);
    return result;
}