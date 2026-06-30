#include <iostream>
#include <memory>
#include <string>
#include <chrono>
#include <limits>
#include <vector>
#include <set>
#include <sstream>

#include "Capture.hpp"
#include "Detector.hpp"
#include "Commander.hpp"
#include "tracking/ByteTracker.hpp"

int main() {
    std::cout << "  UAV PERCEPTION SUITE - SOURCE SELECTION " << std::endl;
    std::cout << " [1] Launch with Live Webcam Feed" << std::endl;
    std::cout << " [2] Run via Pre-recorded Reference Video File" << std::endl;
    std::cout << "Select input option (1 or 2): ";

    int choice = 1;
    std::cin >> choice;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    std::unique_ptr<Capture> camera;
    if (choice == 2) {
        std::string videoPath = "../test_videos/test1.mov";
        std::cout << "[Pipeline] Initializing playback file: " << videoPath << std::endl;
        camera = std::make_unique<Capture>(videoPath);
    } else {
        std::cout << "[Pipeline] Initializing live webcam context [0]" << std::endl;
        camera = std::make_unique<Capture>(0);
    }

    Detector detector("../models/yolov8n.onnx", "../models/coco.names");
    const auto& CLASS_NAMES = detector.getClassNames();

    // ── Mode Selection ──
    std::cout << "\n  SELECT PIPELINE MODE " << std::endl;
    std::cout << " [1] Detection      - bounding boxes only, no drone" << std::endl;
    std::cout << " [2] Classification - bounding boxes + labels, no drone" << std::endl;
    std::cout << " [3] Tracking       - boxes + labels + drone follows target" << std::endl;
    std::cout << "Enter mode (1, 2 or 3): ";
    int mode = 3;
    std::cin >> mode;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    // ── Detection + Tracking Config ──
    std::cout << "\n  OBJECT DETECTION CONFIG " << std::endl;
    std::cout << "Available classes: person, car, dog, cat, laptop, cell phone, book, chair, etc." << std::endl;
    std::cout << "Enter classes to (comma separated, or 'all'): ";
    std::string detectInput;
    std::getline(std::cin, detectInput);

    std::string trackInput = "";
    if (mode == 3) {
        std::cout << "Enter class to TRACK/FOLLOW (drone follows this): ";
        std::getline(std::cin, trackInput);
    }

    // Parse detect classes
    std::set<int> detectClasses;
    bool detectAll = (detectInput == "all");
    if (!detectAll) {
        std::stringstream ss(detectInput);
        std::string token;
        while (std::getline(ss, token, ',')) {
            token.erase(0, token.find_first_not_of(' '));
            token.erase(token.find_last_not_of(' ') + 1);
            for (int i = 0; i < (int)CLASS_NAMES.size(); i++) {
                if (CLASS_NAMES[i] == token) {
                    detectClasses.insert(i);
                    break;
                }
            }
        }
    }

    // Parse track class (only used in mode 3)
    int trackClassId = 0;
    if (mode == 3) {
        trackInput.erase(0, trackInput.find_first_not_of(' '));
        trackInput.erase(trackInput.find_last_not_of(' ') + 1);
        for (int i = 0; i < (int)CLASS_NAMES.size(); i++) {
            if (CLASS_NAMES[i] == trackInput) {
                trackClassId = i;
                break;
            }
        }
    }

    std::string modeLabel = (mode == 1) ? "Detection" : (mode == 2) ? "Classification" : "Tracking";
    std::cout << "[Config] Mode:      " << modeLabel << std::endl;
    std::cout << "[Config] Detecting: " << (detectAll ? "all classes" : detectInput) << std::endl;
    if (mode == 3)
        std::cout << "[Config] Tracking:  " << CLASS_NAMES[trackClassId] << std::endl;

    std::unique_ptr<Tracker> tracker = std::make_unique<ByteTracker>(
        0.4f,   // high confidence threshold
        0.15f,  // low confidence threshold
        0.6f,   // IOU match threshold
        10      // max lost frames before track removed
    );

    // Only init Commander in tracking mode
    std::unique_ptr<Commander> droneCommander;
    if (mode == 3) {
        droneCommander = std::make_unique<Commander>();
    }

    if (!camera->isReady()) {
        std::cerr << "[Fatal Error] Unable to bind to selected video data feed!" << std::endl;
        return -1;
    }

    std::cout << "\nUAV Onboard Perception Engine Active. Press 'q' to quit, 'l' to land." << std::endl;

    auto last_target_seen = std::chrono::steady_clock::now();
    bool target_ever_seen = false;
    const double NO_TARGET_TIMEOUT_SEC = 10.0;

    cv::namedWindow("UAV - Onboard AI Stream", cv::WINDOW_NORMAL);
    cv::resizeWindow("UAV - Onboard AI Stream", 1280, 800);

    int lockedTargetId = -1;

    while (true) {
        cv::Mat frame = camera->getFrame();

        if (!frame.empty()) {
            cv::resize(frame, frame, cv::Size(640, 480));
        }

        if (frame.empty()) {
            if (choice == 2) {
                std::cout << "[Pipeline] End of testing file reached. Restarting video track loop..." << std::endl;
                camera->resetPlayback();
                continue;
            } else {
                std::cerr << "[Pipeline Warning] Empty frame caught. Re-buffering..." << std::endl;
                cv::waitKey(33);
                continue;
            }
        }

        int frameW = frame.cols;
        int frameH = frame.rows;

        cv::line(frame, cv::Point(frameW / 2 - 100, 0), cv::Point(frameW / 2 - 100, frameH), cv::Scalar(255, 0, 0), 1);
        cv::line(frame, cv::Point(frameW / 2 + 100, 0), cv::Point(frameW / 2 + 100, frameH), cv::Scalar(255, 0, 0), 1);

        auto start = std::chrono::high_resolution_clock::now();
        std::vector<Detection> detections = detector.runInference(frame);

        // NMS on raw detections to kill duplicates before tracking
        std::vector<cv::Rect> raw_boxes;
        std::vector<float> raw_scores;
        for (auto& d : detections) {
            raw_boxes.push_back(d.box);
            raw_scores.push_back(d.confidence);
        }
        std::vector<int> nms_idx;
        cv::dnn::NMSBoxes(raw_boxes, raw_scores, 0.3f, 0.45f, nms_idx);

        std::vector<Detection> nms_detections;
        for (int idx : nms_idx) nms_detections.push_back(detections[idx]);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        float inferenceTime = duration.count();
        float fps = 1000.0f / inferenceTime;

        // ── Filter detections by user config ──
        std::vector<Detection> filtered_detections;
        for (auto& d : nms_detections) {
            if (detectAll || detectClasses.count(d.class_id)) {
                filtered_detections.push_back(d);
            }
        }

        // ── Mode 1 & 2: draw detections directly, no tracker, no drone ──
        if (mode == 1 || mode == 2) {
            for (auto& d : filtered_detections) {
                cv::rectangle(frame, d.box, cv::Scalar(0, 255, 0), 2);
                if (mode == 2) {
                    std::string className = (d.class_id < (int)CLASS_NAMES.size())
                        ? CLASS_NAMES[d.class_id] : "obj";
                    std::string label = className + " " + std::to_string((int)(d.confidence * 100)) + "%";
                    cv::putText(frame, label,
                        cv::Point(d.box.x, d.box.y - 10),
                        cv::FONT_HERSHEY_DUPLEX, 0.45, cv::Scalar(0, 255, 0), 1);
                }
            }

            // HUD
            std::string perfText = "Inference: " + std::to_string((int)inferenceTime)
                + "ms | FPS: " + std::to_string((int)fps)
                + " | Detections: " + std::to_string(filtered_detections.size())
                + " | Mode: " + modeLabel;
            cv::putText(frame, perfText, cv::Point(10, frameH - 15),
                cv::FONT_HERSHEY_DUPLEX, 0.45, cv::Scalar(0, 255, 255), 1);

            cv::imshow("UAV - Onboard AI Stream", frame);
            int key = cv::waitKey(33) & 0xFF;
            if (key == 'q') break;
            continue;
        }

        // ── Mode 3: full tracking + drone pipeline ──
        std::vector<Track> tracks = tracker->update(filtered_detections, frame);

        // Find drone follow target (lowest ID of track class)
        Track* target = nullptr;
        if (lockedTargetId != -1) {
            for (auto& t : tracks) {
                if (t.id == lockedTargetId &&
                    t.class_id == trackClassId &&
                    t.state != TrackState::Removed) {
                    target = &t;
                    break;
                }
            }
        }

        if (target == nullptr) {
            int lowest_id = INT_MAX;
            for (auto& t : tracks) {
                if (t.class_id == trackClassId &&
                    t.state != TrackState::Removed &&
                    t.hit_streak >= 3 &&
                    t.id < lowest_id) {
                    lowest_id = t.id;
                    target = &t;
                }
            }
            if (target != nullptr) {
                lockedTargetId = target->id;
                std::cout << "[Tracker] Locked onto ID:" << lockedTargetId << std::endl;
            }
        }

        // Draw all tracked objects
        for (auto& t : tracks) {
            if (t.state == TrackState::Removed || t.state == TrackState::Lost) continue;
            if (t.hit_streak < 3 && t.state == TrackState::New) continue;

            bool isTarget = (target != nullptr && t.id == target->id);

            cv::Scalar color;
            if (isTarget)                        color = cv::Scalar(0, 255, 0);
            else if (t.class_id == trackClassId) color = cv::Scalar(0, 165, 255);
            else                                 color = cv::Scalar(255, 165, 0);

            cv::rectangle(frame, t.box, color, 2);

            std::string className = (t.class_id < (int)CLASS_NAMES.size())
                ? CLASS_NAMES[t.class_id] : "obj";
            std::string label = className + " ID:" + std::to_string(t.id);

            cv::putText(frame, label,
                cv::Point(t.box.x, t.box.y - 10),
                cv::FONT_HERSHEY_DUPLEX, 0.45, color, 1);

            if (isTarget) {
                cv::circle(frame, t.getCenter(), 5, cv::Scalar(0, 0, 255), -1);
            }
        }

        // Commander logic
        if (target != nullptr) {
            last_target_seen = std::chrono::steady_clock::now();
            target_ever_seen = true;

            std::string cmdLog = droneCommander->processTarget(
                target->getCenter(), target->getArea(),
                frameW, frameH, inferenceTime, fps
            );

            std::cout << cmdLog << std::endl;

            size_t splitPos = cmdLog.find('|', cmdLog.find('|') + 1);
            std::string line1 = cmdLog.substr(0, splitPos);
            std::string line2 = splitPos != std::string::npos ? cmdLog.substr(splitPos) : "";

            cv::putText(frame, line1, cv::Point(10, 25),
                cv::FONT_HERSHEY_DUPLEX, 0.45, cv::Scalar(0, 0, 255), 2);
            cv::putText(frame, line2, cv::Point(10, 45),
                cv::FONT_HERSHEY_DUPLEX, 0.45, cv::Scalar(0, 0, 255), 2);

        } else {
            droneCommander->processTarget(
                cv::Point(frameW / 2, frameH / 2), -1.0f,
                frameW, frameH, inferenceTime, fps
            );

            if (target_ever_seen) {
                double seconds_since_target = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - last_target_seen).count();
                double remaining = NO_TARGET_TIMEOUT_SEC - seconds_since_target;

                if (remaining <= 0.0) {
                    std::cout << "[AUTO-LAND] No target for "
                              << (int)NO_TARGET_TIMEOUT_SEC
                              << "s. Initiating landing..." << std::endl;
                    cv::destroyAllWindows();
                    droneCommander->triggerLanding();
                    break;
                }

                std::string noTargetText = "No Target - Landing in "
                    + std::to_string((int)remaining + 1) + "s";
                cv::putText(frame, noTargetText, cv::Point(30, 70),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 165, 255), 2);
            } else {
                cv::putText(frame, "No Target - Hovering", cv::Point(30, 70),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 165, 255), 2);
            }
        }

        // HUD
        int active_tracks = 0;
        for (auto& t : tracks)
            if (t.state == TrackState::Tracked || t.state == TrackState::New)
                active_tracks++;

        std::string perfText = "Inference: " + std::to_string((int)inferenceTime)
            + "ms | FPS: " + std::to_string((int)fps)
            + " | Tracks: " + std::to_string(active_tracks)
            + " | Following: " + CLASS_NAMES[trackClassId];
        cv::putText(frame, perfText, cv::Point(10, frameH - 15),
            cv::FONT_HERSHEY_DUPLEX, 0.45, cv::Scalar(0, 255, 255), 1);

        cv::imshow("UAV - Onboard AI Stream", frame);

        int key = cv::waitKey(33) & 0xFF;
        if (key == 'q') {
            break;
        } else if (key == 'l') {
            std::cout << "[MANUAL] Landing key pressed. Initiating landing sequence..." << std::endl;
            cv::destroyAllWindows();
            droneCommander->triggerLanding();
            break;
        }
    }
    return 0;
}
