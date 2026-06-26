#include <iostream>
#include <memory>
#include <string>
#include <chrono>
#include <limits>
#include <vector>

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

    Detector detector("../models/yolov8n.onnx");

    std::unique_ptr<Tracker> tracker = std::make_unique<ByteTracker>(
        0.4f,   // high confidence threshold
        0.1f,   // low confidence threshold
        0.3f,   // IOU match threshold
        60      // max lost frames before track removed
    );

    Commander droneCommander;

    if (!camera->isReady()) {
        std::cerr << "[Fatal Error] Unable to bind to selected video data feed!" << std::endl;
        return -1;
    }

    std::cout << "UAV Onboard Perception Engine Active. Press 'q' to quit, 'l' to land." << std::endl;

    auto last_target_seen = std::chrono::steady_clock::now();
    bool target_ever_seen = false;
    const double NO_TARGET_TIMEOUT_SEC = 10.0;

    // ── Window setup — outside loop so it only runs once ──
    cv::namedWindow("UAV - Onboard AI Stream", cv::WINDOW_NORMAL);
    cv::resizeWindow("UAV - Onboard AI Stream", 1280, 800);

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
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> duration = end - start;
        float inferenceTime = duration.count();
        float fps = 1000.0f / inferenceTime;

        std::vector<Track> tracks = tracker->update(detections, frame);

        Track* target = nullptr;
        for (auto& t : tracks) {
            if (t.class_id == 0 && t.state != TrackState::Removed) {
                target = &t;
                break;
            }
        }

        if (target != nullptr) {
            last_target_seen = std::chrono::steady_clock::now();
            target_ever_seen = true;

            cv::rectangle(frame, target->box, cv::Scalar(0, 255, 0), 2);
            cv::circle(frame, target->getCenter(), 5, cv::Scalar(0, 0, 255), -1);
            cv::putText(frame,
                "ID:" + std::to_string(target->id),
                cv::Point(target->box.x, target->box.y - 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);

            std::string cmdLog = droneCommander.processTarget(
                target->getCenter(), target->getArea(),
                frameW, frameH, inferenceTime, fps
            );

            std::cout << cmdLog << std::endl;
            size_t splitPos = cmdLog.find('|', cmdLog.find('|') + 1);
            std::string line1 = cmdLog.substr(0, splitPos);
            std::string line2 = splitPos != std::string::npos ? cmdLog.substr(splitPos) : "";

            cv::putText(frame, line1, cv::Point(10, 25),
                cv::FONT_HERSHEY_DUPLEX, 0.45, cv::Scalar(0, 0, 255), 1);
            cv::putText(frame, line2, cv::Point(10, 45),
                cv::FONT_HERSHEY_DUPLEX, 0.45, cv::Scalar(0, 0, 255), 1);

        } else {
            droneCommander.processTarget(
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
                    droneCommander.triggerLanding();
                    break;
                }

                std::string noTargetText = "No Target - Landing in "
                    + std::to_string((int)remaining + 1) + "s";
                cv::putText(frame, noTargetText, cv::Point(30, 50),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 165, 255), 2);
            } else {
                cv::putText(frame, "No Target - Hovering", cv::Point(30, 50),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 165, 255), 2);
            }
        }

        int active_tracks = 0;
        for (auto& t : tracks)
            if (t.state == TrackState::Tracked || t.state == TrackState::New)
                active_tracks++;

        std::string perfText = "Inference: " + std::to_string((int)inferenceTime)
            + "ms | FPS: " + std::to_string((int)fps)
            + " | Tracks: " + std::to_string(active_tracks);
        cv::putText(frame, perfText, cv::Point(30, frameH - 30),
            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);

        cv::imshow("UAV - Onboard AI Stream", frame);

        int waitTime = (choice == 2) ? 33 : 33;
        int key = cv::waitKey(waitTime) & 0xFF;

        if (key == 'q') {
            break;
        } else if (key == 'l') {
            std::cout << "[MANUAL] Landing key pressed. Initiating landing sequence..." << std::endl;
            cv::destroyAllWindows();
            droneCommander.triggerLanding();
            break;
        }
    }
    return 0;
}