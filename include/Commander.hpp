#ifndef COMMANDER_HPP
#define COMMANDER_HPP

#include <mavsdk/mavsdk.h>
#include <mavsdk/plugins/action/action.h>
#include <mavsdk/plugins/offboard/offboard.h>
#include <mavsdk/plugins/telemetry/telemetry.h>
#include <opencv2/opencv.hpp>
#include <memory>
#include <deque>
#include <string>
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>
#include <algorithm>

enum FlightState { GROUNDED, HOVERING, LANDING };

class Commander {
public:
    Commander();
    ~Commander();
    std::string processTarget(cv::Point center, float area, int frameW, int frameH, float latencyMs, float liveFps);
    void triggerLanding();

private:
    std::unique_ptr<mavsdk::Mavsdk> _mavsdk;
    std::shared_ptr<mavsdk::System> _system;
    std::unique_ptr<mavsdk::Action> _action;
    std::unique_ptr<mavsdk::Offboard> _offboard;
    std::unique_ptr<mavsdk::Telemetry> _telemetry;
    std::atomic<float> _live_lidar_distance_m{0.0f};

    // Moving average history buffers for smoothing tracking inputs
    std::deque<float> areaHistory;
    std::deque<float> xHistory; // Changed to float to match the helper template signature
    std::deque<float> yHistory; // Changed to float to match the helper template signature
    
    FlightState state;
    float clearance_m = 0.0f;

    float last_errorX;
    float last_errorArea;

    // Calibration Member States
    float _dynamic_ideal_area;
    int _calibration_frames;
    bool _calibrated;

    std::thread _heartbeat_thread;
    std::atomic<bool> _running{false};
    mavsdk::Offboard::VelocityBodyYawspeed _last_cmd{};
    std::mutex _cmd_mutex;

    std::chrono::steady_clock::time_point _last_target_seen;
    bool _target_ever_seen = false;
    const double NO_TARGET_TIMEOUT_SEC = 10.0;

    const float SAFE_LANDING_CLEARANCE_M = 0.5f; 

    const int SMOOTHING_WINDOW = 10;
    const int X_TOLERANCE = 100;

    // Helper functions
    float getSmoothedValue(std::deque<float>& history, float newValue);
    float getSmoothedArea(float currentArea);
};

#endif