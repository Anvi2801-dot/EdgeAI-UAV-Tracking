#include "Commander.hpp"
#include <iomanip>
#include <sstream>
#include <numeric>
#include <iostream>
#include <thread>
#include <cmath>
#include <algorithm>

Commander::Commander() : state(GROUNDED), last_errorX(0.0f), last_errorArea(0.0f),
                         _dynamic_ideal_area(0.0f), _calibration_frames(0), _calibrated(false) {
    mavsdk::Mavsdk::Configuration config(mavsdk::ComponentType::CompanionComputer);
    _mavsdk = std::make_unique<mavsdk::Mavsdk>(config);

    mavsdk::ConnectionResult connection_result = _mavsdk->add_any_connection("udpout://192.168.110.141:14580");
    if (connection_result != mavsdk::ConnectionResult::Success) {
        std::cerr << "[MAVSDK Error] Connection to vehicle failed!" << std::endl;
        return;
    }

    std::cout << "[MAVSDK] Waiting to discover drone autopilot" << std::endl;
    while (_mavsdk->systems().empty()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    _system = _mavsdk->systems().at(0);
    std::cout << "[MAVSDK] Drone hardware discovered successfully!" << std::endl;

    _action    = std::make_unique<mavsdk::Action>(_system);
    _offboard  = std::make_unique<mavsdk::Offboard>(_system);
    _telemetry = std::make_unique<mavsdk::Telemetry>(_system);

    _telemetry->subscribe_distance_sensor([this](mavsdk::Telemetry::DistanceSensor distance_sensor) {
        if (distance_sensor.current_distance_m > 0.01f) {
            _live_lidar_distance_m.store(distance_sensor.current_distance_m);
        }
    });

    // Wait for system to be armable
    std::cout << "[MAVSDK] Waiting for system ready..." << std::endl;
    while (!_telemetry->health().is_armable) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        std::cout << "[MAVSDK] Not armable yet, waiting..." << std::endl;
    }
    std::cout << "[MAVSDK] System armable!" << std::endl;

    std::this_thread::sleep_for(std::chrono::seconds(5));

    // Hover setpoint — zero velocity, drone holds position
    mavsdk::Offboard::VelocityBodyYawspeed hover{};
    hover.forward_m_s    = 0.0f;
    hover.right_m_s      = 0.0f;
    hover.down_m_s       = 0.0f;
    hover.yawspeed_deg_s = 0.0f;

    // 1. ARM in Hold mode — no offboard yet
    std::cout << "[MAVSDK] Requesting motor arming sequence" << std::endl;
    mavsdk::Action::Result arm_result = _action->arm();
    if (arm_result != mavsdk::Action::Result::Success) {
        std::cerr << "[MAVSDK Error] Arming failed: " << arm_result << std::endl;
        return;
    }
    std::cout << "[MAVSDK] Armed successfully!" << std::endl;

    // 2. Stream pre-offboard hover setpoints (required before offboard->start())
    std::cout << "[MAVSDK] Streaming pre-offboard setpoints" << std::endl;
    for (int i = 0; i < 10; i++) {
        if (!_telemetry->armed()) {
            std::cerr << "[MAVSDK Error] Disarmed while streaming pre-offboard setpoints." << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(2));
            _action->arm();
        }
        _offboard->set_velocity_body(hover);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::cout << "[MAVSDK] Transitioning to Offboard Navigation Mode" << std::endl;
    mavsdk::Offboard::Result offboard_result = _offboard->start();
    if (offboard_result != mavsdk::Offboard::Result::Success) {
        std::cerr << "[MAVSDK Error] Offboard mode switch failed! Error: "
                  << offboard_result << std::endl;
        return;
    }
    std::cout << "[MAVSDK] Offboard mode activated successfully!" << std::endl;

    // 3. Climb via offboard velocity setpoints — no PX4 mission/takeoff subsystem
    const float TAKEOFF_ALTITUDE_M = 2.0f;
    mavsdk::Offboard::VelocityBodyYawspeed climb{};
    climb.forward_m_s    = 0.0f;
    climb.right_m_s      = 0.0f;
    climb.down_m_s       = -1.5f; // negative = climb
    climb.yawspeed_deg_s = 0.0f;

    std::cout << "[MAVSDK] Climbing to takeoff altitude..." << std::endl;
    const float ALTITUDE_REACHED_THRESHOLD_M = TAKEOFF_ALTITUDE_M * 0.9f;
    const auto  TAKEOFF_TIMEOUT              = std::chrono::seconds(60);
    auto        takeoff_wait_start           = std::chrono::steady_clock::now();
    bool        altitude_reached             = false;

    while (std::chrono::steady_clock::now() - takeoff_wait_start < TAKEOFF_TIMEOUT) {
        if (!_telemetry->armed()) {
            std::cerr << "[MAVSDK Error] Vehicle disarmed unexpectedly during climb!" << std::endl;
            return;
        }

        float current_alt = _telemetry->position().relative_altitude_m;
        std::cout << "[MAVSDK] Climbing... altitude: " << current_alt << "m" << std::endl;

        if (current_alt >= ALTITUDE_REACHED_THRESHOLD_M) {
            altitude_reached = true;
            break;
        }

        _offboard->set_velocity_body(climb);
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    if (!altitude_reached) {
        std::cerr << "[MAVSDK Error] Takeoff altitude not reached within "
                  << TAKEOFF_TIMEOUT.count() << "s timeout. Aborting." << std::endl;
        return;
    }
    std::cout << "[MAVSDK] Hover altitude reached." << std::endl;

    // Hold hover before handing control to heartbeat thread
    _offboard->set_velocity_body(hover);

    // 4. START HEARTBEAT THREAD
    _running  = true;
    _last_cmd = hover;
    _heartbeat_thread = std::thread([this]() {
        while (_running) {
            {
                std::lock_guard<std::mutex> lock(_cmd_mutex);
                _offboard->set_velocity_body(_last_cmd);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });

    state = HOVERING;
    std::cout << "[MAVSDK] System linked, armed, and in Offboard mode. Ready for vision targets!" << std::endl;
}

Commander::~Commander() {
    _running = false;
    if (_heartbeat_thread.joinable()) {
        _heartbeat_thread.join();
    }
}

void Commander::triggerLanding() {
    if (state == LANDING || state == GROUNDED) return;

    std::cout << "[MAVSDK] Landing requested. Checking ground clearance" << std::endl;

    auto dist = _telemetry->distance_sensor();
    if (dist.current_distance_m > 0.0f) {
        clearance_m = dist.current_distance_m;
        std::cout << "[MAVSDK] Downward clearance: " << clearance_m << "m" << std::endl;
    } else {
        std::cout << "[MAVSDK] Distance sensor unavailable - using altitude as fallback." << std::endl;
        auto pos = _telemetry->position();
        clearance_m = pos.relative_altitude_m;
        std::cout << "[MAVSDK] Relative altitude: " << clearance_m << "m" << std::endl;
    }

    if (clearance_m < SAFE_LANDING_CLEARANCE_M) {
        std::cerr << "[MAVSDK Warning] Obstacle detected " << clearance_m
                  << "m below! Landing aborted for safety." << std::endl;
        return;
    }

    std::cout << "[MAVSDK] Clearance OK (" << clearance_m << "m). Stopping offboard and landing" << std::endl;

    state    = LANDING;
    _running = false;

    if (_heartbeat_thread.joinable()) {
        _heartbeat_thread.join();
    }

    _offboard->stop();

    mavsdk::Action::Result land_result = _action->land();
    if (land_result != mavsdk::Action::Result::Success) {
        std::cerr << "[MAVSDK Error] Land command failed! Error: " << land_result << std::endl;
        return;
    }

    std::cout << "[MAVSDK] Landing command accepted. Descending" << std::endl;

    // COM_DISARM_LAND is set to 0 in this airframe config, so PX4 will NOT
    // auto-disarm on landing detection — we must detect ground contact
    // ourselves and disarm explicitly, or this loop never exits.
    const float LANDED_ALTITUDE_THRESHOLD_M  = 0.15f;
    const int   LANDED_STABLE_COUNT_REQUIRED = 3;
    const auto  LANDING_TIMEOUT              = std::chrono::seconds(30);
    auto        landing_wait_start           = std::chrono::steady_clock::now();
    int         stable_count                 = 0;

    while (_telemetry->armed()) {
        float alt = _telemetry->position().relative_altitude_m;
        std::cout << "[MAVSDK] Descending... altitude: " << alt << "m" << std::endl;

        if (std::abs(alt) < LANDED_ALTITUDE_THRESHOLD_M) {
            stable_count++;
            if (stable_count >= LANDED_STABLE_COUNT_REQUIRED) {
                std::cout << "[MAVSDK] Ground contact confirmed. Disarming." << std::endl;
                mavsdk::Action::Result disarm_result = _action->disarm();
                if (disarm_result != mavsdk::Action::Result::Success) {
                    std::cerr << "[MAVSDK Error] Disarm command failed! Error: "
                              << disarm_result << ". Retrying." << std::endl;
                    _action->disarm(); // best-effort retry
                }
                break;
            }
        } else {
            stable_count = 0; // altitude moved away from ground band — reset confirmation streak
        }

        if (std::chrono::steady_clock::now() - landing_wait_start > LANDING_TIMEOUT) {
            std::cerr << "[MAVSDK Error] Landing did not complete within "
                      << LANDING_TIMEOUT.count() << "s. Forcing disarm for safety." << std::endl;
            _action->disarm();
            break;
        }

        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    state = GROUNDED;
    std::cout << "[MAVSDK] Landed and disarmed successfully." << std::endl;
}

float Commander::getSmoothedValue(std::deque<float>& history, float newValue) {
    history.push_back(newValue);
    if (history.size() > static_cast<size_t>(SMOOTHING_WINDOW)) {
        history.pop_front();
    }
    return std::accumulate(history.begin(), history.end(), 0.0f) / history.size();
}

float Commander::getSmoothedArea(float currentArea) {
    return getSmoothedValue(areaHistory, currentArea);
}

std::string Commander::processTarget(cv::Point center, float area, int frameW, int frameH, float latencyMs, float liveFps) {
    if (state == LANDING || state == GROUNDED) return "[Landing in progress]";

    if (area < 0.0f) {
        std::lock_guard<std::mutex> lock(_cmd_mutex);
        _last_cmd.forward_m_s    = 0.0f;
        _last_cmd.right_m_s      = 0.0f;
        _last_cmd.down_m_s       = 0.0f;
        _last_cmd.yawspeed_deg_s = 0.0f;
        return "[Perception Lost] Freezing Position Matrix (Safe Hover)";
    }

    float live_lidar = _live_lidar_distance_m.load();

    if (!_calibrated) {
        if (live_lidar <= 0.05f && _calibration_frames < 30) {
            _calibration_frames++;
            std::lock_guard<std::mutex> lock(_cmd_mutex);
            _last_cmd = {};
            return "[Initializing LiDAR Array...] Checking Sensor Telemetry Stream";
        }
        areaHistory.clear();
        xHistory.clear();
        yHistory.clear();
        last_errorX    = static_cast<float>(center.x - (frameW / 2));
        last_errorArea = 0.0f;
        _calibrated    = true;
        std::cout << "[Commander] Hardware link confirmed. Active range: " << live_lidar << "m" << std::endl;
    }

    float smoothX = getSmoothedValue(xHistory, static_cast<float>(center.x));
    getSmoothedArea(area);

    std::stringstream log;
    std::string action = "HOVER";
    int screenCenter   = frameW / 2;

    float errorX = smoothX - screenCenter;
    const float IDEAL_FOLLOW_DISTANCE_M = 2.5f;
    float errorDistance = live_lidar - IDEAL_FOLLOW_DISTANCE_M;

    float d_errorX        = errorX        - last_errorX;
    float d_errorDistance = errorDistance - last_errorArea;
    last_errorX    = errorX;
    last_errorArea = errorDistance;

    const float Kp_yaw   = 0.04f;
    const float Kd_yaw   = 0.02f;
    const float Kp_pitch = 0.45f;
    const float Kd_pitch = 0.08f;

    float yaw_speed   = 0.0f;
    float forward_vel = 0.0f;
    bool  is_turning  = false;
    bool  is_moving   = false;

    if (std::abs(errorX) > X_TOLERANCE) {
        yaw_speed  = (errorX * Kp_yaw) + (d_errorX * Kd_yaw);
        yaw_speed  = std::clamp(yaw_speed, -25.0f, 25.0f);
        is_turning = true;
    }

    if (std::abs(errorDistance) > 0.10f && live_lidar > 0.15f) {
        forward_vel = (errorDistance * Kp_pitch) + (d_errorDistance * Kd_pitch);
        forward_vel = std::clamp(forward_vel, -0.80f, 0.80f);
        is_moving   = true;
    }

    if (is_turning && is_moving) {
        std::string turn_dir = (errorX       > 0) ? "RIGHT"   : "LEFT";
        std::string move_dir = (errorDistance > 0) ? "FORWARD" : "BACKWARD";
        action = "MOVE_" + move_dir + "_ROTATE_" + turn_dir;
    } else if (is_turning) {
        action = (errorX > 0) ? "ROTATE_RIGHT_PD" : "ROTATE_LEFT_PD";
    } else if (is_moving) {
        action = (errorDistance > 0) ? "MOVE_FORWARD_PD" : "MOVE_BACKWARD_PD";
    }

    {
        std::lock_guard<std::mutex> lock(_cmd_mutex);
        _last_cmd.forward_m_s    = forward_vel;
        _last_cmd.right_m_s      = 0.0f;
        _last_cmd.down_m_s       = 0.0f;
        _last_cmd.yawspeed_deg_s = yaw_speed;
    }

    auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    log << "[" << std::put_time(std::localtime(&now), "%H:%M:%S") << "] " << action
        << " | Range: "   << std::fixed << std::setprecision(2) << live_lidar   << "m"
        << " | Pitch: "   << std::fixed << std::setprecision(2) << forward_vel  << "m/s"
        << " | Yaw: "     << std::fixed << std::setprecision(1) << yaw_speed    << "deg/s"
        << " | Latency: " << std::fixed << std::setprecision(1) << latencyMs    << "ms"
        << " | FPS: "     << static_cast<int>(liveFps);

    return log.str();
}
