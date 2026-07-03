cmake_minimum_required(VERSION 3.16)
project(uav)
set(CMAKE_CXX_STANDARD 17)

set(MAVSDK_DIR "C:/mavsdk/lib/cmake/MAVSDK")
find_package(MAVSDK REQUIRED)

set(OpenCV_DIR "C:/opencv/build/x64/vc16/lib")
find_package(OpenCV REQUIRED)

include_directories(include ${OpenCV_INCLUDE_DIRS})


add_executable(uav
    src/main.cpp
    src/Capture.cpp
    src/Detector.cpp
    src/Commander.cpp
    src/tracking/KalmanFilter.cpp
    src/tracking/ByteTracker.cpp
)

target_link_libraries(uav PRIVATE
    ${OpenCV_LIBRARIES}
    MAVSDK::mavsdk
)
