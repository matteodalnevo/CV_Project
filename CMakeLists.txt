#Aaron Prevedello

# define the minimum version of CMake required
cmake_minimum_required(VERSION 3.5.0)
# define the project name
project(Project VERSION 0.1.0)

set(OpenCV_DIR//Users/matteodalnevo/Documents/Open_CV/install/lib/cmake/opencv4)
find_package(Boost REQUIRED COMPONENTS filesystem system)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)
include_directories(${Boost_INCLUDE_DIRS})

# Add executable targets
add_executable(main src/main.cpp
           	src/utils.cpp
           	src/ball_hand_detection.cpp
           	src/ballClassification.cpp
           	src/tableDetection.cpp
           	src/homography_tracking.cpp
           	src/preProcess.cpp)

add_executable(performance src/performance_main.cpp
           	src/utils.cpp
           	src/ball_hand_detection.cpp
           	src/ballClassification.cpp
           	src/tableDetection.cpp
           	src/homography_tracking.cpp
           	src/preProcess.cpp
           	src/performance.cpp)

target_link_libraries(main ${Boost_FILESYSTEM_LIBRARY})
target_link_libraries(performance ${OpenCV_LIBS})
target_link_libraries(main ${Boost_LIBRARIES} ${OpenCV_LIBS})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

