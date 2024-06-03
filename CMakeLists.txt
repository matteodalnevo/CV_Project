# define the minimum version of CMake required
cmake_minimum_required(VERSION 3.5.0)
# define the project name
project(Project VERSION 0.1.0)

include(CTest)
enable_testing()

set(OpenCV_DIR//Users/matteodalnevo/Documents/Open_CV/install/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)

# Add executable targets
add_executable(main src/main.cpp
               src/utils.cpp
               src/ballDetection.cpp)

target_link_libraries(main ${OpenCV_LIBS})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)