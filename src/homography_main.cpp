// Homography_main.cpp
#include "homography.h"
#include "tableDetection.h"
#include "ballDetection.h"
#include "utils.h"
#include "preProcess.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/tracking.hpp>
#include <iostream>
#include <vector>
#include <tuple>
#include <fstream>
#include <sstream>

int main() {
    // Here we load the video, ORIGINAL VIDEO
    std::string TEST = "../data/game1_clip1/";
    std::string VIDEO = "game1_clip1.mp4";
    std::string video_full_path = TEST+VIDEO;

    // Here we have the conversion into frames FRAMES ORIGINAL
    auto [frames, fps] = videoToFrames(video_full_path);

    // PreProcessing Corners
    cv::Mat result_first;
    std::vector<cv::Vec2f> first_detected_lines;
    std::tie(result_first, first_detected_lines) = preProcess(frames[0]);

    // Table Corners
    std::vector<cv::Point2f> footage_corners;
    footage_corners = tableDetection(first_detected_lines);

    // Balls Detection


    // 

    std::vector<cv::Mat> video_frames = homography_track_balls(frames,TEST,footage_corners);

    framesToVideo(video_frames, "MOD_"+VIDEO, fps); // TO BE USED ON TRACK OUT

    return 0;
}
