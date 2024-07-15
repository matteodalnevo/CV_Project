// Homography_main.cpp
#include "homography.h"
#include "tableDetection.h"
#include "ballDetection.h"
#include "utils.h"

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
    std::string TEST = "../data/game3_clip2/";
    std::string VIDEO = "game3_clip2.mp4";
    std::string video_full_path = TEST+VIDEO;

    // Here we have the conversion into frames FRAMES ORIGINAL
    auto [frames, fps] = videoToFrames(video_full_path);

    std::vector<cv::Mat> video_frames = homography_track_balls(frames,TEST);

    framesToVideo(video_frames, "MOD_"+VIDEO, fps); // TO BE USED ON TRACK OUT

    return 0;
}
