// Homography_main.cpp
#include "homography.h"
#include "tableDetection.h"
#include "ballDetection.h"
#include "utils.h"
#include "preProcess.h"
#include "ballClassification.h"

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
#include <string>

int main() {

    std::vector<std::string> imagePaths = {
        "../data/game1_clip1/game1_clip1.mp4",
        "../data/game1_clip2/game1_clip2.mp4",
        "../data/game1_clip3/game1_clip3.mp4",
        "../data/game1_clip4/game1_clip4.mp4",
        "../data/game2_clip1/game2_clip1.mp4",
        "../data/game2_clip2/game2_clip2.mp4",
        "../data/game3_clip1/game3_clip1.mp4",
        "../data/game3_clip2/game3_clip2.mp4",
        "../data/game4_clip1/game4_clip1.mp4",
        "../data/game4_clip2/game4_clip2.mp4"
    };

    for (int i = 0; i < imagePaths.size(); ++i) {

        // Here we have the conversion into frames FRAMES ORIGINAL
        auto [frames, fps] = videoToFrames(imagePaths[i]);
        // PreProcessing Corners
        cv::Mat result_first;
        std::vector<cv::Vec2f> first_detected_lines;
        std::tie(result_first, first_detected_lines) = preProcess(frames.front());

        // Table Corners
        std::vector<cv::Point2f> footage_corners;
        footage_corners = tableDetection(first_detected_lines);

        // Balls Detection
        std::vector<cv::Rect> bboxes_first;
        std::vector<cv::Rect> bboxes_last;

        bboxes_first = ballsDetection(frames.front(), footage_corners);
        bboxes_last = ballsDetection(frames.back(), footage_corners);

        std::vector<BoundingBox> classified_boxes_first;
        std::vector<BoundingBox> classified_boxes_last;

        classified_boxes_first = ballClassification( frames.front(), bboxes_first);
        classified_boxes_last = ballClassification( frames.back(), bboxes_last);
        


    }

    // Ball classification 

    // std::vector<cv::Mat> video_frames = homography_track_balls(frames,TEST,footage_corners);

    // framesToVideo(video_frames, "MOD_"+VIDEO, fps); // TO BE USED ON TRACK OUT

    return 0;
}
