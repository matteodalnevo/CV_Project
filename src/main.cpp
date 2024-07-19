// Dal Nevo Matteo

#include "homography_tracking.h"
#include "tableDetection.h"
#include "ball_hand_detection.h"
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

void processVideo(const std::string &videoPath, int index = -1) {
    std::cout << "\nANALYSING: " << videoPath.substr(8, 11) << std::endl;

    // Here we have the conversion into frames FRAMES ORIGINAL
    std::vector<cv::Mat> frames;
    double fps;
    std::tie(frames, fps) = videoToFrames(videoPath);

    std::cout << "VIDEO EXTRAPOLATION: DONE" << std::endl;

    // PreProcessing Corners
    cv::Mat result_first;
    std::vector<cv::Vec2f> first_detected_lines;
    std::tie(result_first, first_detected_lines) = preProcess(frames.front());

    // Table Corners
    std::vector<cv::Point2f> footage_corners;
    footage_corners = tableDetection(first_detected_lines);

    std::cout << "TABLE DETECTION: DONE" << std::endl;

    // Balls Detection and Hand Segmentation
    std::vector<cv::Rect> bboxes_first;
    std::vector<cv::Rect> bboxes_last;

    cv::Mat hand_first;
    cv::Mat hand_last;
    
    std::tie(bboxes_first, hand_first) = ballsHandDetection(frames.front(), footage_corners);
    std::tie(bboxes_last, hand_last) = ballsHandDetection(frames.back(), footage_corners);

    std::cout << "BALLS DETECTION and HAND SEGMENTATION: DONE" << std::endl;

    // Balls Classification
    std::vector<BoundingBox> classified_boxes_first;
    std::vector<BoundingBox> classified_boxes_last;

    classified_boxes_first = ballClassification(frames.front(), bboxes_first);
    classified_boxes_last = ballClassification(frames.back(), bboxes_last);

    std::cout << "BALLS CLASSIFICATION: DONE" << std::endl;

    // Creation of the video
    std::vector<cv::Mat> video_frames = homography_track_balls(frames, footage_corners, classified_boxes_first, classified_boxes_last);
    std::string outputFileName = index >= 0 ? "Result_" + std::to_string(index) + ".mp4" : "Result.mp4";
    framesToVideo(video_frames, outputFileName, fps); // TO BE USED ON TRACK OUT

    std::cout << "VIDEO SAVING: DONE" << std::endl;

    // Segmentation
    cv::Mat segmentation_first = segmentation(frames.front(), footage_corners, classified_boxes_first, hand_first);
    cv::Mat segmentation_last = segmentation(frames.back(), footage_corners, classified_boxes_last, hand_last);

    cv::Mat first_col, last_col;
    mapGrayscaleMaskToColorImage(segmentation_first, first_col);
    mapGrayscaleMaskToColorImage(segmentation_last, last_col);

    std::cout << "SEGMENTATION: DONE" << std::endl;
    std::cout << "END" << std::endl;

    // Output images
    outputBBImage(frames.front(), footage_corners, classified_boxes_first);
    outputBBImage(frames.back(), footage_corners, classified_boxes_last);

    // Shows the results
    showImage(frames.front(), "Bounding Boxes first frame");
    showImage(frames.back(), "Bounding Boxes last frame");
    showImage(first_col, "Segmentation first frame");
    showImage(last_col, "Segmentation last frame");
    showImage(video_frames.back(), "Game Trajectory last frame");

    //cv::destroyAllWindows();
}

int main(int argc, char* argv[]) {
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

    if (argc > 1) {
        // Argument provided, process the single video file
        std::string videoPath = argv[1];
        processVideo(videoPath);
        cv::waitKey(0);
    } else {
        // No argument provided, loop through the predefined list of paths
        for (int i = 0; i < imagePaths.size(); ++i) {
            processVideo(imagePaths[i], i);
            std::cout << "\nPress any key to proceed to the next clip " << std::endl;
            cv::waitKey(0);
        }
    }

    return 0;
}