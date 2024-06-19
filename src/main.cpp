#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "utils.h"
#include "ballDetection.h"
#include "tableDetection.h"
#include "homography.h"
#include "preProcess.h"


// Function to load, process, display and save images
void processImages(const std::string& path_first, const std::string& path_last, const std::string& output_prefix, const std::string& clip_desc) {
    cv::Mat image_frame_first = cv::imread(path_first);
    cv::Mat image_frame_last = cv::imread(path_last);
    
    if (!image_frame_first.empty()) {
        //cv::Mat result_first = preProcess(image_frame_first);
        cv::Mat result_first;
        std::vector<cv::Vec2f> first_detected_lines;
        std::tie(result_first, first_detected_lines) = preProcess(image_frame_first);
        std::cout << "FIRST FRAME POINTS..." << std::endl;
        tableDetection(first_detected_lines);
        showImage(result_first, clip_desc + " first");
        // cv::imwrite(output_prefix + "_first.png", result_first);
    } else {
        std::cerr << "Error loading image: " << path_first << std::endl;
    }

    if (!image_frame_last.empty()) {
        // cv::Mat result_last = preProcess(image_frame_last);
        cv::Mat result_last;
        std::vector<cv::Vec2f> last_detected_lines;
        std::tie(result_last, last_detected_lines) = preProcess(image_frame_last);
        std::cout << "FIRST FRAME POINTS..." << std::endl;
        tableDetection(last_detected_lines);
        showImage(result_last, clip_desc + " last");
        // cv::imwrite(output_prefix + "_last.png", result_last);
    } else {
        std::cerr << "Error loading image: " << path_last << std::endl;
    }
}

int main() {
    
    // Game 1 Clips
    processImages("../data/game1_clip1/frames/frame_first.png", "../data/game1_clip1/frames/frame_last.png", "game1clip1", "Game 1 Clip 1");
    
    processImages("../data/game1_clip2/frames/frame_first.png", "../data/game1_clip2/frames/frame_last.png", "game1clip2", "Game 1 Clip 2");
    processImages("../data/game1_clip3/frames/frame_first.png", "../data/game1_clip3/frames/frame_last.png", "game1clip3", "Game 1 Clip 3");
    processImages("../data/game1_clip4/frames/frame_first.png", "../data/game1_clip4/frames/frame_last.png", "game1clip4", "Game 1 Clip 4");
    
    // Game 2 Clips
    processImages("../data/game2_clip1/frames/frame_first.png", "../data/game2_clip1/frames/frame_last.png", "game2clip1", "Game 2 Clip 1");
    processImages("../data/game2_clip2/frames/frame_first.png", "../data/game2_clip2/frames/frame_last.png", "game2clip2", "Game 2 Clip 2");
    
    // Game 3 Clips
    processImages("../data/game3_clip1/frames/frame_first.png", "../data/game3_clip1/frames/frame_last.png", "game3clip1", "Game 3 Clip 1");
    processImages("../data/game3_clip2/frames/frame_first.png", "../data/game3_clip2/frames/frame_last.png", "game3clip2", "Game 3 Clip 2");
    
    // Game 4 Clips
    processImages("../data/game4_clip1/frames/frame_first.png", "../data/game4_clip1/frames/frame_last.png", "game4clip1", "Game 4 Clip 1");
    processImages("../data/game4_clip2/frames/frame_first.png", "../data/game4_clip2/frames/frame_last.png", "game4clip2", "Game 4 Clip 2");
    
    return 0;

    /*
    cv::Mat mask_frame_first = cv::imread("../data/game2_clip1/masks/frame_first.png", cv::IMREAD_ANYDEPTH);
    cv::Mat mask_frame_last = cv::imread("../data/game2_clip1/masks/frame_last.png", cv::IMREAD_ANYDEPTH);

    
    std::vector<BoundingBox> bbox_frame_first;

    // Call the function to read bounding boxes
    /* if (readBoundingBoxes("../data/game1_clip1/bounding_boxes/frame_first_bbox.txt", bbox_frame_first)) {
        // Display the data
        for (const auto& bbox : bbox_frame_first) {
            std::cout << bbox.x << " " << bbox.y << " " << bbox.width << " " << bbox.height << " " << bbox.ID << std::endl;
        }
    }
    
    // Create color image
    cv::Mat colorImage;
    // Call the function to segment the mask
    mapGrayscaleMaskToColorImage(mask_frame_first, colorImage);
    
    // Call the function to show an image
    showImage(colorImage, "image");
    */
}