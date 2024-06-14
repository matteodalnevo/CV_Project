#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "utils.h"
#include "ballDetection.h"
#include "tableDetection.h"
#include "homography.h"


int main() {

    // Load the dataset game 1
    cv::Mat image_frame_first = cv::imread("../data/game1_clip1/frames/frame_first.png");
    cv::Mat image_frame_last = cv::imread("../data/game1_clip1/frames/frame_last.png");

    /*
    cv::Mat mask_frame_first = cv::imread("../data/game2_clip1/masks/frame_first.png", cv::IMREAD_ANYDEPTH);
    cv::Mat mask_frame_last = cv::imread("../data/game2_clip1/masks/frame_last.png", cv::IMREAD_ANYDEPTH);

    
    std::vector<BoundingBox> bbox_frame_first;

    // Call the function to read bounding boxes
    if (readBoundingBoxes("../data/game1_clip1/bounding_boxes/frame_first_bbox.txt", bbox_frame_first)) {
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

    // Test the ball detection function
    cv::Mat result = ballDetection(image_frame_last);
    
    // Call the function to show an image
    showImage(result, "Game 1 Clip 1");


    
    image_frame_first = cv::imread("../data/game1_clip2/frames/frame_first.png");
    image_frame_last = cv::imread("../data/game1_clip2/frames/frame_last.png");
    result = ballDetection(image_frame_last);
    showImage(result, "Game 1 Clip 2");

    image_frame_first = cv::imread("../data/game1_clip3/frames/frame_first.png");
    image_frame_last = cv::imread("../data/game1_clip3/frames/frame_last.png");
    result = ballDetection(image_frame_last);
    showImage(result, "Game 1 Clip 3");

    image_frame_first = cv::imread("../data/game1_clip4/frames/frame_first.png");
    image_frame_last = cv::imread("../data/game1_clip4/frames/frame_last.png");
    result = ballDetection(image_frame_last);
    showImage(result, "Game 1 Clip 4");

    /*
    image_frame_first = cv::imread("../data/game2_clip1/frames/frame_first.png");
    image_frame_last = cv::imread("../data/game2_clip1/frames/frame_last.png");
    result = ballDetection(image_frame_last);
    showImage(result, "Game 2 Clip 1");

    image_frame_first = cv::imread("../data/game2_clip2/frames/frame_first.png");
    image_frame_last = cv::imread("../data/game2_clip2/frames/frame_last.png");
    result = ballDetection(image_frame_last);
    showImage(result, "Game 2 Clip 2");

    image_frame_first = cv::imread("../data/game3_clip1/frames/frame_first.png");
    image_frame_last = cv::imread("../data/game3_clip1/frames/frame_last.png");
    result = ballDetection(image_frame_last);
    showImage(result, "Game 3 Clip 1");

    image_frame_first = cv::imread("../data/game3_clip2/frames/frame_first.png");
    image_frame_last = cv::imread("../data/game3_clip2/frames/frame_last.png");
    result = ballDetection(image_frame_last);
    showImage(result, "Game 3 Clip 2");

    image_frame_first = cv::imread("../data/game4_clip1/frames/frame_first.png");
    image_frame_last = cv::imread("../data/game4_clip1/frames/frame_last.png");
    result = ballDetection(image_frame_last);
    showImage(result, "Game 4 Clip 1");

    image_frame_first = cv::imread("../data/game4_clip2/frames/frame_first.png");
    image_frame_last = cv::imread("../data/game4_clip2/frames/frame_last.png");
    result = ballDetection(image_frame_last);
    showImage(result, "Game 4 Clip 2");
    */

    return 0;
}