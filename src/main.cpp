#include <opencv2/opencv.hpp>
#include <iostream>
#include "image_utils.h"


int main() {

    // Load the dataset game 1
    cv::Mat image_frame_first = cv::imread("../data/game1_clip1/frames/frame_first.png");
    cv::Mat image_frame_last = cv::imread("../data/game1_clip1/frames/frame_last.png");

    cv::Mat mask_frame_first = cv::imread("../data/game1_clip1/masks/frame_first.png");
    cv::Mat mask_frame_last = cv::imread("../data/game1_clip1/masks/frame_last.png");


    showImage(image_frame_first, "Display Image");

    return 0;
}