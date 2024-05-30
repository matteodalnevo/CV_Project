// image_utils.cpp
#include "image_utils.h"

void showImage(const cv::Mat& image, const std::string& windowName) {
    // Create a window
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

    // Show the image inside the created window
    cv::imshow(windowName, image);

    // Wait for any keystroke in the window
    cv::waitKey(0);
}
