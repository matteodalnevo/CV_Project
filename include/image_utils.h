// image_utils.h
#ifndef IMAGE_UTILS_H
#define IMAGE_UTILS_H

#include <opencv2/opencv.hpp>
#include <string>

// Function to display an image in a window
void showImage(const cv::Mat& image, const std::string& windowName);

#endif // IMAGE_UTILS_H