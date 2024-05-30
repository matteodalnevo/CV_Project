// utils.h
#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <string>

struct BoundingBox {
    int x, y, width, height, ID;
};

// Function to display an image in a window
void showImage(const cv::Mat& image, const std::string& windowName);

// Function to read bounding boxes from a file
bool readBoundingBoxes(const std::string& filePath, std::vector<BoundingBox>& boundingBoxes);

// Function to get color mapping value
cv::Scalar getColorForValue(int value);

// Function to map from mask to color segmented image
void mapGrayscaleMaskToColorImage(const cv::Mat& grayscaleMask, cv::Mat& colorImage);

#endif // UTILS_H