#include "ballDetection.h"
#include <iostream>

// Usse the corners
cv::Mat ballDetection(const cv::Mat& image, std::vector<cv::Point> vertices) {
    
    // Make a copy of the input image to draw the rectangle on
    cv::Mat result = image.clone();

    // Draw the quadrilateral by connecting the vertices
    std::vector<std::vector<cv::Point>> contours;
    contours.push_back(vertices);
    cv::polylines(result, contours, true, cv::Scalar(0, 255, 255), 3); // Green color with thickness 2

    return result;
}