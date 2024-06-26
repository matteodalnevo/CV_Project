#include "ballDetection.h"
#include <iostream>

// Use the corners
cv::Mat ballDetection(const cv::Mat& image, std::vector<cv::Point> vertices) {
    
    // Make a copy of the input image to draw the rectangle on
    cv::Mat result = image.clone();

    // Draw the quadrilateral by connecting the vertices
    std::vector<std::vector<cv::Point>> contours;
    contours.push_back(vertices);
    cv::polylines(result, contours, true, cv::Scalar(0, 255, 255), 2); // Green color with thickness 2
    
    // Create a mask for the ROI
    cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1); // Initialize mask with zeros (black)

    // Fill the ROI (region of interest) defined by the vertices with white color (255)
    cv::fillPoly(mask, contours, cv::Scalar(255));

    // Create a masked image using the original image and the mask
    cv::Mat maskedImage;
    image.copyTo(maskedImage, mask);

    return maskedImage;
}