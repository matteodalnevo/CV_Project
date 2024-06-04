#include "ballDetection.h"
#include <iostream>

// Function to show an image
cv::Mat ballDetection(const cv::Mat& image) {

    // Convert image to grayscale
    cv::Mat gray(image.rows, image.cols, CV_8UC1);  // Create a grayscale image matrix
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);  // Convert the input image to grayscale
    cv::GaussianBlur(gray, gray, cv::Size(3, 3), 2, 2);  // Apply Gaussian blur to reduce noise

    // detecting circles via Hough transform
    std::vector<cv::Vec3f> circles;  // Vector to store detected circles
    cv::HoughCircles(gray, circles, cv::HOUGH_GRADIENT, 1, gray.rows/32, 50, 13, 6, 10);  // Detect circles using Hough Transform

    // Create an overlay image
    cv::Mat overlay;
    image.copyTo(overlay);

    // plotting detected signs
    for(size_t i=0; i<circles.size(); i++){
        cv::Vec3i c = circles[i];
        cv::Point center = cv::Point(c[0], c[1]);
        int radius = c[2];
        // Define the bounding box
        cv::Rect boundingBox(center.x - radius, center.y - radius, radius * 2, radius * 2);

        // Draw the bounding box
        cv::rectangle(image, boundingBox, cv::Scalar(0, 0, 255), 1);
        // Fill it
        cv::rectangle(overlay, boundingBox, cv::Scalar(0, 0, 255), -1);

        // std::cout << radius << std::endl;
    }
    
    // Blend the overlay with the original image
    double alpha = 0.5; // Transparency factor
    cv::addWeighted(overlay, alpha, image, 1 - alpha, 0, image);

    return image;
}