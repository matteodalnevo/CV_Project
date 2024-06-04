#include "ballDetection.h"
#include <iostream>

// Function to show an image
cv::Mat ballDetection(const cv::Mat& image) {

    // Convert image to grayscale
    cv::Mat gray(image.rows, image.cols, CV_8UC1);  // Create a grayscale image matrix
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);  // Convert the input image to grayscale
    cv::GaussianBlur(gray, gray, cv::Size(9, 9), 2, 2);  // Apply Gaussian blur to reduce noise

    cv::imshow("Blurred image", gray);
    // Wait for any keystroke in the window
    cv::waitKey(0);

    // detecting circles via Hough transform
    std::vector<cv::Vec3f> circles;  // Vector to store detected circles
    cv::HoughCircles(gray, circles, cv::HOUGH_GRADIENT, 1, gray.rows/8, 100, 30, 10, 50);  // Detect circles using Hough Transform

    // plotting detected signs
    cv::Mat circle_plot(image.size(), image.type(), cv::Scalar(0, 0, 0));
    for(size_t i=0; i<circles.size(); i++){
        cv::Vec3i c = circles[i];
        cv::Point center = cv::Point(c[0], c[1]);
        int radius = c[2];
        // Draw filled circles with transparent red on the original image
        cv::circle(circle_plot, center, radius, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
    }
    
    // Convert the potential markings to BGR for visualization
    cv::Mat potential_mark = cv::Mat::zeros(image.size(), CV_8UC3);
    potential_mark.setTo(cv::Scalar(0, 0, 255), circle_plot); // Set potential markings to red

    // Merge the original image with the potential balls
    cv::Mat result;
    cv::addWeighted(image, 0.5, potential_mark, 0.5, 0, result);
    
    return result;
}