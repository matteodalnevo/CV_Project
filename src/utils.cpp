// MATTEO DAL NEVO - ID: 2087919

// utils.cpp
#include "utils.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <vector>

// Function to show an image
void showImage(const cv::Mat& image, const std::string& windowName) {
    // Create a window
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);

    // Show the image inside the created window
    cv::imshow(windowName, image);

    // Wait for any keystroke in the window
    //cv::waitKey(0);
    // cv::destroyAllWindows();
}

// Function to read bounding boxes from a file
bool readBoundingBoxes(const std::string& filePath, std::vector<BoundingBox>& boundingBoxes) {
    std::ifstream inputFile(filePath);

    if (!inputFile.is_open()) {
        std::cerr << "Unable to open file: " << filePath << std::endl;
        return false;
    }

    std::string line;
    while (getline(inputFile, line)) {
        std::istringstream iss(line);
        BoundingBox bbox;
        if (!(iss >> bbox.box.x >> bbox.box.y >> bbox.box.width >> bbox.box.height >> bbox.ID)) {
            std::cerr << "Error parsing line: " << line << std::endl;
            continue;
        }
        boundingBoxes.push_back(bbox);
    }
    inputFile.close();
    return true;
}

// Function to get color mapping value
cv::Scalar getColorForValue(int value) {
    switch (value) {
        case 0: return cv::Scalar(128, 128, 128);
        case 1: return cv::Scalar(255, 255, 255);
        case 2: return cv::Scalar(0, 0, 0);
        case 3: return cv::Scalar(0, 0, 255);
        case 4: return cv::Scalar(255, 0, 0);
        case 5: return cv::Scalar(0, 255, 0);
        default: return cv::Scalar(0, 0, 0); // Handle unknown values
    }
}

// Function to map from mask to color segmented image
void mapGrayscaleMaskToColorImage(const cv::Mat& grayscaleMask, cv::Mat& colorImage) {
    colorImage = cv::Mat::zeros(grayscaleMask.size(), CV_8UC3);

    for (int i = 0; i < grayscaleMask.rows; ++i) {
        for (int j = 0; j < grayscaleMask.cols; ++j) {
            int value = grayscaleMask.at<uchar>(i, j); // Assuming grayscaleMask is single-channel (CV_8UC1)
            cv::Scalar color = getColorForValue(value);
            colorImage.at<cv::Vec3b>(i, j) = cv::Vec3b(color[0], color[1], color[2]);
        }
    }
}

// Function to plot a Ball's Bounding Box in an image based on the ID

void plotBBox(cv::Mat& image, BoundingBox& bbox) {
    // Create a transparent overlay with the same size as the original image
    cv::Mat overlay = cv::Mat::zeros(image.size(), CV_8UC3);
    // Transparency level
    double alpha;
    // Initialize the color
    cv::Scalar color;
    switch (bbox.ID) {
        case 1:
            color = cv::Scalar(255, 255, 255); // White - white ball
            alpha = 0.2;
            break;
        case 2:
            color = cv::Scalar(20, 20, 20); // Black - black ball
            alpha = 1;
            break;
        case 3:
            color = cv::Scalar(0, 0, 255); // Red - solid ball
            alpha = 0.4;
            break;
        case 4:
            color = cv::Scalar(255, 0, 0); // Blue - striped ball
            alpha = 0.4;
            break;
        default:
            break;
    }

    if (bbox.ID != -1) {
        // Draw the Bounding Box contour on the original image
        cv::rectangle(image, bbox.box, color, 2);
        // Draw a filled rectangle on the overlay
        cv::rectangle(overlay, bbox.box, color, cv::FILLED);
        // Blend the overlay with the original image
        cv::addWeighted(overlay, alpha, image, 1.0, 0.0, image);
    }
    
}

// Function that draw the table's profile in Yellow and the Balls Bounding Box
void outputBBImage(cv::Mat& image, std::vector<cv::Point2f> vertices, std::vector<BoundingBox> classified_boxes) {
    
    // Convert the vector from cv::Point2f to cv::Point
    std::vector<cv::Point> points;
    for (const auto& vertex : vertices) {
        points.push_back(cv::Point(static_cast<int>(vertex.x), static_cast<int>(vertex.y)));
    }

    // Draw the quadrilateral by connecting the vertices
    std::vector<std::vector<cv::Point>> contours;
    contours.push_back(points);
    cv::polylines(image, contours, true, cv::Scalar(0, 255, 255), 2); // Yellow color with thickness 2

    for (auto& bbox : classified_boxes) {
        // Plot the Classified Bounding Box
        plotBBox(image, bbox);
    }
}

cv::Mat segmentation(const cv::Mat img, const std::vector<cv::Point2f> footage_corners, const std::vector<BoundingBox> classified_boxes, cv::Mat hand_mask) {
    // Create a black image of the same size as input image
    cv::Mat dark_image = cv::Mat::zeros(img.size(), CV_8UC1);

    // Create a mask with the same size as the image, initialized to black
    cv::Mat mask = cv::Mat::zeros(img.size(), CV_8UC1);

    // Convert the vector of points to a vector of cv::Point for fillPoly
    std::vector<cv::Point> polygon_points(footage_corners.begin(), footage_corners.end());

    // Fill the polygon area on the mask with white color (255)
    cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{polygon_points}, cv::Scalar(255));

    // Create the green color
    cv::Scalar field_color(5); // BGR format

    // Fill the area inside the polygon in the dark image with green color
    dark_image.setTo(field_color, mask);

    // Draw circles centered in the bounding boxes
    for (const auto& box : classified_boxes) {
        // Calculate the center of the bounding box
        cv::Point center(box.box.x + box.box.width / 2, box.box.y + box.box.height / 2);

        // Calculate the radius as half of the smaller dimension (width or height)
        int radius = std::min(box.box.width, box.box.height) / 2;

        // Create the color of the ball
        cv::Scalar ball_color = box.ID;

        // Draw the circle on the dark image
        cv::circle(dark_image, center, radius, ball_color, -1); // -1 to fill the circle
    }

    // Ensure hand_mask is single-channel and same size as dark_image
    cv::Mat hand_mask_gray;
    if (hand_mask.channels() == 3) {
        cv::cvtColor(hand_mask, hand_mask_gray, cv::COLOR_BGR2GRAY);
    } else {
        hand_mask_gray = hand_mask;
    }
    
    // Normalize hand_mask to be binary (0 or 255)
    cv::threshold(hand_mask_gray, hand_mask_gray, 128, 255, cv::THRESH_BINARY);

    // Create a black image of the same size as dark_image
    cv::Mat black_image = cv::Mat::zeros(dark_image.size(), CV_8UC1);

    // Combine dark_image and black_image using hand_mask
    cv::Mat final_image;
    dark_image.copyTo(final_image, hand_mask_gray); // copy the dark_image where hand_mask is white
    black_image.copyTo(final_image, 255 - hand_mask_gray); // copy the black_image where hand_mask is black

    return final_image;
}
