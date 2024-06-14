// utils.cpp
#include "utils.h"
#include <fstream>
#include <sstream>
#include <iostream>

// Function to show an image
void showImage(const cv::Mat& image, const std::string& windowName) {
    // Create a window
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

    // Show the image inside the created window
    cv::imshow(windowName, image);

    // Wait for any keystroke in the window
    cv::waitKey(0);
    cv::destroyAllWindows();
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
        if (!(iss >> bbox.x >> bbox.y >> bbox.width >> bbox.height >> bbox.ID)) {
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
