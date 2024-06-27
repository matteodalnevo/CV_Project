#include "ballDetection.h"
#include <iostream>

cv::Vec3b computeMedianColor(const cv::Mat& image) {
    std::vector<uchar> blue, green, red;

    // Extract pixel values for each channel, ignoring black pixels
    for (int row = 0; row < image.rows; ++row) {
        for (int col = 0; col < image.cols; ++col) {
            cv::Vec3b color = image.at<cv::Vec3b>(row, col);
            if (color != cv::Vec3b(0, 0, 0)) { // Ignore black pixels
                blue.push_back(color[0]);
                green.push_back(color[1]);
                red.push_back(color[2]);
            }
        }
    }

    // Function to find median of a vector
    auto findMedian = [](std::vector<uchar>& channel) -> uchar {
        size_t n = channel.size();
        std::sort(channel.begin(), channel.end());
        if (n % 2 == 0) {
            return (channel[n / 2 - 1] + channel[n / 2]) / 2;
        } else {
            return channel[n / 2];
        }
    };

    // Compute the median for each channel
    uchar medianBlue = findMedian(blue);
    uchar medianGreen = findMedian(green);
    uchar medianRed = findMedian(red);

    return cv::Vec3b(medianBlue, medianGreen, medianRed);
}


cv::Mat segmentByColor(const cv::Mat& image, const cv::Vec3b& medianColor, int threshold) {
    cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);

    // Create mask based on the threshold range around the median color
    for (int row = 0; row < image.rows; ++row) {
        for (int col = 0; col < image.cols; ++col) {
            cv::Vec3b color = image.at<cv::Vec3b>(row, col);
            if (color != cv::Vec3b(0, 0, 0)) { // Ignore black pixels
                if (abs(color[0] - medianColor[0]) < threshold &&
                    abs(color[1] - medianColor[1]) < threshold &&
                    abs(color[2] - medianColor[2]) < threshold) {
                    mask.at<uchar>(row, col) = 255;
                } else {
                    mask.at<uchar>(row, col) = 0;
                }
            }
        }
    }
    
    return mask;
}

void drawCircles(cv::Mat& image, const std::vector<cv::Vec3f>& circles) {
    for (size_t i = 0; i < circles.size(); ++i) {
        cv::Vec3f circle = circles[i];
        cv::Point center(cvRound(circle[0]), cvRound(circle[1]));
        int radius = cvRound(circle[2]);
        // Draw the circle center
        cv::circle(image, center, 3, cv::Scalar(0, 255, 0), -1);
        // Draw the circle outline
        cv::circle(image, center, radius, cv::Scalar(0, 0, 255), 3);
    }

    // Display the image with circles
    cv::imshow("Detected Circles", image);
}

// Use the corners
cv::Mat ballDetection(const cv::Mat& image, std::vector<cv::Point> vertices) {
    
    // Make a copy of the input image to draw the rectangle on
    cv::Mat result = image.clone();

    // Draw the quadrilateral by connecting the vertices
    std::vector<std::vector<cv::Point>> contours;
    contours.push_back(vertices);
    cv::polylines(result, contours, true, cv::Scalar(0, 255, 255), 2); // Green color with thickness 2
    //cv::imshow("Table Contours", result);

    // Create a mask for the ROI
    cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1); // Initialize mask with zeros (black)

    // Fill the ROI (region of interest) defined by the vertices with white color (255)
    cv::fillPoly(mask, contours, cv::Scalar(255));

    // Create a masked image using the original image and the mask
    cv::Mat maskedImage;
    image.copyTo(maskedImage, mask);

   return maskedImage;
}