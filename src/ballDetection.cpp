#include "ballDetection.h"
#include <iostream>

cv::Vec3b computeMedianColor(const cv::Mat& image) {
    std::vector<uchar> blue, green, red;

    // Extract pixel values for each channel, excluding black pixels ([0, 0, 0])
    for (int row = 0; row < image.rows; ++row) {
        for (int col = 0; col < image.cols; ++col) {
            cv::Vec3b bgr = image.at<cv::Vec3b>(row, col);
            if (bgr != cv::Vec3b(0, 0, 0)) { // Skip black pixels
                blue.push_back(bgr[0]);
                green.push_back(bgr[1]);
                red.push_back(bgr[2]);
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


void segmentByColor(const cv::Mat& inputImage, const cv::Vec3b& referenceColor, int T, cv::Mat& mask) {
    // Iterate through each pixel of the input image
    for (int y = 0; y < inputImage.rows; ++y) {
        for (int x = 0; x < inputImage.cols; ++x) {
            // Get the BGR values of the current pixel
            cv::Vec3b pixel = inputImage.at<cv::Vec3b>(y, x);

            // Check if the absolute difference for each channel is within the threshold
            if (std::abs(pixel[0] - referenceColor[0]) <= 2*T &&
                std::abs(pixel[1] - referenceColor[1]) <= 2*T &&
                std::abs(pixel[2] - referenceColor[2]) <= T) {
                mask.at<uchar>(y, x) = 255; // White pixel
            } else {
                mask.at<uchar>(y, x) = 0; // Black pixel
            }
        }
    }
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
    
    // DRAW THE BOX PF THE TABLE

    // Make a copy of the input image to draw the rectangle on
    cv::Mat result = image.clone();

    // Draw the quadrilateral by connecting the vertices
    std::vector<std::vector<cv::Point>> contours;
    contours.push_back(vertices);
    cv::polylines(result, contours, true, cv::Scalar(0, 255, 255), 2); // Green color with thickness 2
    //cv::imshow("Table Contours", result);

    // CUT THE ORIGINAL IMAGE

    // Create a mask for the ROI
    cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1); // Initialize mask with zeros (black)

    // Fill the ROI (region of interest) defined by the vertices with white color (255)
    cv::fillPoly(mask, contours, cv::Scalar(255));

    // Create a masked image using the original image and the mask
    cv::Mat maskedImage;
    image.copyTo(maskedImage, mask);

    int height = maskedImage.rows;
    int part_height = height / 3;

    cv::Mat part1 = maskedImage(cv::Rect(0, 0, image.cols, part_height));
    cv::Mat part2 = maskedImage(cv::Rect(0, part_height, image.cols, part_height));
    cv::Mat part3 = maskedImage(cv::Rect(0, 2 * part_height, image.cols, part_height));

    cv::Vec3b median1 = computeMedianColor(part1);
    
    // Create a grayscale mask image
    cv::Mat segmentedImage(part1.rows, part1.cols, CV_8UC1);
    segmentByColor(part1, median1, 30, segmentedImage);

    return maskedImage;
}