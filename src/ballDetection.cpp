#include "ballDetection.h"
#include <iostream>

cv::Vec3b computeMedianColor(const cv::Mat& image) {
    cv::Mat hsvImage;
    cv::cvtColor(image, hsvImage, cv::COLOR_BGR2HSV); // Convert BGR to HSV
    
    std::vector<uchar> hue, saturation, value;

    // Extract pixel values for each channel, ignoring black pixels
    for (int row = 0; row < hsvImage.rows; ++row) {
        for (int col = 0; col < hsvImage.cols; ++col) {
            cv::Vec3b hsv = hsvImage.at<cv::Vec3b>(row, col);
            if (hsv[2] > 0) { // Ignore black pixels
                hue.push_back(hsv[0]);
                saturation.push_back(hsv[1]);
                value.push_back(hsv[2]);
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
    uchar medianHue = findMedian(hue);
    uchar medianSaturation = findMedian(saturation);
    uchar medianValue = findMedian(value);

    return cv::Vec3b(medianHue, medianSaturation, medianValue);
}


cv::Mat segmentByColor(const cv::Mat& image, const cv::Vec3b& medianColorHSV, int hueThreshold, int saturationThreshold, int valueThreshold) {
    cv::Mat hsvImage, mask;
    cv::cvtColor(image, hsvImage, cv::COLOR_BGR2HSV); // Convert BGR to HSV

    // Create mask based on the threshold range around the median HSV color
    cv::inRange(hsvImage,
                cv::Scalar(medianColorHSV[0] - hueThreshold, medianColorHSV[1] - saturationThreshold, medianColorHSV[2] - valueThreshold),
                cv::Scalar(medianColorHSV[0] + hueThreshold, medianColorHSV[1] + saturationThreshold, medianColorHSV[2] + valueThreshold),
                mask);

    // Apply morphological operations to improve shape
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel); // Closing operation to fill gaps

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

    cv::Vec3b medianColor = computeMedianColor(maskedImage);

    // Set HSV threshold values
    int hueThreshold = 20; // Example value, adjust as needed
    int saturationThreshold = 70; // Example value, adjust as needed
    int valueThreshold = 70; // Example value, adjust as needed
    
    // Segment the image based on computed median HSV color
    cv::Mat segmentedMask = segmentByColor(maskedImage, medianColor, hueThreshold, saturationThreshold, valueThreshold);

    

    return segmentedMask;
}