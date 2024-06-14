#include "ballDetection.h"
#include <iostream>

// function for calc hist
cv::Mat histogramCalculation(const cv::Mat& image);

// Function for ball detection
cv::Mat ballDetection(const cv::Mat& image) {

    // Convert image to grayscale
    cv::Mat gray;  // Create a grayscale image matrix
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);  // Convert the input image to grayscale
    cv::GaussianBlur(gray, gray, cv::Size(5,5), 1.5, 1.5);  // Apply Gaussian blur to reduce noise
    cv::imshow("Gray image", gray);
    /*
    // Convert to float
    cv::Mat floatImage;
    gray.convertTo(floatImage, CV_32F, 1.0 / 255);

    double gamma = 3;
    // Apply gamma correction
    cv::Mat gammaImage;
    cv::pow(floatImage, gamma, gammaImage);

    // Convert back to 8-bit
    gammaImage.convertTo(gray, CV_8U, 255);

    cv::imshow("Gray image modded", gray);
    */
   
    // detecting circles via Hough transform
    std::vector<cv::Vec3f> circles;  // Vector to store detected circles
    cv::HoughCircles(gray, circles, cv::HOUGH_GRADIENT_ALT, 2, gray.rows/64, 70, 0.9, 0, 15);  // Detect circles using Hough Transform

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

        //cv::Mat dst = image(boundingBox);
        //cv::Mat hist = histogramCalculation(gray(boundingBox));
        //cv::imshow("roi", dst);
        //cv::waitKey(0);

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


cv::Mat histogramCalculation(const cv::Mat& image){
    // Calculate histogram
    cv::Mat hist;
    int histSize = 256; // Number of bins
    float range[] = {0, 256}; // Range of pixel values
    const float* histRange = {range};
    cv::calcHist(&image, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);

    // Plot histogram
    int histWidth = 512;
    int histHeight = 400;
    int binWidth = cvRound(static_cast<double>(histWidth) / histSize);
    cv::Mat histImage(histHeight, histWidth, CV_8UC1, cv::Scalar(255));

    // Normalize the histogram
    cv::normalize(hist, hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());

    // Draw the histogram
    for (int i = 0; i < histSize; i++) {
        cv::rectangle(histImage, cv::Point(binWidth * i, histHeight),
                cv::Point(binWidth * (i + 1), histHeight - cvRound(hist.at<float>(i))),
                cv::Scalar(0), -1, 8, 0);
    }

    // Display histogram
    cv::imshow("Histogram (Bins: " + std::to_string(histSize) + ")", histImage);
    //cv::waitKey(0);

    return histImage;
}