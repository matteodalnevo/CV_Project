// DAL NEVO MATTEO - ID: 2087919

#include "preProcess.h"
#include <iostream>
#include <tuple>
#include <vector>
#include <opencv2/opencv.hpp>

// Function to find the color of a specific cluster center
cv::Scalar findClusterColor(const cv::Mat& centers, int index) {
    cv::Vec3f color = centers.at<cv::Vec3f>(index);
    return cv::Scalar(color[0], color[1], color[2]);
}

// Function to find the largest clusters by size
void findLargestClusters(const cv::Mat& labels, int clusterCount, std::vector<int>& largestIndices) {
    std::vector<int> counts(clusterCount, 0);
    // Count the number of occurences of each cluster label
    for (int i = 0; i < labels.rows; i++) {
        counts[labels.at<int>(i)]++;
    }

    // Sort the clusters by their size and take into consideration also the original index
    std::vector<std::pair<int, int>> countIndexPairs;
    for (int i = 0; i < clusterCount; i++) {
        countIndexPairs.emplace_back(counts[i], i);
    }

    // Sort the pairs in descresing order based on the counts
    std::sort(countIndexPairs.rbegin(), countIndexPairs.rend());

    // Extract the indices of the largest clusters
    for (const auto& pair : countIndexPairs) {
        largestIndices.push_back(pair.second);
    }
}

// Function to visualize clusters in an image
cv::Mat visualizeClusters(const cv::Mat& src, const cv::Mat& labels, const cv::Mat& centers) {
    // Create an image with the same properties as the original one
    cv::Mat clusteredImage(src.size(), src.type());
    // Loop for all the image pixels and create the clusterized image
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            int clusterIdx = labels.at<int>(i * src.cols + j);
            cv::Vec3f color = centers.at<cv::Vec3f>(clusterIdx);
            clusteredImage.at<cv::Vec3b>(i, j) = cv::Vec3b(static_cast<uchar>(color[0]), static_cast<uchar>(color[1]), static_cast<uchar>(color[2]));
        }
    }
    return clusteredImage;
}

// Function to calculate Euclidean distance between two colors
double calculateDistance(cv::Vec3b color1, cv::Vec3b color2) {
    // Compute the difference between the BGR color
    int b_diff = color1[0] - color2[0];
    int g_diff = color1[1] - color2[1];
    int r_diff = color1[2] - color2[2];
    // Return the Euclidean distance
    return std::sqrt(static_cast<double>(b_diff * b_diff + g_diff * g_diff + r_diff * r_diff));
}

// Function to create a mask based on color similarity of the table
cv::Mat createColorMask(const cv::Mat& image, cv::Vec3b targetColor, double similarityThreshold) {
    // Create a black grayscale image
    cv::Mat mask(image.size(), CV_8UC1, cv::Scalar(0));
    // Loop over all the image and set as 255 the pixel only if is similar to the table
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            // Extrapolate the pixel value in (y,x)
            cv::Vec3b pixel = image.at<cv::Vec3b>(y, x);
            // Compute the euclidian distance of the pixel value w.r.t. the table target color
            double distance = calculateDistance(pixel, targetColor);
            if (distance <= similarityThreshold) {
                mask.at<uchar>(y, x) = 255; // modify the pixel of the mask
            }
        }
    }
    return mask;
}

// Function to apply morphological operations
cv::Mat applyMorphology(const cv::Mat& mask, int openSize, int closeSize) {

    // Apply the opening operation with an small ellipse
    cv::Mat morphResult;
    cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                                cv::Size(2 * openSize + 1, 2 * openSize + 1),
                                                cv::Point(openSize, openSize));
    cv::morphologyEx(mask, morphResult, cv::MORPH_OPEN, element);

    // Apply the closening operation with a bigger ellipse
    element = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                        cv::Size(2 * closeSize + 1, 2 * closeSize + 1),
                                        cv::Point(closeSize, closeSize));
    cv::morphologyEx(morphResult, morphResult, cv::MORPH_CLOSE, element);

    return morphResult;
}

// Function to detect Hough lines
std::vector<cv::Vec2f> detectHoughLines(const cv::Mat& edges, double lowerThreshold, double upperThreshold) {
    // Compute canny algorithm for extrapolate the table profile
    cv::Mat edgesDetected;
    cv::Canny(edges, edgesDetected, lowerThreshold, upperThreshold);
    // Compute Hogh lines algorithm for detecting the principal lines
    std::vector<cv::Vec2f> lines;
    cv::HoughLines(edgesDetected, lines, 1, CV_PI / 180, 110);

    return lines;
}

// Main pre processing function for detecting the table
std::tuple<cv::Mat, std::vector<cv::Vec2f>> preProcess(const cv::Mat& image) {

    // Reshape the original image and convert it in CV_32F
    cv::Mat data = image.reshape(1, image.total());
    data.convertTo(data, CV_32F);

    // Compute the k means clustering to the new reshaped image
    int clusterCount = 2; // best number of cluster for detecting color of the table
    cv::Mat labels, centers;
    cv::kmeans(data, clusterCount, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0), 3, cv::KMEANS_PP_CENTERS, centers);

    // Find the Cluster that have more pixels
    std::vector<int> largestIndices;
    findLargestClusters(labels, clusterCount, largestIndices);
    
    // Extrapolated the second cluster BGR color (that is always the second one)
    cv::Scalar secondLargestColor = findClusterColor(centers, largestIndices[1]);
    // Compute the clusterized image
    cv::Mat clusteredImage = visualizeClusters(image, labels, centers);

    // Create the color in a 100x100 box, printable for debugging
    cv::Mat colorDisplay(100, 100, CV_8UC3, secondLargestColor);

    // Create a mask of the table based on the detected color
    double similarityThreshold = 70.0; // best value for color range 
    cv::Mat mask = createColorMask(image, cv::Vec3b(secondLargestColor[0], secondLargestColor[1], secondLargestColor[2]), similarityThreshold);

    // Apply two morfologic operations for enhance the color profile
    int openSize = 1; // best parmeter for opening
    int closeSize = 10; // best parmeter for closening
    cv::Mat morphResult = applyMorphology(mask, openSize, closeSize);

    // Detect the main lines of the table
    double upperThreshold = 70; // best lower threshold for canny
    double lowerThreshold = upperThreshold / 2; // best upper threshold for canny
    std::vector<cv::Vec2f> lines = detectHoughLines(morphResult, lowerThreshold, upperThreshold);

    return std::make_tuple(image, lines);
}