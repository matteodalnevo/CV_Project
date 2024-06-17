#include "ballDetection.h"
#include <iostream>

cv::Scalar findClusterColor(const cv::Mat& centers, int index) {
    cv::Vec3f color = centers.at<cv::Vec3f>(index);
    return cv::Scalar(color[0], color[1], color[2]);
}

void findLargestClusters(const cv::Mat& labels, int clusterCount, std::vector<int>& largestIndices) {
    std::vector<int> counts(clusterCount, 0);
    for (int i = 0; i < labels.rows; i++) {
        counts[labels.at<int>(i)]++;
    }

    // Create a vector of pairs (count, index) and sort by count in descending order
    std::vector<std::pair<int, int>> countIndexPairs;
    for (int i = 0; i < clusterCount; i++) {
        countIndexPairs.push_back(std::make_pair(counts[i], i));
    }
    std::sort(countIndexPairs.rbegin(), countIndexPairs.rend());

    // Extract the indices of the largest clusters
    for (const auto& pair : countIndexPairs) {
        largestIndices.push_back(pair.second);
    }
}

cv::Mat visualizeClusters(const cv::Mat& src, const cv::Mat& labels, const cv::Mat& centers) {
    cv::Mat clusteredImage(src.size(), src.type());
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            int clusterIdx = labels.at<int>(i * src.cols + j);
            cv::Vec3f color = centers.at<cv::Vec3f>(clusterIdx);
            clusteredImage.at<cv::Vec3b>(i, j) = cv::Vec3b(static_cast<uchar>(color[0]), static_cast<uchar>(color[1]), static_cast<uchar>(color[2]));
        }
    }
    return clusteredImage;
}

double calculateDistance(cv::Vec3b color1, cv::Vec3b color2) {
    int b_diff = color1[0] - color2[0];
    int g_diff = color1[1] - color2[1];
    int r_diff = color1[2] - color2[2];
    
    // Euclidean distance formula
    return std::sqrt(static_cast<double>(b_diff * b_diff + g_diff * g_diff + r_diff * r_diff));
}

// Function for ball detection
cv::Mat ballDetection(const cv::Mat& image) {
    
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

    // CLUSTERING FOR DETECTION THE TABLE COLOR

    // Convert the image from BGR to a flat 2D array of pixels
    cv::Mat data = image.reshape(1, image.total());
    data.convertTo(data, CV_32F);

    // Define the number of clusters
    int clusterCount = 2;

    // Apply K-means clustering
    cv::Mat labels, centers;
    cv::kmeans(data, clusterCount, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0), 3, cv::KMEANS_PP_CENTERS, centers);

    // Find the largest clusters
    std::vector<int> largestIndices;
    findLargestClusters(labels, clusterCount, largestIndices);

    // Find the color of the second largest cluster
    cv::Scalar secondLargestColor = findClusterColor(centers, largestIndices[1]);

    // Display the second largest color
    std::cout << "Table color (BGR): [" << secondLargestColor[0] << ", " << secondLargestColor[1] << ", " << secondLargestColor[2] << "]" << std::endl;

    // Visualize the clusters
    cv::Mat clusteredImage = visualizeClusters(image, labels, centers);
    cv::imshow("Clustered Image", clusteredImage);

    // Visualize the second largest color
    cv::Mat color_display(100, 100, CV_8UC3, secondLargestColor);
    cv::imshow("Table Color", color_display);

    // MASKING BASED ON THE TABLE COLOR
    
    // Threshold for color similarity (adjust as needed)
    double similarityThreshold = 70.0; // Threshold
    
    // Create a mask
    cv::Mat mask(image.size(), CV_8UC1, cv::Scalar(0));  // Initialize mask image with zeros

    cv::Vec3b targetColor(secondLargestColor[0], secondLargestColor[1], secondLargestColor[2]);

    // Process each pixel in the image
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            cv::Vec3b pixel = image.at<cv::Vec3b>(y, x);
            
            // Calculate distance between pixel color and target color
            double distance = calculateDistance(pixel, targetColor);
            
            // Check if the distance is within the threshold
            if (distance <= similarityThreshold) {
                // Pixel color is similar to target color, keep it unchanged
                mask.at<uchar>(y, x) = 255;
            } else {
                // Pixel color is not similar, set it to black
                mask.at<uchar>(y, x) = 0;
            }
        }
    }
    
    // Display maske image
    cv::imshow("Mask Image", mask);

    // MORPHOLOGICAL OPERATION FOR TABLE SHAPING

    int morph_size = 1; // Adjust size as needed
    cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                                cv::Size(2 * morph_size + 1, 2 * morph_size + 1),
                                                cv::Point(morph_size, morph_size));

    
    cv::Mat morphResult;
    cv::morphologyEx(mask, morphResult, cv::MORPH_OPEN, element);
    cv::imshow("Morphologically Processed Mask OPEN", morphResult);

    
    morph_size = 10; // Adjust size as needed
    element = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                                cv::Size(2 * morph_size + 1, 2 * morph_size + 1),
                                                cv::Point(morph_size, morph_size));

    // Apply dilation followed by erosion (closing)
    cv::morphologyEx(morphResult, morphResult, cv::MORPH_CLOSE, element);
    cv::imshow("Morphologically Processed Mask CLOSE", morphResult);
    

    // HOUGH LINES DETECTION

    // Canny detector
    cv::Mat edges;
    double upper_threshold = 70;
    double lower_threshold = lower_threshold/2;
    cv::Canny(morphResult, edges, lower_threshold, upper_threshold);
    cv::imshow("Edges", edges);

    // Apply Hough Line Transform
    std::vector<cv::Vec2f> lines;
    cv::HoughLines(edges, lines, 1, CV_PI / 180, 110);
    

    // Draw the lines
    for (size_t i = 0; i < lines.size(); i++) {
        float rho = lines[i][0];
        float theta = lines[i][1];
        double a = cos(theta);
        double b = sin(theta);
        double x0 = a * rho;
        double y0 = b * rho;
        cv::Point pt1(cvRound(x0 + 1000 * (-b)), cvRound(y0 + 1000 * (a)));
        cv::Point pt2(cvRound(x0 - 1000 * (-b)), cvRound(y0 - 1000 * (a)));
        cv::line(image, pt1, pt2, cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
    }
    
   return image;
}