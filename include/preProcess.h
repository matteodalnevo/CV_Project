// DAL NEVO MATTEO - ID: 2087919

// preProcess.h
#ifndef PRE_PROCESS_H
#define PRE_PROCESS_H

#include <opencv2/highgui.hpp>
#include <tuple>
#include <vector>

/**
 * @brief Finds the color of a specific cluster center
 * 
 * This function retrieves the color of a cluster center from the k-means clustering results
 * 
 * @param centers The matrix containing cluster centers as cv::Vec3f
 * @param index The index of the cluster center to retrieve
 * @return The color of the cluster center as cv::Scalar (BGR format)
 */
cv::Scalar findClusterColor(const cv::Mat& centers, int index);

/**
 * @brief Finds the largest clusters by size from k-means clustering labels
 * 
 * This function identifies and sorts clusters by their size, and populates the provided vector with the indices of the largest clusters
 * 
 * @param labels The matrix of cluster labels assigned to each pixel
 * @param clusterCount The total number of clusters
 * @param largestIndices A vector to be populated with indices of the largest clusters
 */
void findLargestClusters(const cv::Mat& labels, int clusterCount, std::vector<int>& largestIndices);

/**
 * @brief Visualizes clusters in an image by coloring each pixel according to its cluster center
 * 
 * This function creates an image where each pixel is colored according to the cluster it belongs to
 * 
 * @param src The source image to be clustered
 * @param labels The matrix of cluster labels assigned to each pixel
 * @param centers The matrix of cluster centers
 * @return The clustered image with pixels colored according to their cluster center
 */
cv::Mat visualizeClusters(const cv::Mat& src, const cv::Mat& labels, const cv::Mat& centers);

/**
 * @brief Calculates the Euclidean distance between two colors
 * 
 * This function computes the Euclidean distance between two BGR color values
 * 
 * @param color1 The first color (BGR format)
 * @param color2 The second color (BGR format)
 * @return The Euclidean distance between the two colors
 */
double calculateDistance(cv::Vec3b color1, cv::Vec3b color2);

/**
 * @brief Creates a mask based on color similarity to a target color
 * 
 * This function generates a binary mask where pixels similar to the target color are set to 255, and all other pixels are set to 0
 * 
 * @param image The input image.
 * @param targetColor The target color for mask creation (BGR format)
 * @param similarityThreshold The maximum Euclidean distance for a pixel to be considered similar
 * @return The binary mask where pixels similar to the target color are white
 */
cv::Mat createColorMask(const cv::Mat& image, cv::Vec3b targetColor, double similarityThreshold);

/**
 * @brief Applies morphological operations (opening and closing) to a binary mask
 * 
 * This function applies morphological opening followed by closing to enhance the mask
 * 
 * @param mask The binary mask to be processed
 * @param openSize The size of the structuring element used for the opening operation
 * @param closeSize The size of the structuring element used for the closing operation
 * @return The processed mask after applying morphological operations
 */
cv::Mat applyMorphology(const cv::Mat& mask, int openSize, int closeSize);

/**
 * @brief Detects lines in an edge-detected image using the Hough Transform
 * 
 * This function detects lines in an image using the Hough Line Transform
 * 
 * @param edges The edge-detected image
 * @param lowerThreshold The lower threshold for the Canny edge detector
 * @param upperThreshold The upper threshold for the Canny edge detector
 * @return A vector of lines detected in the image, represented as cv::Vec2f
 */
std::vector<cv::Vec2f> detectHoughLines(const cv::Mat& edges, double lowerThreshold, double upperThreshold);

/**
 * @brief Main preprocessing function for detecting the table in an image
 * 
 * This function performs clustering, color detection, morphological processing, and line detection to identify the table in an image
 * 
 * @param image The input image to be processed
 * @return A tuple containing the original image and a vector of detected lines (cv::Vec2f)
 */
std::tuple<cv::Mat, std::vector<cv::Vec2f>> preProcess(const cv::Mat& image);


#endif // PRE_PROCESS_H
