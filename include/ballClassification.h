// DAL NEVO MATTEO - ID: 2087919

// ballClassification.h
#ifndef BALLCLASSIFICATION_H
#define BALLCLASSIFICATION_H

#include "utils.h"

#include <opencv2/highgui.hpp>
#include <vector>


/**
 * @brief Calculates the percentage of white pixels in an image
 * 
 * This function converts the image to grayscale (if needed), applies a binary threshold, and computes the percentage of white pixels.
 * 
 * @param image The input image
 * @param threshold The threshold value for binary thresholding
 * @return The percentage of white pixels
 */
double calculateWhitePixelPercentage(const cv::Mat& image, const int threshold);

/**
 * @brief Calculates the percentage of black pixels in an image
 * 
 * This function converts the image to grayscale (if needed), applies a binary threshold with inversion, and computes the percentage of black pixels
 * 
 * @param image The input image
 * @param threshold The threshold value for binary thresholding
 * @return The percentage of black pixels
 */
double calculateBlackPixelPercentage(const cv::Mat& image, const int threshold);

/**
 * @brief Adjusts the dimensions of a bounding box
 * 
 * This function enlarges or shrinks the bounding box by a specified shift value
 * 
 * @param bbox The bounding box to adjust
 * @param shift The amount to adjust the bounding box dimensions [Pixels]
 */
void adjustBoundingBox(BoundingBox& bbox, int shift);

/**
 * @brief Computes the indexes of white and black balls based on pixel percentages
 * 
 * This function analyzes the percentage of white and black pixels in bounding boxes to classify them as white or black balls
 * 
 * @param image The input image
 * @param Bbox The vector of bounding boxes to be analyzed
 * @param whiteindex Output index of the bounding box with the highest white percentage
 * @param blackindex Output index of the bounding box with the highest black percentage
 */
void computeWhiteBlackBallIndexes(const cv::Mat& image, std::vector<BoundingBox>& Bbox, int& whiteindex, int& blackindex);

/**
 * @brief Processes a bounding box image to create a mask based on a color range
 * 
 * This function creates a mask for a bounding box based on color similarity and computes the percentage of white pixels in the masked region
 * 
 * @param bbox The bounding box to process
 * @param image The input image
 * @param shift The amount to adjust the bounding box dimensions
 * @param color The target color for masking (BGR format)
 * @param val The range value for color similarity
 * @param white_binary_threshold The threshold value for binary thresholding in the masked image
 * @return The percentage of white pixels in the masked region
 */
double processBoundingBox(BoundingBox& bbox, const cv::Mat& image, const int shift, const cv::Vec3b& color, const int val, const int white_binary_threshold);

/**
 * @brief Classifies balls in the image as solid or stripe balls in a cascade procedure
 * 
 * This function classifies bounding boxes in the image into solid or stripe balls based on pixel percentages and color characteristics
 * 
 * @param image The input image
 * @param Bbox The vector of bounding boxes to classify
 */
void classifySolidStripeBalls(const cv::Mat& image, std::vector<BoundingBox>& Bbox);

/**
 * @brief Main function for ball classification
 * 
 * This function initializes bounding boxes from rectangles, classify white and black balls, classifies solid and stripe balls, and returns the classified bounding boxes
 * 
 * @param image The input image
 * @param Bbox_rect The vector of rectangles representing initial bounding boxes
 * @return A vector of bounding boxes with their classified IDs
 */
std::vector<BoundingBox> ballClassification(cv::Mat& image, const std::vector<cv::Rect>& Bbox_rect);


#endif // BALLCLASSIFICATION_H
