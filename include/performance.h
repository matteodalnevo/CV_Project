// ballDetection.h
#ifndef PERFORMANCE_H
#define PERFORMANCE_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include "utils.h"

/** @brief Compute the metric mIoU on the segmented images

 */
double segmentationMeanIoU(const cv::Mat &groundTruth, const cv::Mat &segmentedImg);

/** @brief Compute intersection over union between two bounding boxes epressed as cv::Rect

 */
//double segmentationMeanIoU(BoundingBox bb1, BoundingBox bb2);

std::vector<int> pairBoxesIndices(std::vector<std::vector<double>> distMatrix, std::vector<BoundingBox> groundTruth, std::vector<BoundingBox> myGuess);

std::vector<double> computeVectorIoUFromPairings(std::vector<int> pairingsVector, std::vector<BoundingBox> groundTruth, std::vector<BoundingBox> myGuess);

std::vector<BoundingBox> splitBbSingleClass(std::vector<BoundingBox> bb, int index);

void printMatrix(const std::vector<std::vector<double>>& matrix); 

double euclideanDist(cv::Point a, cv::Point b);

cv::Point boxCenter(BoundingBox pt);

std::vector<std::vector<double>> plotDistMatrix(std::vector<BoundingBox> boundingBox, std::vector<BoundingBox> groundTruth);

double IoUfromBbRect(BoundingBox bb1, BoundingBox bb2);

double computeAP(std::vector<double> vectorIoU, std::vector<BoundingBox> groundtruth);

double APfromSingleBbClass(std::vector<BoundingBox> groundTruth, std::vector<BoundingBox> myGuess);

double boxesMeanAP(std::vector<BoundingBox> groundTruth, std::vector<BoundingBox> outAlgo);

#endif // PERFORMANCE_H