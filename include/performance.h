// Prevedello Aaron

#ifndef PERFORMANCE_H
#define PERFORMANCE_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include "utils.h"

/** @brief Struct containing vector of resulting IoU over all the images, one vector for each class.
 *  Used for accumulating IoU results before computing final mean value 
 */
struct performanceMIou {
    std::vector<double> class1;
    std::vector<double> class2;
    std::vector<double> class3;
    std::vector<double> class4;
    std::vector<double> class5;
    std::vector<double> class6;
};

/** @brief Struct to accumulate values of ground truth boxes on the images, one value for each class.
 *  Useful to compute final mAP result.
 */
struct groundTruthLengths {
    int class1;
    int class2;
    int class3;
    int class4;
};

/** @brief Struct to accumulate IoU results of all the prediction on the images, one vector for each class.
 *  Useful to accumulate all the IoU values before computing precision-recall curve and AP for each class separately.
 */
struct vectorsOfIoUStruct {
    std::vector<double> class1;
    std::vector<double> class2;
    std::vector<double> class3;
    std::vector<double> class4;
};

/** @brief Compute the vector of Intersection over Union values for the six classes on the images
 * @param groundTruth reference segmented image
 * @param segmentedImg predicted segmented image
 * @return A vector containing six values of IoU, one for each class
 */
std::vector<double> segmentationIoU(const cv::Mat &groundTruth, const cv::Mat &segmentedImg);

/** @brief Given matrix of distances pair each predicted box to a ground truth
 *  @param distMatrix matrix of distances between ground truth and predicted boxes
 *  @return vector of pairings indeces, index of vector is prediction box, element of the vector is the ground truth box
 */
std::vector<int> pairBoxesIndices(const std::vector<std::vector<double>> distMatrix, const std::vector<BoundingBox> groundTruth, const std::vector<BoundingBox> myGuess);

/** @brief Given the vector of predictions and paired ground truth, compute vector of IoU for each prediction
 *  @param pairingsVector Vector of indices of the ground truth box assigned to each prediction
 *  @param groundTruth vector of the ground truth boxes
 *  @param myGuess vector of predicted bounding boxes
 *  @return a vector of IoU values, one for each prediction box
 */
std::vector<double> computeVectorIoUFromPairings(const std::vector<int> pairingsVector, const std::vector<BoundingBox> groundTruth, const std::vector<BoundingBox> myGuess);

/** @brief Extract a vector of bounding boxes of the desired class (values from 1 to 4).
 *  If there are no boxes for the specified class, return a vector with a single dummy box classified with the fake 
 *  id=7, so that it will be discarded later.
 * @param bb vector of bounding boxes to be elaborated
 */
std::vector<BoundingBox> splitBbSingleClass(const std::vector<BoundingBox> &bb, const int index);

/** @brief plot a matrix on scree
 * @param matrix matrix to be printed
 */
void printMatrix(const std::vector<std::vector<double>>& matrix); 

/** @brief Compute the euclidean distance between to cv::Point
 * @param a first cv::Point
 * @param b second cv::Point
 * @return double distance between the two points
 */
double euclideanDist(const cv::Point a,const cv::Point b);

/** @brief Compute the center of a BoundingBox object
 * @param pt BoundingBox struct
 * @return cv::Point coordinate of the center 
 */
cv::Point boxCenter(const BoundingBox pt);

/** @brief Compute the matrix of distances between guesses and ground truth, GT along the rows, guesses along the columns
 * @param boundingBox vector of predicted bounding boxes
 * @param groundTruth vector of ground truth bounding boxes 
 * @return a matrix of distances between centers of each pair of boxes
 */
std::vector<std::vector<double>> plotDistMatrix(const std::vector<BoundingBox> &boundingBox, const std::vector<BoundingBox> &groundTruth);

/** @brief compute IoU value for boundingBoxes objects expressed as cv::Rect
 * @param bb1 first BoundingBox
 * @param bb2 second BoundingBox 
 * @return IoU value 
 */
double IoUfromBbRect(const BoundingBox bb1, const BoundingBox bb2);

/** @brief Given the total vector of IoU values for a single class, compute AP on the whole set of images
 *  @param vectorIoU vector of IoU values for each prediction of a single class, on all the images of the dataset
 *  @param totalObjects number of ground truth balls of a single class to be detected, on all the images of the dataset
 *  @return Average Precision on the single class: integral of the precisio recall curve
 */
double computeAP(const std::vector<double> &vectorIoU, const int totalObjects);

/** @brief Compute the vector of predictions IoU given a single pair of vectors (ground truth, predictions)  
 *  @param groundTruth vector of ground thrut boxes of a single class
 *  @param myGuess vector of predicted boxes of a single class
 *  @return Single vector representing the IoU of each prediction of a single class
 */
std::vector<double> singleClassVectorIoU(const std::vector<BoundingBox> &groundTruth, const std::vector<BoundingBox> &myGuess);

/** @brief Given the total boxes of an image (containing all the classes), splits into single classes and compute vectors of Iou separately, 
 *  then return also number of objects to be detected for each class
 *  @param gruondTruth vector of correct bounding boxes
 *  @param outAlgo vector of estimated boxes
 *  @return Two structs, containing the vectors of IoU values, one for each class, and the number of ground truth boxes, respectively
 */
std::tuple<vectorsOfIoUStruct, groundTruthLengths> computeVectorsOfIoU(const std::vector<BoundingBox> &groundTruth, const std::vector<BoundingBox> &outAlgo);

/** @brief Given the structure containing the performance result on segmented images, divided by class, compute the final value of mIoU 
 *  @param iouStructure structure containing six vectors, each one containing the IoU value of a single class for a single image
 *  @return finale performance result mIoU
 */
double finalMIou(const performanceMIou &iouStructure);

/** @brief Insert the values of iouVector into the structure that stores the values of all the images
 * @param iouStructure Structure where to append the vectors
 * @param iouVector vector of Iou values to append
 */
void accumulateIouValues(performanceMIou &iouStructure, std::vector<double> iouVector);

/** @brief Accumulate IoU values and ground truth lengths for a single image
 *  @param globalAccumulator Struct where to insert the actual values
 *  @param 
 */
//void accumulateIoUAndGT (AccumulatorForMAP& globalAccumulator, const vectorsOfIoUStruct& IoUtotals, const groundTruthLengths& lengthTotals);

#endif // PERFORMANCE_H