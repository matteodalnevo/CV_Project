#include "performance.h"
#include "utils.h"
#include <iostream>
#include <numeric>


// IOU ON SEMGENTED IMAGE

std::vector<double> segmentationIoU(const cv::Mat &groundTruth, const cv::Mat &segmentedImg) {
    // Define vectors where to store temp values of intersection and union for each class, and vector of outputs
    std::vector<double> intersezione(6, 0), unione(6, 0);
    std::vector<double> IoU(6, 0.0f);

    // Iteration over each class
    for (int k = 0; k < 6; k++){

        // Iteration over each pixel of the image
        for (int i = 0; i < groundTruth.rows; i++) {

            for (int j = 0; j < groundTruth.cols; j++) {

                // Check if the pixel belongs to the class k on one of the two images
                if ((groundTruth.at<uchar>(i, j) == k) || (segmentedImg.at<uchar>(i, j) == k) && !((groundTruth.at<uchar>(i, j) == k) && (segmentedImg.at<uchar>(i, j) == k) )) {
                    unione[k] +=1;
                }
                // Check if the pixel belongs to class k on both the images
                if ((groundTruth.at<uchar>(i, j) == k) && (segmentedImg.at<uchar>(i, j) == k) ) {
                    intersezione[k] +=1;
                    // unione[k] += 1;
                }
            }
        }
        
        // Check if we are dividing by zero, namely the class is present in the image
        if(unione[k] > 0) {
            IoU[k] = intersezione[k] / unione[k];
        } 
        
        // Otherwise set the value of IoU to -1, later will be discarded
        else IoU[k] = -1;
        //std::cout << "Unione       (" << k << ") : " << unione[k] << std::endl;
        //std::cout << "Intersezione (" << k << ") : " << intersezione[k] << std::endl;
    }
    return IoU; 
}


// DISTANCE MATRIX

std::vector<std::vector<double>> plotDistMatrix(const std::vector<BoundingBox> &boundingBox, const std::vector<BoundingBox> &groundTruth) {
    
    // Define matrix dimensions and empty matrix
    int n = boundingBox.size();
    int m = groundTruth.size();
    std::vector<std::vector<double>> costMatrix(n, std::vector<double>(m, 0));

    // Iterate along vector of detected bounding boxes
    for (size_t i = 0; i < n; ++i) {
        // Iterate along vector of ground truth bounding boxes
        for (size_t j = 0; j < m; ++j) {

            // Compute coordinates of the boxes center
            cv::Point gt = boxCenter(boundingBox[i]);
            cv::Point bb = boxCenter(groundTruth[j]);

            // Compute distance between the two points
            costMatrix[i][j] = euclideanDist(gt, bb);            
        }
    }

    // Return total matrix of distances between each pair of boxes
    return costMatrix;
}


// BOXES CENTERS

cv::Point boxCenter(const BoundingBox pt) {

    // Compute coordinate of the center of the bounding box
    int x = pt.box.x + (pt.box.width / 2);
    int y = pt.box.y + (pt.box.height / 2);

    return cv::Point(x, y);
}


// EUCLIDEAN DISTANCE

double euclideanDist(const cv::Point a, const cv::Point b) {
    return std::sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}


// PRINT MATRIX ON SCREEN

void printMatrix(const std::vector<std::vector<double>>& matrix) {
    // Iterate along each element of the matrix
    for (const auto& row : matrix) {
        for (const auto& elem : row) {
            // Print values on a row
            std::cout << elem << "      ";
        }
        // Go to next row
        std::cout << "|" << std::endl;
    }
}


// EXTRACT A VECTOR OF BOXES FOR A SINGLE CLASS

std::vector<BoundingBox> splitBbSingleClass(const std::vector<BoundingBox> &bb, const int index) {

    // Define empty vector to be returned
    std::vector<BoundingBox> single_class;

    // Iterate over the input vector 
    for (size_t i = 0; i < bb.size(); ++i) {

        // Extract the boundingBox if it is of the desired class
        if (bb[i].ID == index) {
            single_class.push_back(bb[i]);
        }
    }

    // Check if the vector is empty, namely no detections for this class
    if (single_class.empty()) {

        // Print on screen that no object is detected for this class
        // std::cout << "No detections for the class " << index << std::endl;
        
        // Create a fake boundingBox that will be discarded in the next functions
        BoundingBox false_val;
        false_val.box = cv::Rect(0, 0, 0, 0);
        // Value of a fake class, in order to detect and discard this value later
        const int fake_ID = 7;
        false_val.ID = fake_ID;
        single_class.push_back(false_val);
    }

    return single_class;
}


// COMPUTE IoU BETWEEN TWO BOUNDING BOXES   

double IoUfromBbRect(const BoundingBox bb1, const BoundingBox bb2) { 
    // Compute intersection of the two cv::Rect objects and its area
    cv::Rect intersection = bb1.box & bb2.box; 
    double intArea = intersection.area();

    // Check if the area of intersection is zero
    if(intArea == 0) {
        return 0.0;
    }

    // Compute the union between the two cv::Rect objects and its area
    cv::Rect unionRect = bb1.box | bb2.box;
    double unionArea = unionRect.area();

    // Compute value of the IoU    
    double IoU = intArea / unionArea;

    return IoU;
}


// PAIR EACH DETECTED BOX TO A GROUND TRUTH BOX

std::vector<int> pairBoxesIndices(const std::vector<std::vector<double>> distMatrix, const std::vector<BoundingBox> groundTruth, const std::vector<BoundingBox> myGuess) {
    
    // Define number of rows (predicted boxes) and columns (ground truth boxes)
    size_t num_rows = distMatrix.size();
    size_t num_cols = distMatrix[0].size();
    
    // Vector that stores index of closest ground truth box, for each predicted box
    std::vector<int> assignCounter(num_rows, -1);

    // Auxiliary vector to track the assigned ground truths
    std::vector<bool> gtAssigned(num_cols, false); 

    // Pair each guess to the closest ground truth, iterate along vector of predictions
    for (size_t i = 0; i < num_rows; ++i) {

        // Define variable to store minimum distance and index of the closest ground truth box
        double minPair = std::numeric_limits<double>::max();
        int minIdx = -1;

        // Iterate over all the ground truth boxes to find the closest predicted box
        for (size_t j = 0; j < num_cols; ++j) {

            // If the current distance is less than the minimum found so far, update the pairing
            if (distMatrix[i][j] < minPair) {
                minPair = distMatrix[i][j];
                minIdx = j;
            }
        } 

        // Store the index of closest ground truth box
        assignCounter[i] = minIdx;
    }

    // Resolve conflicts in the case in which multiple guesses are assigned to the same ground truth: keep the closest box
    // Iterate along ground truth boxes
    for (size_t j = 0; j < num_cols; ++j) {

        // Vector to store pairs of (distance, predicted box indx) for all the predicted boxes assigned to the current ground truth box 
        std::vector<std::pair<double, int>> candidates;


        for (size_t i = 0; i < num_rows; ++i) {
            
            
            if (assignCounter[i] == j) {
                
                // Insert the current prediction as candidate for pairing
                candidates.push_back({distMatrix[i][j], i});
            }
        }

        // If there are multiple candidates for the current ground truth box
        if (!candidates.empty()) {

            // Sort the candidates based on distances
            std::sort(candidates.begin(), candidates.end());

            // Keep the closest predicted box and unassign the rest
            for (size_t k = 1; k < candidates.size(); ++k) {
                assignCounter[candidates[k].second] = -1;
            }
        }
    } 
    
    // Optional print of the assignments for debugging
    // for(size_t i = 0; i < assignCounter.size(); ++i) {
    //     std::cout << "BB " << i << "  ---------------> GT " << assignCounter[i] << std::endl;
    // }

    return assignCounter;
}


//COMPUTE VECTOR OF IOU VALUES 

std::vector<double> computeVectorIoUFromPairings(const std::vector<int> pairingsVector, const std::vector<BoundingBox> groundTruth, const std::vector<BoundingBox> myGuess) {
    
    // Define output vector
    std::vector<double> vecIoU;
    
    //Iterate along vector of pairings 
    for (size_t z = 0; z < pairingsVector.size(); ++z) {

        //If the pairing exists
        if (pairingsVector[z] != -1) {

            // Compute IoU value and append the value to the output vector 
            vecIoU.push_back(IoUfromBbRect(myGuess[z], groundTruth[pairingsVector[z]]));
        } 
        else {

            // Otherwise if the prediction is not paired, append IoU = 0
            vecIoU.push_back(0.0);
        }
    }

    return vecIoU;
}


// COMPUTE AVERAGE PRECISION FOR A SINGLE CLASS ON ALL THE IMAGES

double computeAP(const std::vector<double> &vectorIoU, const int totalObjects) {
    
    // Initialize cumulative true positive and false positive vectors
    int cumTP = 0, cumFP = 0;

    // Initialize vectors to store precision and recall values
    std::vector<double> vectorPrecision, vectorRecall;

    // Define the constant threshold on the IoU to assign true and false positives
    const int IoU_Threshold = 0.5;
    
    // Variable to store the Average Precision 
    double AP = 0;
    
    // Iterate over the vector of IoU values 
    for (size_t i = 0; i < vectorIoU.size(); ++i) {

        // If Iou value is -1 skip this detection
        if(vectorIoU[i] == -1 ) {
            // std::cout << "IoU = -1: continue" << std::endl;
            continue;
        }

        // If IoU is greather than threshold, prediction is considered as true positive
        else if (vectorIoU[i] > IoU_Threshold) {
            // Increment cumulative true and false positive count
            cumTP += 1;
        } 

        // If IoU is less or equal than threshold, prediction is considered as false positive
        else {
            // Increment cumulative true and false positive count
            cumFP += 1;
        }

        // Compute precision and recall values
        double precision = static_cast<double>(cumTP) / (cumTP + cumFP);
        double recall = static_cast<double>(cumTP) / totalObjects;

        // Store the values into a vector
        vectorPrecision.push_back(precision);
        vectorRecall.push_back(recall);
    }
    

    // Define recall levels for interpolation using PASCAL VOC 11-point interpolation, and vector of interpolated precisions for each recall level
    std::vector<double> recallLevels = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
    std::vector<double> interpolatedPrecisions(recallLevels.size(), 0);
    

    // Loop over each recall level to calculate interpolated precision
    for (size_t i = 0; i < recallLevels.size(); ++i) {
        // Initialize maximum precision for the current recall level
        double maxPrecision = 0;

        // Loop over all recall values to find the maximum precision for recall >= current recall level
        for (size_t j = 0; j < vectorRecall.size(); ++j) {
            if (vectorRecall[j] >= recallLevels[i]) {

                maxPrecision = std::max(maxPrecision, vectorPrecision[j]);
            }
        }

        // Store the maximum precision found
        interpolatedPrecisions[i] = maxPrecision;
    }
    
    // Sum all the interpolated precision values
    for (const auto& precision : interpolatedPrecisions) {
        AP += precision;

    }

    // Compute the Average Precision value    
    AP /= recallLevels.size();
    
    // Optional: Print the value of AP 
    //std::cout << "AP value: " << AP << std::endl;

    return AP;
} 


// COMPLETE COMPUTATION OF IOU VECTORS FOR A SINGLE IMAGE

std::tuple<vectorsOfIoUStruct, groundTruthLengths> computeVectorsOfIoU(const std::vector<BoundingBox> &groundTruth, const std::vector<BoundingBox> &outAlgo) {
    
    // Optional: Print on screen all the classification labels
    //std::cout << "Print of classification labels" << std::endl;
    //for(int k = 0; k < outAlgo.size(); ++k) {
    //    std::cout << outAlgo[k].ID << "     ";
    //}
    //std::cout << std::endl;

    // Extract a vector for each class, for the predictions and the ground truth boxes
    std::vector<BoundingBox> bb_gt_1 = splitBbSingleClass(groundTruth, 1);
    std::vector<BoundingBox> bb_gt_2 = splitBbSingleClass(groundTruth, 2);
    std::vector<BoundingBox> bb_gt_3 = splitBbSingleClass(groundTruth, 3);
    std::vector<BoundingBox> bb_gt_4 = splitBbSingleClass(groundTruth, 4);

    std::vector<BoundingBox> bb_out_1 = splitBbSingleClass(outAlgo, 1);
    std::vector<BoundingBox> bb_out_2 = splitBbSingleClass(outAlgo, 2);
    std::vector<BoundingBox> bb_out_3 = splitBbSingleClass(outAlgo, 3);
    std::vector<BoundingBox> bb_out_4 = splitBbSingleClass(outAlgo, 4);

    //Compute the vector of IoU for each class separately and store into the struct
    vectorsOfIoUStruct collectionOfIoU;
    collectionOfIoU.class1 = singleClassVectorIoU(bb_gt_1, bb_out_1);
    collectionOfIoU.class2 = singleClassVectorIoU(bb_gt_2, bb_out_2);
    collectionOfIoU.class3 = singleClassVectorIoU(bb_gt_3, bb_out_3);
    collectionOfIoU.class4 = singleClassVectorIoU(bb_gt_4, bb_out_4);

    //Compute the number of ground truth for each class separately and store into the struct
    groundTruthLengths collectionOfLenghts;
    collectionOfLenghts.class1 = bb_gt_1.size();
    collectionOfLenghts.class2 = bb_gt_2.size();  
    collectionOfLenghts.class3 = bb_gt_3.size();
    collectionOfLenghts.class4 = bb_gt_4.size();

    // Return struct of the vectors of IoU values, and struct of number of ground truth objects, one value for each class
    return std::make_tuple(collectionOfIoU, collectionOfLenghts);
}

//Compute the vector of IoU values for a single class
std::vector<double> singleClassVectorIoU(const std::vector<BoundingBox> &groundTruth, const std::vector<BoundingBox> &myGuess) {
    
    // Fake class ID in order to detect a class with no prediction
    const int fake_id = 7;

    // If the vector has this fake classification, return IoU = -1 that will be skipped during the AP computation
    if((myGuess.size() == 1) && (myGuess[0].ID == fake_id)) {
        const std::vector<double> fake_val = {-1};
        return fake_val;
    }

    //Compute the matrix of distances between boxes for a single class
    std::vector<std::vector<double>> costMatrix = plotDistMatrix(myGuess, groundTruth);

    //Pair the boxes based on the minimum distance and compute the final vector of IoU    
    std::vector<int> vettorePairings = pairBoxesIndices(costMatrix, myGuess, groundTruth);
    std::vector<double> vettoreIoU = computeVectorIoUFromPairings(vettorePairings, groundTruth, myGuess);

    return vettoreIoU;
}

// AVERAGE OF ALL THE IOU RESULTS ON SEGMENTED IMAGES

double finalMIou(const performanceMIou &iouStructure) {

    // Define vector where to store the mean of IoU of all the image for a single class
    std::vector<double> meanForEachClass;

    // Compute the mean IoU for each class separately
    meanForEachClass.push_back(std::accumulate(iouStructure.class1.begin(), iouStructure.class1.end(), 0.0) / iouStructure.class1.size());
    meanForEachClass.push_back(std::accumulate(iouStructure.class2.begin(), iouStructure.class2.end(), 0.0) / iouStructure.class2.size());
    meanForEachClass.push_back(std::accumulate(iouStructure.class3.begin(), iouStructure.class3.end(), 0.0) / iouStructure.class3.size());
    meanForEachClass.push_back(std::accumulate(iouStructure.class4.begin(), iouStructure.class4.end(), 0.0) / iouStructure.class4.size());
    meanForEachClass.push_back(std::accumulate(iouStructure.class5.begin(), iouStructure.class5.end(), 0.0) / iouStructure.class5.size());
    meanForEachClass.push_back(std::accumulate(iouStructure.class6.begin(), iouStructure.class6.end(), 0.0) / iouStructure.class6.size());

    // Compute total mean 
    double final_perf = (std::accumulate(meanForEachClass.begin(), meanForEachClass.end(), 0.0) / meanForEachClass.size());

    return final_perf;
}

// INSERT IOU OF THE CURRENT SEGMENTED IMAGE ON THE STRUCTURE

void accumulateIouValues(performanceMIou &iouStructure, std::vector<double> iouVector) {

    // Push back the values into the vectors, one value for each class
    iouStructure.class1.push_back(iouVector[0]);
    iouStructure.class2.push_back(iouVector[1]);
    iouStructure.class3.push_back(iouVector[2]);
    
    // For solid and striped balls check that the Iou is not -1, 
    // that would mean that that class is not present nor in the ground truth segmented mask, nor in our classification
    if (iouVector[3] != -1) {
        iouStructure.class4.push_back(iouVector[3]);
    }
    if (iouVector[4] != -1 ) {
        iouStructure.class5.push_back(iouVector[4]);
    }
    
    iouStructure.class6.push_back(iouVector[5]);

}