#include "performance.h"
#include "utils.h"
#include <iostream>
#include <numeric>

double segmentationMeanIoU(const cv::Mat &groundTruth, const cv::Mat &segmentedImg) {
    std::vector<double> intersezione(6, 0), unione(6, 0);
    std::vector<double> IoU(6, 0.0f);
    for (int k = 0; k < 6; k++){
        //std::cout << "Debug 1" << std::endl;
        for (int i = 0; i < groundTruth.rows; i++) {
            //std::cout << "Debug 2" << std::endl;
            for (int j = 0; j < groundTruth.cols; j++) {
                if ((groundTruth.at<uchar>(i, j) == k) || (segmentedImg.at<uchar>(i, j) == k) && !((groundTruth.at<uchar>(i, j) == k) && (segmentedImg.at<uchar>(i, j) == k) )) {
                    unione[k] +=1;
                }
                if ((groundTruth.at<uchar>(i, j) == k) && (segmentedImg.at<uchar>(i, j) == k) ) {
                    intersezione[k] +=1;
                    //unione[k] += 1;
                }
            }
        }
        IoU[k] = intersezione[k] / unione[k];
        //std::cout << "Unione       (" << k << ") : " << unione[k] << std::endl;
        //std::cout << "Intersezione (" << k << ") : " << intersezione[k] << std::endl;
    }
    double sum = std::accumulate(IoU.begin(), IoU.end(), 0.0);
    double mIoU = sum / static_cast<double>(IoU.size());
    return mIoU; 
}

//Compute the matrix of distances between guesses and ground truth, GT along the rows, guesses along the columns
std::vector<std::vector<double>> plotDistMatrix(std::vector<BoundingBox> boundingBox, std::vector<BoundingBox> groundTruth) {
    int n = boundingBox.size();
    int m = groundTruth.size();
    std::vector<std::vector<double>> costMatrix(n, std::vector<double>(m, 0));

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            cv::Point gt = boxCenter(boundingBox[i]);
            cv::Point bb = boxCenter(groundTruth[j]);
            costMatrix[i][j] = euclideanDist(gt, bb);            
        }
    }
    return costMatrix;
}

cv::Point boxCenter(BoundingBox pt) {
    int x = pt.box.x + (pt.box.width / 2);
    int y = pt.box.y + (pt.box.height / 2);

    return cv::Point(x, y);
}

double euclideanDist(cv::Point a, cv::Point b) {
    return std::sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

void printMatrix(const std::vector<std::vector<double>>& matrix) {
    for (const auto& row : matrix) {
        for (const auto& elem : row) {
            std::cout << elem << "      ";
        }
        std::cout << "|" << std::endl;
    }
}

std::vector<BoundingBox> splitBbSingleClass(std::vector<BoundingBox> bb, int index) {
    std::vector<BoundingBox> single_class;
    for (size_t i = 0; i < bb.size(); ++i) {
        if (bb[i].ID == index) {
            single_class.push_back(bb[i]);
        }
    }
    return single_class;
}

double IoUfromBbRect(BoundingBox bb1, BoundingBox bb2) { 
    cv::Rect intersection = bb1.box & bb2.box;

    if(intersection.area() == 0) {
        return 0.0;
    }
    cv::Rect unionRect = bb1.box | bb2.box;

    double intArea = intersection.area();
    double unionArea = unionRect.area();
    double IoU = intArea / unionArea;

    return IoU;
}

// STANDARD FUNCTION that assign each prediction to a ground truth bounding box,
// outputs a vector that includes the pairings: number of cell is the groundtruth index,
// value of the cell is the Guess index. If -1 means unassigned. Multiple guesses
// assigned to the same GT must be corrected.
/* std::vector<int> pairBoxesIndices(const std::vector<std::vector<double>> distMatrix, std::vector<BoundingBox> groundTruth, std::vector<BoundingBox> myGuess) {
    size_t num_rows = distMatrix.size();
    size_t num_cols = distMatrix[0].size();
    int minPair, minIdx;
    std::vector<int> assignCounter(std::max(num_rows, num_cols), -1);
    std::vector<int> groundTruthAssigned(num_cols);

    //Pair each ground truth with the nearest predicted box
    for(int i = 0; i < num_rows; ++i) {
        minPair = distMatrix[i][0];
        minIdx = 0;
        for(int j = 0; j < num_cols; ++j) {
            if(distMatrix[i][j] < minPair) {
                minPair = distMatrix[i][j];
                minIdx = j;
            }
        }
        assignCounter[i] = minIdx;        
    }
    return assignCounter;
} */

std::vector<int> pairBoxesIndices(const std::vector<std::vector<double>> distMatrix, std::vector<BoundingBox> groundTruth, std::vector<BoundingBox> myGuess) {
    size_t num_rows = distMatrix.size();
    size_t num_cols = distMatrix[0].size();
    std::vector<int> assignCounter(num_rows, -1);
    std::vector<bool> gtAssigned(num_cols, false); // to track assigned ground truths

    // Pair each guess to the closest ground truth
    for (size_t i = 0; i < num_rows; ++i) {
        double minPair = std::numeric_limits<double>::max();
        int minIdx = -1;
        for (size_t j = 0; j < num_cols; ++j) {
            if (distMatrix[i][j] < minPair) {
                minPair = distMatrix[i][j];
                minIdx = j;
            }
        }
        assignCounter[i] = minIdx;
    }

    // Resolve conflicts: if multiple guesses are assigned to the same ground truth, keep the closest
    for (size_t j = 0; j < num_cols; ++j) {
        std::vector<std::pair<double, int>> candidates;
        for (size_t i = 0; i < num_rows; ++i) {
            if (assignCounter[i] == j) {
                candidates.push_back({distMatrix[i][j], i});
            }
        }
        if (!candidates.empty()) {
            std::sort(candidates.begin(), candidates.end());
            for (size_t k = 1; k < candidates.size(); ++k) {
                assignCounter[candidates[k].second] = -1;
            }
        }
    } 

    for(size_t i = 0; i < assignCounter.size(); ++i) {
        //std::cout << "BB " << i << "  ---------------> GT " << assignCounter[i] << std::endl;
    }

    return assignCounter;
}

// std::vector<double> computeVectorIoUFromPairings(std::vector<int> pairingsVector, std::vector<BoundingBox> groundTruth, std::vector<BoundingBox> myGuess) {
//     std::vector<double> vecIoU;
//     
//     std::cout << "Now print vecIoU " << std::endl;
//     for(size_t z = 0; z < pairingsVector.size(); ++z) {
//         if(pairingsVector[z] != -1) {
//             vecIoU.push_back(IoUfromBbRect(myGuess[z], groundTruth[(pairingsVector[z])]));
//         }
//         else vecIoU.push_back(0.0);        
//     }   
//     return vecIoU;
// 
// } 

std::vector<double> computeVectorIoUFromPairings(std::vector<int> pairingsVector, std::vector<BoundingBox> groundTruth, std::vector<BoundingBox> myGuess) {
    std::vector<double> vecIoU;
    
    //std::cout << "Now print vecIoU " << std::endl;
    for (size_t z = 0; z < pairingsVector.size(); ++z) {
        if (pairingsVector[z] != -1) {
            vecIoU.push_back(IoUfromBbRect(myGuess[z], groundTruth[pairingsVector[z]]));
        } else {
            vecIoU.push_back(0.0);
        }
    }
    return vecIoU;
}


//Function that accumulates FP and TP given the vector of IoU for a single class
/* double computeAP(std::vector<double> vectorIoU, std::vector<BoundingBox> groundtruth) {
    float cumTP = 0, cumFP = 0;
    std::vector<float> vectorPrecision(1, 0), vectorRecall(1, 0);
    int totalObjects = groundtruth.size();
    float AP = 0, width = 0, height = 0,  precision, recall;

    for(size_t i = 0; i < vectorIoU.size(); ++i) {
        if(vectorIoU[i] > 0.5) {
            cumTP += 1;
        }
        else cumFP += 1;

        precision = cumTP / (cumFP + cumTP);
        recall = cumTP / totalObjects;

        vectorPrecision.push_back(precision);
        vectorRecall.push_back(recall);

        //std::cout << "Debug " << totalObjects << std::endl;

        std::cout << "cumTP: " << cumTP << "   cumFP: " << cumFP << std::endl;
        std::cout << "Precision: " << precision << "   Recall: " << recall << std::endl;

        width = vectorRecall[(i+1)] - vectorRecall[i];
        height = vectorPrecision[(i+1)];// - vectorPrecision[i];
        AP += (width * height); 
        std::cout << "AP 00: " << AP << std::endl;
    }
    std::cout << "Total objects " << totalObjects << std::endl;
    return AP;
} */

double computeAP(const std::vector<double> vectorIoU, std::vector<BoundingBox> groundtruth) {
    std::vector<float> cumTP(vectorIoU.size(), 0), cumFP(vectorIoU.size(), 0);
    std::vector<float> vectorPrecision, vectorRecall;
    int totalObjects = groundtruth.size();
    
    float AP = 0;
    
    for (size_t i = 0; i < vectorIoU.size(); ++i) {
        if (vectorIoU[i] > 0.5) {
            cumTP[i] = (i == 0) ? 1 : cumTP[i - 1] + 1;
            cumFP[i] = (i == 0) ? 0 : cumFP[i - 1];
        } else {
            cumTP[i] = (i == 0) ? 0 : cumTP[i - 1];
            cumFP[i] = (i == 0) ? 1 : cumFP[i - 1] + 1;
        }

        float precision = cumTP[i] / (cumTP[i] + cumFP[i]);
        float recall = cumTP[i] / totalObjects;

        vectorPrecision.push_back(precision);
        vectorRecall.push_back(recall);
    }
    
    std::vector<float> recallLevels = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
    std::vector<float> interpolatedPrecisions(recallLevels.size(), 0);
    
    for (size_t i = 0; i < recallLevels.size(); ++i) {
        float maxPrecision = 0;
        for (size_t j = 0; j < vectorRecall.size(); ++j) {
            if (vectorRecall[j] >= recallLevels[i]) {
                maxPrecision = std::max(maxPrecision, vectorPrecision[j]);
            }
        }
        interpolatedPrecisions[i] = maxPrecision;
    }
    
    for (const auto& precision : interpolatedPrecisions) {
        AP += precision;
    }
    
    AP /= recallLevels.size();
    //std::cout << "AP 99: " << AP << std::endl;

    return AP;
} 


double boxesMeanAP(std::vector<BoundingBox> groundTruth, std::vector<BoundingBox> outAlgo) {
    std::vector<BoundingBox> bb_gt_1 = splitBbSingleClass(groundTruth, 1);
    std::vector<BoundingBox> bb_gt_2 = splitBbSingleClass(groundTruth, 2);
    std::vector<BoundingBox> bb_gt_3 = splitBbSingleClass(groundTruth, 3);
    std::vector<BoundingBox> bb_gt_4 = splitBbSingleClass(groundTruth, 4);

    std::vector<BoundingBox> bb_out_1 = splitBbSingleClass(outAlgo, 1);
    std::vector<BoundingBox> bb_out_2 = splitBbSingleClass(outAlgo, 2);
    std::vector<BoundingBox> bb_out_3 = splitBbSingleClass(outAlgo, 3);
    std::vector<BoundingBox> bb_out_4 = splitBbSingleClass(outAlgo, 4);

    std::vector<double> IoU;
    IoU.push_back(APfromSingleBbClass(bb_gt_1, bb_out_1));
    IoU.push_back(APfromSingleBbClass(bb_gt_2, bb_out_2));
    IoU.push_back(APfromSingleBbClass(bb_gt_3, bb_out_3));
    IoU.push_back(APfromSingleBbClass(bb_gt_4, bb_out_4));                                                                                                                            

    return std::accumulate(IoU.begin(), IoU.end(), 0.0) / IoU.size();
}

double APfromSingleBbClass(std::vector<BoundingBox> groundTruth, std::vector<BoundingBox> myGuess) {
    std::vector<std::vector<double>> costMatrix = plotDistMatrix(myGuess, groundTruth);

    //std::cout << "Now print the distances matrix" << std::endl;
    //printMatrix(costMatrix);
    //cv::waitKey(0);

    std::vector<int> vettorePairings = pairBoxesIndices(costMatrix, myGuess, groundTruth);
    std::vector<double> vettoreIoU = computeVectorIoUFromPairings(vettorePairings, groundTruth, myGuess);
    double AP = computeAP(vettoreIoU, groundTruth);
    std::cout << "AP: " << AP << std::endl;
    //cv::waitKey(0);

    return AP;
}