// PREVEDELLO AARON ID: 2089401

#include "homography_tracking.h"
#include "tableDetection.h"
#include "ball_hand_detection.h"
#include "utils.h"
#include "preProcess.h"
#include "ballClassification.h"
#include "performance.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/tracking.hpp>
#include <iostream>
#include <vector>
#include <tuple>
#include <fstream>
#include <sstream>
#include <string>
#include <numeric>

int main(int argc, char* argv[]) {

    // Verify that no arguments are passed to the function
    if (argc != 1 ) {
        std::cout << "\nError: you cannot pass any argument to this executable" << std::endl;
        return 1;
    }

    // List of paths for video clips
    std::vector<std::string> imagePaths = {
        "../data/game1_clip1/game1_clip1.mp4",
        "../data/game1_clip2/game1_clip2.mp4",
        "../data/game1_clip3/game1_clip3.mp4",
        "../data/game1_clip4/game1_clip4.mp4",
        "../data/game2_clip1/game2_clip1.mp4",
        "../data/game2_clip2/game2_clip2.mp4",
        "../data/game3_clip1/game3_clip1.mp4",
        "../data/game3_clip2/game3_clip2.mp4",
        "../data/game4_clip1/game4_clip1.mp4",
        "../data/game4_clip2/game4_clip2.mp4"
    };

    // List of paths for masks of first frames
    std::vector<std::string> groundTruthFirstPaths = {
        "../data/game1_clip1/masks/frame_first.png",
        "../data/game1_clip2/masks/frame_first.png",
        "../data/game1_clip3/masks/frame_first.png",
        "../data/game1_clip4/masks/frame_first.png",
        "../data/game2_clip1/masks/frame_first.png",
        "../data/game2_clip2/masks/frame_first.png",
        "../data/game3_clip1/masks/frame_first.png",
        "../data/game3_clip2/masks/frame_first.png",
        "../data/game4_clip1/masks/frame_first.png",
        "../data/game4_clip2/masks/frame_first.png"
    };

    // List of paths for masks of last frames
    std::vector<std::string> groundTrutLastPaths = {
        "../data/game1_clip1/masks/frame_last.png",
        "../data/game1_clip2/masks/frame_last.png",
        "../data/game1_clip3/masks/frame_last.png",
        "../data/game1_clip4/masks/frame_last.png",
        "../data/game2_clip1/masks/frame_last.png",
        "../data/game2_clip2/masks/frame_last.png",
        "../data/game3_clip1/masks/frame_last.png",
        "../data/game3_clip2/masks/frame_last.png",
        "../data/game4_clip1/masks/frame_last.png",
        "../data/game4_clip2/masks/frame_last.png"
    };

    // List of paths for the boxes of first frames
    std::vector<std::string> boxesFirstFramePaths = {
        "../data/game1_clip1/bounding_boxes/frame_first_bbox.txt",
        "../data/game1_clip2/bounding_boxes/frame_first_bbox.txt",
        "../data/game1_clip3/bounding_boxes/frame_first_bbox.txt",
        "../data/game1_clip4/bounding_boxes/frame_first_bbox.txt",
        "../data/game2_clip1/bounding_boxes/frame_first_bbox.txt",
        "../data/game2_clip2/bounding_boxes/frame_first_bbox.txt",
        "../data/game3_clip1/bounding_boxes/frame_first_bbox.txt",
        "../data/game3_clip2/bounding_boxes/frame_first_bbox.txt",
        "../data/game4_clip1/bounding_boxes/frame_first_bbox.txt",
        "../data/game4_clip2/bounding_boxes/frame_first_bbox.txt"
    };

    // List of paths for the boxes of first frames
    std::vector<std::string> boxesLastFramePaths = {
        "../data/game1_clip1/bounding_boxes/frame_last_bbox.txt",
        "../data/game1_clip2/bounding_boxes/frame_last_bbox.txt",
        "../data/game1_clip3/bounding_boxes/frame_last_bbox.txt",
        "../data/game1_clip4/bounding_boxes/frame_last_bbox.txt",
        "../data/game2_clip1/bounding_boxes/frame_last_bbox.txt",
        "../data/game2_clip2/bounding_boxes/frame_last_bbox.txt",
        "../data/game3_clip1/bounding_boxes/frame_last_bbox.txt",
        "../data/game3_clip2/bounding_boxes/frame_last_bbox.txt",
        "../data/game4_clip1/bounding_boxes/frame_last_bbox.txt",
        "../data/game4_clip2/bounding_boxes/frame_last_bbox.txt"
    };

//Vectors that store the complete list of IoU values from the segmented images
performanceMIou globalIouAccumulatiorSegmentation;
std::vector<double> segmentationPerformance_class1, segmentationPerformance_class2, segmentationPerformance_class3, segmentationPerformance_class4, segmentationPerformance_class5, segmentationPerformance_class6; 

//Vectors that store the complete list of IoU values for each single class
std::vector<double> globalMAPaccumulator_class1, globalMAPaccumulator_class2, globalMAPaccumulator_class3, globalMAPaccumulator_class4;

//Accumulator of number of total ground truth objects for each class
int totalObjects_class1 = 0, totalObjects_class2 = 0, totalObjects_class3 = 0, totalObjects_class4 = 0;

// Iteration along each image of the whole dataset
for (int i = 0; i < imagePaths.size(); ++i) {

        // Here we have the conversion into frames FRAMES ORIGINAL
        std::vector<cv::Mat> frames;
        double fps;
        std::tie(frames, fps) = videoToFrames(imagePaths[i]);

        // PreProcessing Corners
        cv::Mat result_first;
        std::vector<cv::Vec2f> first_detected_lines;
        std::tie(result_first, first_detected_lines) = preProcess(frames.front());

        // Obtain Table Corners
        std::vector<cv::Point2f> footage_corners;
        footage_corners = tableDetection(first_detected_lines);

        // Define vectors and images where to store output of balls & hand detection
        std::vector<cv::Rect> bboxes_first;
        std::vector<cv::Rect> bboxes_last;
        cv::Mat hand_first;
        cv::Mat hand_last;
        
        // Execute balls and hand detection and store outputs
        std::tie(bboxes_first, hand_first) = ballsHandDetection(frames.front(), footage_corners);
        std::tie(bboxes_last, hand_last) = ballsHandDetection(frames.back(), footage_corners);

        //Define and compute vector of correct Bounding Boxes with classification
        std::vector<BoundingBox> classified_boxes_first;
        std::vector<BoundingBox> classified_boxes_last;
        classified_boxes_first = ballClassification( frames.front(), bboxes_first);
        classified_boxes_last = ballClassification( frames.back(), bboxes_last);

        //showImage(frames.front(), "First frame");
        //showImage(frames.back(), "Last frame"); 
        //cv::waitKey(0);

        // Compute segmented image 
        cv::Mat segmentation_first = segmentation(frames.front(),footage_corners, classified_boxes_first, hand_first );
        cv::Mat segmentation_last = segmentation(frames.back(),footage_corners, classified_boxes_last, hand_last);

        //Map the segmented image from gray scale to BGR
        cv::Mat first_col, last_col;
        mapGrayscaleMaskToColorImage( segmentation_first, first_col);
        mapGrayscaleMaskToColorImage( segmentation_last, last_col);

        // cv::imshow("Our segmentation First", first_col);
        // cv::imshow("Our segmentation Last", last_col);

        // cv::waitKey(0);

        //Extract ground truth masks 
        cv::Mat segmentation_gt_first = cv::imread(groundTruthFirstPaths[i], cv::IMREAD_ANYDEPTH);
        cv::Mat segmentation_gt_last  = cv::imread(groundTrutLastPaths[i], cv::IMREAD_ANYDEPTH);

        //Convert ground truth masks from gray scale to BGR
        cv::Mat result_segm_first, result_segm_last;
        mapGrayscaleMaskToColorImage( segmentation_gt_first, result_segm_first);
        mapGrayscaleMaskToColorImage( segmentation_gt_last, result_segm_last);

        // cv::imshow("Ground Truth First", result_segm_first);
        // cv::imshow("Ground Truth Last", result_segm_last);

        // FIRST METRIC: mean Intersection over Union computed on segmented masks

        // Structure of accumulator value of segmentation performance on the single clip
        performanceMIou singleClipAccumulator;

        //Compute vector of IoU for each class on the segmented images
        std::vector<double> IoU_first = segmentationIoU(segmentation_gt_first, segmentation_first);
        std::vector<double> IoU_last = segmentationIoU(segmentation_gt_last, segmentation_last);

        // Accumulate segmentation performances on each image
        accumulateIouValues(singleClipAccumulator, IoU_first);
        accumulateIouValues(singleClipAccumulator, IoU_last);

        // Compute mIoU on the two images of the single clip
        double single_clip_IoU = finalMIou(singleClipAccumulator);

        // Optional for debugging: Print the values of IoU for each class of the first and last frame of the current clip
        // std::cout << "Segmentation IoU vector for first image " << std::endl;
        // for(int z = 0; z < IoU_first.size(); ++z) {
        //     std::cout << "  " << IoU_first[z];
        // }
        // std::cout << "\n" << std::endl;
        // std::cout << "Segmentation IoU vector for last image " << std::endl;
        // for(int z = 0; z < IoU_last.size(); ++z) {
        //     std::cout << "  " << IoU_last[z];
        // }
        // std::cout << "\n" << std::endl;
        //cv::waitKey(0);

        //Accumulate IoU for each class on a struct containing 6 different vectors
        accumulateIouValues(globalIouAccumulatiorSegmentation, IoU_first);
        accumulateIouValues(globalIouAccumulatiorSegmentation, IoU_last); 

        // END OF FIRST METRIC

        //Compute mean AP over ball localization and classification
        //Extract vector of correct bounding boxes
        std::vector<BoundingBox> groundTruthBoxesFirst, groundTruthBoxesLast;
        readBoundingBoxes(boxesFirstFramePaths[i], groundTruthBoxesFirst);
        readBoundingBoxes(boxesLastFramePaths[i], groundTruthBoxesLast);

        // SECOND METRIC: mean Average Precision omputed on vectors of BoundingBoxes

        //Define struct where to store IoU values and number of total object to be detected for a single image
        vectorsOfIoUStruct IoUtotals;
        groundTruthLengths lengthTotals;

        // Define structs to store results of mAP for the two images of a single clip
        std::vector<double> singleClipAccForMap_class1, singleClipAccForMap_class2, singleClipAccForMap_class3, singleClipAccForMap_class4; 
        int singleClipGT_class1 = 0, singleClipGT_class2 = 0, singleClipGT_class3 = 0, singleClipGT_class4 = 0;

        //Compute the IoU vectors for all the detections of the first frame
        std::tie(IoUtotals, lengthTotals) = computeVectorsOfIoU(groundTruthBoxesFirst, classified_boxes_first);

        //Accumulate the values for a single image on the complete vector for the first frame
        globalMAPaccumulator_class1.insert(globalMAPaccumulator_class1.end(), IoUtotals.class1.begin(), IoUtotals.class1.end());
        globalMAPaccumulator_class2.insert(globalMAPaccumulator_class2.end(), IoUtotals.class2.begin(), IoUtotals.class2.end());
        globalMAPaccumulator_class3.insert(globalMAPaccumulator_class3.end(), IoUtotals.class3.begin(), IoUtotals.class3.end());
        globalMAPaccumulator_class4.insert(globalMAPaccumulator_class4.end(), IoUtotals.class4.begin(), IoUtotals.class4.end());
        totalObjects_class1 += lengthTotals.class1;
        totalObjects_class2 += lengthTotals.class2;
        totalObjects_class3 += lengthTotals.class3;
        totalObjects_class4 += lengthTotals.class4;

        // Accumulate values for MAP computations on single first frame
        singleClipAccForMap_class1.insert(singleClipAccForMap_class1.end(), IoUtotals.class1.begin(), IoUtotals.class1.end());
        singleClipAccForMap_class2.insert(singleClipAccForMap_class2.end(), IoUtotals.class2.begin(), IoUtotals.class2.end());
        singleClipAccForMap_class3.insert(singleClipAccForMap_class3.end(), IoUtotals.class3.begin(), IoUtotals.class3.end());
        singleClipAccForMap_class4.insert(singleClipAccForMap_class4.end(), IoUtotals.class4.begin(), IoUtotals.class4.end());
        singleClipGT_class1 += lengthTotals.class1;
        singleClipGT_class1 += lengthTotals.class2; 
        singleClipGT_class1 += lengthTotals.class3;
        singleClipGT_class1 += lengthTotals.class4;

        //Compute the IoU vectors for all the detections of the first frame
        std::tie(IoUtotals, lengthTotals) = computeVectorsOfIoU(groundTruthBoxesLast, classified_boxes_last);

        //Accumulate the values for a single image on the complete vector for the first frame
        globalMAPaccumulator_class1.insert(globalMAPaccumulator_class1.end(), IoUtotals.class1.begin(), IoUtotals.class1.end());
        globalMAPaccumulator_class2.insert(globalMAPaccumulator_class2.end(), IoUtotals.class2.begin(), IoUtotals.class2.end());
        globalMAPaccumulator_class3.insert(globalMAPaccumulator_class3.end(), IoUtotals.class3.begin(), IoUtotals.class3.end());
        globalMAPaccumulator_class4.insert(globalMAPaccumulator_class4.end(), IoUtotals.class4.begin(), IoUtotals.class4.end());
        totalObjects_class1 += lengthTotals.class1;
        totalObjects_class2 += lengthTotals.class2;
        totalObjects_class3 += lengthTotals.class3;
        totalObjects_class4 += lengthTotals.class4;

        // Accumulate values for computation of single image MAP performance
        singleClipAccForMap_class1.insert(singleClipAccForMap_class1.end(), IoUtotals.class1.begin(), IoUtotals.class1.end());
        singleClipAccForMap_class2.insert(singleClipAccForMap_class2.end(), IoUtotals.class2.begin(), IoUtotals.class2.end());
        singleClipAccForMap_class3.insert(singleClipAccForMap_class3.end(), IoUtotals.class3.begin(), IoUtotals.class3.end());
        singleClipAccForMap_class4.insert(singleClipAccForMap_class4.end(), IoUtotals.class4.begin(), IoUtotals.class4.end());
        singleClipGT_class1 += lengthTotals.class1;
        singleClipGT_class1 += lengthTotals.class2; 
        singleClipGT_class1 += lengthTotals.class3;
        singleClipGT_class1 += lengthTotals.class4;

        // Computation of MAP for a single videoclip
        std::vector<double> singleClipAP;
        singleClipAP.push_back(computeAP(singleClipAccForMap_class1, singleClipGT_class1));
        singleClipAP.push_back(computeAP(singleClipAccForMap_class2, singleClipGT_class2)); 
        singleClipAP.push_back(computeAP(singleClipAccForMap_class3, singleClipGT_class3));
        singleClipAP.push_back(computeAP(singleClipAccForMap_class4, singleClipGT_class4));
        double sumOfSingleClipAP = std::accumulate(singleClipAP.begin(), singleClipAP.end(), 0.0);
        double singleClipMAP = sumOfSingleClipAP / singleClipAP.size();

        // END OF SECOND METRIC

        std::cout << "Clip " << imagePaths[i] << "          OK\n" << std::endl;
        std::cout << "Current clip mIoU: " << single_clip_IoU << std::endl;
        std::cout << "Current clip mAP:  " << singleClipMAP << std::endl;
        std::cout << "------------------------------------------------------" << std::endl;   

    }

    //Computation of mAP value given the complete vectors of value for all the dataset
    std::vector<double> vectorOfAP;
    vectorOfAP.push_back(computeAP(globalMAPaccumulator_class1, totalObjects_class1));
    vectorOfAP.push_back(computeAP(globalMAPaccumulator_class2, totalObjects_class2));
    vectorOfAP.push_back(computeAP(globalMAPaccumulator_class3, totalObjects_class3));
    vectorOfAP.push_back(computeAP(globalMAPaccumulator_class4, totalObjects_class4));
    double sumOfAP = std::accumulate(vectorOfAP.begin(), vectorOfAP.end(), 0.0);
    double mAP = sumOfAP / vectorOfAP.size();

    //Print on screen the final performance measure
    std::cout << "mAP on the whole dataset is " << mAP << std::endl;

    //Computation of the mean Intersection over Union on the whole dataset
    double final_mIoU = finalMIou(globalIouAccumulatiorSegmentation);
    std::cout << "mIoU on the whole dataset is " << final_mIoU << std::endl;

    return 0;
}
