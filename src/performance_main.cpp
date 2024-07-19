// Homography_main.cpp
#include "homography.h"
#include "tableDetection.h"
#include "ballDetection.h"
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

double computeMAPfromStruct(accumulationForAPvalues bigStruct) {
    std::vector<double> resultsVec;

    resultsVec.push_back(computeAP(bigStruct.ious_class1, bigStruct.totalDetects_class1));
    resultsVec.push_back(computeAP(bigStruct.ious_class2, bigStruct.totalDetects_class2));
    resultsVec.push_back(computeAP(bigStruct.ious_class3, bigStruct.totalDetects_class3));
    resultsVec.push_back(computeAP(bigStruct.ious_class4, bigStruct.totalDetects_class4));

    double sum = std::accumulate(resultsVec.begin(), resultsVec.end(), 0.0); 
    return (sum / static_cast<double>(resultsVec.size()));
}

void assignIouForMAP(const vectorsOfIoUStruct IouValsStruct, const groundTruthLengths totalDetections, accumulationForAPvalues &completeStruct) {
    
    completeStruct.ious_class1.insert(completeStruct.ious_class1.end(), IouValsStruct.class1.begin(), IouValsStruct.class1.end());
    completeStruct.ious_class2.insert(completeStruct.ious_class2.end(), IouValsStruct.class2.begin(), IouValsStruct.class2.end());
    completeStruct.ious_class3.insert(completeStruct.ious_class3.end(), IouValsStruct.class3.begin(), IouValsStruct.class3.end());
    completeStruct.ious_class4.insert(completeStruct.ious_class4.end(), IouValsStruct.class4.begin(), IouValsStruct.class4.end());

    completeStruct.totalDetects_class1 += totalDetections.class1;
    completeStruct.totalDetects_class2 += totalDetections.class2;
    completeStruct.totalDetects_class3 += totalDetections.class3;
    completeStruct.totalDetects_class4 += totalDetections.class4;
}

void accumulateIouValues(performanceMIou &iouStructure, std::vector<double> iouVector) {
    iouStructure.class1.push_back(iouVector[0]);
    iouStructure.class2.push_back(iouVector[1]);
    iouStructure.class3.push_back(iouVector[2]);
    
    if (iouVector[3] != -1) {
        iouStructure.class4.push_back(iouVector[3]);
    }
    if (iouVector[4] != -1 ) {
        iouStructure.class5.push_back(iouVector[4]);
    }
    
    iouStructure.class6.push_back(iouVector[5]);

}


cv::Mat segmentation(const cv::Mat img, const std::vector<cv::Point2f> footage_corners, const std::vector<BoundingBox> classified_boxes, cv::Mat hand_mask) {
    // Create a black image of the same size as input image
    cv::Mat dark_image = cv::Mat::zeros(img.size(), CV_8UC1);

    // Create a mask with the same size as the image, initialized to black
    cv::Mat mask = cv::Mat::zeros(img.size(), CV_8UC1);

    // Convert the vector of points to a vector of cv::Point for fillPoly
    std::vector<cv::Point> polygon_points(footage_corners.begin(), footage_corners.end());

    // Fill the polygon area on the mask with white color (255)
    cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{polygon_points}, cv::Scalar(255));

    // Create the green color
    cv::Scalar field_color(5); // BGR format

    // Fill the area inside the polygon in the dark image with green color
    dark_image.setTo(field_color, mask);

    // Draw circles centered in the bounding boxes
    for (const auto& box : classified_boxes) {
        // Calculate the center of the bounding box
        cv::Point center(box.box.x + box.box.width / 2, box.box.y + box.box.height / 2);

        // Calculate the radius as half of the smaller dimension (width or height)
        int radius = std::min(box.box.width, box.box.height) / 2;

        // Create the color of the ball
        cv::Scalar ball_color = box.ID;

        // Draw the circle on the dark image
        cv::circle(dark_image, center, radius, ball_color, -1); // -1 to fill the circle
    }

    // Ensure hand_mask is single-channel and same size as dark_image
    cv::Mat hand_mask_gray;
    if (hand_mask.channels() == 3) {
        cv::cvtColor(hand_mask, hand_mask_gray, cv::COLOR_BGR2GRAY);
    } else {
        hand_mask_gray = hand_mask;
    }
    
    // Normalize hand_mask to be binary (0 or 255)
    cv::threshold(hand_mask_gray, hand_mask_gray, 128, 255, cv::THRESH_BINARY);

    // Create a black image of the same size as dark_image
    cv::Mat black_image = cv::Mat::zeros(dark_image.size(), CV_8UC1);

    // Combine dark_image and black_image using hand_mask
    cv::Mat final_image;
    dark_image.copyTo(final_image, hand_mask_gray); // copy the dark_image where hand_mask is white
    black_image.copyTo(final_image, 255 - hand_mask_gray); // copy the black_image where hand_mask is black

    return final_image;
}

cv::Vec3b computeMedianColor(const cv::Mat& image) {
    std::vector<uchar> blue, green, red;

    // Extract pixel values for each channel, excluding black pixels ([0, 0, 0])
    for (int row = 0; row < image.rows; ++row) {
        for (int col = 0; col < image.cols; ++col) {
            cv::Vec3b bgr = image.at<cv::Vec3b>(row, col);
            if (bgr != cv::Vec3b(0, 0, 0)) { // Skip black pixels
                blue.push_back(bgr[0]);
                green.push_back(bgr[1]);
                red.push_back(bgr[2]);
            }
        }
    }

    // Function to find median of a vector
    auto findMedian = [](std::vector<uchar>& channel) -> uchar {
        size_t n = channel.size();
        std::sort(channel.begin(), channel.end());
        if (n % 2 == 0) {
            return (channel[n / 2 - 1] + channel[n / 2]) / 2;
        } else {
            return channel[n / 2];
        }
    };

    // Compute the median for each channel
    uchar medianBlue = findMedian(blue);
    uchar medianGreen = findMedian(green);
    uchar medianRed = findMedian(red);

    return cv::Vec3b(medianBlue, medianGreen, medianRed);
}

cv::Vec3b ROItable(const cv::Mat& image, std::vector<cv::Point2f> vertices) {

    // Make a copy of the input image to draw the rectangle on
    cv::Mat result = image.clone();
    
    std::vector<cv::Point> points;
    for (const auto& vertex : vertices) {
        points.push_back(cv::Point(static_cast<int>(vertex.x), static_cast<int>(vertex.y)));
    }

    // Draw the quadrilateral by connecting the vertices
    std::vector<std::vector<cv::Point>> contours;
    contours.push_back(points);
    cv::polylines(result, contours, true, cv::Scalar(0, 255, 255), 2); // Yellow color with thickness 2

    //cv::imshow("Table Contours", result);

    // CUT THE ORIGINAL IMAGE

    // Create a mask for the ROI
    cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1); // Initialize mask with zeros (black)

    // Fill the ROI (region of interest) defined by the vertices with white color (255)
    cv::fillPoly(mask, contours, cv::Scalar(255));

    // Create a masked image using the original image and the mask
    cv::Mat maskedImage;
    image.copyTo(maskedImage, mask);
    
    
    cv::Vec3b tableColor = computeMedianColor(maskedImage);
    
    //std::cout << "BGR: " << tableColor[0] << " " << tableColor[1] << " " << tableColor[2] << " " << std::endl;
    
    return tableColor;
}

int main() {

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
performanceMIou iouAccumulator;
accumulationForAPvalues structOfTotalIoUaccumulation;
std::vector<double> segmentationPerformance_class1; 
std::vector<double> segmentationPerformance_class2;
std::vector<double> segmentationPerformance_class3;
std::vector<double> segmentationPerformance_class4;
std::vector<double> segmentationPerformance_class5;
std::vector<double> segmentationPerformance_class6; 

//Vectors that store the complete list of IoU values for each single class
std::vector<double> globalMAPaccumulator_class1;
std::vector<double> globalMAPaccumulator_class2;
std::vector<double> globalMAPaccumulator_class3;
std::vector<double> globalMAPaccumulator_class4;

//Accumulator of number of total ground truth objects for each class
int totalObjects_class1 = 0, totalObjects_class2 = 0, totalObjects_class3 = 0, totalObjects_class4 = 0;

for (int i = 0; i < imagePaths.size(); ++i) {

        // Here we have the conversion into frames FRAMES ORIGINAL
        std::vector<cv::Mat> frames;
        double fps;
        std::tie(frames, fps) = videoToFrames(imagePaths[i]);

        // PreProcessing Corners
        cv::Mat result_first;
        std::vector<cv::Vec2f> first_detected_lines;
        std::tie(result_first, first_detected_lines) = preProcess(frames.front());

        // Table Corners
        std::vector<cv::Point2f> footage_corners;
        footage_corners = tableDetection(first_detected_lines);

        // Color Table
        cv::Vec3b tableColor = ROItable(frames.front(), footage_corners);

        std::vector<cv::Rect> bboxes_first;
        std::vector<cv::Rect> bboxes_last;

        cv::Mat hand_first;
        cv::Mat hand_last;
        
        std::tie(bboxes_first, hand_first) = ballsDetection(frames.front(), footage_corners);
        std::tie(bboxes_last, hand_last) = ballsDetection(frames.back(), footage_corners);

        //Define and compute vector of correct Bounding Boxes with classification
        std::vector<BoundingBox> classified_boxes_first;
        std::vector<BoundingBox> classified_boxes_last;
        classified_boxes_first = ballClassification( frames.front(), bboxes_first, tableColor);
        classified_boxes_last = ballClassification( frames.back(), bboxes_last, tableColor);

        //showImage(frames.front(), "First frame");
        //showImage(frames.back(), "Last frame"); 
        //cv::waitKey(0);

        // Compute segmented image 
        cv::Mat segmentation_first = segmentation(frames.front(),footage_corners, classified_boxes_first, hand_first );
        cv::Mat segmentation_last = segmentation(frames.back(),footage_corners, classified_boxes_last, hand_last);

        //Convert segmented image from gray scale to BGR
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

        //Compute vector of IoU for each class on the segmented images
        std::vector<double> IoU_first = segmentationIoU(segmentation_gt_first, segmentation_first);
        std::vector<double> IoU_last = segmentationIoU(segmentation_gt_last, segmentation_last);

        //Print the values of IoU for each class of the first and last frame of the current clip
        // std::cout << "Segmentation IoU vector for first image " << std::endl;
        // for(int z = 0; z < IoU_first.size(); ++z) {
        //     std::cout << "  " << IoU_first[z];
        // }
        // std::cout << std::endl;
        // std::cout << "Segmentation IoU vector for last image " << std::endl;
        // for(int z = 0; z < IoU_last.size(); ++z) {
        //     std::cout << "  " << IoU_last[z];
        // }
        // std::cout << std::endl;
        // cv::waitKey(0);

        //Accumulate IoU for each class on a struct containing 6 different vectors
        accumulateIouValues(iouAccumulator, IoU_first);
        accumulateIouValues(iouAccumulator, IoU_last); 

        //Compute mean AP over ball localization and classification
        //Extract vector of correct bounding boxes
        std::vector<BoundingBox> groundTruthBoxesFirst, groundTruthBoxesLast;
        readBoundingBoxes(boxesFirstFramePaths[i], groundTruthBoxesFirst);
        readBoundingBoxes(boxesLastFramePaths[i], groundTruthBoxesLast);

        //Define struct where to store IoU values and number of total object to be detected for a single image
        vectorsOfIoUStruct IoUtotals;
        groundTruthLengths lengthTotals;

        //Compute the IoU vectors for all the detections of the first frame
        std::tie(IoUtotals, lengthTotals) = computeVectorsOfIoU(groundTruthBoxesFirst, classified_boxes_first);

        //Accumulate the values for a single image on the complete vector for the first frame
        assignIouForMAP(IoUtotals, lengthTotals, structOfTotalIoUaccumulation);
        globalMAPaccumulator_class1.insert(globalMAPaccumulator_class1.end(), IoUtotals.class1.begin(), IoUtotals.class1.end());
        globalMAPaccumulator_class2.insert(globalMAPaccumulator_class2.end(), IoUtotals.class2.begin(), IoUtotals.class2.end());
        globalMAPaccumulator_class3.insert(globalMAPaccumulator_class3.end(), IoUtotals.class3.begin(), IoUtotals.class3.end());
        globalMAPaccumulator_class4.insert(globalMAPaccumulator_class4.end(), IoUtotals.class4.begin(), IoUtotals.class4.end());

        totalObjects_class1 += lengthTotals.class1;
        totalObjects_class2 += lengthTotals.class2;
        totalObjects_class3 += lengthTotals.class3;
        totalObjects_class4 += lengthTotals.class4;

        //Compute the IoU vectors for all the detections of the first frame
        std::tie(IoUtotals, lengthTotals) = computeVectorsOfIoU(groundTruthBoxesLast, classified_boxes_last);
        //Accumulate the values for a single image on the complete vector for the first frame
        assignIouForMAP(IoUtotals, lengthTotals, structOfTotalIoUaccumulation);
        globalMAPaccumulator_class1.insert(globalMAPaccumulator_class1.end(), IoUtotals.class1.begin(), IoUtotals.class1.end());
        globalMAPaccumulator_class2.insert(globalMAPaccumulator_class2.end(), IoUtotals.class2.begin(), IoUtotals.class2.end());
        globalMAPaccumulator_class3.insert(globalMAPaccumulator_class3.end(), IoUtotals.class3.begin(), IoUtotals.class3.end());
        globalMAPaccumulator_class4.insert(globalMAPaccumulator_class4.end(), IoUtotals.class4.begin(), IoUtotals.class4.end());
 
        totalObjects_class1 += lengthTotals.class1;
        totalObjects_class2 += lengthTotals.class2;
        totalObjects_class3 += lengthTotals.class3;
        totalObjects_class4 += lengthTotals.class4; 

        std::cout << "##Clip " << imagePaths[i] << "          OK" << std::endl;
        std::cout << "-------------------------------------------------------------------------------------------------------" << std::endl;   

    }

    //Computation of mAP value given the complete vectors of value for all the dataset
    std::vector<double> vectorOfAP;
    vectorOfAP.push_back(computeAP(globalMAPaccumulator_class1, totalObjects_class1));
    vectorOfAP.push_back(computeAP(globalMAPaccumulator_class2, totalObjects_class2));
    vectorOfAP.push_back(computeAP(globalMAPaccumulator_class3, totalObjects_class3));
    vectorOfAP.push_back(computeAP(globalMAPaccumulator_class4, totalObjects_class4));
    double sumOfAP = std::accumulate(vectorOfAP.begin(), vectorOfAP.end(), 0.0);
    double mAP_ugly = sumOfAP / vectorOfAP.size();
    double mAP = computeMAPfromStruct(structOfTotalIoUaccumulation);

    //Print on screen the final performance measure
    std::cout << "mAP on the whole dataset is " << mAP << "     The ugly value is " << mAP_ugly << std::endl;

    // std::cout << "Final IoU values for class 4: solid ball" << std::endl;
    // for(int k = 0; k < iouAccumulator.class4.size(); ++k) {
    //     std::cout << iouAccumulator.class4[k] << "      ";
    // }
    // std::cout << std::endl;
    //Computation of final value of mean IoU

    //Computation of the mean Intersection over Union on the whole dataset
    double final_mIoU = finalMIou(iouAccumulator);
    std::cout << "mIoU on the whole dataset is " << final_mIoU << std::endl;

    return 0;
}