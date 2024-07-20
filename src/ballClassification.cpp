// DAL NEVO MATTEO - ID: 2087919

#include "ballClassification.h"
#include <iostream>
#include "utils.h"

// Function to calculate the percentage of white pixels in an image
double calculateWhitePixelPercentage(const cv::Mat& image, const int threshold) {

    // Convert to grayscale if not already
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image;
    }

    // Apply a binary threshold to get a binary image
    cv::Mat binary;
    cv::threshold(gray, binary, threshold, 255, cv::THRESH_BINARY);

    // Optional for debugging
    // cv::imshow("bin", binary);
    // cv::waitKey(0);

    // Count the white pixels
    int whitePixelCount = cv::countNonZero(binary);
    
    // Calculate the total number of pixels
    int totalPixels = image.rows * image.cols;

    // Calculate the percentage of white pixels
    double whitePixelPercentage = (static_cast<double>(whitePixelCount) / totalPixels) * 100.0;

    return whitePixelPercentage;
}

// Function to calculate the percentage of black pixels in an image
double calculateBlackPixelPercentage(const cv::Mat& image, const int threshold) {
    // Convert to grayscale if not already
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image;
    }

    // Apply a binary threshold to get a binary image
    cv::Mat binary;
    cv::threshold(gray, binary, threshold, 255, cv::THRESH_BINARY_INV); // Adjust the threshold value if needed

    // Optional for debugging
    // cv::imshow("gray", gray);
    // showImage(binary, "bin");
    // cv::waitKey(0);

    // Count the black pixels (which are white in the binary image)
    int blackPixelCount = cv::countNonZero(binary);

    // Calculate the total number of pixels
    int totalPixels = image.rows * image.cols;

    // Calculate the percentage of black pixels
    double blackPixelPercentage = (static_cast<double>(blackPixelCount) / totalPixels) * 100.0;

    return blackPixelPercentage;
}

// Function to increase or decreased the bounding box dimension
void adjustBoundingBox(BoundingBox& bbox, int shift) {
    bbox.box.x -= shift;
    bbox.box.y -= shift;
    bbox.box.width += shift * 2;
    bbox.box.height += shift * 2;
}

// Function to compute white and black ball indexes
void computeWhiteBlackBallIndexes(const cv::Mat& image, std::vector<BoundingBox>& Bbox, int& whiteindex, int& blackindex) {
    // Initilize to zero the percentage of black and with pixels
    double maxWhitePercentage = 0.0;
    double maxBlackPercentage = 0.0;
    
    for (int i = 0; i < Bbox.size(); i++) {
    	// Adjust the bounding box (increase them)
        const int shift = 3; // value for enlarging the Bounding Box for better metrics computation
        adjustBoundingBox(Bbox[i], shift);
        
        // Crop the ball ROI
        cv::Mat ballROI = image(Bbox[i].box);
        
        // Compute the metrics for classification
        const int white_binary_threshold = 220;  // good parameters for high white pixels
        const int black_binary_threshold = 40; // good parameters for low black pixels
        
        double whitePixelPercentage = calculateWhitePixelPercentage(ballROI, white_binary_threshold);
        double blackPixelPercentage = calculateBlackPixelPercentage(ballROI, black_binary_threshold);
        
        // Select the whitest ball
        if (whitePixelPercentage > maxWhitePercentage) {
            maxWhitePercentage = whitePixelPercentage;
            whiteindex = i;
        }
        // Select the darkest ball
        if (blackPixelPercentage + (100 - whitePixelPercentage) > maxBlackPercentage) {
            maxBlackPercentage = blackPixelPercentage + (100 - whitePixelPercentage);
            blackindex = i;
        }
        
        // Adjust the bounding box (reduce them as normal)
        adjustBoundingBox(Bbox[i], -shift);
    }
}

// Function that process the Bounding Box image creating a mask based on a color range
double processBoundingBox(BoundingBox& bbox, const cv::Mat& image, const int shift, const cv::Vec3b& color, const int val, const int white_binary_threshold) {
    // Adjust the bounding box (Increase)
    adjustBoundingBox(bbox, shift);
    // Crop the ball
    cv::Mat ballROI = image(bbox.box);

    // Initialize lower and upper color range
    cv::Scalar lower_color(color[0] - val, color[1] - val, color[2] - val);
    cv::Scalar upper_color(color[0] + val, color[1] + val, color[2] + val);

    // Masking the image component based on the color
    cv::Mat mask;
    cv::inRange(ballROI, lower_color, upper_color, mask);

    // Convert the original image in grayscale and make the bitwise and with the mask
    cv::Mat gray, result;
    cv::cvtColor(ballROI, gray, cv::COLOR_BGR2GRAY);
    cv::bitwise_and(gray, gray, result, mask);

    // Compute the metric for Classification
    double whitePixelPercentage = calculateWhitePixelPercentage(result, white_binary_threshold);
    
    return whitePixelPercentage;
}

// Function to classify solid and stripe balls
void classifySolidStripeBalls(const cv::Mat& image, std::vector<BoundingBox>& Bbox) {

    const int shift = 3; // value for enlarging the Bounding Box for better metrics computation

    // Classify solid balls
    for (auto& bbox : Bbox) {
        if (bbox.ID == -1) {
            // Adjust the bounding box (Increase)
            adjustBoundingBox(bbox, shift);

            // Crop the ball
            cv::Mat ballROI = image(bbox.box);

            // Compute the metric for Classification
            const int white_binary_threshold = 200; // good parameters for high white pixels
            double whitePixelPercentage = calculateWhitePixelPercentage(ballROI, white_binary_threshold);

            // Initialize the threshold for solid ball
            const double white_threshold = 0.6; // good threshold for some of the solid balls
            if (whitePixelPercentage < white_threshold) {
                bbox.ID = 3; // Solid balls ID
            }

            // Adjust the bounding box (reduce them as normal)
            adjustBoundingBox(bbox, -shift);
        }
    }

    // Classify stripe balls
    for (auto& bbox : Bbox) {
        if (bbox.ID == -1) {
            // Initialize the parameters for masking the white component and compute the percentage
            const cv::Vec3b color(175, 242, 252); // BGR color triplet for white
            const int val = 40; // large threshold for color range
            const int white_binary_threshold = 200; // good parameters for high white pixels
            const int id = 4; // ID param for stripe balls
            double whitePixelPercentage = processBoundingBox(bbox, image, shift, color, val, white_binary_threshold);

            // Initialize the threshold for stripe ball
            const double white_threshold = 0.8; // good threshold for some of the stripe balls
            if (whitePixelPercentage > white_threshold) {
                bbox.ID = id; // Stripe balls ID
            }

            // Adjust the bounding box (reduce them as normal)
            adjustBoundingBox(bbox, -shift);
        }
    }

    // Classify other solid balls
    for (auto& bbox : Bbox) {
        if (bbox.ID == -1) {
            // Initialize the parameters for masking the white component and compute the percentage
            const cv::Vec3b color(86, 131, 140); // BGR color triplet for white
            const int val = 40; // large threshold for color range
            const int white_binary_threshold = 0; // parameters for considering the remaining component in the image
            const int id = 3; // ID param for solid balls
            double whitePixelPercentage = processBoundingBox(bbox, image, shift, color, val, white_binary_threshold);

            const double low_white_threshold = 2.85; // good low threshold for other solid balls
            const double high_white_threshold = 7.3; // good high threshold for other solid balls
            
            // Classify other solid balls
            if (whitePixelPercentage > low_white_threshold && whitePixelPercentage < high_white_threshold) {
                bbox.ID = id; // Solid balls ID
            }

            // Adjust the bounding box (reduce them as normal)
            adjustBoundingBox(bbox, -shift);
        }
    }

    // Classify remaining balls
    for (auto& bbox : Bbox) {
        if (bbox.ID == -1) {
            // Adjust the bounding box (Increase)
            adjustBoundingBox(bbox, shift);

            // Crop the ball
            cv::Mat ballROI = image(bbox.box);

            // Compute the metrics for Classification
            const int white_binary_threshold = 200; // good parameters for high white pixels
            double whitePixelPercentage = calculateWhitePixelPercentage(ballROI, white_binary_threshold);

            // Initialize the threshold for solid and stripe ball
            const double white_threshold = 2.0; // good threshold for the remaining balls

            // Classify the remaining balls
            if (whitePixelPercentage < white_threshold) {
                bbox.ID = 3; // Solid balls ID
            } else {
                bbox.ID = 4; // Stripe balls ID
            }

            // Adjust the bounding box (reduce them as normal)
            adjustBoundingBox(bbox, -shift);
        }
    }
    
    // Remove fake Bouning Box
    for (int i = 0; i < Bbox.size(); i++) {
        // Initialize the parameters for masking the white component and compute the percentage
        const cv::Vec3b color(150, 120, 50); // BGR color triplet for table color
        const int val = 20; // small threshold for color range
        const int white_binary_threshold = 100; // parameters for considering the remaining component in the image
        double whitePixelPercentage = processBoundingBox(Bbox[i], image, shift, color, val, white_binary_threshold);

        const double white_threshold = 79; // best threshold for white percentage
            
        // Classify a fake Bounding Box
        if (whitePixelPercentage > white_threshold) {
            Bbox[i].ID = -1;
        }

        // Adjust the bounding box (reduce them as normal)
        adjustBoundingBox(Bbox[i], -shift);
    }
}

// MAIN FUNCTION FOR BALL CLASSIFICATION
std::vector<BoundingBox> ballClassification(cv::Mat& image, const std::vector<cv::Rect>& Bbox_rect) {
    
    // Initialize a vector of Bounding Box
    std::vector<BoundingBox> Bbox;
         
    for (auto& bbox : Bbox_rect) {
        BoundingBox temp;
        temp.box = bbox;
        temp.ID = -1;
        Bbox.push_back(temp);
    }
    
    // Iitialize the indexes for black and white balls
    int whiteindex = -1;
    int blackindex = -1;

    computeWhiteBlackBallIndexes(image, Bbox, whiteindex, blackindex);

    if (whiteindex != -1) {
        Bbox[whiteindex].ID = 1;
    }

    if (blackindex != -1) {
        Bbox[blackindex].ID = 2;
    }

    classifySolidStripeBalls(image, Bbox);
    
    return Bbox;
}