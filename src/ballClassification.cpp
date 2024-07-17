#include "ballClassification.h"
#include <iostream>
#include "utils.h"

// Function to calculate the percentage of white pixels in an image
double calculateWhitePixelPercentage(const cv::Mat& image, int threshold) {
    // Convert to grayscale if not already
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image;
    }

    // Apply a binary threshold to get a binary image
    cv::Mat binary;
    cv::threshold(gray, binary, threshold, 255, cv::THRESH_BINARY); // You can adjust the threshold value

    //cv::imshow("bin", binary);
    //cv::waitKey(0);

    // Count the white pixels
    int whitePixelCount = cv::countNonZero(binary);
    
    // Calculate the total number of pixels
    int totalPixels = image.rows * image.cols;

    // Calculate the percentage of white pixels
    double whitePixelPercentage = (static_cast<double>(whitePixelCount) / totalPixels) * 100.0;
    return whitePixelPercentage;
}

// Function to calculate the percentage of black pixels in an image
double calculateBlackPixelPercentage(const cv::Mat& image, int threshold) {
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

    //cv::imshow("gray", gray);
    //showImage(binary, "bin");
    //cv::waitKey(0);

    // Count the black pixels (which are white in the binary image)
    int blackPixelCount = cv::countNonZero(binary);

    // Calculate the total number of pixels
    int totalPixels = image.rows * image.cols;

    // Calculate the percentage of black pixels
    double blackPixelPercentage = (static_cast<double>(blackPixelCount) / totalPixels) * 100.0;
    return blackPixelPercentage;
}

// Function to plot the Bounding Box in an image based on the Classification ID
void plotBBox(cv::Mat& image, BoundingBox& bbox) {
    cv::Scalar color;
    if (bbox.ID == 1) {
        color = cv::Scalar(0, 255, 0); // Red - solid ball
        cv::rectangle(image, bbox.box, color, 2);
    } else if (bbox.ID == 2) {
        color = cv::Scalar(0, 0, 0); // Blue - Stride ball
        cv::rectangle(image, bbox.box, color, 2);
    } else if (bbox.ID == 3) {
        color = cv::Scalar(0, 0, 255); // Red - solid ball
        cv::rectangle(image, bbox.box, color, 2);
    } else if (bbox.ID == 4) {
        color = cv::Scalar(255, 0, 0); // Blue - Stride ball
        cv::rectangle(image, bbox.box, color, 2);
    } else {
    	color = cv::Scalar(255, 255, 0); // Clear Blue - Stride ball
        cv::rectangle(image, bbox.box, color, 2);
    }
}

// Function to calculate the Variance of an image
double computeImageVariance(const cv::Mat& image) {
    // Convert to grayscale if not already
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image;
    }

    // Calculate the mean of the pixel values
    cv::Scalar meanScalar = cv::mean(gray);
    double mean = meanScalar[0];

    // Calculate the variance
    double variance = 0.0;
    for (int i = 0; i < gray.rows; ++i) {
        for (int j = 0; j < gray.cols; ++j) {
            double pixelValue = static_cast<double>(gray.at<uchar>(i, j));
            variance += (pixelValue - mean) * (pixelValue - mean);
        }
    }
    variance /= (gray.rows * gray.cols);

    return variance;
}

// Function apply K means clustering
cv::Mat applyKMeans(const cv::Mat& src, cv::Mat& dst, int clusterCount) {
    // Convert image to data points
    cv::Mat samples(src.rows * src.cols, 3, CV_32F);
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            for (int z = 0; z < 3; z++) {
                samples.at<float>(y + x * src.rows, z) = src.at<cv::Vec3b>(y, x)[z];
            }
        }
    }

    // Apply K-means clustering
    cv::Mat labels, centers;
    int attempts = 5;
    cv::kmeans(samples, clusterCount, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0), attempts, cv::KMEANS_PP_CENTERS, centers);

    // Create output image
    dst.create(src.size(), src.type());
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            int cluster_idx = labels.at<int>(y + x * src.rows, 0);
            dst.at<cv::Vec3b>(y, x)[0] = centers.at<float>(cluster_idx, 0);
            dst.at<cv::Vec3b>(y, x)[1] = centers.at<float>(cluster_idx, 1);
            dst.at<cv::Vec3b>(y, x)[2] = centers.at<float>(cluster_idx, 2);
        }
    }
    
    return centers;
}


// Function to calculate Euclidean distance between two BGR colors
float colorDistance(const cv::Scalar& color1, const cv::Scalar& color2) {
    float diffB = color1[0] - color2[0];
    float diffG = color1[1] - color2[1];
    float diffR = color1[2] - color2[2];
    return std::sqrt(diffB * diffB + diffG * diffG + diffR * diffR);
}


// Function to count pixels in an image that match a target color
int countPixels(const cv::Mat& image, const cv::Scalar& targetColor) {
    int count = 0;
    cv::Vec3b targetBGR(targetColor[0], targetColor[1], targetColor[2]);

    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            cv::Vec3b pixel = image.at<cv::Vec3b>(y, x);
            if (pixel == targetBGR) {
                count++;
            }
        }
    }

    return count;
}

// Function to adjust the bounding box
void adjustBoundingBox(BoundingBox& bbox, int shift) {
    bbox.box.x -= shift;
    bbox.box.y -= shift;
    bbox.box.width += shift * 2;
    bbox.box.height += shift * 2;
}


// MAIN FUNCTION FOR BALL CLASSIFICATION
std::vector<BoundingBox> ballClassification(cv::Mat& image, const std::vector<cv::Rect>& Bbox_rect, const cv::Vec3b& tableColor) {

	// Initialize a vector of Bounding Box
        std::vector<BoundingBox> Bbox;
         
        for (auto& bbox : Bbox_rect) {
            BoundingBox temp;
            temp.box = bbox;
            temp.ID = -1;
            Bbox.push_back(temp);
        }
        
        // Cretate an image to save the result
        cv::Mat result;
        image.copyTo(result);
        
        // Initilize to zero the percentage of black and with pixels
        double maxWhitePercentage = 0.0;
        double maxBlackPercentage = 0.0;
        
        // Iitialize the indexes for black and white balls
        int whiteindex;
        int blackindex;
        
        // COMPUTE THE WHITE AND BLACK BALL INDEXES for all the Bounding boxes
        
        for (int i = 0; i < Bbox.size(); i++) {
            
            // Adjust the bounding box (increase them)
            int shift = 3; // 3
            adjustBoundingBox(Bbox[i], shift);
            
            // Crop the ball ROI
            cv::Mat ballROI = image(Bbox[i].box);
            
            // Compute the metrics for classification
            double whitePixelPercentage = calculateWhitePixelPercentage(ballROI, 220); // 220
            double blackPixelPercentage = calculateBlackPixelPercentage(ballROI, 40); // 40
            
            // Select the whitest ball
            if (whitePixelPercentage > maxWhitePercentage){
                maxWhitePercentage = whitePixelPercentage;
                whiteindex = i;
            }
            
            // Select the darkest ball
            if (blackPixelPercentage + (100 - whitePixelPercentage)  > maxBlackPercentage){
                maxBlackPercentage = blackPixelPercentage + (100 - whitePixelPercentage);
                blackindex = i;
            }
            
            // Adjust the bounding box (reduce them as normal)
            shift = -3; // 3
            adjustBoundingBox(Bbox[i], shift);
        }
        
        // Classify the balls given the index
        Bbox[whiteindex].ID = 1;
        Bbox[blackindex].ID = 2;
        
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        
        // CLASSIFY SOLID AND STRIPE BALLS
           
        for (auto& bbox : Bbox) {
        
            if (bbox.ID == -1){
        	
		    // Adjust the bounding box (Increase)
		    int shift = 3;
		    adjustBoundingBox(bbox, shift);
		    
		    // Crop the ball
		    cv::Mat ballROI = image(bbox.box);
		    
		    // Compute the metrics for Classification
		    double whitePixelPercentage = calculateWhitePixelPercentage(ballROI, 200);
		    double blackPixelPercentage = calculateBlackPixelPercentage(ballROI, 30);
		    double variance = computeImageVariance(ballROI);
		    // double sum = computeSumOfPixels(ballROI);
		    
		    // Initialize the threshold for solid and stripe ball
		    double threshold = 0.6;
		    
		    // Classify some of the solid balls
		    if (whitePixelPercentage < threshold) {
		    	bbox.ID = 3;
		    }
		    
		    // Adjust the bounding box (reduce them as normal)
		    shift = -3; // 3
        	    adjustBoundingBox(bbox, shift);
		    
		    // std::cout << "White %: " << whitePixelPercentage << " " << "ID: " << bbox.ID << std::endl;
	    }
       }
       
       for (auto& bbox : Bbox) {
        
            if (bbox.ID == -1){
        	
		    // Adjust the bounding box (Increase)
		    int shift = 3;
		    adjustBoundingBox(bbox, shift);
		    
		    // Crop the ball
		    cv::Mat ballROI = image(bbox.box);
		    
		    // Compute the metrics for Classification
		    double whitePixelPercentage = calculateWhitePixelPercentage(ballROI, 200);
		    double blackPixelPercentage = calculateBlackPixelPercentage(ballROI, 30);
		    double variance = computeImageVariance(ballROI);
		    // double sum = computeSumOfPixels(ballROI);
		    
		    // Initialize the threshold for solid and stripe ball
		    double threshold = 2;
		    
		    // Classify some of the stipe balls
		    if (whitePixelPercentage > threshold) {
		    	bbox.ID = 4;
		    }
		    
		    
		    else {
		    	bbox.ID = 3;
		    }
		    
		    // Adjust the bounding box (reduce them as normal)
		    shift = -3; // 3
        	    adjustBoundingBox(bbox, shift);
		    
		    std::cout << "White %: " << whitePixelPercentage << " " << "ID: " << bbox.ID << std::endl;
	    }
        }
        
        
        for (auto& bbox : Bbox) {
        	// Plot the Classified Bounding Box
        	plotBBox(image, bbox);
        }
        
        std::cout << std::endl;
        
    return Bbox;
}




/*
        
		    
		    // Apply K means clustering
		    cv::Mat segmented_ROI, centers;
		    int clusterCount = 3; 
		    centers = applyKMeans(ballROI, segmented_ROI, clusterCount); 
		    
		    
		    
		    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		    
		    // Extract and display the cluster centers as BGR values
		    std::vector<cv::Scalar> cluster_colors;
		    for (int i = 0; i < clusterCount; i++) {
			cv::Scalar color(centers.at<float>(i, 0), centers.at<float>(i, 1), centers.at<float>(i, 2));
			cluster_colors.push_back(color);
		    }
		    
		    std::cout << "Cluster Centers (BGR values):" << std::endl;
		    for (int i = 0; i < clusterCount; i++) {
			std::cout << "Cluster " << i + 1 << ": " << cluster_colors[i] << std::endl;
		    }
		    
		    
		    // Identify the cluster center closest to the target color
		    cv::Scalar targetColor(tableColor[0], tableColor[1], tableColor[2]);
		    int excludeIndex = -1;
		    float minDist = std::numeric_limits<float>::max();
		    for (int i = 0; i < clusterCount; ++i) {
			float dist = colorDistance(cluster_colors[i], targetColor);
			if (dist < minDist) {
			    minDist = dist;
			    excludeIndex = i;
			}
		    }
		    
		    // Identify the two remaining colors
		    std::vector<cv::Scalar> remainingColors;
		    for (int i = 0; i < clusterCount; ++i) {
			if (i != excludeIndex) {
			    remainingColors.push_back(cluster_colors[i]);
			}
		    }
		    
		    int totalNum = 0; 
		    std::vector<int> pixelNum;
		    for (int i = 0; i < 2; ++i) {
		    	pixelNum.push_back(countPixels(segmented_ROI, remainingColors[i]));
		    	totalNum += pixelNum[i];
		    	//std::cout << "pixel number:" << pixelNum[i] << std::endl;
		    }
		    
		    double percentage = (static_cast<double>(pixelNum[0]) / totalNum) * 100.0;
		    
		    // std::cout << "pixel percentage:" << percentage << std::endl;
		    
		    
		    
		    // Create an image to display the remaining colors in boxes
		    int boxSize = 200;
		    cv::Mat displayImage(boxSize, boxSize * 2, CV_8UC3, cv::Scalar(0, 0, 0));

		    cv::rectangle(displayImage, cv::Rect(0, 0, boxSize * 2 * percentage/100, boxSize), remainingColors[0], cv::FILLED);
		    cv::rectangle(displayImage, cv::Rect(boxSize * 2 * percentage/100, 0, boxSize * 2* (1 - percentage/100), boxSize), remainingColors[1], cv::FILLED);
		    
		    
		    //showImage(displayImage,"image box");
		    //showImage(ballROI,"roi image");
		    //showImage(segmented_ROI,"roi clust");
		    //cv::waitKey();
		    
		    std::cout << "percentage: " << whitePixelPercentage << "  " << blackPixelPercentage << "  " << variance << "  " << bbox.ID << "  " << std::endl;
		    
		}
	}
		*/
