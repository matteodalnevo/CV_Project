#include "ball_hand_detection.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <set>
#include <vector>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <numeric>
#include <cmath>

// MEDIAN_COLOR

cv::Vec3b MedianColor(const cv::Mat& image) {

    // Compute the median color in uchar format
    std::vector<uchar> blue, green, red;

    // Extract pixel values for each channel, excluding black pixels ([0, 0, 0])
    // Loop for all the rows
    for (int row = 0; row < image.rows; ++row) {
        
        // Loop for all the column
        for (int col = 0; col < image.cols; ++col) {
            cv::Vec3b bgr = image.at<cv::Vec3b>(row, col);
            
            // skip the black color
            if (bgr != cv::Vec3b(0, 0, 0)) { 
                
                // Load the vectors
                blue.push_back(bgr[0]);
                green.push_back(bgr[1]);
                red.push_back(bgr[2]);
            }
        }
    }

    // Function to find the median of a vector
    auto findMedian = [](std::vector<uchar>& channel) -> uchar {
        const int n = channel.size();
        
        // Sorting of the channel component
        std::sort(channel.begin(), channel.end());

        // Even case 
        if (n % 2 == 0) {
            return (channel[n / 2 - 1] + channel[n / 2]) / 2;

        // Odd case
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


// TABLE_COLOR

cv::Vec3b tableColor(const cv::Mat& image, std::vector<cv::Point2f> vertices) {

    // Make a copy of the input image to draw the rectangle on
    cv::Mat result = image.clone();
    
    // Corners of the table 
    std::vector<cv::Point> points;
    for (const auto& vertex : vertices) {
        points.push_back(cv::Point(static_cast<int>(vertex.x), static_cast<int>(vertex.y)));
    }

    // Draw the quadrilateral by connecting the vertices
    std::vector<std::vector<cv::Point>> contours;
    contours.push_back(points);
    cv::polylines(result, contours, true, cv::Scalar(0, 255, 255), 2); // Yellow color with thickness 2

    // Create a mask for the ROI
    cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1); // Initialize mask with zeros (black)

    // Fill the ROI (region of interest) defined by the vertices with white color (255)
    cv::fillPoly(mask, contours, cv::Scalar(255));

    // Create a masked image using the original image and the mask
    cv::Mat maskedImage;
    image.copyTo(maskedImage, mask);
    
    // Compute the median color
    cv::Vec3b tableColor = MedianColor(maskedImage);
    
    return tableColor;
}


// BEST_HOMOG_DETECTION

cv::Mat best_homog_detection(std::vector<cv::Point2f> footage_table_corners, std::vector<cv::Point2f> scheme_table_corners) {
    // Compute 4 different homografy matricies w.r.t. rotated corners, two was enough however some noise could happened and ruin the results 
    cv::Mat H1 = cv::findHomography(footage_table_corners, scheme_table_corners); // Calculate the first homography matrix
    std::rotate(footage_table_corners.rbegin(), footage_table_corners.rbegin() + 1, footage_table_corners.rend()); // rotation of the corners
    cv::Mat H2 = cv::findHomography(footage_table_corners, scheme_table_corners); // Calculate the second homography matrix
    std::rotate(footage_table_corners.rbegin(), footage_table_corners.rbegin() + 1, footage_table_corners.rend()); // rotation of the corners
    cv::Mat H3 = cv::findHomography(footage_table_corners, scheme_table_corners); // Calculate the third homography matrix
    std::rotate(footage_table_corners.rbegin(), footage_table_corners.rbegin() + 1, footage_table_corners.rend()); // rotation os the corners
    cv::Mat H4 = cv::findHomography(footage_table_corners, scheme_table_corners); // Calculate the fourth homography matrix
    
    // Determine all the four error (difference) for each hessian matrix along the diagonal (stretching component, if 1 on the diagonal no stretching) 
    double e1 = std::pow((H1.at<double>(0, 0) - 1), 2) + std::pow((H1.at<double>(1, 1) - 1), 2) + std::pow((H1.at<double>(2, 2) - 1), 2);
    double e2 = std::pow((H2.at<double>(0, 0) - 1), 2) + std::pow((H2.at<double>(1, 1) - 1), 2) + std::pow((H2.at<double>(2, 2) - 1), 2);
    double e3 = std::pow((H3.at<double>(0, 0) - 1), 2) + std::pow((H3.at<double>(1, 1) - 1), 2) + std::pow((H3.at<double>(2, 2) - 1), 2);
    double e4 = std::pow((H4.at<double>(0, 0) - 1), 2) + std::pow((H4.at<double>(1, 1) - 1), 2) + std::pow((H4.at<double>(2, 2) - 1), 2);

    //// Select the correct Homography matrix, the one with lower error (difference)
    if (e1 < e2 && e1 < e3 && e1 < e4) {
        return H1;
    }
    if (e2 < e1 && e2 < e3 && e2 < e4) {
        return H2;
    }
    if (e3 < e1 && e3 < e2 && e3 < e4) {
        return H3;
    }
    if (e4 < e1 && e4 < e3 && e4 < e2) {
        return H4;
    }
    else return H1;
}


// DRAW_POLYGON

void drawPolygon(cv::Mat& image, const std::vector<cv::Point2f>& polygon, const cv::Scalar& color, const int thickness) {
    // Check on the dimensions, it needs to be 4 
    if (polygon.size() < 4) return;

    // Draw the polygon with the lines
    for (int i = 0; i < 4; ++i) {
        cv::Point2f pt1 = polygon[i];
        cv::Point2f pt2 = polygon[(i + 1) % polygon.size()];

        // Line drawing
        cv::line(image, cv::Point2f(pt1.x, pt1.y), cv::Point2f(pt2.x, pt2.y), color, thickness);
    }
}


// COMPUTE_SOBEL

cv::Mat computeSobel(const cv::Mat& gray) {
    
    // First order derivative with sobel 
    cv::Mat sobel;
    cv::Mat g_x;
    cv::Mat g_y;
    cv::Mat a_x;
    cv::Mat a_y;

    // Application of sobel functions
    cv::Sobel(gray, g_x, CV_16S, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
    cv::Sobel(gray, g_y, CV_16S, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);

    cv::convertScaleAbs(g_x, a_x);
    cv::convertScaleAbs(g_y, a_y);

    // Normalization to have similar intensity ranges 
    cv::normalize(a_x, a_x, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::normalize(a_y, a_y, 0, 255, cv::NORM_MINMAX, CV_8U);

    cv::addWeighted(a_x, 0.5, a_y, 0.5, 0, sobel);

    return sobel;
}


// DETECT_CIRCLES

std::vector<cv::Vec3f> detectCircles(cv::Mat& image, const int dp, const int minDist, const int param1, const int param2, const int minRadius, const int maxRadius, const int threshold_value, const cv::Size gauss_ker) {
   
    cv::Mat gray;
    cv::Mat sobel;
    cv::Mat binary;

    // Since the utilized images derived from different color spaces convert the image to grayscale if needed
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } 
    else {
        gray = image;
    }

    // Compute Sobel edges with the apposite created function
    sobel = computeSobel(gray);

    // Apply a threshold to get a binary image, this is done to divide into usefull and useless points (section)
    cv::threshold(sobel, binary, threshold_value, 255, cv::THRESH_BINARY);

    // Create the vectore where store the circles
    std::vector<cv::Vec3f> circles;

    // Apply GaussianBlur to reduce noise and improve circle detection
    const double sigx = 1.2; // Gaussian kernel standard deviation in X direction
    const double sigy = 1.2; // Gaussian kernel standard deviation in Y direction
    cv::GaussianBlur(binary, binary, gauss_ker, sigx, sigy);

    // Apply Hough Circle Transform on the binary image
    cv::HoughCircles(binary, circles, cv::HOUGH_GRADIENT, dp, minDist, param1, param2, minRadius, maxRadius);

    return circles;
}


// DRAW_BOUNDINGBOXES

void drawBoundingBoxes(cv::Mat& image, const std::vector<cv::Rect>& boundingBoxes) {
    // Draw detected bounding boxes
    for (const auto& box : boundingBoxes) {
        cv::rectangle(image, box, cv::Scalar(0, 0, 255), 2); // Red color for detected bounding boxes
    }
}


// MERGE CIRCLES

std::vector<cv::Vec3f> mergeCircles(const std::vector<cv::Vec3f>& circles, float radius_threshold) {
    // Vector where load all the cleaned circles
    std::vector<cv::Vec3f> mergedCircles;

    // Needed to see if a circle it has already been visited
    std::vector<bool> visited(circles.size(), false);

    // Loop for all the circles
    for (int i = 0; i < circles.size(); ++i) {

        // Check if the cirlce_1 has benn visited
        if (visited[i]) continue;

        // Center of the circle_1 
        cv::Point2f center_i = cv::Point2f(circles[i][0], circles[i][1]);
        
        // Radius of the circle_1
        int radius_i = circles[i][2];

        int count = 1; // Counter for the number of merged circles into a unique one
        
        // Sum of the single parameters of the circle_1 (needed to be averaged at the end)
        float sum_x = circles[i][0]; // sum_x now contain just the x value of the circle_1
        float sum_y = circles[i][1]; // sum_y now contain just the x value of the circle_1
        float sum_r = circles[i][2]; // sum_z now contain just the x value of the circle_1
        
        // Loop for the remaining circles that follow the one selected in the above (outter) loop
        // Important note: in the following there is written circle_2, however in successive iteration could be more: circle_3, ..., circle_n 
        for (int j = i + 1; j < circles.size(); ++j) {

            // Check if the circle_2 has been visited
            if (visited[j]) continue;

            // Center of the circle_2
            cv::Point2f center_j = cv::Point2f(circles[j][0], circles[j][1]);
            
            // Radius of the circle_2
            int radius_j = circles[j][2];
            
            // Check on the distance between the circle_1 and circle_2 if smaller than the radius_threshold, merge the circles
            if (cv::norm(center_i - center_j) < radius_threshold) {

                // Sum the x,y,r values of the circle_2 to circle_1
                sum_x += circles[j][0];
                sum_y += circles[j][1];
                sum_r += circles[j][2];

                // Increment of the variable that count how many circles has been merged together
                count++;

                // Set the circle_2 has been visited
                visited[j] = true;
            }
        }
        
        // Set the circle_1 has been visited
        visited[i] = true;

        // Vector of the merged circles
        mergedCircles.push_back(cv::Vec3f(sum_x / count, sum_y / count, sum_r / count));
    }

    return mergedCircles;
}


// IS_POINT_INSIDE_POLYGON

bool isPointInsidePolygon(const cv::Point2f& point, const std::vector<cv::Point2f>& polygon) {
    // Boolean check if the input point is inside of the polygon described by the vector of points in input 
    return cv::pointPolygonTest(polygon, point, false) >= 0;
}


// LOAD_POLYGON_VERTICES (not used in the final code)

std::vector<cv::Point2f> loadPolygonVertices(const std::string& filepath) {
    
    // Vector where store the polygon verticies
    std::vector<cv::Point2f> polygon_original;
    
    // Reading from txt file
    std::ifstream infile(filepath);

    // Check if it can be loaded
    if (!infile.is_open()) {
        std::cerr << "Could not open the note file txt!\n";
        return polygon_original;
    }

    // x,y coordinate 
    int x, y;
    while (infile >> x >> y) {
        polygon_original.push_back(cv::Point2f(x, y));
    }

    infile.close();

    return polygon_original;
}


// LOAD_CORRECT_BOUNDINGBOXES (not used in the final code)

std::vector<cv::Rect> loadCorrectBoundingBoxes(const std::string& filepath) {

    // Correct bounding boxes in the txt file 
    std::vector<cv::Rect> boundingBoxes;
    std::ifstream infile(filepath);

    // Check if the file could be read 
    if (!infile.is_open()) {
        std::cerr << "Could not open the file!\n";
        return boundingBoxes;
    }

    // Parameters of the bounding boxes
    int x, y, w, h, i;
    while (infile >> x >> y >> w >> h >> i) {
        boundingBoxes.push_back(cv::Rect(x, y, w, h));
    }

    infile.close();
    return boundingBoxes;
}


// CLOSE_COLOR

bool closeColor(const cv::Vec3f& color1, const cv::Vec3f& color2, const float margin) {

    // Boolean check if the difference between the two color is under or equal to the margin threshold 
    return std::abs(color1[0] - color2[0]) <= margin &&
           std::abs(color1[1] - color2[1]) <= margin &&
           std::abs(color1[2] - color2[2]) <= margin;
}


// FALSE_POSITIVE

bool falsePositive(const cv::Mat& image, const cv::Rect& bbox, const cv::Vec3b& mean, const float threshold, const float margin) {

    // Consider the bounding box area (image)
    cv::Mat roi = image(bbox);

    // Calsulation of the all pixels to average
    const int totalPixels = roi.rows * roi.cols;
    
    // Variable that count how many pixels has the color close the mean 
    int closePixels = 0;

    // Loop for each row of the bounding box area in the image
    for (int y = 0; y < roi.rows; ++y) {

        // Loop for each column of the bounding box area in the image
        for (int x = 0; x < roi.cols; ++x) {

            // Take the color of that pixel
            cv::Vec3b pixel = roi.at<cv::Vec3b>(y, x);

            // Check if the pixel is close to the mean color
            if (closeColor(cv::Vec3f(pixel[0], pixel[1], pixel[2]), cv::Vec3f(mean[0], mean[1], mean[2]), margin)) {

                // Update the counter
                ++closePixels;
            }
        }
    }
    return static_cast<float>(closePixels) / totalPixels > threshold;
}


// FILTER_BOUNDINGBOXES

std::vector<cv::Rect> filterBoundingBoxes(const cv::Mat& image, const std::vector<cv::Rect>& bboxes, const std::vector<cv::Point2f>& vertices, const float threshold, const float margin) {
    
    // Vector where storing the output filtered boundung boxes
    std::vector<cv::Rect> filteredBboxes;

    // Compute the mean color of the table
    cv::Vec3b mean = tableColor(image, vertices);

    // Utilize the false positive function to check it and in case af true result, not store that bounding box 
    for (const auto& bbox : bboxes) {
        if (!falsePositive(image, bbox, mean, threshold, margin)) {
            filteredBboxes.push_back(bbox);
        }
    }
    return filteredBboxes;
}


// CALCULATE_DISTANCE

double calculateDistance(const cv::Point2f p1, const cv::Point2f p2) {
    // Calculation of the distance between the bounding boxes by readind the x and y coordinate     
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}


// INTERSECTION_AREA

double intersectionArea(const cv::Rect& r1, const cv::Rect& r2) {

    // overlapped segment along x 
    const int x_overlap = std::max(0, std::min(r1.x + r1.width, r2.x + r2.width) - std::max(r1.x, r2.x));
    
    // overlapped segment along y
    const int y_overlap = std::max(0, std::min(r1.y + r1.height, r2.y + r2.height) - std::max(r1.y, r2.y));
    
    // Overlapping area 
    return x_overlap * y_overlap;
}


// MERGED_BOUNDING_BOXES 

cv::Rect mergedBoundingBoxes(const cv::Rect& r1, const cv::Rect& r2) {
    
    // Consider the minimum x and y value 
    const int x1 = std::min(r1.x, r2.x);
    const int y1 = std::min(r1.y, r2.y);

    // Consider the maximum x+w and y+h value
    const int x2 = std::max(r1.x + r1.width, r2.x + r2.width);
    const int y2 = std::max(r1.y + r1.height, r2.y + r2.height);

    // Biggest rect possible
    return cv::Rect(x1, y1, x2 - x1, y2 - y1);
}


// MERGE_BOUNDING_BOXES

void mergeBoundingBoxes(std::vector<cv::Rect>& boundingBoxes, const int pixeldistance, const float dimdifference, const float sharedarea) {

    // Loop over all the bounding boxes
    for (int i = 0; i < boundingBoxes.size(); ++i) {

        // Loop over the remaining bounding boxes following the above one 
        for (int j = i + 1; j < boundingBoxes.size();) {

            // Retrive the areas of the Bounding Boxes, in this case since them are rectangle can be compute with .area()
            double areaI = boundingBoxes[i].area();
            double areaJ = boundingBoxes[j].area();

            // Centers of the bounding boxe
            cv::Point2f centerI(boundingBoxes[i].x + boundingBoxes[i].width / 2.0,
                                boundingBoxes[i].y + boundingBoxes[i].height / 2.0);

            cv::Point2f centerJ(boundingBoxes[j].x + boundingBoxes[j].width / 2.0,
                                boundingBoxes[j].y + boundingBoxes[j].height / 2.0);

            // Call to the function for Distance between centers
            double centerDistance = calculateDistance(centerI, centerJ);

            // Check if one box is within the neighborhood of the other (pixeldistance) and has a smaller percentage (dimdifference) area w.r.t the other one
            if (centerDistance < pixeldistance) {
                if ((areaI < dimdifference * areaJ) || (areaJ < dimdifference * areaI)) {

                    // If Both condition are satisfied, it is very likely that it is the same balls, noise problem, so we merge them
                    boundingBoxes[i] = mergedBoundingBoxes(boundingBoxes[i], boundingBoxes[j]);
                    boundingBoxes.erase(boundingBoxes.begin() + j);
                    continue;
                }
            }

            // Intersection area calculation
            double intersectArea = intersectionArea(boundingBoxes[i], boundingBoxes[j]);

            // Check if they share more than the threshold value, if so it very likely that are the same ball, so we merge them  
            if (intersectArea > sharedarea * std::min(areaI, areaJ)) {
                boundingBoxes[i] = mergedBoundingBoxes(boundingBoxes[i], boundingBoxes[j]);
                boundingBoxes.erase(boundingBoxes.begin() + j);
            } else {
                ++j;
            }
        }
    }
    
}


// HAND_MASK_FILTERING

cv::Mat HandMaskFiltering( std::vector<cv::Rect>& bbox, const cv::Mat& image, const std::vector<cv::Point2f>& areaOfInterest, const double threshold_hand) {

    // Convert the image to HSV color space
    cv::Mat hsvImage;
    cv::cvtColor(image, hsvImage, cv::COLOR_BGR2HSV);

    // Split the HSV image into its channels, we are interested into the Saturation channel
    std::vector<cv::Mat> hsvChannels;
    cv::split(hsvImage, hsvChannels);

    // Get the saturation channel
    cv::Mat saturation = hsvChannels[1];

    // A Gaussian blur it is applied to the saturation channel (image)
    const cv::Size gaussKer = cv::Size(3, 3); // Little smoothing 3*3 kernel
    const double sigmaX = 1.5; // Gaussian kernel standard deviation in X direction
    const double sigmaY = 1.5; // Gaussian kernel standard deviation in Y direction
    cv::GaussianBlur(saturation, saturation, gaussKer, sigmaX, sigmaY);

    // Apply a threshold to the saturation channel to create a mask
    cv::Mat mask;
    cv::threshold(saturation, mask, threshold_hand, 255, cv::THRESH_BINARY);

    // Creation of a mask of the same size as the input image initialized to zero
    cv::Mat maskROI = cv::Mat::zeros(mask.size(), mask.type());

    // Convert area of interest coordinate from Point2f to Point for applying fillPoly later
    std::vector<cv::Point> points;
    for (const auto& point : areaOfInterest) {
        points.push_back(cv::Point(static_cast<int>(point.x), static_cast<int>(point.y)));
    }

    // Fill the area of polygon in the mask with white
    std::vector<std::vector<cv::Point>> pts = { points };
    cv::fillPoly(maskROI, pts, cv::Scalar(255));

    // Apply the polygon mask to the original mask
    cv::Mat finalMask;
    cv::bitwise_and(mask, maskROI, finalMask);

    // Create an output image initialized to black
    cv::Mat outputImage = cv::Mat::zeros(image.size(), image.type());

    // Find contours in the final mask
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(finalMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Color the contours
    for (int i = 0; i < contours.size(); ++i) {
        cv::drawContours(outputImage, contours, static_cast<int>(i), cv::Scalar(255, 255, 255), -1); // White color
    }

    // Till here we have found the the hand and connected with the back ground, instead the table is fully white 
    // Now we need to separate the hand from ht ebackground, obtaining as result just a black area that identify the hand

    // Create a mask of the polygon areaOfInterest
    cv::Mat polygonMask = cv::Mat::zeros(image.size(), CV_8UC1);
    cv::fillPoly(polygonMask, pts, cv::Scalar(255)); // white color

    // Set everything outside the polygon to white in the output image
    cv::Mat whiteBackground(image.size(), image.type(), cv::Scalar(255, 255, 255));
    outputImage.copyTo(whiteBackground, polygonMask);

    // Now we have obtained a mask that remove just the hand

    // 
    auto BlackArea = [&whiteBackground](const cv::Rect& box) {

        // Loop for each row of the bounding box
        for (int y = box.y; y < box.y + box.height; ++y) {

            // Loop for each column of the bounding box 
            for (int x = box.x; x < box.x + box.width; ++x) {

                // Take the color of the mask (or white or black)
                cv::Vec3b color = whiteBackground.at<cv::Vec3b>(y, x);

                // Check if is black, if so turn true 
                if (color == cv::Vec3b(0, 0, 0)) {

                    // Found a black pixel in the bounding box
                    return true; 
                }
            }
        }

    // No black pixel found 
    return false; 
    };

    // Remove bounding boxes that intersect with the black area (hand) of the whiteBackground
    bbox.erase(std::remove_if(bbox.begin(), bbox.end(), BlackArea), bbox.end());

    return whiteBackground;


}



// BALLS_HAND_DETECTION
std::tuple<std::vector<cv::Rect>, cv::Mat> ballsHandDetection(const cv::Mat& img, const std::vector<cv::Point2f> polygon) {

    // PARAMETERS
    
    // The scheme _corners vector contain the original corner of the scheme image (blue or green area)
    const std::vector<cv::Point2f> scheme_corners = {cv::Point2f(82, 81), cv::Point2f(82, 756), cv::Point2f(1384, 756), cv::Point2f(1384, 81)};

    // The smaller _corners vector contain the playable field corner of the scheme image (so no border)
    const std::vector<cv::Point2f> smaller_corners = {cv::Point2f(94, 104), cv::Point2f(94, 733), cv::Point2f(1365, 733), cv::Point2f(1365, 104)};

    // The scheme_holes vector contain the holes/pocket of the scheme image
    const std::vector<cv::Point2f> scheme_holes = {cv::Point2f(94, 100), cv::Point2f(94, 737), cv::Point2f(1365, 737), cv::Point2f(1365, 100),   cv::Point2f(730, 100), cv::Point2f(730, 737) };


    // Hough Circle Parameters

    // dp Inverse ratio of the accumulator resolution to the image resolution (1 accumulator has the same resolution as the input image)
    const int dp = 1; 

    // Minimum distance between the centers of the detected circles
    const int minDist = 13; 

    // First method-specific parameter, in our case it does not modify the results 
    const int param1 = 100;

    // It is the accumulator threshold for the circle centers, this value ensure a good amount of balls detection with a contained number of false positive
    const int param2 = 16;

    // Minimum circle radius
    const int minRadius = 6;

    // Maximum circle radius
    const int maxRadius = 16;

    // Threshold value for the biary image conversion 
    const int threshold_binary_image_value = 55;

    // Gaussian kernel dimensions for the gaussian blur 
    const cv::Size gauss_ker(11,11);

    // Filtering Parameters
    
    // Circle Merge distance 
    const float merge_circle_distance = 8;
    
    // Bounding box merge pixel distance value (connected to the merge_dim_difference)
    const int merge_pixel_distance = 16;

    // Bounding box Size different threshold percentage form merging: 60% (connected to the merge_pixel_distance)
    const float merge_dim_difference = 0.6; 
    
    // Shared area threshold percentage, above it will merge the bboxes
    const float sharedarea = 0.7;

    // Percentage of pixels of a color similar to the table one allowed to be considered valid: 40%
    const float pixelsofcolor = 0.40;

    // The variance of each single color component for the similarity color with the table 
    const int margincolor = 25;

    // Dimensions of the squares to be inserted in the hole/pocket positions 
    const int squareSize = 37;


    // Check to the input image
    if (img.empty()) {
        std::cerr << "Could not open or find the image!\n";
        return {};
    }
 
    // Check to the polygon input
    if (polygon.empty()) {
        std::cerr << "Could not load polygon vertices!\n";
        return {};
    }

    // Clones of the img (they will be modified)
    cv::Mat image = img.clone();
    cv::Mat out = img.clone();

    // Calculation of the best homography matrix 
    cv::Mat H = best_homog_detection(polygon, scheme_corners);
    
    // Retrive the inverse matrix 
    cv::Mat H_inv;
    cv::invert(H, H_inv);
    
    // Find the correspond holes and corner of the real image w.r.t. the scheme ones 
    std::vector<cv::Point2f> footage_holes, smaller_corners_footage;
    cv::perspectiveTransform(scheme_holes, footage_holes, H_inv);
    cv::perspectiveTransform(smaller_corners, smaller_corners_footage, H_inv);

    // Draw the squares in the real footage for hiding the holes in the real footage 
    for (const cv::Point2f& vertex : footage_holes) {
        cv::Point2f topLeft(vertex.x - squareSize / 2, vertex.y - squareSize / 2);
        cv::rectangle(image, cv::Rect(topLeft, cv::Size(squareSize, squareSize)), cv::Scalar(0, 255, 0), cv::FILLED);
    }

    // Display image is the clone of image, containing the squares 
    cv::Mat displayImage = image.clone();

    // Vector that will be loaded with all the circles that will be find in the nexts steps
    std::set<cv::Vec3f, Vec3fComparator> unifiedCircles;

    // Convertion of the image into different color spaces, selection of the miningfull channels, detection of the circles, insertion into the unifiedCircles vector

    // BGR, R component 
    std::vector<cv::Mat> channels; 
    cv::split(displayImage, channels); // Split the channels
    std::vector<cv::Vec3f> redCircles = detectCircles(channels[2], dp, minDist, param1, param2, minRadius, maxRadius, threshold_binary_image_value, gauss_ker);
    unifiedCircles.insert(redCircles.begin(), redCircles.end());
    
    //YUV, V component
    cv::Mat yuvImage;
    cv::cvtColor(displayImage, yuvImage, cv::COLOR_BGR2YUV);
    std::vector<cv::Mat> yuvChannels;
    cv::split(yuvImage, yuvChannels); // Split the channels
    std::vector<cv::Vec3f> vCircles = detectCircles(yuvChannels[2], dp, minDist, param1, param2, minRadius, maxRadius, threshold_binary_image_value, gauss_ker);
    unifiedCircles.insert(vCircles.begin(), vCircles.end());
    
    //HSV 
    // Value component
    cv::Mat hsvImage;
    cv::cvtColor(displayImage, hsvImage, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> hsvChannels;
    cv::split(hsvImage, hsvChannels); // Split the channels
    std::vector<cv::Vec3f> valueCircles = detectCircles(hsvChannels[2], dp, minDist, param1, param2, minRadius, maxRadius, threshold_binary_image_value, gauss_ker);
    unifiedCircles.insert(valueCircles.begin(), valueCircles.end());

    //Saturation component
    cv::Mat hsvImage2;
    cv::cvtColor(displayImage, hsvImage2, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> hsvChannelssat;
    cv::split(hsvImage2, hsvChannelssat);
    std::vector<cv::Vec3f> satCircles = detectCircles(hsvChannelssat[1], dp, minDist, param1, param2, minRadius, maxRadius, threshold_binary_image_value, gauss_ker);
    unifiedCircles.insert(satCircles.begin(), satCircles.end());


    // MERGE of all the circles
    std::vector<cv::Vec3f> mergedCircles = mergeCircles(std::vector<cv::Vec3f>(unifiedCircles.begin(), unifiedCircles.end()), merge_circle_distance);

    // FILTER circles
    std::vector<cv::Vec3f> filteredCircles;
    for (const auto& circle : mergedCircles) {
        cv::Point2f center = cv::Point2f(circle[0], circle[1]);
        if (isPointInsidePolygon(center, smaller_corners_footage)) {
            filteredCircles.push_back(circle);
        }
    }


    std::vector<cv::Rect> detectedBoundingBoxes;
    for (const auto& circle : filteredCircles) {
        cv::Point2f topLeft(circle[0] - circle[2], circle[1] - circle[2]);
        int width = 2 * circle[2];
        int height = 2 * circle[2];
        detectedBoundingBoxes.push_back(cv::Rect(topLeft.x, topLeft.y, width, height));
    }

    // REMOVE OUTLIERS BBOXES
    mergeBoundingBoxes(detectedBoundingBoxes, merge_pixel_distance, merge_dim_difference, sharedarea);

    // REMOVE FALSE POSITIVES
    //ColorMean means;

    //std::vector<cv::Rect> filteredBboxes = detectedBoundingBoxes;
    std::vector<cv::Rect> filteredBboxes = filterBoundingBoxes(displayImage, detectedBoundingBoxes,smaller_corners_footage, pixelsofcolor, margincolor);

    // Hand mask
    int threshold_hand = 100;

    cv::Mat hand = HandMaskFiltering(filteredBboxes, displayImage, smaller_corners_footage , threshold_hand);
    
    // Draw both detected and correct bounding boxes
    drawBoundingBoxes(displayImage, filteredBboxes);
    
    // Draw the polygon on the image
    drawPolygon(displayImage, smaller_corners_footage, cv::Scalar(0, 255, 0), 2); // Green color with thickness 2

    //cv::imshow("Hough Circles", displayImage);
    //cv::waitKey(0);
    //cv::destroyAllWindows();

    return std::make_tuple(filteredBboxes, hand);
}
