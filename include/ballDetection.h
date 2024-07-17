// ballDetection.h
#ifndef BALLDETECTION_H
#define BALLDETECTION_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include <set>
#include <vector>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <numeric>
#include <cmath>


// Compute the median
cv::Vec3b MedianColor(const cv::Mat& image);

// Find the color of the table
cv::Vec3b table(const cv::Mat& image, std::vector<cv::Point2f> vertices);


// Print the bounding boxes
void printBoundingBoxes(const std::vector<cv::Rect>& filteredBboxes);

cv::Mat best_homog_detection(std::vector<cv::Point2f> footage_table_corners, std::vector<cv::Point2f> scheme_table_corners);

// DRAWPOLYGON, not needed in the final code OK
void drawPolygon(cv::Mat& image, const std::vector<cv::Point2f>& polygon, const cv::Scalar& color, int thickness);

// Necessary structure to be able to compare the elements OK
struct Vec3fComparator {
    bool operator() (const cv::Vec3f& a, const cv::Vec3f& b) const {
        if (a[0] != b[0]) return a[0] < b[0];
        if (a[1] != b[1]) return a[1] < b[1];
        return a[2] < b[2];
    }
};

// Function to compute Sobel function OK
cv::Mat computeSobel(const cv::Mat& gray);

// Function to detect circles using Hough Transform OK
std::vector<cv::Vec3f> detectCircles(cv::Mat& image, int dp, int minDist, int param1, int param2, int minRadius, int maxRadius, int threshold_value, cv::Size gauss_ker);

// Function to draw bounding boxes around detected circles, just intermidiate function to check OK
void drawBoundingBoxes(cv::Mat& image, const std::vector<cv::Rect>& boundingBoxes);

// Function to merge circles that are close to each other with radius_threshold OK 
std::vector<cv::Vec3f> mergeCircles(const std::vector<cv::Vec3f>& circles, float radius_threshold);

// Function to check if a point is inside a polygon, corners of the table OK
bool isPointInsidePolygon(const cv::Point2f& point, const std::vector<cv::Point2f>& polygon);

// Function to load polygon vertices from a text file OK
std::vector<cv::Point2f> loadPolygonVertices(const std::string& filepath);

// Function to load correct bounding boxes from a text file OK
std::vector<cv::Rect> loadCorrectBoundingBoxes(const std::string& filepath);

// Definition of the particular means of the false positives in BGR format 
struct ColorMean {
    cv::Vec3f black = {44, 41, 35};
    cv::Vec3f blue1 = {144, 106, 26};
    cv::Vec3f blue2 = {82, 46, 3};
    cv::Vec3f blue3 = {140, 115, 40};
    cv::Vec3f grey = {86, 82, 85};
    cv::Vec3f green = {76, 102, 18};
    cv::Vec3f pink1 = {164, 177, 243};
    cv::Vec3f pink2 = {79, 77, 89};
    cv::Vec3f pink3 = {167, 197, 223};
    cv::Vec3f brown = {111, 144, 168};
};

// Check if the color is closer to the particular ones
bool isColorClose(const cv::Vec3f& color1, const cv::Vec3f& color2, float margin);

// Check on the false positives, derived from the color matching 
bool isFalsePositive(const cv::Mat& image, const cv::Rect& bbox, const cv::Vec3b& mean, float threshold, float margin);

// Filtered based on color 
std::vector<cv::Rect> filterBoundingBoxes(const cv::Mat& image, const std::vector<cv::Rect>& bboxes, const std::vector<cv::Point2f>& vertices, float threshold, float margin);

// Distance between bounding boxes
double calculateDistance(cv::Point2f p1, cv::Point2f p2);

// Check on the intersection area
double intersectionArea(const cv::Rect& r1, const cv::Rect& r2);

// Merged rectangle
cv::Rect mergeBoundingBoxes(const cv::Rect& r1, const cv::Rect& r2);

// Merging function
void mergeBoundingBoxes(std::vector<cv::Rect>& boundingBoxes, int& pixeldistance, float& dimdifference, float& sharedarea);

//hand detection
void HandMask(std::vector<cv::Rect>& bbox, cv::Mat image, const std::vector<cv::Point2f>& areaOfInterest, double threshold_hand);

// Overall function
std::vector<cv::Rect> ballsDetection(cv::Mat img, std::vector<cv::Point2f> polygon);


#endif // BALLDETECTION_H
