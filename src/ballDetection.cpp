#include "ballDetection.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <set>
#include <vector>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <numeric>
#include <cmath>

// Print the bounding boxes
void printBoundingBoxes(const std::vector<cv::Rect>& filteredBboxes) {
    for (std::vector<cv::Rect>::const_iterator it = filteredBboxes.begin(); it != filteredBboxes.end(); ++it) {
        std::cout << "Rect(x=" << it->x << ", y=" << it->y 
                  << ", width=" << it->width << ", height=" << it->height << ")" << std::endl;
    }
}

// BEST_HOMOG OK
cv::Mat best_homog_detection(std::vector<cv::Point2f> footage_table_corners, std::vector<cv::Point2f> scheme_table_corners) {
    cv::Mat H1 = cv::findHomography(footage_table_corners, scheme_table_corners); // Calculate the first homography matrix
    std::rotate(footage_table_corners.rbegin(), footage_table_corners.rbegin() + 1, footage_table_corners.rend());
    cv::Mat H2 = cv::findHomography(footage_table_corners, scheme_table_corners); // Calculate the second homography matrix
    std::rotate(footage_table_corners.rbegin(), footage_table_corners.rbegin() + 1, footage_table_corners.rend());
    cv::Mat H3 = cv::findHomography(footage_table_corners, scheme_table_corners); // Calculate the third homography matrix
    std::rotate(footage_table_corners.rbegin(), footage_table_corners.rbegin() + 1, footage_table_corners.rend());
    cv::Mat H4 = cv::findHomography(footage_table_corners, scheme_table_corners); // Calculate the fourth homography matrix
    
    // Determine all the four error for each hessian matrix along the diagonal (stretching component)
    double e1 = std::pow((H1.at<double>(0, 0) - 1), 2) + std::pow((H1.at<double>(1, 1) - 1), 2) + std::pow((H1.at<double>(2, 2) - 1), 2);
    double e2 = std::pow((H2.at<double>(0, 0) - 1), 2) + std::pow((H2.at<double>(1, 1) - 1), 2) + std::pow((H2.at<double>(2, 2) - 1), 2);
    double e3 = std::pow((H3.at<double>(0, 0) - 1), 2) + std::pow((H3.at<double>(1, 1) - 1), 2) + std::pow((H3.at<double>(2, 2) - 1), 2);
    double e4 = std::pow((H4.at<double>(0, 0) - 1), 2) + std::pow((H4.at<double>(1, 1) - 1), 2) + std::pow((H4.at<double>(2, 2) - 1), 2);

    // Select the correct Homography matrix, the one with lower error
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
    return H1;
}

// DRAWPOLYGON, not needed in the final code OK
void drawPolygon(cv::Mat& image, const std::vector<cv::Point2f>& polygon, const cv::Scalar& color, int thickness) {
    // Check on the dimensions, it needs to be 4 
    if (polygon.size() < 4) return;

    // Draw the polygon with the lines
    for (int i = 0; i < 4; ++i) {
        cv::Point2f pt1 = polygon[i];
        cv::Point2f pt2 = polygon[(i + 1) % polygon.size()];
        cv::line(image, cv::Point2f(pt1.x, pt1.y), cv::Point2f(pt2.x, pt2.y), color, thickness);
    }
}

// Necessary structure to be able to compare the elements OK
//struct Vec3fComparator {
//    bool operator() (const cv::Vec3f& a, const cv::Vec3f& b) const {
//        if (a[0] != b[0]) return a[0] < b[0];
//        if (a[1] != b[1]) return a[1] < b[1];
//        return a[2] < b[2];
//    }
//};

// Function to compute Sobel function OK
cv::Mat computeSobel(const cv::Mat& gray) {
    cv::Mat sobel;
    cv::Mat g_x;
    cv::Mat g_y;
    cv::Mat a_x;
    cv::Mat a_y;

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

// Function to detect circles using Hough Transform OK
std::vector<cv::Vec3f> detectCircles(cv::Mat& image, int dp, int minDist, int param1, int param2, int minRadius, int maxRadius, int threshold_value, cv::Size gauss_ker) {
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

    // Apply a threshold to get a binary image, this is done to divide into use and useless points (section)
    cv::threshold(sobel, binary, threshold_value, 255, cv::THRESH_BINARY);

    // Create the vectore where store the circles
    std::vector<cv::Vec3f> circles;

    // Apply GaussianBlur to reduce noise and improve circle detection
    cv::GaussianBlur(binary, binary, gauss_ker, 1, 1); //1 1

    // Apply Hough Circle Transform on the binary image
    cv::HoughCircles(binary, circles, cv::HOUGH_GRADIENT, dp, minDist, param1, param2, minRadius, maxRadius);

    return circles;
}

// Function to draw bounding boxes around detected circles, just intermidiate function to check OK
void drawBoundingBoxes(cv::Mat& image, const std::vector<cv::Rect>& boundingBoxes) {
    // Draw correct bounding boxes with a different color
    // Correct resulst (ground truth)
    // Draw detected bounding boxes
    for (const auto& box : boundingBoxes) {
        cv::rectangle(image, box, cv::Scalar(0, 0, 255), 2); // Red color for detected bounding boxes
    }
}

// Function to merge circles that are close to each other with radius_threshold OK 
std::vector<cv::Vec3f> mergeCircles(const std::vector<cv::Vec3f>& circles, float radius_threshold) {
    // Vector where load all the cleaned circles
    std::vector<cv::Vec3f> mergedCircles;

    // Needed to see if a circle it has already been visited
    std::vector<bool> visited(circles.size(), false);

    for (int i = 0; i < circles.size(); ++i) {
        if (visited[i]) continue;

        cv::Point2f center_i = cv::Point2f(circles[i][0], circles[i][1]);
        int radius_i = circles[i][2];

        int count = 1;
        float sum_x = circles[i][0];
        float sum_y = circles[i][1];
        float sum_r = circles[i][2];

        for (int j = i + 1; j < circles.size(); ++j) {
            if (visited[j]) continue;

            cv::Point2f center_j = cv::Point2f(circles[j][0], circles[j][1]);
            int radius_j = circles[j][2];

            if (cv::norm(center_i - center_j) < radius_threshold) {
                sum_x += circles[j][0];
                sum_y += circles[j][1];
                sum_r += circles[j][2];
                count++;
                visited[j] = true;
            }
        }

        visited[i] = true;

        mergedCircles.push_back(cv::Vec3f(sum_x / count, sum_y / count, sum_r / count));
    }

    return mergedCircles;
}

// Function to check if a point is inside a polygon, corners of the table OK
bool isPointInsidePolygon(const cv::Point2f& point, const std::vector<cv::Point2f>& polygon) {
    return cv::pointPolygonTest(polygon, point, false) >= 0;
}

// Function to load polygon vertices from a text file OK
std::vector<cv::Point2f> loadPolygonVertices(const std::string& filepath) {
    // The original polygon derive from the txt
    std::vector<cv::Point2f> polygon_original;

    std::ifstream infile(filepath);

    // Check if it can be loaded
    if (!infile.is_open()) {
        std::cerr << "Could not open the file!\n";
        return polygon_original;
    }

    int x, y;
    while (infile >> x >> y) {
        polygon_original.push_back(cv::Point2f(x, y));
    }

    infile.close();

    return polygon_original;
}

// Function to load correct bounding boxes from a text file OK
std::vector<cv::Rect> loadCorrectBoundingBoxes(const std::string& filepath) {
    // Correct bounding boxes in the txt file 
    std::vector<cv::Rect> boundingBoxes;
    std::ifstream infile(filepath);

    // Check 
    if (!infile.is_open()) {
        std::cerr << "Could not open the file!\n";
        return boundingBoxes;
    }

    int x, y, w, h, i;
    while (infile >> x >> y >> w >> h >> i) {
        boundingBoxes.push_back(cv::Rect(x, y, w, h));
    }

    infile.close();
    return boundingBoxes;
}

//// Definition of the particular means of the false positives in BGR format 
//struct ColorMean {
//    cv::Vec3f black = {44, 41, 35};
//    cv::Vec3f blue1 = {144, 106, 26};
//    cv::Vec3f blue2 = {82, 46, 3};
//    cv::Vec3f blue3 = {140, 115, 40};
//    cv::Vec3f grey = {86, 82, 85};
//    cv::Vec3f green = {76, 102, 18};
//    cv::Vec3f pink1 = {164, 177, 243};
//    cv::Vec3f pink2 = {79, 77, 89};
//    cv::Vec3f pink3 = {167, 197, 223};
//    cv::Vec3f brown = {111, 144, 168};
//};

// Check if the color is closer to the particular ones
bool isColorClose(const cv::Vec3f& color1, const cv::Vec3f& color2, float margin) {
    return std::abs(color1[0] - color2[0]) <= margin &&
           std::abs(color1[1] - color2[1]) <= margin &&
           std::abs(color1[2] - color2[2]) <= margin;
}

// Check on the false positives, derived from the color matching 
bool isFalsePositive(const cv::Mat& image, const cv::Rect& bbox, const ColorMean& means, float threshold, float margin) {
    cv::Mat roi = image(bbox);
    int totalPixels = roi.rows * roi.cols;

    std::vector<cv::Vec3f> meanColors = {means.blue1, means.blue2, means.blue3, means.grey, means.green, means.pink1, means.pink2, means.brown};
    for (const auto& meanColor : meanColors) {
        int closePixels = 0;
        for (int y = 0; y < roi.rows; ++y) {
            for (int x = 0; x < roi.cols; ++x) {
                cv::Vec3b pixel = roi.at<cv::Vec3b>(y, x);
                cv::Vec3f pixelF = {static_cast<float>(pixel[0]), static_cast<float>(pixel[1]), static_cast<float>(pixel[2])};

                // Check if the pixel is close to one of the mean colors
                if (isColorClose(pixelF, meanColor, margin)) {
                    ++closePixels;
                }
            }
        }
        if (static_cast<float>(closePixels) / totalPixels > threshold) {
            return true;
        }
    }
    return false;
}

// Filtered based on color 
std::vector<cv::Rect> filterBoundingBoxes(const cv::Mat& image, const std::vector<cv::Rect>& bboxes, const ColorMean& means, float threshold, float margin) {
    std::vector<cv::Rect> filteredBboxes;
    for (const auto& bbox : bboxes) {
        if (!isFalsePositive(image, bbox, means, threshold, margin)) {
            filteredBboxes.push_back(bbox);
        }
    }
    return filteredBboxes;
}

// Distance between bounding boxes
double calculateDistance(cv::Point2f p1, cv::Point2f p2) {
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

// Check on the intersection area
double intersectionArea(const cv::Rect& r1, const cv::Rect& r2) {
    int x_overlap = std::max(0, std::min(r1.x + r1.width, r2.x + r2.width) - std::max(r1.x, r2.x));
    int y_overlap = std::max(0, std::min(r1.y + r1.height, r2.y + r2.height) - std::max(r1.y, r2.y));
    return x_overlap * y_overlap;
}

// Merged rectangle
cv::Rect mergeBoundingBoxes(const cv::Rect& r1, const cv::Rect& r2) {
    int x1 = std::min(r1.x, r2.x);
    int y1 = std::min(r1.y, r2.y);
    int x2 = std::max(r1.x + r1.width, r2.x + r2.width);
    int y2 = std::max(r1.y + r1.height, r2.y + r2.height);
    return cv::Rect(x1, y1, x2 - x1, y2 - y1);
}

// Merging function
void mergeBoundingBoxes(std::vector<cv::Rect>& boundingBoxes, int& pixeldistance, float& dimdifference, float& sharedarea) {
    for (int i = 0; i < boundingBoxes.size(); ++i) {
        for (int j = i + 1; j < boundingBoxes.size();) {
            double areaI = boundingBoxes[i].area();
            double areaJ = boundingBoxes[j].area();

            // Centers of the bb
            cv::Point2f centerI(boundingBoxes[i].x + boundingBoxes[i].width / 2.0,
                                boundingBoxes[i].y + boundingBoxes[i].height / 2.0);
            cv::Point2f centerJ(boundingBoxes[j].x + boundingBoxes[j].width / 2.0,
                                boundingBoxes[j].y + boundingBoxes[j].height / 2.0);

            // Call to the function for Distance between centers
            double centerDistance = calculateDistance(centerI, centerJ);

            // Check if one box is within the neighborhood of the other and is 30% smaller
            if (centerDistance < pixeldistance) {
                if ((areaI < dimdifference * areaJ) || (areaJ < dimdifference * areaI)) {
                    boundingBoxes[i] = mergeBoundingBoxes(boundingBoxes[i], boundingBoxes[j]);
                    boundingBoxes.erase(boundingBoxes.begin() + j);
                    continue;
                }
            }

            // Intersection area
            double intersectArea = intersectionArea(boundingBoxes[i], boundingBoxes[j]);

            // Check if they share more a threshold value
            if (intersectArea > sharedarea * std::min(areaI, areaJ)) {
                boundingBoxes[i] = mergeBoundingBoxes(boundingBoxes[i], boundingBoxes[j]);
                boundingBoxes.erase(boundingBoxes.begin() + j);
            } else {
                ++j;
            }
        }
    }
    
}

// Overall function
std::vector<cv::Rect> ballsDetection(cv::Mat img, std::vector<cv::Point2f> polygon) {

    std::vector<cv::Point2f> scheme_corners = {cv::Point2f(82, 81), cv::Point2f(82, 756), cv::Point2f(1384, 756), cv::Point2f(1384, 81)}; // 3,4
    std::vector<cv::Point2f> smaller_corners = {cv::Point2f(94, 100), cv::Point2f(94, 737), cv::Point2f(1365, 737), cv::Point2f(1365, 100)};

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Parameters for Hough Circle Transform
    int dp = 1;
    int minDist = 13; 
    int param1 = 100;
    int param2 = 16;
    int minRadius = 6;
    int maxRadius = 16;
    int threshold_value = 55;
    cv::Size gauss_ker(9, 9);

    // Other
    float merge_factor = 8.0; // circle distances to merge
    int squareSize = 35;    // square size of the block on the hole

    // Remove overlapping bbox 
    int pxeldistance = 16; // Distances of the border
    float dimdifference = 0.6; // size difference of the box
    float sharedarea = 0.6;

    // Color check false positive
    float pixelsofcolor = 0.10;
    int margincolor = 10;

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    cv::Mat image = img.clone();

    if (image.empty()) {
        std::cerr << "Could not open or find the image!\n";
        return {};
    }
 
    if (polygon.empty()) {
        std::cerr << "Could not load polygon vertices!\n";
        return {};
    }

    cv::Mat H = best_homog_detection(polygon, scheme_corners);
    cv::Mat H_inv;
    cv::invert(H, H_inv);
    
    std::vector<cv::Point2f> scheme_holes = {cv::Point2f(94, 100), cv::Point2f(94, 737), cv::Point2f(1365, 737), cv::Point2f(1365, 100), cv::Point2f(730, 100), cv::Point2f(730, 737) };
        

    std::vector<cv::Point2f> footage_holes, smaller_corners_footage;
    cv::perspectiveTransform(scheme_holes, footage_holes, H_inv);
    cv::perspectiveTransform(smaller_corners, smaller_corners_footage, H_inv);

    for (const cv::Point2f& vertex : footage_holes) {
        cv::Point2f topLeft(vertex.x - squareSize / 2, vertex.y - squareSize / 2);
        cv::rectangle(image, cv::Rect(topLeft, cv::Size(squareSize, squareSize)), cv::Scalar(0, 255, 0), cv::FILLED);
    }

    
    cv::Mat displayImage = image.clone();


    std::vector<cv::Mat> channels;
    cv::split(displayImage, channels);

    std::set<cv::Vec3f, Vec3fComparator> unifiedCircles;

    // RGB Color scheme
    // RED component
    std::vector<cv::Vec3f> redCircles = detectCircles(channels[2], dp, minDist, param1, param2, minRadius, maxRadius, threshold_value, gauss_ker);
    unifiedCircles.insert(redCircles.begin(), redCircles.end());
    
    // BLUE component
    std::vector<cv::Vec3f> blueCircles = detectCircles(channels[0], dp, minDist, param1, param2, minRadius, maxRadius, threshold_value, gauss_ker);
    unifiedCircles.insert(blueCircles.begin(), blueCircles.end());

    //HSV 
    // Value component
    cv::Mat hsvImage;
    cv::cvtColor(displayImage, hsvImage, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> hsvChannels;
    cv::split(hsvImage, hsvChannels);
    std::vector<cv::Vec3f> valueCircles = detectCircles(hsvChannels[2], dp, minDist, param1, param2, minRadius, maxRadius, threshold_value, gauss_ker);
    unifiedCircles.insert(valueCircles.begin(), valueCircles.end());

    //Saturation component
    cv::Mat hsvImage2;
    cv::cvtColor(displayImage, hsvImage2, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> hsvChannelssat;
    cv::split(hsvImage2, hsvChannelssat);
    std::vector<cv::Vec3f> satCircles = detectCircles(hsvChannelssat[1], dp, minDist, param1, param2, minRadius, maxRadius, threshold_value, gauss_ker);
    unifiedCircles.insert(satCircles.begin(), satCircles.end());


    // MERGE of all the circles
    std::vector<cv::Vec3f> mergedCircles = mergeCircles(std::vector<cv::Vec3f>(unifiedCircles.begin(), unifiedCircles.end()), merge_factor);

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
    mergeBoundingBoxes(detectedBoundingBoxes, pxeldistance, dimdifference, sharedarea);

    // REMOVE FALSE POSITIVES
    ColorMean means;
    std::vector<cv::Rect> filteredBboxes = filterBoundingBoxes(displayImage, detectedBoundingBoxes, means, pixelsofcolor, margincolor);

    // Draw both detected and correct bounding boxes
    drawBoundingBoxes(displayImage, filteredBboxes);
    
    // Draw the polygon on the image
    drawPolygon(displayImage, smaller_corners_footage, cv::Scalar(0, 255, 0), 2); // Green color with thickness 2

    cv::imshow("Hough Circles", displayImage);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return filteredBboxes;
}
