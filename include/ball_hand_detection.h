// ballDetection.h
#ifndef BALL_HAND_DETECTION_H
#define BALL_HAND_DETECTION_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include <set>
#include <vector>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <numeric>
#include <cmath>


/**
 * @brief Computes the median color of an image, excluding black pixels ([0, 0, 0]).
 * 
 * This function calculates the median color of the given image by extracting the pixel 
 * values for each color channel (blue, green, and red), ignoring black pixels. It then 
 * determines the median value for each channel and combines them into a single median color.
 *
 * @param image A constant reference to a cv::Mat object representing the input image.
 *              The image is expected to be in BGR format.
 * 
 * @return cv::Vec3b A vector of three unsigned characters representing the median color 
 *                   in BGR format.
 */
cv::Vec3b MedianColor(const cv::Mat& image);


/**
 * @brief Computes the median color of a specific quadrilateral region in an image.
 * 
 * This function takes an image and a set of vertices defining a quadrilateral region of interest (ROI).
 * It creates a mask to isolate this region, draws the quadrilateral on a copy of the image, and computes 
 * the median color of the pixels within the ROI. The median color is calculated using the `MedianColor` 
 * function.
 *
 * @param image A constant reference to a cv::Mat object representing the input image.
 *              The image is expected to be in BGR format.
 * @param vertices A vector of cv::Point2f objects representing the vertices of the quadrilateral 
 *                 region of interest.
 * 
 * @return cv::Vec3b A vector of three unsigned characters representing the median color 
 *                   of the specified region in BGR format.
 */
cv::Vec3b tableColor(const cv::Mat& image, std::vector<cv::Point2f> vertices);


/**
 * @brief Computes the best homography matrix based on minimizing the stretching error.
 * 
 * This function calculates four different homography matrices by rotating the corners of 
 * the footage table and compares the diagonal stretching components of each matrix to 
 * determine the one with the lowest error. The homography matrix with the smallest error 
 * is returned.
 *
 * @param footage_table_corners A vector of cv::Point2f objects representing the corners 
 *                              of the table in the footage.
 * @param scheme_table_corners A vector of cv::Point2f objects representing the corners 
 *                             of the table in the scheme.
 * 
 * @return cv::Mat The homography matrix with the smallest stretching error.
 */
cv::Mat best_homog_detection(std::vector<cv::Point2f> footage_table_corners, std::vector<cv::Point2f> scheme_table_corners);


/**
 * @brief Draws a polygon on an image.
 * 
 * This function takes an image and draws a polygon on it using the specified color and thickness.
 * The polygon is defined by a vector of points. The function ensures that the polygon has at 
 * least 4 points before attempting to draw it.
 *
 * @param image A reference to a cv::Mat object representing the image on which the polygon 
 *              will be drawn.
 * @param polygon A constant reference to a vector of cv::Point2f objects representing the 
 *                vertices of the polygon.
 * @param color A constant reference to a cv::Scalar object representing the color of the 
 *              polygon's lines.
 * @param thickness An integer representing the thickness of the polygon's lines.
 */
void drawPolygon(cv::Mat& image, const std::vector<cv::Point2f>& polygon, const cv::Scalar& color, const int thickness);


/**
 * @brief Comparator for cv::Vec3f elements to enable their comparison.
 * 
 * This structure defines a custom comparator for cv::Vec3f elements, allowing 
 * them to be compared based on their components. The comparison is performed 
 * lexicographically: first by the first component, then by the second if the 
 * first components are equal, and finally by the third if the first and second 
 * components are equal.
 */
struct Vec3fComparator {
    bool operator() (const cv::Vec3f& a, const cv::Vec3f& b) const {
        if (a[0] != b[0]) return a[0] < b[0];
        if (a[1] != b[1]) return a[1] < b[1];
        return a[2] < b[2];
    }
};


/**
 * @brief Computes the Sobel gradient magnitude of a grayscale image.
 * 
 * This function calculates the Sobel gradient magnitude of a given grayscale image. 
 * It uses the Sobel operator to compute the gradient in both the x and y directions, 
 * normalizes the results, and then combines them to produce the final gradient magnitude 
 * image. The output image highlights edges by representing the gradient magnitude.
 *
 * @param gray A constant reference to a cv::Mat object representing the input grayscale 
 *             image. The image is expected to be in 8-bit single-channel format.
 * 
 * @return cv::Mat A cv::Mat object representing the gradient magnitude image, with edge 
 *                 information highlighted. The image is in 8-bit single-channel format.
 */
cv::Mat computeSobel(const cv::Mat& gray);


/**
 * @brief Detects circles in an image using the Hough Circle Transform.
 * 
 * This function performs circle detection on the input image using the Hough Circle 
 * Transform method. It first converts the image to grayscale if it is in color, then 
 * applies the Sobel operator to detect edges, followed by binary thresholding to isolate 
 * potential circle regions. Gaussian blur is applied to reduce noise before performing 
 * the Hough Circle Transform to detect circles.
 * 
 * @param image A reference to a cv::Mat object representing the input image. The image 
 *              can be in color or grayscale.
 * @param dp Inverse ratio of the accumulator resolution to the image resolution. A value 
 *           of 1 means the accumulator has the same resolution as the input image.
 * @param minDist Minimum distance between the centers of detected circles. This helps 
 *                 avoid detecting multiple circles that are too close to each other.
 * @param param1 First method-specific parameter for the Hough Circle Transform, typically 
 *                the higher threshold for the Canny edge detector.
 * @param param2 Second method-specific parameter for the Hough Circle Transform, typically 
 *                the accumulator threshold for the circle centers at the detection stage.
 * @param minRadius Minimum circle radius to be detected.
 * @param maxRadius Maximum circle radius to be detected.
 * @param threshold_value Threshold value used for binary thresholding of the Sobel edge image.
 * @param gauss_ker Size of the Gaussian kernel used for blurring the binary image to reduce noise.
 * 
 * @return std::vector<cv::Vec3f> A vector of cv::Vec3f objects, each representing a detected 
 *                                 circle. Each cv::Vec3f contains the (x, y) coordinates of the 
 *                                 circle center and the circle radius.
 */
std::vector<cv::Vec3f> detectCircles(cv::Mat& image, const int dp, const int minDist, const int param1, const int param2, const int minRadius, const int maxRadius, const int threshold_value, const cv::Size gauss_ker);


/**
 * @brief Draws bounding boxes on an image.
 * 
 * This function draws rectangles around detected objects on the input image. Each rectangle 
 * corresponds to a bounding box provided in the `boundingBoxes` vector. The rectangles are 
 * drawn in red with a specified thickness to highlight the detected regions.
 * 
 * @param image A reference to a cv::Mat object representing the image on which the bounding 
 *              boxes will be drawn. The image will be modified in place.
 * @param boundingBoxes A constant reference to a vector of cv::Rect objects, where each 
 *                      cv::Rect represents a detected bounding box to be drawn on the image.
 * 
 * @return void This function does not return any value.
 */
void drawBoundingBoxes(cv::Mat& image, const std::vector<cv::Rect>& boundingBoxes);


/**
 * @brief Merges overlapping or close circles into a single circle.
 * 
 * This function takes a list of detected circles and merges those that are close to each other 
 * based on a specified radius threshold. It calculates the average center and radius for circles 
 * that are deemed to be overlapping or close, and returns a vector of the merged circles.
 * 
 * @param circles A constant reference to a vector of cv::Vec3f objects, each representing 
 *                a detected circle. Each cv::Vec3f contains the (x, y) coordinates of the 
 *                circle center and the circle radius.
 * @param radius_threshold The maximum distance between the centers of two circles for them to 
 *                         be considered as overlapping or close and thus merged.
 * 
 * @return std::vector<cv::Vec3f> A vector of cv::Vec3f objects representing the merged circles. 
 *                                 Each cv::Vec3f contains the averaged (x, y) coordinates and 
 *                                 radius of the merged circles.
 */
std::vector<cv::Vec3f> mergeCircles(const std::vector<cv::Vec3f>& circles, float radius_threshold);


/**
 * @brief Determines if a point is inside a polygon.
 * 
 * This function uses `cv::pointPolygonTest` to check if a given point is within 
 * or on the boundary of a polygon.
 * 
 * @param point The point to check.
 * @param polygon The vertices of the polygon.
 * 
 * @return bool `true` if the point is inside or on the polygon; `false` otherwise.
 */
bool isPointInsidePolygon(const cv::Point2f& point, const std::vector<cv::Point2f>& polygon);


/**
 * @brief Loads polygon vertices from a file.
 * 
 * This function reads vertex coordinates from a text file and stores them as 
 * `cv::Point2f` objects in a vector. Each line in the file should contain an 
 * x and y coordinate.
 * 
 * @param filepath The path to the text file containing the vertex coordinates.
 * 
 * @return std::vector<cv::Point2f> A vector of polygon vertices.
 */
std::vector<cv::Point2f> loadPolygonVertices(const std::string& filepath);


/**
 * @brief Loads bounding boxes from a file.
 * 
 * This function reads bounding box parameters from a text file and stores them 
 * as `cv::Rect` objects in a vector. Each line in the file should contain the 
 * x, y coordinates, width, and height of a bounding box.
 * 
 * @param filepath The path to the text file containing the bounding box data.
 * 
 * @return std::vector<cv::Rect> A vector of bounding boxes.
 */
std::vector<cv::Rect> loadCorrectBoundingBoxes(const std::string& filepath);


/**
 * @brief Checks if two colors are close to each other.
 * 
 * This function compares two colors and determines if their difference in each 
 * channel (R, G, B) is within a specified margin.
 * 
 * @param color1 The first color.
 * @param color2 The second color.
 * @param margin The maximum allowed difference between the two colors.
 * 
 * @return bool `true` if the colors are within the margin; `false` otherwise.
 */
bool closeColor(const cv::Vec3f& color1, const cv::Vec3f& color2, const float margin);


/**
 * @brief Determines if a bounding box contains a significant amount of color close to a mean color.
 * 
 * This function checks if the proportion of pixels within a bounding box that are close to a 
 * specified mean color exceeds a given threshold. It compares pixel colors in the bounding box 
 * area to the mean color using a margin to determine "closeness."
 * 
 * @param image The image containing the bounding box.
 * @param bbox The bounding box to analyze.
 * @param mean The mean color to compare against.
 * @param threshold The proportion of close pixels required to consider it a false positive.
 * @param margin The maximum allowed difference for color closeness.
 * 
 * @return bool `true` if the proportion of close pixels exceeds the threshold; `false` otherwise.
 */
bool falsePositive(const cv::Mat& image, const cv::Rect& bbox, const cv::Vec3b& mean, const float threshold, const float margin);


/**
 * @brief Filters bounding boxes based on color similarity.
 * 
 * This function filters out bounding boxes from the list that contain a significant amount 
 * of color similar to the mean color of a specified region. It uses the `falsePositive` 
 * function to determine if a bounding box should be excluded.
 * 
 * @param image The image containing the bounding boxes.
 * @param bboxes A vector of bounding boxes to filter.
 * @param vertices The vertices defining the region for color mean calculation.
 * @param threshold The proportion of close pixels required to exclude a bounding box.
 * @param margin The maximum allowed color difference for similarity.
 * 
 * @return std::vector<cv::Rect> A vector of filtered bounding boxes.
 */
std::vector<cv::Rect> filterBoundingBoxes(const cv::Mat& image, const std::vector<cv::Rect>& bboxes, const std::vector<cv::Point2f>& vertices, const float threshold, const float margin);


/**
 * @brief Calculates the Euclidean distance between two points.
 * 
 * This function computes the straight-line distance between two points in 2D space using 
 * the Euclidean distance formula.
 * 
 * @param p1 The first point.
 * @param p2 The second point.
 * 
 * @return double The Euclidean distance between the two points.
 */
double calculateDistance(const cv::Point2f p1, const cv::Point2f p2);


/**
 * @brief Computes the area of intersection between two rectangles.
 * 
 * This function calculates the area of overlap between two rectangles. It determines the 
 * intersecting region by finding the overlapping segments along the x and y axes.
 * 
 * @param r1 The first rectangle.
 * @param r2 The second rectangle.
 * 
 * @return double The area of intersection between the two rectangles.
 */
double intersectionArea(const cv::Rect& r1, const cv::Rect& r2);


/**
 * @brief Computes the smallest bounding box that contains both input rectangles.
 * 
 * This function calculates a bounding box that encompasses both given rectangles by 
 * expanding to include the extents of both. It determines the minimum and maximum x and 
 * y coordinates to create a new rectangle that covers the area of both input rectangles.
 * 
 * @param r1 The first rectangle.
 * @param r2 The second rectangle.
 * 
 * @return cv::Rect The merged bounding box covering both input rectangles.
 */
cv::Rect mergedBoundingBoxes(const cv::Rect& r1, const cv::Rect& r2);


/**
 * @brief Merges bounding boxes based on proximity and overlap criteria.
 * 
 * This function iterates through a list of bounding boxes and merges those that are close to 
 * each other and meet specified criteria for area similarity and overlap. Bounding boxes are 
 * merged if they are within a certain pixel distance, have area dimensions that are within a 
 * specified ratio, or if they share a significant overlap area.
 * 
 * @param boundingBoxes A vector of bounding boxes to be merged.
 * @param pixeldistance The maximum distance between centers of boxes for them to be considered for merging.
 * @param dimdifference The maximum ratio of area dimensions for merging boxes with significant size difference.
 * @param sharedarea The minimum overlap ratio required to merge bounding boxes.
 */
void mergeBoundingBoxes(std::vector<cv::Rect>& boundingBoxes, const int pixeldistance, const float dimdifference, const float sharedarea);


/**
 * @brief Filters out bounding boxes that overlap with detected hand regions.
 * 
 * This function processes an image to identify hand regions by converting it to HSV color space 
 * and analyzing the saturation channel. A mask is created based on a saturation threshold and 
 * refined using a polygonal area of interest. Bounding boxes that overlap with detected hand regions 
 * are removed from the provided list. The function returns an image with the hand regions highlighted.
 * 
 * @param bbox Vector of bounding boxes to be filtered.
 * @param image The input image on which filtering is applied.
 * @param areaOfInterest Polygonal region where hand detection is focused.
 * @param threshold_hand Threshold value for saturation to create the mask.
 * 
 * @return An image with hand regions highlighted against a white background.
 */
cv::Mat HandMaskFiltering(std::vector<cv::Rect>& bbox, const cv::Mat& image, const std::vector<cv::Point2f>& areaOfInterest, const double threshold_hand);


/**
 * @brief Detects and filters balls and hand regions in an image.
 * 
 * This function detects circles (representing balls) in an image using multiple color channels 
 * and applies a series of filtering and merging steps. It also removes detected bounding boxes 
 * that overlap with hand regions, identified via a saturation threshold in the HSV color space. 
 * The function performs the following tasks:
 * - Transforms the image perspective to align with a predefined scheme.
 * - Detects circles using Hough Circle Transform in various color spaces.
 * - Filters and merges detected circles to create bounding boxes.
 * - Applies a color filter to remove boxes with colors similar to the table.
 * - Filters out bounding boxes that overlap with hand regions.
 * 
 * @param img The input image for processing.
 * @param polygon The polygon defining the area of interest in the image.
 * 
 * @return A tuple containing:
 *         - A vector of filtered bounding boxes around detected balls.
 *         - An image where hand regions are marked (black areas) and the rest is white.
 */
std::tuple<std::vector<cv::Rect>, cv::Mat> ballsHandDetection(const cv::Mat& img, const std::vector<cv::Point2f> polygon);


#endif // BALLDETECTION_H
