// MATTEO DAL NEVO - ID: 2087919

// utils.h
#ifndef UTILS_H
#define UTILS_H

#include <opencv2/highgui.hpp>
#include <string>
#include <vector>


/**
 * @brief Represents a bounding box with an ID
 * 
 * This struct defines a bounding box with its position, sizeand an ID
 */
struct BoundingBox {
    cv::Rect box; // The bounding box defined by a cv::Rect
    int ID; // An identifier for the bounding box
};

/**
 * @brief Displays an image in a window
 * 
 * This function creates a window with the specified name and displays the given image in that window
 * The window will remain open until a key is pressed
 * 
 * @param image The image to be displayed
 * @param windowName The name of the window where the image will be shown
 */
void showImage(const cv::Mat& image, const std::string& windowName);

/**
 * @brief Reads bounding boxes from a file
 * 
 * This function reads bounding box data from a specified file and populates the provided vector
 * with `BoundingBox` objects
 * 
 * @param filePath The path to the file containing bounding box data
 * @param boundingBoxes A vector to be populated with the read bounding boxes
 * @return `true` if the file was successfully read and parsed, `false` otherwise
 */
bool readBoundingBoxes(const std::string& filePath, std::vector<BoundingBox>& boundingBoxes);

/**
 * @brief Retrieves a color based on an integer value
 * 
 * This function maps an integer value to a specific color. The color is returned as a `cv::Scalar`
 * 
 * @param value The integer value used to determine the color
 * @return The color corresponding to the provided value
 */
cv::Scalar getColorForValue(int value);

/**
 * @brief Maps a grayscale mask to a color image
 * 
 * This function converts a grayscale mask image to a color image by mapping grayscale values to specific colors
 * 
 * @param grayscaleMask The input grayscale mask image
 * @param colorImage The output color image that will be created from the grayscale mask
 */
void mapGrayscaleMaskToColorImage(const cv::Mat& grayscaleMask, cv::Mat& colorImage);

/**
 * @brief Draws a bounding box on an image
 * 
 * This function draws a bounding box on the given image with a color and transparency based on the bounding box's ID
 * The bounding box is drawn on a transparent overlay and blended with the original image
 * 
 * @param image The image on which to draw the bounding box. It is modified in place
 * @param bbox The bounding box to be drawn, including its position and ID
 */
void plotBBox(cv::Mat& image, BoundingBox& bbox);

/**
 * @brief Draws a table's profile and classified bounding boxes on an image
 * 
 * This function takes an image and draws a yellow quadrilateral based on the given vertices
 * It also draws classified bounding boxes using the plotBBox function
 * 
 * @param image The image on which to draw the profile and bounding boxes. It is modified in place
 * @param vertices A vector of points representing the vertices of the table's profile
 *                  These points will be connected to form a yellow quadrilateral
 * @param classified_boxes A vector of BoundingBox objects representing the classified bounding boxes that will be drawn on the image
 * 
 */
void outputBBImage(cv::Mat& image, std::vector<cv::Point2f> vertices, std::vector<BoundingBox> classified_boxes);

/** MARCO PANIZZZO 
 * @brief Segments an image by filling a polygon area with a specific color and drawing circles
 *        at the center of classified bounding boxes. It also applies a hand mask to blend the result
 * 
 * @param img The input image to be segmented
 * @param footage_corners The vector of points defining the corners of the polygon to be filled
 * @param classified_boxes The vector of BoundingBox objects representing classified objects
 * @param hand_mask The mask image representing the areas to be removed
 * @return The segmented image with the polygon area filled and circles drawn, blended using the hand mask
 */
cv::Mat segmentation(const cv::Mat img, const std::vector<cv::Point2f> footage_corners, const std::vector<BoundingBox> classified_boxes, cv::Mat hand_mask);


#endif // UTILS_H
