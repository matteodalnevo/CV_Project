// tableDetection.h
#ifndef TABLEDETECTION_H
#define TABLEDETECTION_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <tuple>

//cv::Mat computeHist(cv::Mat image, int bins_number); 
//static void computeMeanShift(int, void*);
//std::tuple<cv::Vec2f, cv::Vec2f, cv::Vec2f, cv::Vec2f> findRepresentativeLine(std::vector<cv::Vec2f> horizontalLine, std::vector<cv::Vec2f> verticalLine, float meanVert, float meanHoriz);


/** @brief Split lines between horizontal and vertical lines, based on theta.
 *  Lines are assumed to be expressed by parameters rho and theta, like output of function cv::HoughLines 
 *  @param lines vector of all the lines that must be splitted
 *  @param horizontalLines vector where to insert the horizontal lines founded 
 *  @param verticalLines vector where to insert the vertical lines founded
 */
std::tuple<float, float> splitHorVertLines(const std::vector<cv::Vec2f> &lines, std::vector<cv::Vec2f> &horizontalLines, std::vector<cv::Vec2f> &verticalLines );


/** @brief Draw a set of lines on the given image wioth the specified color
 *  @param 
 */
static void drawLines(std::vector<cv::Vec2f> lines, cv::Mat img, cv::Scalar colour);

/** @brief Given horizontal and vertical lines, and their mean values, split into top, bottom, left, right. Output also the mean values.

 */
std::tuple<std::vector<cv::Vec2f>, std::vector<cv::Vec2f>, std::vector<cv::Vec2f>, std::vector<cv::Vec2f>> findGroupOfLines(std::vector<cv::Vec2f> horizontalLine, std::vector<cv::Vec2f> verticalLine, float meanVert, float meanHoriz);


/** @brief Draw the single line on the given image, you can also choose the color.

 */
static void drawSingleLine(cv::Vec2f lines, cv::Mat img, cv::Scalar colour);

/** @brief Compute the coordinates of the interception between two lines. Internally compute slope and intercept of the two lines.
 * 
 * @param lines in format of rho and theta

 */
cv::Point2f computeIntercept (cv::Vec2f line1, cv::Vec2f line2);


/** @brief Given the four lines for the table, compute coordinates of the four corners of the table.

 */
std::tuple<cv::Point2f, cv::Point2f, cv::Point2f, cv::Point2f> computeCorners(cv::Vec2f topHoriz, cv::Vec2f lowHoriz, cv::Vec2f leftVert, cv::Vec2f rightVert);

std::vector<cv::Point2f> tableDetection(std::vector<cv::Vec2f> lines);


/** @brief Stupid check on left and right vertical lines, very hard-coded.

 */
static void checkLeftRight (cv::Vec2f &left, cv::Vec2f &right); 

/** @brief Given a vector of lines, compute the medium value that represents the set.

 */
cv::Vec2f findMediumLine(std::vector<cv::Vec2f> lineVector); 

cv::Mat enhanceColourContrast (cv::Mat input_img);

cv::Mat computeMask(cv::Mat image); 

#endif // TABLEDETECTION_H