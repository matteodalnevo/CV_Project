// Prevedello Aaron

#ifndef TABLEDETECTION_H
#define TABLEDETECTION_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <tuple>

/** @brief Split lines between horizontal and vertical lines, based on theta.
 *  Lines are assumed to be expressed by parameters rho and theta, like output of function cv::HoughLines 
 *  @param lines vector of all the lines that must be splitted
 *  @param horizontalLines vector where to insert the horizontal lines founded 
 *  @param verticalLines vector where to insert the vertical lines founded
 *  @return Mean values of rho for both horizontal and vertical lines
 */
std::tuple<float, float> splitHorVertLines(const std::vector<cv::Vec2f> &lines, std::vector<cv::Vec2f> &horizontalLines, std::vector<cv::Vec2f> &verticalLines );


/** @brief Draw a set of lines on the given image wioth the specified color
 *  @param lines lines to be plotted
 *  @param img img where to plot the lines
 *  @param colour colour of the lines
 */
static void drawLines(std::vector<cv::Vec2f> lines, cv::Mat img, cv::Scalar colour);

/** @brief Given horizontal and vertical lines, and their mean values, split into top, bottom, left, right.
 * @param horizontalLine vector of horizontal lines to be splitted
 * @param verticalLine vector of vertical lines to be splitted
 * @param meanVert mean value of rho of the group of vertical lines passed
 * @param meanHoriz mean value of rho of the group of horizontal lines passed
 * @return Four vectors of lines: top, bottom, left, right
 */
std::tuple<std::vector<cv::Vec2f>, std::vector<cv::Vec2f>, std::vector<cv::Vec2f>, std::vector<cv::Vec2f>> findGroupOfLines(const std::vector<cv::Vec2f> &horizontalLine, const std::vector<cv::Vec2f> &verticalLine,const float meanVert, const float meanHoriz);


/** @brief Draw the single line on the given image, you can also choose the color.
 *  @param line line to be plotted
 *  @param img img where to plot the line
 *  @param colour colour of the line
 */
static void drawSingleLine(cv::Vec2f lines, cv::Mat img, cv::Scalar colour);

/** @brief Compute the coordinates of the interception between two lines. 
 * Internally compute slope and intercept of the two lines.
 * @param line1 first line 
 * @param line2 second line
 */
cv::Point2f computeIntercept (cv::Vec2f line1, cv::Vec2f line2);

/** @brief Given the four lines for the table, already splitted in left, right, top, bottom, 
 * compute coordinates of the four corners of the table.
 * @param topHoriz horizontal line on top
 * @param lowHOriz horizontal line bottom
 * @param leftVert left vertical line
 * @param rightVert right vertical line
 * @return Returns the four corners, from top left in counter-clock wise order
 */
std::tuple<cv::Point2f, cv::Point2f, cv::Point2f, cv::Point2f> computeCorners(cv::Vec2f topHoriz, cv::Vec2f lowHoriz, cv::Vec2f leftVert, cv::Vec2f rightVert);

/** @brief Check if left and right lines are swapped, and correct them in case it is needed.
 * @param left left line
 * @param right right line
 */
static void checkLeftRight (cv::Vec2f &left, cv::Vec2f &right); 

/** @brief Given a vector of lines, compute the mean value of theta and rho.
 * @param lineVector vector of lines
 * @return Return mean rho and theta 
 */
cv::Vec2f findMediumLine(const std::vector<cv::Vec2f> &lineVector); 

/** @brief Function that takes in input a vector of lines and extrapolate the corners of the table.
 * @param lines vector of lines detected on the image
 * @return vector of points representing the corners of the table
 */
std::vector<cv::Point2f> tableDetection(std::vector<cv::Vec2f> lines);

#endif // TABLEDETECTION_H