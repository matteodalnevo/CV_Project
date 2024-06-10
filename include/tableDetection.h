// tableDetection.h
#ifndef TABLEDETECTION_H
#define TABLEDETECTION_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

cv::Mat computeHist(cv::Mat image, int bins_number); 
static void computeMeanShift(int, void*);
std::tuple<float, float> splitHorVertLines(std::vector<cv::Vec2f> lines, std::vector<cv::Vec2f> &horizontalLines, std::vector<cv::Vec2f> &verticalLines );
static void drawLines(std::vector<cv::Vec2f> lines, cv::Mat img, cv::Scalar colour);
std::tuple<std::vector<cv::Vec2f>, std::vector<cv::Vec2f>, std::vector<cv::Vec2f>, std::vector<cv::Vec2f>> findGroupOfLines(std::vector<cv::Vec2f> horizontalLine, std::vector<cv::Vec2f> verticalLine, float meanVert, float meanHoriz);
std::tuple<cv::Vec2f, cv::Vec2f, cv::Vec2f, cv::Vec2f> findRepresentativeLine(std::vector<cv::Vec2f> horizontalLine, std::vector<cv::Vec2f> verticalLine, float meanVert, float meanHoriz);
static void drawSingleLine(cv::Vec2f lines, cv::Mat img, cv::Scalar colour);
cv::Point computeIntercept (cv::Vec2f line1, cv::Vec2f line2);
std::tuple<cv::Point, cv::Point, cv::Point, cv::Point> computeCorners(cv::Vec2f topHoriz, cv::Vec2f lowHoriz, cv::Vec2f leftVert, cv::Vec2f rightVert);
cv::Mat tableDetection(const cv::Mat& image);
static void checkLeftRight (cv::Vec2f &left, cv::Vec2f &right); 
cv::Vec2f findMediumLine(std::vector<cv::Vec2f> lineVector);    

#endif // TABLEDETECTION_H