// ballDetection.h
#ifndef BALLDETECTION_H
#define BALLDETECTION_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

cv::Mat ballDetection(const cv::Mat& image, std::vector<cv::Point> vertices);

#endif // BALLDETECTION_H