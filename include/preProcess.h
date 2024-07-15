// preProcess.h
#ifndef PREPROCESS_H
#define PREPROCESS_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <tuple>

std::tuple<cv::Mat, std::vector<cv::Vec2f>> preProcess(const cv::Mat& image);


#endif // PREPROCESS_H