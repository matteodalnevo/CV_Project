// ballClassification.h
#ifndef BALLCLASSIFICATION_H
#define BALLCLASSIFICATION_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include "utils.h"

std::vector<BoundingBox> ballClassification(cv::Mat& image, const std::vector<cv::Rect>& Bbox_rect, const cv::Vec3b& tableColor);

#endif // BALLCLASSIFICATION_H