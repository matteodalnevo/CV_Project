#include "ballClassification.h"
#include <iostream>
#include "utils.h"

// Function to calculate the percentage of white pixels in an image
double calculateWhitePixelPercentage(const cv::Mat& image, int threshold) {
    // Convert to grayscale if not already
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image;
    }

    // Apply a binary threshold to get a binary image
    cv::Mat binary;
    cv::threshold(gray, binary, threshold, 255, cv::THRESH_BINARY); // You can adjust the threshold value

    //cv::imshow("bin", binary);
    //cv::waitKey(0);

    // Count the white pixels
    int whitePixelCount = cv::countNonZero(binary);
    
    // Calculate the total number of pixels
    int totalPixels = image.rows * image.cols;

    // Calculate the percentage of white pixels
    double whitePixelPercentage = (static_cast<double>(whitePixelCount) / totalPixels) * 100.0;
    return whitePixelPercentage;
}

double calculateBlackPixelPercentage(const cv::Mat& image) {
    // Convert to grayscale if not already
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image;
    }

    // Apply a binary threshold to get a binary image
    cv::Mat binary;
    cv::threshold(gray, binary, 43, 255, cv::THRESH_BINARY_INV); // Adjust the threshold value if needed

    //cv::imshow("gray", gray);
    //cv::imshow("bin", binary);
    //cv::waitKey(0);

    // Count the black pixels (which are white in the binary image)
    int blackPixelCount = cv::countNonZero(binary);

    // Calculate the total number of pixels
    int totalPixels = image.rows * image.cols;

    // Calculate the percentage of black pixels
    double blackPixelPercentage = (static_cast<double>(blackPixelCount) / totalPixels) * 100.0;
    return blackPixelPercentage;
}

// Function to classify the ball based on the percentage of white pixels
void classifyWhiteBall(cv::Mat& image, BoundingBox& bbox) {
    cv::rectangle(image, bbox.box, cv::Scalar(0, 255, 0), 2);
    bbox.ID = 1;
}

void classifyBlackBall(cv::Mat& image, BoundingBox& bbox) {
    cv::rectangle(image, bbox.box, cv::Scalar(0, 0, 0), 2);
    bbox.ID = 2;
}

void classifyOtherBall(cv::Mat& image, BoundingBox& bbox, double variance, double threshold) {
    cv::Scalar color;
    if (variance < threshold) {
        color = cv::Scalar(0, 0, 255); // Red - solid ball
        cv::rectangle(image, bbox.box, color, 2);
        bbox.ID = 3;
    } else {
        color = cv::Scalar(255, 0, 0); // Blue - Stride ball
        cv::rectangle(image, bbox.box, color, 2);
        bbox.ID = 4;
    }
    
}

double computeImageVariance(const cv::Mat& image) {
    // Convert to grayscale if not already
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image;
    }

    // Calculate the mean of the pixel values
    cv::Scalar meanScalar = cv::mean(gray);
    double mean = meanScalar[0];

    // Calculate the variance
    double variance = 0.0;
    for (int i = 0; i < gray.rows; ++i) {
        for (int j = 0; j < gray.cols; ++j) {
            double pixelValue = static_cast<double>(gray.at<uchar>(i, j));
            variance += (pixelValue - mean) * (pixelValue - mean);
        }
    }
    variance /= (gray.rows * gray.cols);

    return variance;
}

void applyKMeans(const cv::Mat& src, cv::Mat& dst, int clusterCount) {
    // Convert image to data points
    cv::Mat samples(src.rows * src.cols, 3, CV_32F);
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            for (int z = 0; z < 3; z++) {
                samples.at<float>(y + x * src.rows, z) = src.at<cv::Vec3b>(y, x)[z];
            }
        }
    }

    // Apply K-means clustering
    cv::Mat labels;
    int attempts = 5;
    cv::Mat centers;
    cv::kmeans(samples, clusterCount, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0), attempts, cv::KMEANS_PP_CENTERS, centers);

    // Create output image
    dst.create(src.size(), src.type());
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            int cluster_idx = labels.at<int>(y + x * src.rows, 0);
            dst.at<cv::Vec3b>(y, x)[0] = centers.at<float>(cluster_idx, 0);
            dst.at<cv::Vec3b>(y, x)[1] = centers.at<float>(cluster_idx, 1);
            dst.at<cv::Vec3b>(y, x)[2] = centers.at<float>(cluster_idx, 2);
        }
    }
}

double computeRecursiveMean(double new_value) {
    static double current_mean = 0.0;
    static int count = 0;

    count++;
    current_mean += (new_value - current_mean) / count;

    return current_mean;
}


//////////////////////////////////////////

struct Point {
    double value;
    int cluster;

    Point(double value = 0) : value(value), cluster(-1) {}
};

double euclideanDistance(double a, double b) {
    return std::abs(a - b);
}

void initializeCentroids(std::vector<double>& centroids, const std::vector<Point>& points, int k) {
    std::vector<int> usedIndexes;
    int n = points.size();

    for (int i = 0; i < k; ++i) {
        int index;
        do {
            index = std::rand() % n;
        } while (std::find(usedIndexes.begin(), usedIndexes.end(), index) != usedIndexes.end());
        centroids[i] = points[index].value;
        usedIndexes.push_back(index);
    }
}

void assignClusters(std::vector<Point>& points, const std::vector<double>& centroids) {
    for (auto& point : points) {
        double minDistance = std::numeric_limits<double>::max();
        int closestCentroid = -1;

        for (int i = 0; i < centroids.size(); ++i) {
            double distance = euclideanDistance(point.value, centroids[i]);
            if (distance < minDistance) {
                minDistance = distance;
                closestCentroid = i;
            }
        }

        point.cluster = closestCentroid;
    }
}

void updateCentroids(std::vector<double>& centroids, const std::vector<Point>& points, int k) {
    std::vector<int> clusterSizes(k, 0);
    std::vector<double> newCentroids(k, 0.0);

    for (const auto& point : points) {
        newCentroids[point.cluster] += point.value;
        ++clusterSizes[point.cluster];
    }

    for (int i = 0; i < k; ++i) {
        if (clusterSizes[i] != 0) {
            newCentroids[i] /= clusterSizes[i];
        } else {
            newCentroids[i] = centroids[i];
        }
    }

    centroids = newCentroids;
}

bool hasConverged(const std::vector<double>& oldCentroids, const std::vector<double>& newCentroids, double tolerance) {
    for (int i = 0; i < oldCentroids.size(); ++i) {
        if (euclideanDistance(oldCentroids[i], newCentroids[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

void kMeansClustering(std::vector<Point>& points, int k, int maxIterations = 100, double tolerance = 1e-4) {
    std::vector<double> centroids(k);
    initializeCentroids(centroids, points, k);

    for (int iter = 0; iter < maxIterations; ++iter) {
        std::vector<double> oldCentroids = centroids;

        assignClusters(points, centroids);
        updateCentroids(centroids, points, k);

        if (hasConverged(oldCentroids, centroids, tolerance)) {
            break;
        }
    }
}

void findClusterThresholds(const std::vector<Point>& points, const std::vector<double>& centroids, std::vector<double>& thresholds) {
    thresholds.clear();
    std::vector<double> sortedCentroids = centroids;
    std::sort(sortedCentroids.begin(), sortedCentroids.end());

    for (int i = 1; i < sortedCentroids.size(); ++i) {
        double threshold = (sortedCentroids[i - 1] + sortedCentroids[i]) / 2.0;
        thresholds.push_back(threshold);
    }
}

//////////////////////////////////////

std::vector<BoundingBox> ballClassification(cv::Mat& image, const std::vector<cv::Rect>& Bbox_rect) {

        std::vector<BoundingBox> Bbox;

        for (auto& bbox : Bbox_rect) {
            BoundingBox temp;
            temp.box = bbox;
            temp.ID = 2;
            Bbox.push_back(temp);
        }

        cv::Mat result;
        image.copyTo(result);

        BoundingBox bbox_best_white;
        BoundingBox bbox_best_black;

        double maxWhitePercentage = 0.0;
        double maxBlackPercentage = 0.0;

        double meanWhite3 = 0.0;
        double meanWhite4 = 0.0;

        std::vector<Point> points;

        for (auto& bbox : Bbox) {
            cv::Mat ballROI = image(bbox.box);
            double whitePixelPercentage = calculateWhitePixelPercentage(ballROI, 230);
            double blackPixelPercentage = calculateBlackPixelPercentage(ballROI);
            double variance = computeImageVariance(ballROI);
            

            // std::cout << "percentage: " << whitePixelPercentage << "  " << blackPixelPercentage << "  " << variance << "  " << bbox.ID << std::endl;

            if (whitePixelPercentage > maxWhitePercentage){
                maxWhitePercentage = whitePixelPercentage;
                bbox_best_white = bbox;
            }

            if (blackPixelPercentage + (100 - whitePixelPercentage)  > maxBlackPercentage){
                maxBlackPercentage = blackPixelPercentage + (100 - whitePixelPercentage);
                bbox_best_black = bbox;
            }
            

            cv::Mat segmented_image;
            int clusterCount = 3; // Number of clusters
            // Apply K-means clustering
            applyKMeans(ballROI, segmented_image, clusterCount);

            // showImage(segmented_image,"roi clust");

            whitePixelPercentage = calculateWhitePixelPercentage(segmented_image, 200);
            blackPixelPercentage = calculateBlackPixelPercentage(segmented_image);
            variance = computeImageVariance(segmented_image);

            if (bbox.ID == 3 || bbox.ID == 4){
                points.push_back(variance);
            }
            
            // std::cout << "percentage: " << whitePixelPercentage << "  " << blackPixelPercentage << "  " << variance << "  " << bbox.ID << std::endl;

            // classifyOtherBall(image, bbox, whitePixelPercentage, variance);

        }
        
        classifyWhiteBall(image, bbox_best_white);
        classifyBlackBall(image, bbox_best_black);

        /*
        int k = 2;
        kMeansClustering(points, k);

        std::vector<double> centroids(k);
        for (int i = 0; i < k; ++i) {
            centroids[i] = points[i].value;
        }

        std::vector<double> thresholds;
        findClusterThresholds(points, centroids, thresholds);

        std::cout << "Thresholds between clusters: ";
        for (int i = 0; i < thresholds.size(); ++i) {
            std::cout << thresholds[i] << std::endl;
        }

        for (auto& bbox : Bbox) {
            cv::Mat ballROI = image(bbox.box);
            double whitePixelPercentage = calculateWhitePixelPercentage(ballROI, 230);
            double blackPixelPercentage = calculateBlackPixelPercentage(ballROI);
            double variance = computeImageVariance(ballROI);

            classifyOtherBall(image, bbox, variance, thresholds[0]);
        }

        */

    return Bbox;
}