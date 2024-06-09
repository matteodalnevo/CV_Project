#include "tableDetection.h"
#include "utils.h"
#include <iostream>
#include <tuple>

cv::Mat hsv_img, outMeanShift;
int sp, sr;

cv::Mat computeHist(cv::Mat image, int bins_number); 
static void computeMeanShift(int, void*);
std::tuple<float, float> splitLines(std::vector<cv::Vec2f> lines, std::vector<cv::Vec2f> &horizontalLines, std::vector<cv::Vec2f> &verticalLines );
static void drawLines(std::vector<cv::Vec2f> lines, cv::Mat img, cv::Scalar colour);
std::tuple<std::vector<cv::Vec2f>, std::vector<cv::Vec2f>, std::vector<cv::Vec2f>, std::vector<cv::Vec2f>> findGroupOfLines(std::vector<cv::Vec2f> horizontalLine, std::vector<cv::Vec2f> verticalLine, float meanVert, float meanHoriz);
std::tuple<cv::Vec2f, cv::Vec2f, cv::Vec2f, cv::Vec2f> findRepresentativeLine(std::vector<cv::Vec2f> horizontalLine, std::vector<cv::Vec2f> verticalLine, float meanVert, float meanHoriz);
static void drawSingleLine(cv::Vec2f lines, cv::Mat img, cv::Scalar colour);

// Function to show an image
cv::Mat tableDetection(const cv::Mat& image) {
 
    int lowThreshold = 40, upperThreshold = 80, blur_kernel = 3;
    cv::Mat img_gray, detected_edges, out_hough;
    cvtColor(image, img_gray, cv::COLOR_BGR2GRAY);
    cv::blur( img_gray, detected_edges, cv::Size(blur_kernel, blur_kernel) );
    cv::Canny( detected_edges, detected_edges, lowThreshold, upperThreshold, 3 );

    // Copy edges to the images that will display the results in BGR
    cvtColor(detected_edges, out_hough, cv::COLOR_GRAY2BGR);

    // Standard Hough Line Transform
    std::vector<cv::Vec2f> lines, verticalLines, horizontalLines; // will hold the results of the detection
    HoughLines(detected_edges, lines, 1, CV_PI/180, 180, 0, 0 ); // runs the actual detection
    auto [meanHoriz, meanVert] = splitLines(lines, horizontalLines, verticalLines);
    std::cout << "Mean Rho Horiz: " << meanHoriz << "   Mean Rho Vert: " << meanVert << std::endl;
    //auto [topHoriz, lowHoriz, leftVert, rightVert] = findGroupOfLines(horizontalLines, verticalLines, meanVert, meanHoriz);
    auto [topHoriz, lowHoriz, leftVert, rightVert] = findRepresentativeLine(horizontalLines, verticalLines, meanVert, meanHoriz);
    //std::cout << "VERTICAL LINES: " << std::endl;
    drawSingleLine(topHoriz, out_hough, cv::Scalar(255, 0, 255));
    //std::cout << "HORIZONTAL LINES: " << std::endl;
    drawSingleLine(lowHoriz, out_hough, cv::Scalar(0, 255, 0));
    drawSingleLine(leftVert, out_hough, cv::Scalar(0, 255, 255));
    drawSingleLine(rightVert, out_hough, cv::Scalar(255, 255, 0));
    std::cout << "OK" << std::endl;
    cv::waitKey(0);
    
    return image;
}

cv::Mat computeHist(cv::Mat image, int bins_number) {
    // Set histogram parameters
        int histSize[] = {bins_number}; // Number of bins
        float range[] = {0, 256}; // Pixel value range (0-255)
        const float* histRange[] = {range};
        int channels[] = {0}; // Channel index

        // Compute histogram
        cv::Mat hist;
        cv::calcHist(&image, 1, channels, cv::Mat(), hist, 1, histSize, histRange);

        // Normalize the histogram
        cv::normalize(hist, hist, 0, 255, cv::NORM_MINMAX);

        // Create a black image to draw the histogram
        int histWidth = 512;
        int histHeight = 400;
        cv::Mat histImage(histHeight, histWidth, CV_8UC1, cv::Scalar(0));

        // Draw histogram
        int binWidth = cvRound((double)histWidth / 256);
        for (int i = 0; i < 256; ++i) {
            cv::rectangle(histImage, cv::Point(binWidth * i, histHeight),
                          cv::Point(binWidth * i + binWidth, histHeight - cvRound(hist.at<float>(i))),
                          cv::Scalar(255), -1);
        }
    return histImage;
}
 
static void computeMeanShift(int, void* ){
    cv::pyrMeanShiftFiltering(hsv_img, outMeanShift, sp, sr);
    cv::cvtColor(outMeanShift, outMeanShift, cv::COLOR_HSV2BGR);
    cv::imshow("OutMeanShift", outMeanShift);
 }

std::tuple<float, float> splitLines(std::vector<cv::Vec2f> lines, std::vector<cv::Vec2f> &horizontalLines, std::vector<cv::Vec2f> &verticalLines ){
    float sumRhoVert = 0, sumRhoHoriz = 0, meanVert = 0, meanHoriz = 0;
    for (size_t i = 0; i < lines.size(); i++) {
        float theta = lines[i][1];      
        if ((theta >= CV_PI / 2.5 && theta <= 3 * CV_PI / 4) || (theta >= 5 * CV_PI / 4 && theta <= 7 * CV_PI / 4)) {
            // Horizontal lines
            horizontalLines.push_back(lines[i]);
            sumRhoHoriz += lines[i][0];
        } else if ((theta >= 0 && theta <= CV_PI / 2.5) || (theta >= 7 * CV_PI / 4 && theta <= 2 * CV_PI) || 
                   (theta >= 3 * CV_PI / 4 && theta <= 5 * CV_PI / 4)) {
            // Vertical lines
            verticalLines.push_back(lines[i]);
            sumRhoVert += lines[i][0];
        }
    }
    meanHoriz = sumRhoHoriz / horizontalLines.size();
    meanVert = sumRhoVert / verticalLines.size();
    //std::cout << "Mean Rho Horiz: " << meanHoriz << "   Mean Rho Vert: " << meanVert << std::endl;
    return std::make_tuple(meanHoriz, meanVert);
}

static void drawLines(std::vector<cv::Vec2f> lines, cv::Mat img, cv::Scalar colour){
    // Draw the lines
    for( size_t i = 0; i < lines.size(); i++ )
        {
        float rho = lines[i][0], theta = lines[i][1];
        float theta_deg = theta * (180/M_PI);
        float slope = -cos(theta)/sin(theta), intercept = rho / sin(theta);
        std::cout << "Rho: " << rho << "        Theta: " << theta_deg << std::endl;
        //std::cout << "Slope: " << slope << "        Intercept: " << intercept << std::endl;
        cv::Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;
        pt1.x = cvRound(x0 + 1000*(-b));
        pt1.y = cvRound(y0 + 1000*(a));
        pt2.x = cvRound(x0 - 1000*(-b));
        pt2.y = cvRound(y0 - 1000*(a));
        line( img, pt1, pt2, colour, 3, cv::LINE_AA);
        }
    cv::imshow("Detected Lines (in red)", img);
    cv::waitKey(0);
}

static void drawSingleLine(cv::Vec2f lines, cv::Mat img, cv::Scalar colour){
    // Draw the lines
    float rho = lines[0], theta = lines[1];
    float theta_deg = theta * (180/M_PI);
    float slope = -cos(theta)/sin(theta), intercept = rho / sin(theta);
    std::cout << "Rho: " << rho << "        Theta: " << theta_deg << std::endl;
    //std::cout << "Slope: " << slope << "        Intercept: " << intercept << std::endl;
    cv::Point pt1, pt2;
    double a = cos(theta), b = sin(theta);
    double x0 = a*rho, y0 = b*rho;
    pt1.x = cvRound(x0 + 1000*(-b));
    pt1.y = cvRound(y0 + 1000*(a));
    pt2.x = cvRound(x0 - 1000*(-b));
    pt2.y = cvRound(y0 - 1000*(a));
    line( img, pt1, pt2, colour, 3, cv::LINE_AA);
    cv::imshow("Detected Lines (in red)", img);
    cv::waitKey(0);
}

std::tuple<std::vector<cv::Vec2f>, std::vector<cv::Vec2f>, std::vector<cv::Vec2f>, std::vector<cv::Vec2f>> findGroupOfLines(std::vector<cv::Vec2f> horizontalLine, std::vector<cv::Vec2f> verticalLine, float meanVert, float meanHoriz) {
    std::vector<cv::Vec2f> topHoriz, lowHoriz, leftVert, rightVert;
    for( size_t i = 0; i < verticalLine.size(); i++ ) {
        if (verticalLine[i][0] >= meanVert) {
            rightVert.push_back(verticalLine[i]);
        }
        else {
            leftVert.push_back(verticalLine[i]);
        }
    }
    for( size_t i = 0; i < horizontalLine.size(); i++ ) {
        if (horizontalLine[i][0] >= meanHoriz) {
            lowHoriz.push_back(horizontalLine[i]);
        }
        else {
            topHoriz.push_back(horizontalLine[i]);
        }
    }
    return std::make_tuple(topHoriz, lowHoriz, leftVert, rightVert);
}

std::tuple<cv::Vec2f, cv::Vec2f, cv::Vec2f, cv::Vec2f> findRepresentativeLine(std::vector<cv::Vec2f> horizontalLine, std::vector<cv::Vec2f> verticalLine, float meanVert, float meanHoriz) {
    std::vector<cv::Vec2f> topHoriz, lowHoriz, leftVert, rightVert;
    float meanRhoTop = 0, meanRhoLow = 0, meanRhoLeft = 0, meanRhoRight = 0;
    float meanThetaTop = 0, meanThetaLow = 0, meanThetaLeft = 0, meanThetaRight = 0;
    for( size_t i = 0; i < verticalLine.size(); i++ ) {
        if (verticalLine[i][0] >= meanVert) {
            rightVert.push_back(verticalLine[i]);
            meanRhoRight += verticalLine[i][0];
            meanThetaRight += verticalLine[i][1];
        }
        else {
            leftVert.push_back(verticalLine[i]);
            meanRhoLeft += verticalLine[i][0];
            meanThetaLeft += verticalLine[i][1];
        }
    }
    meanRhoRight = meanRhoRight / rightVert.size();
    meanRhoLeft = meanRhoLeft / leftVert.size();
    meanThetaRight = meanThetaRight / rightVert.size();
    meanThetaLeft = meanThetaLeft / leftVert.size();
    for( size_t i = 0; i < horizontalLine.size(); i++ ) {
        if (horizontalLine[i][0] >= meanHoriz) {
            lowHoriz.push_back(horizontalLine[i]);
            meanRhoLow += horizontalLine[i][0];
            meanThetaLow += horizontalLine[i][1];
        }
        else {
            topHoriz.push_back(horizontalLine[i]);
            meanRhoTop += horizontalLine[i][0];
            meanThetaTop += horizontalLine[i][1];
        }
    }
    meanRhoTop = meanRhoTop / topHoriz.size();
    meanRhoLow = meanRhoLow/ lowHoriz.size();
    meanThetaTop = meanThetaTop / topHoriz.size();
    meanThetaLow = meanThetaLow / lowHoriz.size();
    return std::make_tuple(cv::Vec2f(meanRhoTop, meanThetaTop), cv::Vec2f(meanRhoLow, meanThetaLow), cv::Vec2f(meanRhoLeft, meanThetaLeft), cv::Vec2f(meanRhoRight, meanThetaRight));
}