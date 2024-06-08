#include "tableDetection.h"
#include "utils.h"
#include <iostream>

cv::Mat hsv_img, outMeanShift;
int sp, sr;

cv::Mat computeHist(cv::Mat image, int bins_number); 
static void computeMeanShift(int, void*);
static void splitLines(std::vector<cv::Vec2f> lines, std::vector<cv::Vec2f> &horizontalLines, std::vector<cv::Vec2f> &verticalLines);
static void drawLines(std::vector<cv::Vec2f> lines, cv::Mat img, cv::Scalar colour);

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
    splitLines(lines, horizontalLines, verticalLines);
    std::cout << "VERTICAL LINES: " << std::endl;
    drawLines(verticalLines, out_hough, cv::Scalar(255, 0, 0));
    std::cout << "HORIZONTAL LINES: " << std::endl;
    drawLines(horizontalLines, out_hough, cv::Scalar(0, 255, 0));
    std::cout << "OK" << std::endl;
    
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

 static void splitLines(std::vector<cv::Vec2f> lines, std::vector<cv::Vec2f> &horizontalLines, std::vector<cv::Vec2f> &verticalLines ){
    
    for (size_t i = 0; i < lines.size(); i++) {
        float theta = lines[i][1];
        //if ((theta >= CV_PI / 4 && theta <= 3 * CV_PI / 4) || (theta >= 5 * CV_PI / 4 && theta <= 7 * CV_PI / 4)) {
        if ((theta >= CV_PI / 2.5 && theta <= 3 * CV_PI / 4) || (theta >= 5 * CV_PI / 4 && theta <= 7 * CV_PI / 4)) {
            // Horizontal lines
            horizontalLines.push_back(lines[i]);
        } else if ((theta >= 0 && theta <= CV_PI / 2.5) || (theta >= 7 * CV_PI / 4 && theta <= 2 * CV_PI) || 
                   (theta >= 3 * CV_PI / 4 && theta <= 5 * CV_PI / 4)) {
            // Vertical lines
            verticalLines.push_back(lines[i]);
        }
    }
}

static void drawLines(std::vector<cv::Vec2f> lines, cv::Mat img, cv::Scalar colour){
    // Draw the lines
    for( size_t i = 0; i < lines.size(); i++ )
        {
        float rho = lines[i][0], theta = lines[i][1];
        float theta_deg = theta * (180/M_PI);
        float slope = -cos(theta)/sin(theta), intercept = rho / sin(theta);
        //std::cout << "Rho: " << rho << "        Theta: " << theta_deg << std::endl;
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