#include "tableDetection.h"
#include "utils.h"
#include <iostream>

cv::Mat hsv_img, outMeanShift;
int sp, sr;

cv::Mat computeHist(cv::Mat image, int bins_number); 
static void computeMeanShift(int, void*);

// Function to show an image
cv::Mat tableDetection(const cv::Mat& image) {

    //Conversion to gray scale image
    cv::Mat gray_img;
    cv::cvtColor(image, gray_img, cv::COLOR_BGR2GRAY);

    //Blur of the image
    int blur_kernel_dim = 7;
    cv::Mat gauss_img07;
    cv::GaussianBlur(gray_img, gauss_img07, cv::Size(blur_kernel_dim, blur_kernel_dim), 0);

    //showImage(image, "First Image");

    //Otsu's thresholding
    //cv::Mat otsu_out;
    //cv::threshold(gray_img, otsu_out, 125, 255, cv::THRESH_OTSU);
    //showImage(otsu_out, "Thresholded image");

    //Conversion to HSV and then pyrMeanShiftFiltering
    cv::cvtColor(image, hsv_img, cv::COLOR_BGR2HSV);
    cv::imshow("First Image", image);
    cv::createTrackbar( "Spatial window radius:", "First Image", &sp, 100, computeMeanShift );
    cv::createTrackbar( "Colour window radius:", "First Image", &sr, 255, computeMeanShift );
    cv::waitKey(0);

    //Histogram evaluation
    //cv::Mat istogramma = computeHist(gauss_img07, 256);
    //showImage(istogramma, "Histogram");

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