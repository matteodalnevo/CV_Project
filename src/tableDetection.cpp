#include "tableDetection.h"
#include "utils.h"
#include <iostream>
#include <tuple>


// Function to show an image
std::vector<cv::Point2f> tableDetection(const std::vector<cv::Vec2f> lines) {
    
    std::vector<cv::Vec2f> verticalLines, horizontalLines;

    //First split between horizontal and vertical lines
    float meanHoriz, meanVert;
    std::tie(meanHoriz, meanVert) = splitHorVertLines(lines, horizontalLines, verticalLines);

    //Second split between top, bottom, left, right
    std::vector<cv::Vec2f> topHoriz, lowHoriz, leftVert, rightVert;
    std::tie(topHoriz, lowHoriz, leftVert, rightVert) = findGroupOfLines(horizontalLines, verticalLines, meanVert, meanHoriz);

    //Drawing of the 4 groups of lines
    // drawLines(topHoriz, out_hough, cv::Scalar(255, 0, 255));
    // drawLines(lowHoriz, out_hough, cv::Scalar(0, 255, 0));
    // drawLines(leftVert, out_hough, cv::Scalar(0, 255, 255));
    // drawLines(rightVert, out_hough, cv::Scalar(255, 255, 0));
    
    //Identify the mean line for each group
    cv::Vec2f vertSx, vertDx, horizUp, horizBot;
    vertSx = findMediumLine(leftVert);
    vertDx = findMediumLine(rightVert);
    horizUp = findMediumLine(topHoriz);
    horizBot = findMediumLine(lowHoriz);    
    checkLeftRight(vertSx, vertDx);   
    
    cv::Point2f pt1, pt2, pt3, pt4;
    std::tie(pt1, pt2, pt3, pt4) = computeCorners(horizUp, horizBot, vertSx, vertDx);

    std::vector<cv::Point2f> vertices = {pt1, pt2, pt3, pt4};

    // std::cout << "PT1: x = " << pt1.x << "  y = " << pt1.y << std::endl;
    // std::cout << "PT2: x = " << pt2.x << "  y = " << pt2.y << std::endl;
    // std::cout << "PT3: x = " << pt3.x << "  y = " << pt3.y << std::endl;
    // std::cout << "PT4: x = " << pt4.x << "  y = " << pt4.y << std::endl;

    //std::cout << "Table Detection OK" << std::endl;
    
    return vertices;
}


// SPLIT LINES IN HORIZONTAL AND VERTICAL 

std::tuple<float, float> splitHorVertLines(const std::vector<cv::Vec2f> &lines, std::vector<cv::Vec2f> &horizontalLines, std::vector<cv::Vec2f> &verticalLines ){
    
    // Upper and lower thresholds to detect horizontal lines
    const double low_threshold_horizontal_1 = CV_PI / 2.5;
    const double up_threshold_horizontal_1 = 3 * CV_PI / 4;
    const double low_threshold_horizontal_2 = 5 * CV_PI / 4;
    const double up_threshold_horizontal_2 = 7 * CV_PI / 4;

    //Upper and lower thresholds to detect vertical lines
    const double low_threshold_vertical_1 = 0;
    const double up_threshold_vertical_1 = CV_PI / 2.5;
    const double low_threshold_vertical_2 = 7 * CV_PI / 4;
    const double up_threshold_vertical_2 = 2 * CV_PI;
    const double low_threshold_vertical_3 = 3 * CV_PI / 4;
    const double up_threshold_vertical_3 = 5 * CV_PI / 4;

    // Define sums and mean variables to store results
    float sumRhoVert = 0, sumRhoHoriz = 0, meanVert = 0, meanHoriz = 0;

    // Iterate along all the lines received as input
    for (size_t i = 0; i < lines.size(); i++) {

        // Assign theta value of the current line 
        float theta = lines[i][1];      

        // Split lines in horizontal and vertical based on the value of theta
        // Horizontal lines
        if ((theta >= low_threshold_horizontal_1 && theta <= up_threshold_horizontal_1) || (theta >= low_threshold_horizontal_2 && theta <= up_threshold_horizontal_2)) {
            // Add the current line to the vector of horizontal lines and accumulate the sum of Rho values
            horizontalLines.push_back(lines[i]);
            sumRhoHoriz += lines[i][0];
        } 
        
        // Vertical lines
        else if ((theta >= low_threshold_vertical_1 && theta <= up_threshold_vertical_1) || (theta >= low_threshold_vertical_2 && theta <= up_threshold_vertical_2) 
                    || (theta >= low_threshold_vertical_3 && theta <= up_threshold_vertical_3)) {
            // Add the current line to the vector of horizontal lines and accumulate the sum of Rho values
            verticalLines.push_back(lines[i]);
            sumRhoVert += lines[i][0];
        }
    }

    // Compute mean value of Rho for horizontal and vertical lines
    meanHoriz = sumRhoHoriz / horizontalLines.size();
    meanVert = sumRhoVert / verticalLines.size();

    // Optional for debugging: print on screen the mean values of Rho for horizontal and vertical lines
    //std::cout << "Mean Rho Horiz: " << meanHoriz << "   Mean Rho Vert: " << meanVert << std::endl;
    return std::make_tuple(meanHoriz, meanVert);
}


// DRAW A GROUP OF LINES

static void drawLines(std::vector<cv::Vec2f> lines, cv::Mat img, cv::Scalar colour){

    // Iterate along the vector of lines
    for( size_t i = 0; i < lines.size(); i++ ) {
        
        // Extract rho (distance from origin to the line) and theta (angle of the line) from the current line
        float rho = lines[i][0], theta = lines[i][1];

        // Optional for debugging: compute theta in degrees, slope of the line and intercept with y-axis, and print
        float theta_deg = theta * (180/M_PI);
        float slope = -cos(theta)/sin(theta), intercept = rho / sin(theta);
        //std::cout << "Rho: " << rho << "        Theta: " << theta << std::endl;
        //std::cout << "Slope: " << slope << "        Intercept: " << intercept << std::endl;
        
        // Define two points to represent the line 
        cv::Point2f pt1, pt2;
        double a = cos(theta), b = sin(theta); 
        double x0 = a * rho, y0 = b * rho;   
        
        // Compute x and y coordinates of the first and second point
        pt1.x = cvRound(x0 + 1000 * (-b));     
        pt1.y = cvRound(y0 + 1000 * (a));      
        pt2.x = cvRound(x0 - 1000 * (-b));     
        pt2.y = cvRound(y0 - 1000 * (a));     

        // Write the line that connects the two points on the image
        cv::line( img, pt1, pt2, colour, 3, cv::LINE_AA);
        }

    // Show the image
    cv::imshow("Detected Lines ", img);
    cv::waitKey(0);
}



static void drawSingleLine(cv::Vec2f lines, cv::Mat img, cv::Scalar colour){
    
    // Extract rho (distance from origin to the line) and theta (angle of the line) from the current line
    float rho = lines[0], theta = lines[1];

    // Optional for debugging: compute theta in degrees, slope of the line and intercept with y-axis, and print
    float theta_deg = theta * (180/M_PI);
    float slope = -cos(theta)/sin(theta), intercept = rho / sin(theta);
    //std::cout << "Rho: " << rho << "        Theta: " << theta_deg << std::endl;
    //std::cout << "Slope: " << slope << "        Intercept: " << intercept << std::endl;

    // Define two points to represent the line 
    cv::Point2f pt1, pt2;
    double a = cos(theta), b = sin(theta);
    double x0 = a*rho, y0 = b*rho;

    // Compute x and y coordinates of the first and second point
    pt1.x = cvRound(x0 + 1000*(-b));
    pt1.y = cvRound(y0 + 1000*(a));
    pt2.x = cvRound(x0 - 1000*(-b));
    pt2.y = cvRound(y0 - 1000*(a));

    // Write the line that connects the two points on the image
    cv::line( img, pt1, pt2, colour, 3, cv::LINE_AA);
    cv::imshow("Draw single line", img);
    cv::waitKey(0);
}

std::tuple<std::vector<cv::Vec2f>, std::vector<cv::Vec2f>, std::vector<cv::Vec2f>, std::vector<cv::Vec2f>> findGroupOfLines(std::vector<cv::Vec2f> horizontalLine, std::vector<cv::Vec2f> verticalLine, float meanVert, float meanHoriz) {
    
    // Define vectors to store the four lines
    std::vector<cv::Vec2f> topHoriz, lowHoriz, leftVert, rightVert;
    float lastTheta, lastRho;
    float meanRhoTop = 0, meanRhoLow = 0, meanRhoLeft = 0, meanRhoRight = 0;
    float meanThetaTop = 0, meanThetaLow = 0, meanThetaLeft = 0, meanThetaRight = 0;
    for( size_t i = 0; i < verticalLine.size(); i++ ) {
        if (verticalLine[i][0] >= meanVert) {            
            rightVert.push_back(verticalLine[i]);
            lastTheta = verticalLine[i][1];
            lastRho = verticalLine[i][0];
            if ((abs(lastTheta - rightVert[0][1]) > 0.2617)) {  //15 deg
                rightVert.pop_back();
                //std::cout << "Miao" << std::endl;
            }
                
            //meanRhoRight += verticalLine[i][0];
            //meanThetaRight += verticalLine[i][1];
        }
        else {
            leftVert.push_back(verticalLine[i]);
            lastTheta = verticalLine[i][1];
            lastRho = verticalLine[i][0];
            if ((abs(lastTheta - leftVert[0][1]) > 0.2617)) {  //15 deg
                leftVert.pop_back();
                //std::cout << "Miao" << std::endl;
            }
            //meanRhoLeft += verticalLine[i][0];
            //meanThetaLeft += verticalLine[i][1];
        }
    }
    for( size_t i = 0; i < horizontalLine.size(); i++ ) {
        if (horizontalLine[i][0] >= meanHoriz) {
            lowHoriz.push_back(horizontalLine[i]);
            lastRho = horizontalLine[i][0];
            lastTheta = horizontalLine[i][1];
            if ((abs(lastTheta - lowHoriz[0][1]) > 0.2617) || (abs(lastRho - lowHoriz[0][0]) > 75)) {  //15 deg
                lowHoriz.pop_back();
                //std::cout << "Miao" << std::endl;
            }
            //meanRhoLow += horizontalLine[i][0];
            //meanThetaLow += horizontalLine[i][1];
        }
        else {
            topHoriz.push_back(horizontalLine[i]);
            lastRho = horizontalLine[i][0];
            lastTheta = horizontalLine[i][1];
            if ((abs(lastTheta - topHoriz[0][1]) > 0.2617) || (abs(lastRho - topHoriz[0][0]) > 75)) { //15 deg
                topHoriz.pop_back();
                //std::cout << "Miao" << std::endl;
            }
            //meanRhoTop += horizontalLine[i][0];
            //meanThetaTop += horizontalLine[i][1];
        }
    }

    meanRhoRight = meanRhoRight / rightVert.size();
    meanRhoLeft = meanRhoLeft / leftVert.size();
    meanThetaRight = meanThetaRight / rightVert.size();
    meanThetaLeft = meanThetaLeft / leftVert.size();

    meanRhoTop = meanRhoTop / topHoriz.size();
    meanRhoLow = meanRhoLow/ lowHoriz.size();
    meanThetaTop = meanThetaTop / topHoriz.size();
    meanThetaLow = meanThetaLow / lowHoriz.size();

    return std::make_tuple(topHoriz, lowHoriz, leftVert, rightVert);
}

/* std::tuple<cv::Vec2f, cv::Vec2f, cv::Vec2f, cv::Vec2f> findRepresentativeLine(std::vector<cv::Vec2f> horizontalLine, std::vector<cv::Vec2f> verticalLine, float meanVert, float meanHoriz) {
    std::vector<cv::Vec2f> topHoriz, lowHoriz, leftVert, rightVert;
    float meanRhoTop = 0, meanRhoLow = 0, meanRhoLeft = 0, meanRhoRight = 0;
    float meanThetaTop = 0, meanThetaLow = 0, meanThetaLeft = 0, meanThetaRight = 0;
    for( size_t i = 0; i < verticalLine.size(); i++ ) {
        if ((verticalLine[i][0] >= meanVert)) {
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
    //if rho<0, wrong detection of left and right vertical lines -> swap
    if (meanRhoLeft <= 0) {
        float temp1;
        temp1 = meanRhoRight;
        meanRhoRight = meanRhoLeft;
        meanRhoLeft = temp1;

        temp1 = meanThetaRight;
        meanThetaRight = meanThetaLeft;
        meanThetaLeft = temp1;
    }

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
} */

cv::Vec2f findMediumLine(std::vector<cv::Vec2f> lineVector) {
    float rho = 0, theta = 0;
    for (size_t i = 0; i < lineVector.size(); i++) {
        rho += lineVector[i][0];
        theta += lineVector[i][1];
    }
    rho = rho / lineVector.size();
    theta = theta / lineVector.size();

    return cv::Vec2f(rho, theta);
}

static void checkLeftRight (cv::Vec2f &left, cv::Vec2f &right) {
        if (left[0] <= 0) {
        float temp1;
        temp1 = right[0];
        right[0] = left[0];
        left[0] = temp1;

        temp1 = right[1];
        right[1] = left[1];
        left[1] = temp1;
    }
}

cv::Point2f computeIntercept (cv::Vec2f line1, cv::Vec2f line2) {
    //compute slope and intercept from rho, theta
    float m1 = 0, m2 = 0, q1 = 0, q2 = 0;
    m1 = -(cos(line1[1]) / sin(line1[1]));
    m2 = -(cos(line2[1]) / sin(line2[1]));
    q1 = line1[0] / sin(line1[1]);
    q2 = line2[0] / sin(line2[1]);
    if ( m1 == -INFINITY )
    {   
        //std::cout << "ENTERED THE IF" << std::endl;
        float x = cvRound(line1[0]);
        float y = cvRound(m2 * x + q2);

        return cv::Point2f(x, y); 
    }

    //std::cout << "m1: " << m1 << "  q1: " << q1 << std::endl;
    //std::cout << "m2: " << m2 << " q2: " << q2 << std::endl;

    //compute intercept cordinates
    float x = cvRound((q2 - q1) / (m1 - m2));
    float y = cvRound(m1 * x + q1);

    //std::cout << "Intercept x: " << x << "       y: " << y << std::endl;

    return cv::Point2f(x, y);
}

std::tuple<cv::Point2f, cv::Point2f, cv::Point2f, cv::Point2f> computeCorners(cv::Vec2f topHoriz, cv::Vec2f lowHoriz, cv::Vec2f leftVert, cv::Vec2f rightVert) {
    //Compute intercept coordinates
    cv::Point2f pt1, pt2, pt3, pt4;
    pt1 = computeIntercept(leftVert, topHoriz);         //top left
    pt2 = computeIntercept(rightVert, topHoriz);       //top right
    pt3 = computeIntercept(rightVert, lowHoriz);       //bottom right
    pt4 = computeIntercept(leftVert, lowHoriz);       //bottom left

    //std::cout << "Pt1 x: " << pt1.x << "       y: " << pt1.y << std::endl;
    //std::cout << "Pt2 x: " << pt2.x << "       y: " << pt2.y << std::endl;
    //std::cout << "Pt3 x: " << pt3.x << "       y: " << pt3.y << std::endl;
    //std::cout << "Pt4 x: " << pt4.x << "       y: " << pt4.y << std::endl;

    return std::make_tuple(pt1,pt4,pt3, pt2); //from top-left in counterclockwise order
}

cv::Mat computeMask(cv::Mat image) {
    int rows = image.rows, cols = image.cols, thresholdz = 50;
    std::vector<cv::Vec3b> color_acc;

    // Accumulate colors that are not black
    for (int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            if(image.at<cv::Vec3b>(i, j)[0] != 0 ||
               image.at<cv::Vec3b>(i, j)[1] != 0 ||
               image.at<cv::Vec3b>(i, j)[2] != 0) {
                color_acc.push_back(image.at<cv::Vec3b>(i, j));
            }
        }
    }

    // Calculate the mean color
    cv::Vec3d sumColor(0, 0, 0);
    for(const auto& color : color_acc) {
        sumColor[0] += color[0];
        sumColor[1] += color[1];
        sumColor[2] += color[2];
    }
    cv::Vec3b meanColor;
    if (!color_acc.empty()) {
        meanColor[0] = sumColor[0] / color_acc.size();
        meanColor[1] = sumColor[1] / color_acc.size();
        meanColor[2] = sumColor[2] / color_acc.size();
    }

    // Initialize the masked image with the same size and type as the input image
    cv::Mat maskedImg = image.clone();

    // Apply the mask based on the threshold
    for (int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            cv::Vec3b temp = image.at<cv::Vec3b>(i, j);
            cv::Vec3b diff = temp - meanColor;
            if(diff[0] < thresholdz && diff[1] < thresholdz && diff[2] < thresholdz) {
                maskedImg.at<cv::Vec3b>(i, j) = meanColor;
            }
        }
    }

    return maskedImg;
}

cv::Mat enhanceColourContrast (cv::Mat input_img) {
    cv::Mat out_img;
    std::vector<cv::Mat> hsv_mid;
    cv::cvtColor(input_img, out_img, cv::COLOR_BGR2HSV);
    cv::split(out_img, hsv_mid);
    hsv_mid[1] *= 2;
    hsv_mid[2] *= 0.8;
    cv::merge(hsv_mid, out_img);
    cv::cvtColor(out_img, out_img, cv::COLOR_HSV2BGR);

    return out_img;
}