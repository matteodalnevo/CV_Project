#include "tableDetection.h"
#include "utils.h"
#include <iostream>
#include <tuple>


// TABLE DETECTION

std::vector<cv::Point2f> tableDetection(const std::vector<cv::Vec2f> lines) {
    
    std::vector<cv::Vec2f> verticalLines, horizontalLines;

    //First split between horizontal and vertical lines
    float meanHoriz, meanVert;
    std::tie(meanHoriz, meanVert) = splitHorVertLines(lines, horizontalLines, verticalLines);

    //Second split between top, bottom, left, right
    std::vector<cv::Vec2f> topHoriz, lowHoriz, leftVert, rightVert;
    std::tie(topHoriz, lowHoriz, leftVert, rightVert) = findGroupOfLines(horizontalLines, verticalLines, meanVert, meanHoriz);

    // Optional for debugging: draw the 4 groups of lines
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

    // Check if left and vertical lines have to be splitted
    checkLeftRight(vertSx, vertDx);   
    
    // Define and compute the table corners
    cv::Point2f pt1, pt2, pt3, pt4;
    std::tie(pt1, pt2, pt3, pt4) = computeCorners(horizUp, horizBot, vertSx, vertDx);

    std::vector<cv::Point2f> vertices = {pt1, pt2, pt3, pt4};

    // Optional for debugging: print of the corners coordinates
    // std::cout << "PT1: x = " << pt1.x << "  y = " << pt1.y << std::endl;
    // std::cout << "PT2: x = " << pt2.x << "  y = " << pt2.y << std::endl;
    // std::cout << "PT3: x = " << pt3.x << "  y = " << pt3.y << std::endl;
    // std::cout << "PT4: x = " << pt4.x << "  y = " << pt4.y << std::endl;
<<<<<<< Updated upstream

    //std::cout << "Table Detection OK" << std::endl;
=======
>>>>>>> Stashed changes
    
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

std::tuple<std::vector<cv::Vec2f>, std::vector<cv::Vec2f>, std::vector<cv::Vec2f>, std::vector<cv::Vec2f>> findGroupOfLines(const std::vector<cv::Vec2f> &horizontalLine, const std::vector<cv::Vec2f> &verticalLine, const float meanVert, const float meanHoriz) {
    
    // Define vectors to store the four lines
    std::vector<cv::Vec2f> topHoriz, lowHoriz, leftVert, rightVert;
    float lastTheta, lastRho;

    // Thresholds to discard lines representing the sticks of the players 
    const float threshold_orientation = 15 * CV_PI / 180.0;
    const int threshold_distance = 75;

    // Iterate through each vertical line
    for( size_t i = 0; i < verticalLine.size(); i++ ) {
        // If the line is on the right side, add it to the rightVert vector 
        if (verticalLine[i][0] >= meanVert) { 
                     
            rightVert.push_back(verticalLine[i]);
            lastTheta = verticalLine[i][1];
            lastRho = verticalLine[i][0];

            // If the current line is not parallel to the group mean, then it is discarded
            if ((abs(lastTheta - rightVert[0][1]) > threshold_orientation)) {  //15 deg
                rightVert.pop_back();
            }
        }
        // If the line is on the left side, add it to the leftVert vector
        else {
            leftVert.push_back(verticalLine[i]);
            lastTheta = verticalLine[i][1];
            lastRho = verticalLine[i][0];
            if ((abs(lastTheta - leftVert[0][1]) > threshold_orientation)) { 
                leftVert.pop_back();
            }
        }
    }

    // Iterate through each horizontal line
    for( size_t i = 0; i < horizontalLine.size(); i++ ) {
        // If the line is on the lower side, add it to the lowHoriz vector
        if (horizontalLine[i][0] >= meanHoriz) {
            lowHoriz.push_back(horizontalLine[i]);
            lastRho = horizontalLine[i][0];
            lastTheta = horizontalLine[i][1];

            // Check if the angle difference exceeds 15 degrees (0.2617 radians) or rho difference exceeds 75 units, if so, remove the line
            if ((abs(lastTheta - lowHoriz[0][1]) > threshold_orientation) || (abs(lastRho - lowHoriz[0][0]) > threshold_distance)) {  
                lowHoriz.pop_back();
            }
        }
        // If the line is on the upper side, add it to the topHoriz vector
        else {
            topHoriz.push_back(horizontalLine[i]);
            lastRho = horizontalLine[i][0];
            lastTheta = horizontalLine[i][1];

            // Check if the angle difference exceeds 15 degrees (0.2617 radians) or rho difference exceeds 75 units, if so, remove the line
            if ((abs(lastTheta - topHoriz[0][1]) > threshold_orientation) || (abs(lastRho - topHoriz[0][0]) > threshold_distance)) { 
                topHoriz.pop_back();
            }
        }
    }

    return std::make_tuple(topHoriz, lowHoriz, leftVert, rightVert);
}


// FIND MEDIUM LINE FROM THE GROUP

cv::Vec2f findMediumLine(const std::vector<cv::Vec2f> &lineVector) {

    // Define values for accumulating values 
    float rho = 0, theta = 0;

    // Iterate over vector of lines
    for (size_t i = 0; i < lineVector.size(); i++) {
        // Accumulate parameters of the current line
        rho += lineVector[i][0];
        theta += lineVector[i][1];
    }

    // Compute mean values of the lines 
    rho = rho / lineVector.size();
    theta = theta / lineVector.size();

    return cv::Vec2f(rho, theta);
}


// INVERT LEFT AND RIGHT LINES

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


// COMPUTE INTERCEPTION POINT BETWEEN TWO LINES

cv::Point2f computeIntercept (cv::Vec2f line1, cv::Vec2f line2) {
    
    // Initialize variables to store slopes and intercepts of the two lines
    float m1 = 0, m2 = 0, q1 = 0, q2 = 0; 
    
    // Compute slope and intercept from rho, theta of the two lines
    m1 = -(cos(line1[1]) / sin(line1[1]));
    m2 = -(cos(line2[1]) / sin(line2[1]));
    q1 = line1[0] / sin(line1[1]);
    q2 = line2[0] / sin(line2[1]);

    // heck if the first line is vertical 
    if ( m1 == -INFINITY ) {   
        
        float x = cvRound(line1[0]);
        float y = cvRound(m2 * x + q2);
        
        // Optional for debugging: 
        //std::cout << "ENTERED THE IF" << std::endl;
        
        // Return the intersection point
        return cv::Point2f(x, y); 
    }

    // Check if the second line is vertical 
    if (m2 == -INFINITY) {
        
        float x = cvRound(line2[0]);
        float y = cvRound(m1 * x + q1);

        // Return the intersection point
        return cv::Point2f(x, y);
    }

    // Optional for Debugging: print on screen slope and intercept of the two lines
    //std::cout << "m1: " << m1 << "  q1: " << q1 << std::endl;
    //std::cout << "m2: " << m2 << " q2: " << q2 << std::endl;

    // Compute intercept cordinates
    float x = cvRound((q2 - q1) / (m1 - m2));
    float y = cvRound(m1 * x + q1);

    //std::cout << "Intercept x: " << x << "       y: " << y << std::endl;

    return cv::Point2f(x, y);
}


// COMPUTE FOUR CORNERS

std::tuple<cv::Point2f, cv::Point2f, cv::Point2f, cv::Point2f> computeCorners(cv::Vec2f topHoriz, cv::Vec2f lowHoriz, cv::Vec2f leftVert, cv::Vec2f rightVert) {
    //Compute intercept coordinates
    cv::Point2f pt1, pt2, pt3, pt4;
    pt1 = computeIntercept(leftVert, topHoriz);         //top left
    pt2 = computeIntercept(rightVert, topHoriz);       //top right
    pt3 = computeIntercept(rightVert, lowHoriz);       //bottom right
    pt4 = computeIntercept(leftVert, lowHoriz);       //bottom left

    // Optional for debugging: print the points
    //std::cout << "Pt1 x: " << pt1.x << "       y: " << pt1.y << std::endl;
    //std::cout << "Pt2 x: " << pt2.x << "       y: " << pt2.y << std::endl;
    //std::cout << "Pt3 x: " << pt3.x << "       y: " << pt3.y << std::endl;
    //std::cout << "Pt4 x: " << pt4.x << "       y: " << pt4.y << std::endl;

    return std::make_tuple(pt1, pt4, pt3, pt2); //from top-left in counterclockwise order
}