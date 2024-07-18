// Homography_main.cpp
#include "homography.h"
#include "tableDetection.h"
#include "ballDetection.h"
#include "utils.h"
#include "preProcess.h"
#include "ballClassification.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/tracking.hpp>
#include <iostream>
#include <vector>
#include <tuple>
#include <fstream>
#include <sstream>
#include <string>

cv::Mat segmentation(const cv::Mat img, const std::vector<cv::Point2f> footage_corners, const std::vector<BoundingBox> classified_boxes, cv::Mat hand_mask) {
    // Create a black image of the same size as input image
    cv::Mat dark_image = cv::Mat::zeros(img.size(), CV_8UC1);

    // Create a mask with the same size as the image, initialized to black
    cv::Mat mask = cv::Mat::zeros(img.size(), CV_8UC1);

    // Convert the vector of points to a vector of cv::Point for fillPoly
    std::vector<cv::Point> polygon_points(footage_corners.begin(), footage_corners.end());

    // Fill the polygon area on the mask with white color (255)
    cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{polygon_points}, cv::Scalar(255));

    // Create the green color
    cv::Scalar field_color(5); // BGR format

    // Fill the area inside the polygon in the dark image with green color
    dark_image.setTo(field_color, mask);

    // Draw circles centered in the bounding boxes
    for (const auto& box : classified_boxes) {
        // Calculate the center of the bounding box
        cv::Point center(box.box.x + box.box.width / 2, box.box.y + box.box.height / 2);

        // Calculate the radius as half of the smaller dimension (width or height)
        int radius = std::min(box.box.width, box.box.height) / 2;

        // Create the color of the ball
        cv::Scalar ball_color = box.ID;

        // Draw the circle on the dark image
        cv::circle(dark_image, center, radius, ball_color, -1); // -1 to fill the circle
    }

    // Ensure hand_mask is single-channel and same size as dark_image
    cv::Mat hand_mask_gray;
    if (hand_mask.channels() == 3) {
        cv::cvtColor(hand_mask, hand_mask_gray, cv::COLOR_BGR2GRAY);
    } else {
        hand_mask_gray = hand_mask;
    }
    
    // Normalize hand_mask to be binary (0 or 255)
    cv::threshold(hand_mask_gray, hand_mask_gray, 128, 255, cv::THRESH_BINARY);

    // Create a black image of the same size as dark_image
    cv::Mat black_image = cv::Mat::zeros(dark_image.size(), CV_8UC1);

    // Combine dark_image and black_image using hand_mask
    cv::Mat final_image;
    dark_image.copyTo(final_image, hand_mask_gray); // copy the dark_image where hand_mask is white
    black_image.copyTo(final_image, 255 - hand_mask_gray); // copy the black_image where hand_mask is black

    return final_image;
}

cv::Vec3b computeMedianColor(const cv::Mat& image) {
    std::vector<uchar> blue, green, red;

    // Extract pixel values for each channel, excluding black pixels ([0, 0, 0])
    for (int row = 0; row < image.rows; ++row) {
        for (int col = 0; col < image.cols; ++col) {
            cv::Vec3b bgr = image.at<cv::Vec3b>(row, col);
            if (bgr != cv::Vec3b(0, 0, 0)) { // Skip black pixels
                blue.push_back(bgr[0]);
                green.push_back(bgr[1]);
                red.push_back(bgr[2]);
            }
        }
    }

    // Function to find median of a vector
    auto findMedian = [](std::vector<uchar>& channel) -> uchar {
        size_t n = channel.size();
        std::sort(channel.begin(), channel.end());
        if (n % 2 == 0) {
            return (channel[n / 2 - 1] + channel[n / 2]) / 2;
        } else {
            return channel[n / 2];
        }
    };

    // Compute the median for each channel
    uchar medianBlue = findMedian(blue);
    uchar medianGreen = findMedian(green);
    uchar medianRed = findMedian(red);

    return cv::Vec3b(medianBlue, medianGreen, medianRed);
}

cv::Vec3b ROItable(const cv::Mat& image, std::vector<cv::Point2f> vertices) {

    // Make a copy of the input image to draw the rectangle on
    cv::Mat result = image.clone();
    
    std::vector<cv::Point> points;
    for (const auto& vertex : vertices) {
        points.push_back(cv::Point(static_cast<int>(vertex.x), static_cast<int>(vertex.y)));
    }

    // Draw the quadrilateral by connecting the vertices
    std::vector<std::vector<cv::Point>> contours;
    contours.push_back(points);
    cv::polylines(result, contours, true, cv::Scalar(0, 255, 255), 2); // Yellow color with thickness 2

    //cv::imshow("Table Contours", result);

    // CUT THE ORIGINAL IMAGE

    // Create a mask for the ROI
    cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1); // Initialize mask with zeros (black)

    // Fill the ROI (region of interest) defined by the vertices with white color (255)
    cv::fillPoly(mask, contours, cv::Scalar(255));

    // Create a masked image using the original image and the mask
    cv::Mat maskedImage;
    image.copyTo(maskedImage, mask);
    
    
    cv::Vec3b tableColor = computeMedianColor(maskedImage);
    
    //std::cout << "BGR: " << tableColor[0] << " " << tableColor[1] << " " << tableColor[2] << " " << std::endl;
    
    return tableColor;
}

int main() {

    std::vector<std::string> imagePaths = {
        "../data/game1_clip1/game1_clip1.mp4",
        "../data/game1_clip2/game1_clip2.mp4",
        "../data/game1_clip3/game1_clip3.mp4",
        "../data/game1_clip4/game1_clip4.mp4",
        "../data/game2_clip1/game2_clip1.mp4",
        "../data/game2_clip2/game2_clip2.mp4",
        "../data/game3_clip1/game3_clip1.mp4",
        "../data/game3_clip2/game3_clip2.mp4",
        "../data/game4_clip1/game4_clip1.mp4",
        "../data/game4_clip2/game4_clip2.mp4"
    };

    for (int i = 0; i < imagePaths.size(); ++i) {

        // Here we have the conversion into frames FRAMES ORIGINAL
        std::vector<cv::Mat> frames;
        double fps;
        std::tie(frames, fps) = videoToFrames(imagePaths[i]);

        // PreProcessing Corners
        cv::Mat result_first;
        std::vector<cv::Vec2f> first_detected_lines;
        std::tie(result_first, first_detected_lines) = preProcess(frames.front());

        // Table Corners
        std::vector<cv::Point2f> footage_corners;
        footage_corners = tableDetection(first_detected_lines);

        // Color Table
        cv::Vec3b tableColor = ROItable(frames.front(), footage_corners);

        std::vector<cv::Rect> bboxes_first;
        std::vector<cv::Rect> bboxes_last;

        cv::Mat hand_first;
        cv::Mat hand_last;
        
        std::tie(bboxes_first, hand_first) = ballsDetection(frames.front(), footage_corners);
        std::tie(bboxes_last, hand_last) = ballsDetection(frames.back(), footage_corners);

        std::vector<BoundingBox> classified_boxes_first;
        std::vector<BoundingBox> classified_boxes_last;

        classified_boxes_first = ballClassification( frames.front(), bboxes_first, tableColor);
        classified_boxes_last = ballClassification( frames.back(), bboxes_last, tableColor);
        
        // segmentation 
        cv::Mat segmentation_first = segmentation(frames.front(),footage_corners, classified_boxes_first, hand_first );
        cv::Mat segmentation_last = segmentation(frames.back(),footage_corners, classified_boxes_last, hand_last);

        cv::Mat first_col, last_col;
        mapGrayscaleMaskToColorImage( segmentation_first, first_col);
        mapGrayscaleMaskToColorImage( segmentation_last, last_col);

        cv::imshow("segmentation_first", first_col);
        cv::imshow("segmentation_last", last_col);

        cv::waitKey(0);

        std::vector<cv::Mat> video_frames = homography_track_balls(frames,footage_corners,classified_boxes_first, classified_boxes_last);
        
        std::cout<<"\nVideo Creation"<<std::endl;
        framesToVideo(video_frames, "Result_"+std::to_string(i)+".mp4", fps); // TO BE USED ON TRACK OUT

        cv::waitKey(0);
        cv::destroyAllWindows();

    }

    // 

    // 

    return 0;
}
