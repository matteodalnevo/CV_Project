// Dal Nevo Matteo

#include "homography_tracking.h"
#include "tableDetection.h"
#include "ball_hand_detection.h"
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

void processVideo(const std::string &videoPath, int index = -1) {


    std::cout << "\nANALYSING: " << videoPath.substr(videoPath.find_last_of("/\\") + 1) << std::endl;

    // Here we have the conversion into frames FRAMES ORIGINAL
    std::vector<cv::Mat> frames;

    // frames per second 
    double fps;

    // Compute the function and retrive the frames and the fps
    std::tie(frames, fps) = videoToFrames(videoPath);
    
    // Check if the loading phase of the video succeded (fps should be != from 0)
    // The error handling is deploied to the videoToFrames function that return an error comment
    // the following one is just to exit from this function
    if(fps == 0.0) {
        return; // error
    } 
    
    std::cout << "VIDEO EXTRAPOLATION: DONE" << std::endl;

    // PreProcessing Corners
    cv::Mat result_first;
    std::vector<cv::Vec2f> first_detected_lines;

    // Execute the preprocessing on the image and detect a raw group of lines 
    // Collecting the variable 'result_first' (table mask) is useful only for debugging
    std::tie(result_first, first_detected_lines) = preProcess(frames.front());

    // Table Corners
    std::vector<cv::Point2f> footage_corners;

    // Filter the lines provided from the preprocess and find the corners of the table 
    footage_corners = tableDetection(first_detected_lines);

    std::cout << "TABLE DETECTION: DONE" << std::endl;

    // Vectors of rect where storing the bboxes of the circles from the detection
    std::vector<cv::Rect> bboxes_first; // fisrt frame
    std::vector<cv::Rect> bboxes_last; // last frame

    // Images where storing the hand segmentation 
    cv::Mat hand_first; // first frame
    cv::Mat hand_last; // last frame

    // Balls detection and hand segmentation of first and last frames
    std::tie(bboxes_first, hand_first) = ballsHandDetection(frames.front(), footage_corners);
    std::tie(bboxes_last, hand_last) = ballsHandDetection(frames.back(), footage_corners);

    std::cout << "BALLS DETECTION and HAND SEGMENTATION: DONE" << std::endl;

    // vectors of bounding boxes for the classification
    std::vector<BoundingBox> classified_boxes_first;
    std::vector<BoundingBox> classified_boxes_last;

    // Ball classification for first and last frames
    classified_boxes_first = ballClassification(frames.front(), bboxes_first);
    classified_boxes_last = ballClassification(frames.back(), bboxes_last);

    std::cout << "BALLS CLASSIFICATION: DONE" << std::endl;

    // Creation of the video
    std::vector<cv::Mat> video_frames = homography_track_balls(frames, footage_corners, classified_boxes_first, classified_boxes_last);
    
    // Utilization of a Ternary operator to save the video clip  
    std::string outputFileName = index >= 0 ? "../Results_from_processing/Result_" + std::to_string(index) + ".mp4" : "../Results_from_processing/Result.mp4";
    framesToVideo(video_frames, outputFileName, fps);

    std::cout << "VIDEO SAVING: DONE" << std::endl;

    // Segmentation on the first and last frames
    cv::Mat segmentation_first = segmentation(frames.front(), footage_corners, classified_boxes_first, hand_first);
    cv::Mat segmentation_last = segmentation(frames.back(), footage_corners, classified_boxes_last, hand_last);

    // Mapping of the segmented images to a BGR readable images
    cv::Mat first_col, last_col;
    mapGrayscaleMaskToColorImage(segmentation_first, first_col);
    mapGrayscaleMaskToColorImage(segmentation_last, last_col);

    std::cout << "SEGMENTATION: DONE" << std::endl;
    std::cout << "END" << std::endl;

    // Generation of original image with detected table boundaries and balls bounding boxes
    outputBBImage(frames.front(), footage_corners, classified_boxes_first);
    outputBBImage(frames.back(), footage_corners, classified_boxes_last);

    // Shows the results
    showImage(frames.front(), "Bounding Boxes first frame");
    showImage(frames.back(), "Bounding Boxes last frame");
    showImage(first_col, "Segmentation first frame");
    showImage(last_col, "Segmentation last frame");
    showImage(video_frames.back(), "Game Trajectory last frame");

    //cv::destroyAllWindows();
}

int main(int argc, char* argv[]) {
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

    // No argument provided, loop through the predefined list of clips
    if (argc == 1 ) {
        for (int i = 0; i < imagePaths.size(); ++i) {

            // Overall function that compute al the requested tasks
            processVideo(imagePaths[i], i);
            if (i != imagePaths.size()-1) {
                std::cout << "\nPress any key to proceed to the next clip " << std::endl;
            }
            else  {
                std::cout << "\nPress any key to end the execution of the program" << std::endl;
            }            
            cv::waitKey(0);
        }

        return 0;
    }

    // Argument provided, process the single video file
    if (argc == 2) {
        std::string videoPath = argv[1];
        
        // Overall function that compute al the requested tasks
        processVideo(videoPath);
        std::cout << "\nPress any key to end the execution of the program " << std::endl;
        cv::waitKey(0);
    } 

    // More arguments provided turn an error
    else {
        std::cerr << "Error: you must provide one single argument" << std::endl;
        return 1;
    }

    return 0;
}
