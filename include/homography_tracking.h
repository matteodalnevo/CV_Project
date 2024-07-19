// homography_tracking.h
#ifndef HOMOGRAPHY_TRACKING_H
#define HOMOGRAPHY_TRACKING_H

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
#include "utils.h"


// BALL_READ_DATA_FROM_INPUT 
/**
 * @brief Reads ball data (bounding boxes) from the input and stores it in the provided vectors subdividing the rectangle description and the color.
 *
 * @param classified_boxes vector of bounding boxes
 * @param balls_footage Reference to a vector of Rect objects where the bounding boxes of the balls will be stored.
 * @param color Reference to a vector of integers where the color codes of the balls will be stored.
 */
void BALLreadDataFromInput(std::vector<BoundingBox> classified_boxes, std::vector<cv::Rect>& balls_footage, std::vector<int>& color);



// TABLE_READ_POINTS_FROM_FILE 
/**
 * @brief Reads cartesian points from a file and stores them in the provided vector.
 *
 * @param filename Const reference to a string representing the name of the file to read from.
 * @param points Reference to a vector of Point2f objects where the points will be stored.
 */
void TABLEreadPointsFromFile(const std::string& corner_txt, std::vector<cv::Point2f>& points);




// CHECK_MOVED_BALLS
/**
 * @brief Check if the ball has moved during the game.
 * 
 * This function compares two rectangles representing the position of the ball 
 * in two different frames beyond a specific tolerance. This function it is a part of a main function that divide all the moved and not moved balls.
 * 
 * @param rect1 The first rectangle representing the ball's position and size in the first frame.
 * @param rect2 The second rectangle representing the ball's position and size in the second frame.
 * @param tolerance The allowable difference in position and size to still consider the ball as not moved. Default is 5.
 * @return `true` if the ball has moved beyond the specified tolerance, `false` otherwise.
 */
bool check_moved_balls(const cv::Rect& rect1, const cv::Rect& rect2, const int tolerance);



// VIDEO_TO_FRAMES
/**
 * @brief Converts a video into frames.
 * 
 * This function takes the path to a video file, capture the video, and extracts all the single frames.
 * The extracted frames are stored in a vector. The frames per second (FPS) of the video is also returned to have the capability of reconstructing the video.
 * 
 * @param video_full_path The full path to the video file.
 * @return A tuple containing a vector of frames (each frame is a `cv::Mat` object) and a double representing the FPS of the video.
 *         If the video cannot be opened, an empty vector and 0.0 are returned.
 */
std::tuple<std::vector<cv::Mat>,const double> videoToFrames(const std::string& video_full_path);



// FRAMES_TO_VIDEO
/**
 * @brief Converts a sequence of frames into a video file.
 * 
 * This function takes a vector of frames, an output filename, and the desired frames per second (FPS),
 * and creates a video.
 * 
 * @param video_frames A vector of frames (each frame is a `cv::Mat` object) to be written to the video.
 * @param output_filename The name of the output video file.
 * @param fps The frames per second (FPS) for the output video.
 */
void framesToVideo(const std::vector<cv::Mat>& video_frames, const std::string& output_filename, const double fps);



// BEST_HOMOG
/**
 * @brief Finds the best homography matrix to correctly align the passed corners, 4 different homography matrix are computed.
 * 
 * This function takes two sets of points representing corners of the tables. The first one derive from the footage (real image), instead the second set derive from the image scheme of the table. Then it calculates four possible homography matrices by rotating the first set of points (2 was enough).
 * It then determines the best homography matrix based on the smallest error (smallest difference from the identity matrix w.r.t. the diagonal element).
 * 
 * @param footage_table_corners A vector of points representing the corners in the real footage.
 * @param scheme_table_corners A vector of points representing the corners in the scheme table.
 * @return The best homography matrix (as a `cv::Mat`) that aligns the corners from the first image to the second image.
 */
cv::Mat best_homog(std::vector<cv::Point2f> footage_table_corners, std::vector<cv::Point2f> scheme_table_corners);



// DRAW_TRAJECTORY
/**
 * @brief Draws lines representing the trajectory of the mooving ball.
 * 
 * This function takes an image (frame) and a vector of points representing a trajectory and draws lines between the points in the specified color. The lines are drawn only if the points are within a specific region of the image that represents the playable space in the table scheme.
 * 
 * @param image The image (frame) on which to draw the trajectory.
 * @param trajectory A vector of points (`cv::Point2f`) representing the trajectory to be drawn.
 * @param color The color of the ball, this also specify the color of the trajectory lines. The color is selected based on the following:
 *              - 1: Green
 *              - 2: Black
 *              - 3: Light-Blue
 *              - 4: Red
 */
void drawTrajectory(cv::Mat& image, const std::vector<cv::Point2f>& trajectory, int color);



// DRAW_MOVING_BALL
/**
 * @brief Draws moving ball and calls the function to draw its trajectory.
 * 
 * This function draws circles representing moving ball on an image based on their trajectory and specified color.
 * The function uses different colors to represent different types of balls.
 * 
 * @param image The image (frame)on which to draw the moving ball.
 * @param trajectories_scheme A vector of vectors of points (`cv::Point2f`) representing the trajectory of the ball.
 * @param color A vector of integers specifying the color type for each ball. The color type is determined based on the following:
 *              - 1: White ball
 *              - 2: Black ball (8 ball)
 *              - 3: Solid ball
 *              - 4: Striped ball
 * @param i The index of the ball in the `trajectories_scheme` and `color` vectors.
 * @param j The index of the current point in the trajectory of the ball.
 */
void drawMovingBall(cv::Mat& image, const std::vector<std::vector<cv::Point2f>>& trajectories_scheme, const std::vector<int>& color, int i, int j);



// DRAW_STATIC_BALLS
/**
 * @brief Draws static balls on an image (frame).
 * 
 * This function takes an image (frame) and vectors representing the centers an the colors of static balls. It draws the balls at the specified centers with the corresponding colors.
 * 
 * @param image The image on which to draw the static balls.
 * @param centers_scheme A vector of points (`cv::Point2f`) representing the centers of the static balls.
 * @param color_just_draw A vector of integers specifying the color type for each ball. The color type is determined based on the following:
 *                        - 1: White ball
 *                        - 2: Black ball (8 ball)
 *                        - 3: Solid ball
 *                        - 4: Striped ball
 */
void drawStaticBalls(cv::Mat& image, const std::vector<cv::Point2f>& centers_scheme, const std::vector<int>& color_just_draw);



// HSV_PREPROCESSING
/**
 *@brief Preprocesses a sequence of video frames by adjusting saturation and value components in HSV color space.
 *
 *@param video_frames Vector of BGR video frames.
 *@param processed_frames Vector of output frames after HSV adjustment.
 */
void HSV_preprocessing(const std::vector<cv::Mat>& video_frames, std::vector<cv::Mat>& processed_frames);



// IDENTIFY_MOVED_BALLS
/**
 * @brief Classifies balls based on their movement or not between between two frames (first and last).
 *
 * @param balls_footage_first Vector of cv::Rect representing the bounding boxes of balls in the first frame.
 * @param balls_footage_last Vector of cv::Rect representing the bounding boxes of balls in the last frame.
 * @param color_first Vector of integers representing the colors of the balls in the first frame.
 * @param balls_just_draw Output vector where the bounding boxes of balls that have not moved will be stored.
 * @param color_just_draw Output vector where the colors of the balls that have not moved will be stored.
 * @param centers_just_draw Output vector where the centers of the balls that have not moved will be stored.
 * @param balls_footage Output vector where the bounding boxes of balls that have moved will be stored.
 * @param color Output vector where the colors of the balls that have moved will be stored.
 */
void identifyMovedBalls(const std::vector<cv::Rect>& balls_footage_first,
                   const std::vector<cv::Rect>& balls_footage_last,
                   const std::vector<int>& color_first,
                   std::vector<cv::Rect>& balls_just_draw,
                   std::vector<int>& color_just_draw,
                   std::vector<cv::Point2f>& centers_just_draw,
                   std::vector<cv::Rect>& balls_footage,
                   std::vector<int>& color);



// RESIZE_AND_COPY_TO_FRAME
/**
 * @brief Resizes a frame and attaches it to a specified region in the cloned video frames.
 *
 * @param table_scheme_mod Reference to the table scheme matrix that will be resized and copied.
 * @param footage_homography Const reference to the footage homography matrix used to determine new dimensions.
 * @param cloned_video_frames Vector of Mat objects representing the cloned video frames.
 * @param j Index of the frame in cloned_video_frames where the resized table scheme will be copied.
 */
void resizeAndCopyToFrame(cv::Mat& table_scheme_mod, 
                          const cv::Mat& footage_homography, 
                          std::vector<cv::Mat>& cloned_video_frames, 
                          int j);
                   
                   

// HOMOGRAPHY_TRACK_BALLS                   
/**
 * @brief Tracks balls in video frames, draws their trajectories, and returns the modified frames.
 *
 * This function it is the main function that take and utilized most of the smaller function provided above.
 * @param video_frames Vector of Mat objects representing the original video frames.
 * @param footage_table_corners vector of point that identify the corners of the table deriving from the video
 * @param classified_boxes_first vector containing the bounding boxes of the balls in the first frame
 * @param classified_boxes_last vector containing the bounding boxes of the balls in the last frame
 * @return A vector of Mat objects representing the modified video frames with drawn trajectories.
 */
std::vector<cv::Mat> homography_track_balls(std::vector<cv::Mat> video_frames, std::vector<cv::Point2f> footage_table_corners, std::vector<BoundingBox> classified_boxes_first, std::vector<BoundingBox> classified_boxes_last);                   
                   
                   
#endif // HOMOGRAPHY_H
