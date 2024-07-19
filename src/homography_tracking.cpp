#include "homography_tracking.h"
#include "utils.h"


// BALL_READ_DATA_FROM_INPUT
void BALLreadDataFromInput(std::vector<BoundingBox> classified_boxes, std::vector<cv::Rect>& balls_footage, std::vector<int>& color) {
    const int augmentation = 12; // The offset to apply to the corner of bounding box to move it away from the ball
    
    for ( int i = 0; i < classified_boxes.size(); ++i){
        
        // Creation of a vector with of rect identifying the bounding box dimension
        cv::Rect rect(classified_boxes[i].box.x-augmentation, classified_boxes[i].box.y-augmentation, classified_boxes[i].box.width+2*augmentation, classified_boxes[i].box.height+2*augmentation);
        balls_footage.push_back(rect);
        
        // Creation of a vector containing the color of the balls derived by th ebounding boxes 
        color.push_back(classified_boxes[i].ID);
    }
}



// TABLE_READ_DATA_FROM_FILE 
void TABLEreadPointsFromFile(const std::string& corner_txt, std::vector<cv::Point2f>& points) {
    
    // Read and the corners of the scheme table saved in a txt file 
    std::ifstream infile(corner_txt);
    if (!infile.is_open()) {
        std::cerr << "Error opening txt corner file" << corner_txt << std::endl;
        return;
    }

    // creation of the coordinate variable 
    float x, y;
    while (infile >> x >> y) {
        cv::Point2f point(x, y);
        points.push_back(point);
    }

    infile.close();
}



// CHECK_MOVED_BALLS
bool check_moved_balls(const cv::Rect& rect1, const cv::Rect& rect2, const int tolerance) {

    // Check if the difference between two rectangle is higher than a tollerance value that identify the possible motion of the bounding box of a ball 
    return (std::abs(rect1.x - rect2.x) > tolerance || std::abs(rect1.y - rect2.y) > tolerance || // coordinate
            std::abs(rect1.width - rect2.width) > tolerance || std::abs(rect1.height - rect2.height) > tolerance); // dimensions
}



// VIDEO_TO_FRAMES
std::tuple<std::vector<cv::Mat>,const double> videoToFrames(const std::string& video_full_path) {
    
    // Load the video from the path
    cv::VideoCapture video(video_full_path); // Video

    if (!video.isOpened()) { // Check the video availability 
        std::cerr << "Error while opening the video " << video_full_path <<std::endl;
        return std::make_tuple(std::vector<cv::Mat>(), 0.0); // Return empty vector and fps 0.0
    }

    // FPS for future reconstruction from vector of frames
    const double fps = video.get(cv::CAP_PROP_FPS); 

    // Vector where store all the frames
    std::vector<cv::Mat> frames; // Vector for storing the frames

    // Single frame
    cv::Mat frame;

    // Load of frames
    while (true) {
        video >> frame; // Takes one frame at a time

        if (frame.empty()) { // Checks if it reaches the end
            break;
        }
        
        frames.push_back(frame.clone()); // Loads the vector with frames
    }
    video.release(); // Release the video capture

    // return a tuple containing the frames and the fps
    return std::make_tuple(frames, fps);
}



// FRAMES_TO_VIDEO
void framesToVideo(const std::vector<cv::Mat>& video_frames, const std::string& output_filename, const double fps) {
    
    // Check if the frames vector is empty or not 
    if (video_frames.empty()) {
        std::cerr << "No frames for the construction of the video" << std::endl;
        return;
    }
    
    // Writing the video from the frames in the format mp4 
    const int type_of_video = cv::VideoWriter::fourcc('m', 'p', '4', 'v'); // Video properties 
    cv::Size frame_size(video_frames[0].cols, video_frames[0].rows);

    // Initialization of the VideoWriter with the above characteristic
    cv::VideoWriter writer(output_filename, type_of_video, fps, frame_size);

    // Check the writing operation for the creation of the video 
    if (!writer.isOpened()) {
        std::cerr << "Error during the creation of the output video" << std::endl;
        return;
    }

    // writing of the frames
    for (const auto& frame : video_frames) {
        writer.write(frame); // write the frames into the video
    }

    writer.release();

    std::cout << "\nVideo Correctly Saved \n" << std::endl;
}



// BEST_HOMOG
cv::Mat best_homog(std::vector<cv::Point2f> footage_table_corners, std::vector<cv::Point2f> scheme_table_corners) {
    // Compute 4 different homografy matricies w.r.t. rotated corners, two was enough however some noise could happened and ruin the results 
    cv::Mat H1 = cv::findHomography(footage_table_corners, scheme_table_corners); // Calculate the first homography matrix
    std::rotate(footage_table_corners.rbegin(), footage_table_corners.rbegin() + 1, footage_table_corners.rend()); // rotation of the corners
    cv::Mat H2 = cv::findHomography(footage_table_corners, scheme_table_corners); // Calculate the second homography matrix
    std::rotate(footage_table_corners.rbegin(), footage_table_corners.rbegin() + 1, footage_table_corners.rend()); // rotation of the corners
    cv::Mat H3 = cv::findHomography(footage_table_corners, scheme_table_corners); // Calculate the third homography matrix
    std::rotate(footage_table_corners.rbegin(), footage_table_corners.rbegin() + 1, footage_table_corners.rend()); // rotation os the corners
    cv::Mat H4 = cv::findHomography(footage_table_corners, scheme_table_corners); // Calculate the fourth homography matrix
    
    // Determine all the four error (difference) for each hessian matrix along the diagonal (stretching component, if 1 on the diagonal no stretching) 
    double e1 = std::pow((H1.at<double>(0, 0) - 1), 2) + std::pow((H1.at<double>(1, 1) - 1), 2) + std::pow((H1.at<double>(2, 2) - 1), 2);
    double e2 = std::pow((H2.at<double>(0, 0) - 1), 2) + std::pow((H2.at<double>(1, 1) - 1), 2) + std::pow((H2.at<double>(2, 2) - 1), 2);
    double e3 = std::pow((H3.at<double>(0, 0) - 1), 2) + std::pow((H3.at<double>(1, 1) - 1), 2) + std::pow((H3.at<double>(2, 2) - 1), 2);
    double e4 = std::pow((H4.at<double>(0, 0) - 1), 2) + std::pow((H4.at<double>(1, 1) - 1), 2) + std::pow((H4.at<double>(2, 2) - 1), 2);

    //// Select the correct Homography matrix, the one with lower error (difference)
    if (e1 < e2 && e1 < e3 && e1 < e4) {
        return H1;
    }
    if (e2 < e1 && e2 < e3 && e2 < e4) {
        return H2;
    }
    if (e3 < e1 && e3 < e2 && e3 < e4) {
        return H3;
    }
    if (e4 < e1 && e4 < e3 && e4 < e2) {
        return H4;
    }
    else return H1;
}



// DRAW_TRAJECTORY
void drawTrajectory(cv::Mat& image, const std::vector<cv::Point2f>& trajectory, int color) {
    
    // Playable limit space in the table scheme image
    const int left_limit = 96; // x coordinate limit 
    const int right_limit = 1371; // x coordinate limit
    const int bottom_limit = 743; // y coordinate limit
    const int upper_limit = 94; // y coorindate limit  
    
    // Draw the segment from two point: actual and previous (position of the center of the balls) 
    for (int k = 1; k < trajectory.size(); ++k) {
        cv::Point current_point = trajectory[k]; 
        cv::Point previous_point = trajectory[k - 1];

        // Check if the point is within the playable field 
        if (current_point.x > left_limit && current_point.x < right_limit &&
            current_point.y > upper_limit && current_point.y < bottom_limit) {

            cv::Scalar line_color;

            // Draw the color based on the input color parameter
            switch (color) {
                case 1:
                    line_color = cv::Scalar(0, 255, 0); // Green for the white ball
                    break;

                case 2:
                    line_color = cv::Scalar(0, 0, 0);   // Black for the black ball
                    break;

                case 3:
                    line_color = cv::Scalar(179, 137, 52); // Light-Blue for the half
                    break;

                case 4:
                    line_color = cv::Scalar(0, 0, 255); // Red for the solid
                    break;
            }

            // Draw the line on the image
            cv::line(image, previous_point, current_point, line_color, 4);
        }
    }
}



// DRAW_MOVING_BALL
void drawMovingBall(cv::Mat& image, const std::vector<std::vector<cv::Point2f>>& trajectories_scheme, const std::vector<int>& color, int i, int j) {
	
    // First it is drawn a black circled area, than on top of that a smaller colored circled area, so we can obtain a black contour
    const int big_radius = 18; // Good dimension of radious to be clearly seen in the scheme 
    const int small_radius = 12; // Good dimension of radious to be clearly seen in the scheme 
    
    switch (color[i]) {
        // First it is drawn a black circled area and than on top of that a smaller colored circled area so we can obtain a black contour
            case 1: // white -> white
                cv::circle(image, trajectories_scheme[i][j], big_radius, cv::Scalar(0, 0, 0), -1); // BLACK
                cv::circle(image, trajectories_scheme[i][j], small_radius, cv::Scalar(255, 255, 255), -1); // WHITE
                break;

            case 2: // black 8 -> black
                cv::circle(image, trajectories_scheme[i][j], big_radius, cv::Scalar(0, 0, 0), -1); // BLACK
                cv::circle(image, trajectories_scheme[i][j], small_radius, cv::Scalar(0, 0, 0), -1);// BLACK
                break;

            case 3: // solid -> blue
                cv::circle(image, trajectories_scheme[i][j], big_radius, cv::Scalar(0, 0, 0), -1); // BLACK
                cv::circle(image, trajectories_scheme[i][j], small_radius, cv::Scalar(179, 137, 52), -1); // LIGHT BLUE (pure blue too dark to be easly recognize)
                break;

            case 4: // half -> red
                cv::circle(image, trajectories_scheme[i][j], big_radius, cv::Scalar(0, 0, 0), -1); // BLACK
                cv::circle(image, trajectories_scheme[i][j], small_radius, cv::Scalar(0, 0, 255), -1); // RED
                break;
        }
}



// DRAW_STATIC_BALL
void drawStaticBalls(cv::Mat& image, const std::vector<cv::Point2f>& centers_scheme, const std::vector<int>& color_just_draw) {

    // First it is drawn a black circled area, than on top of that a smaller colored circled area, so we can obtain a black contour
    const int big_radius = 18; // Good dimension of radious to be clearly seen in the scheme 
    const int small_radius = 12; // Good dimension of radious to be clearly seen in the scheme 

    for (int h = 0; h < color_just_draw.size(); ++h) {
        switch (color_just_draw[h]) {
        // First it is drawn a black circled area and than on top of that a smaller colored circled area so we can obtain a black contour
            case 1: // white -> white
                cv::circle(image, centers_scheme[h], big_radius, cv::Scalar(0, 0, 0), -1);
                cv::circle(image, centers_scheme[h], small_radius, cv::Scalar(255, 255, 255), -1);
                break;

            case 2: // black 8 -> black
                cv::circle(image, centers_scheme[h], big_radius, cv::Scalar(0, 0, 0), -1);
                cv::circle(image, centers_scheme[h], small_radius, cv::Scalar(0, 0, 0), -1);
                break;

            case 3: // solid -> blue
                cv::circle(image, centers_scheme[h], big_radius, cv::Scalar(0, 0, 0), -1);
                cv::circle(image, centers_scheme[h], small_radius, cv::Scalar(179, 137, 52), -1);
                break;

            case 4: // half -> red
                cv::circle(image, centers_scheme[h], big_radius, cv::Scalar(0, 0, 0), -1);
                cv::circle(image, centers_scheme[h], small_radius, cv::Scalar(0, 0, 255), -1);
                break;
        }
    }
}



// HSV_PREPROCESSING 
void HSV_preprocessing(const std::vector<cv::Mat>& video_frames, std::vector<cv::Mat>& processed_frames) {
    
    // Preporocess to highlight the balls, it is computed for all the frames    
    for (const auto& frame : video_frames) {

        // Single image for saving the converted frame in hsv
        cv::Mat hsvImage;

        // Vector for storing all of the frames
        std::vector<cv::Mat> hsv_mid;

        // Convert the original image to HSV
        cv::cvtColor(frame, hsvImage, cv::COLOR_BGR2HSV);

        // Split the HSV image into its components
        cv::split(hsvImage, hsv_mid);

        // Modify the saturation and value components
        hsv_mid[1] *= 1;  // Scale saturation (no change)
        hsv_mid[2] *= 2;  // Scale value (double)

        // Merge the modified components back into the HSV image
        cv::merge(hsv_mid, hsvImage);

        // Convert the HSV image back to BGR
        cv::Mat modifiedImage;
        cv::cvtColor(hsvImage, modifiedImage, cv::COLOR_HSV2BGR);

        // Add the modified frame to the final video
        processed_frames.push_back(modifiedImage.clone());
    }
}



// IDENTIFY_MOVED_BALLS
void identifyMovedBalls(const std::vector<cv::Rect>& balls_footage_first,
                          const std::vector<cv::Rect>& balls_footage_last,
                          const std::vector<int>& color_first,
                          std::vector<cv::Rect>& balls_just_draw,
                          std::vector<int>& color_just_draw,
                          std::vector<cv::Point2f>& centers_just_draw,
                          std::vector<cv::Rect>& balls_footage,
                          std::vector<int>& color) {
    
    // Value that will be provided to the check_moved_balls function to check if there are difference between firs and last frame
    const int tollerance_of_motion = 5; 
    
    // Take all the bounding information about the balls in the first frame and check if appear also in the last frame, classify them into moved and not moved (normal or just draw) 
    for (int i = 0; i < balls_footage_first.size(); ++i) {

        const cv::Rect& rect_first = balls_footage_first[i];
        bool found = false;

        // Compute on all the balls (bounding boxes) 
        for (const auto& rect_last : balls_footage_last) {

            // Function that check the bounding box between the frame first and last
            if (!check_moved_balls(rect_first, rect_last, tollerance_of_motion)) { 
                
                found = true;
                balls_just_draw.push_back(rect_first);

                // Find the corresponding color and push to color_just_draw
                size_t index = i;
                color_just_draw.push_back(color_first[index]);

                // Not moved balls
                cv::Point2f center(rect_first.x + rect_first.width / 2, rect_first.y + rect_first.height / 2);
                centers_just_draw.push_back(center);

                break;
            }
        }

        if (!found) {
            balls_footage.push_back(rect_first);

            // Find the corresponding color and push to colo

            size_t index = i;
            color.push_back(color_first[index]);
        }
    }
}



// RESIZE_AND_COPY_TO_FRAME
void resizeAndCopyToFrame(cv::Mat& table_scheme_mod, const cv::Mat& footage_homography, std::vector<cv::Mat>& cloned_video_frames, int j) {
    
    // Calculate new height based on footage_homography.rows
    const int rescaler_factor = 3.5; // This values keep the scheme big enough to appriciate the details 
    int new_Height = footage_homography.rows / rescaler_factor; // THe new height of the scheme image
    double aspectRatio = static_cast<double>(table_scheme_mod.cols) / table_scheme_mod.rows; // maintain aspect ratio
    int new_Width = static_cast<int>(new_Height * aspectRatio); // New width based on the aspect ratio 

    // Resize table_scheme_mod to the new Dimensions
    cv::Mat table_scheme_mod_small;
    cv::resize(table_scheme_mod, table_scheme_mod_small, cv::Size(new_Width, new_Height));

    // Calculate ROI for the frame
    cv::Rect roi(0, footage_homography.rows - new_Height, new_Width, new_Height);

    // Copy resized table_scheme_mod_small to cloned_video_frames[j] at the specified ROI, the result it is the small scheme on the bottom left
    table_scheme_mod_small.copyTo(cloned_video_frames[j](roi));
}



// HOMOGRAPHY_TRACK_BALLS MAIN FUNCTION 
std::vector<cv::Mat> homography_track_balls(std::vector<cv::Mat> video_frames, std::vector<cv::Point2f> footage_table_corners, std::vector<BoundingBox> classified_boxes_first, std::vector<BoundingBox> classified_boxes_last ) {
    
    // Constant Parameter identifying the coordinate of the PLAYABLE in the scheme image
    const int left_limit = 96; // left limit x coordinate
    const int right_limit = 1371; // right limit x coordinate
    const int bottom_limit = 743; // bottom limit y coordinate
    const int upper_limit = 94; // upper limit y coordinate

    // Copy of the video frames
    std::vector<cv::Mat> cloned_video_frames;
    cloned_video_frames.reserve(video_frames.size());
    for (const auto& frame : video_frames) {
        cloned_video_frames.push_back(frame.clone());
    }
    // Modified copy of the video, each frames it is preprocessed by the function HSV_preprocessing the value componente it is doubled
    std::vector<cv::Mat> final_video;
    final_video.reserve(video_frames.size());
    HSV_preprocessing(video_frames, final_video);  


    // HOMOGRAPHY PART
    // The scheme of the table is loaded from the following path
    cv::Mat const table_scheme = cv::imread("../data/eight_ball_table/Table.jpg"); // Scheme of the table
    // The corners of the image scheme that identify the CORNER of the green or blue area of the field  
    const  std::vector<cv::Point2f>  scheme_corners = {cv::Point2f(82, 81), cv::Point2f(82, 756),cv::Point2f(1384, 756), cv::Point2f(1384, 81)};
    // First frame used to find the homography
    cv::Mat footage_homography = final_video[0]; // Camera does not change   

    // Find Homography transformation
    cv::Mat Homog = best_homog(footage_table_corners,scheme_corners); // This function provide the best homography matrix w.r.t. the four possible rotation of the corners
    std::cout<<"\nThe best Homography matrix found is: \n"<<std::endl;
    std::cout<<Homog<<std::endl;


    // TRACKING
    std::vector<cv::Rect> balls_footage_first;
    std::vector<cv::Rect> balls_footage_last;
    std::vector<cv::Rect> balls_footage;
    std::vector<cv::Rect> balls_just_draw;
    std::vector<int> color_first;
    std::vector<int> color_last;
    std::vector<int> color;
    std::vector<int> color_just_draw;
    std::vector<cv::Point2f> centers_just_draw; 

    // Read data from the inputs (CLASSIFIED BOUNDING BOXES)
    BALLreadDataFromInput(classified_boxes_first, balls_footage_first, color_first);
    BALLreadDataFromInput(classified_boxes_last, balls_footage_last, color_last);

    // Find which are the moving balls during a game, to speed up the tracker
    identifyMovedBalls(balls_footage_first, balls_footage_last, color_first, balls_just_draw,
                         color_just_draw, centers_just_draw, balls_footage, color);


    std::vector<std::vector<cv::Point2f>> trajectories(balls_footage.size());
    std::vector<std::vector<cv::Point2f>> trajectories_scheme(balls_footage.size());

        
    // Create a new tracker for each ball
    std::vector<cv::Ptr<cv::Tracker>> trackers_ball;

    // The chosen tracker is the CSRT (better results)
    for (const auto& bbox : balls_footage) {
        cv::Ptr<cv::TrackerCSRT> tracker = cv::TrackerCSRT::create();
        tracker->init(final_video[0], bbox);
        trackers_ball.push_back(tracker);
    }

    // Boolean vector where saving the state of the moving ball (once in the hole can't came back (rumor))
    std::vector<bool> ball_in(balls_footage.size(), false);

    std::cout<<"\nStart Tracking the balls along the video"<<std::endl;

    // Loop over all the frames 
    for (int j = 0; j < final_video.size(); ++j) {
        cv::Mat table_scheme_mod = table_scheme.clone();
        
        // Loop over all the balls 
        for (int i = 0; i < balls_footage.size(); ++i) {

            // Check the state of the ball 
            if (!ball_in[i]) {
                
                // Control on the functioning of the tracker (if not tuned correctly, it will stop and fail, so need to be checked)
                bool check_tracker = trackers_ball[i]->update(final_video[j], balls_footage[i]);

                // If Correctly working  save the center of the ball in the real footage of the game 
                if (check_tracker) {
                    cv::Point2f center_of_ball(balls_footage[i].x + balls_footage[i].width / 2, balls_footage[i].y + balls_footage[i].height / 2);
                    trajectories[i].push_back(center_of_ball); // store into trajectories all the centers of the real balls ( moving balls)

                    // Transformation with the omography to retrive the related scheme trajectories to insert to the scheme
                    cv::perspectiveTransform(trajectories[i], trajectories_scheme[i], Homog);

                    // Check if the center of the ball in the scheme are inside of the playable area of the table scheme 
                    if (trajectories_scheme[i][j].x > left_limit && trajectories_scheme[i][j].x < right_limit &&
                        trajectories_scheme[i][j].y > upper_limit && trajectories_scheme[i][j].y < bottom_limit) {                        
                        // If everything is correct draw it as ball 
                        drawMovingBall(table_scheme_mod, trajectories_scheme, color, i, j);

                    } else {
                        // If the center of the ball is outside the playable field it is considered in (goal/ scored point) and so no drawed as a ball
                        ball_in[i] = true;
                        std::cout << "\nDuring the video one ball went into the pocket." << std::endl;
                    }
                } else {
                    // Return a warning if the tracker fail in a specify frame
                    std::cerr << "\nTracking failure in frame " << i << "  " << j << std::endl;
                }
            }

            // Draw the trajectory in the scheme image
            drawTrajectory(table_scheme_mod, trajectories_scheme[i], color[i]);

            // Now remain just the unmoved balls 

            // Vector where stored the centers of the unmoved balls to be inserted into the scheme
            std::vector<cv::Point2f> centers_scheme;
            
            // Transformation from the footage static balls to the shceme ones 
            cv::perspectiveTransform(centers_just_draw, centers_scheme, Homog); 

            // Draw the static ball 
            drawStaticBalls(table_scheme_mod, centers_scheme, color_just_draw);

            cv::Mat table_scheme_mod_small;

            // Resized version of the scheme image and overwriting on the original videp            
            resizeAndCopyToFrame(table_scheme_mod, footage_homography, cloned_video_frames, j);
        }
    }


    return cloned_video_frames;
}


