#include "homography.h"


// BALL_READ_DATA_FROM_FILE
void BALLreadDataFromFile(const std::string& filename, std::vector<cv::Rect>& balls_footage, std::vector<int>& color) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error opening file " << filename << std::endl;
        return;
    }

    int x, y, width, height, color_code;
    while (infile >> x >> y >> width >> height >> color_code) {
        cv::Rect rect(x-15, y-15, width+30, height+30);
        balls_footage.push_back(rect);
        color.push_back(color_code);
    }

    infile.close();
}




// TABLE_READ_DATA_FROM_FILE
void TABLEreadPointsFromFile(const std::string& filename, std::vector<cv::Point2f>& points) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error opening file " << filename << std::endl;
        return;
    }

    float x, y;
    while (infile >> x >> y) {
        cv::Point2f point(x, y);
        points.push_back(point);
    }

    infile.close();
}




// CHECK_MOVED_BALLS
bool check_moved_balls(const cv::Rect& rect1, const cv::Rect& rect2, int tolerance) {
    return (std::abs(rect1.x - rect2.x) > tolerance ||
            std::abs(rect1.y - rect2.y) > tolerance ||
            std::abs(rect1.width - rect2.width) > tolerance ||
            std::abs(rect1.height - rect2.height) > tolerance); // boolean check on the difference
}



// VIDEO_TO_FRAMES
std::tuple<std::vector<cv::Mat>, double> videoToFrames(const std::string& video_full_path) {
    cv::VideoCapture video(video_full_path); // Video

    if (!video.isOpened()) { // Check on the video availability 
        std::cerr << "Error opening the video" << std::endl;
        return std::make_tuple(std::vector<cv::Mat>(), 0.0); // Return empty vector and fps 0.0
    }

    double fps = video.get(cv::CAP_PROP_FPS); // FPS for future reconstruction 

    std::vector<cv::Mat> frames; // Vector for storing the frames

    cv::Mat frame;

    while (true) {
        video >> frame; // Takes one frame at a time

        if (frame.empty()) { // Checks if it reaches the end
            break;
        }
        
        frames.push_back(frame.clone()); // Loads the vector with frames
    }
    video.release(); // Release the video capture

    return std::make_tuple(frames, fps);
}



// FRAMES_TO_VIDEO
void framesToVideo(const std::vector<cv::Mat>& video_frames, const std::string& output_filename, double fps) {
    if (video_frames.empty()) {
        std::cerr << "No frames" << std::endl;
        return;
    }
    
    // Writing the video from the frames
    int codec = cv::VideoWriter::fourcc('m', 'p', '4', 'v'); // Video properties 
    cv::Size frame_size(video_frames[0].cols, video_frames[0].rows);

    // Initialization of the VideoWriter with the above characteristic
    cv::VideoWriter writer(output_filename, codec, fps, frame_size);

    // Check the writing operation 
    if (!writer.isOpened()) {
        std::cerr << "Error opening output video" << std::endl;
        return;
    }

    for (const auto& frame : video_frames) {
        writer.write(frame); // write the frames into the video
    }

    writer.release();

    std::cout << "\nThe processed video has been correctly saved \n" << std::endl;
}



// BEST_HOMOG
cv::Mat best_homog(std::vector<cv::Point2f> footage_table_corners, std::vector<cv::Point2f> scheme_table_corners) {
    cv::Mat H1 = cv::findHomography(footage_table_corners, scheme_table_corners); // Calculate the first homography matrix
    std::rotate(footage_table_corners.rbegin(), footage_table_corners.rbegin() + 1, footage_table_corners.rend());
    cv::Mat H2 = cv::findHomography(footage_table_corners, scheme_table_corners); // Calculate the second homography matrix
    std::rotate(footage_table_corners.rbegin(), footage_table_corners.rbegin() + 1, footage_table_corners.rend());
    cv::Mat H3 = cv::findHomography(footage_table_corners, scheme_table_corners); // Calculate the third homography matrix
    std::rotate(footage_table_corners.rbegin(), footage_table_corners.rbegin() + 1, footage_table_corners.rend());
    cv::Mat H4 = cv::findHomography(footage_table_corners, scheme_table_corners); // Calculate the fourth homography matrix
    
    // Determine all the four error for each hessian matrix along the diagonal (stretching component)
    double e1 = std::pow((H1.at<double>(0, 0) - 1), 2) + std::pow((H1.at<double>(1, 1) - 1), 2) + std::pow((H1.at<double>(2, 2) - 1), 2);
    double e2 = std::pow((H2.at<double>(0, 0) - 1), 2) + std::pow((H2.at<double>(1, 1) - 1), 2) + std::pow((H2.at<double>(2, 2) - 1), 2);
    double e3 = std::pow((H3.at<double>(0, 0) - 1), 2) + std::pow((H3.at<double>(1, 1) - 1), 2) + std::pow((H3.at<double>(2, 2) - 1), 2);
    double e4 = std::pow((H4.at<double>(0, 0) - 1), 2) + std::pow((H4.at<double>(1, 1) - 1), 2) + std::pow((H4.at<double>(2, 2) - 1), 2);

    // Select the correct Homography matrix, the one with lower error
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
    return H1;
}



// DRAW_TRAJECTORY
void drawTrajectory(cv::Mat& image, const std::vector<cv::Point2f>& trajectory, int color) {
    
    // Playable limit space in the table scheme image
    const int left_limit = 96;
    const int right_limit = 1371;
    const int bottom_limit = 743;
    const int upper_limit = 94;
    
    for (int k = 1; k < trajectory.size(); ++k) {
        cv::Point current_point = trajectory[k];
        cv::Point previous_point = trajectory[k - 1];

        // Check if the point is within the specified region
        if (current_point.x > left_limit && current_point.x < right_limit &&
            current_point.y > upper_limit && current_point.y < bottom_limit) {

            cv::Scalar line_color;

            // Select the color based on the input parameter
            switch (color) {
                case 1:
                    line_color = cv::Scalar(0, 255, 0); // Green
                    break;
                case 2:
                    line_color = cv::Scalar(0, 0, 0);   // Black
                    break;
                case 3:
                    line_color = cv::Scalar(179, 137, 52); // Light-Blue
                    break;
                case 4:
                    line_color = cv::Scalar(0, 0, 255); // Red
                    break;
                default:
                    continue; // Skip unknown colors
            }

            // Draw the line on the image
            cv::line(image, previous_point, current_point, line_color, 4);
        }
    }
}




// DRAW_MOVING_BALL
void drawMovingBall(cv::Mat& image, const std::vector<std::vector<cv::Point2f>>& trajectories_scheme, const std::vector<int>& color, int i, int j) {
	
    // First it is drawn a black circled area and than on top of that a smaller colored circled area so we can obtain a black contour
    if (color[i] == 1) { // white -> white
        cv::circle(image, trajectories_scheme[i][j], 18, cv::Scalar(0, 0, 0), -1);
        cv::circle(image, trajectories_scheme[i][j], 12, cv::Scalar(255, 255, 255), -1);
        }
    
    else if (color[i] == 2) { // black 8 -> black
        cv::circle(image, trajectories_scheme[i][j], 18, cv::Scalar(0, 0, 0), -1);
        cv::circle(image, trajectories_scheme[i][j], 12, cv::Scalar(0, 0, 0), -1);
        } 
    
    else if (color[i] == 3) { // solid -> blue 
        cv::circle(image, trajectories_scheme[i][j], 18, cv::Scalar(0, 0, 0), -1);
        cv::circle(image, trajectories_scheme[i][j], 12, cv::Scalar(179, 137, 52), -1);
        } 
    
    else if (color[i] == 4) { // half -> red
        cv::circle(image, trajectories_scheme[i][j], 18, cv::Scalar(0, 0, 0), -1);
        cv::circle(image, trajectories_scheme[i][j], 12, cv::Scalar(0, 0, 255), -1);
        }
    
}



// DRAW_STATIC_BALL
void drawStaticBalls(cv::Mat& image, const std::vector<cv::Point2f>& centers_scheme, const std::vector<int>& color_just_draw) {
    for (int h = 0; h < color_just_draw.size(); ++h) {
        switch (color_just_draw[h]) {
        // First it is drawn a black circled area and than on top of that a smaller colored circled area so we can obtain a black contour
            case 1: // white -> white
                cv::circle(image, centers_scheme[h], 18, cv::Scalar(0, 0, 0), -1);
                cv::circle(image, centers_scheme[h], 12, cv::Scalar(255, 255, 255), -1);
                break;
            case 2: // black 8 -> black
                cv::circle(image, centers_scheme[h], 18, cv::Scalar(0, 0, 0), -1);
                cv::circle(image, centers_scheme[h], 12, cv::Scalar(0, 0, 0), -1);
                break;
            case 3: // solid -> blue
                cv::circle(image, centers_scheme[h], 18, cv::Scalar(0, 0, 0), -1);
                cv::circle(image, centers_scheme[h], 12, cv::Scalar(179, 137, 52), -1);
                break;
            case 4: // half -> red
                cv::circle(image, centers_scheme[h], 18, cv::Scalar(0, 0, 0), -1);
                cv::circle(image, centers_scheme[h], 12, cv::Scalar(0, 0, 255), -1);
                break;
        }
    }
}



// HSV_PREPROCESSING 
void HSV_preprocessing(const std::vector<cv::Mat>& video_frames, std::vector<cv::Mat>& processed_frames) {
    for (const auto& frame : video_frames) {
        cv::Mat hsvImage;
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



// CLASSIFY_BALLS
void classifyBalls(const std::vector<cv::Rect>& balls_footage_first,
                          const std::vector<cv::Rect>& balls_footage_last,
                          const std::vector<int>& color_first,
                          std::vector<cv::Rect>& balls_just_draw,
                          std::vector<int>& color_just_draw,
                          std::vector<cv::Point2f>& centers_just_draw,
                          std::vector<cv::Rect>& balls_footage,
                          std::vector<int>& color) {
    // Take all the bounding information about the balls in the first frame and check if appear also in the last frame, classify them into moved and not moved (normal or just draw) 
    for (size_t i = 0; i < balls_footage_first.size(); ++i) {
        const cv::Rect& rect_first = balls_footage_first[i];
        bool found = false;

        for (const auto& rect_last : balls_footage_last) {
            if (!check_moved_balls(rect_first, rect_last)) {
                found = true;
                balls_just_draw.push_back(rect_first);

                // Find the corresponding color and push to color_just_draw
                size_t index = i;
                color_just_draw.push_back(color_first[index]);

                cv::Point2f center(rect_first.x + rect_first.width / 2, rect_first.y + rect_first.height / 2);
                centers_just_draw.push_back(center);
                break;
            }
        }

        if (!found) {
            balls_footage.push_back(rect_first);

            // Find the corresponding color and push to color
            size_t index = i;
            color.push_back(color_first[index]);
        }
    }
}



// RESIZE_AND_COPY_TO_FRAME
void resizeAndCopyToFrame(cv::Mat& table_scheme_mod, const cv::Mat& footage_homography, std::vector<cv::Mat>& cloned_video_frames, int j) {
    // Calculate new height based on footage_homography.rows
    const int rescaler_factor = 3.5;
    int newHeight = footage_homography.rows / rescaler_factor;

    // Calculate new width to maintain aspect ratio
    double aspectRatio = static_cast<double>(table_scheme_mod.cols) / table_scheme_mod.rows;
    int newWidth = static_cast<int>(newHeight * aspectRatio);

    // Resize table_scheme_mod to new dimensions
    cv::Mat table_scheme_mod_small;
    cv::resize(table_scheme_mod, table_scheme_mod_small, cv::Size(newWidth, newHeight));

    // Calculate ROI for the frame
    cv::Rect roi(0, footage_homography.rows - newHeight, newWidth, newHeight);

    // Copy resized table_scheme_mod_small to cloned_video_frames[j] at the specified ROI
    table_scheme_mod_small.copyTo(cloned_video_frames[j](roi));
}



// HOMOGRAPHY_TRACK_BALLS
std::vector<cv::Mat> homography_track_balls(std::vector<cv::Mat> video_frames, std::string TEST) {
    
    const int left_limit = 96;
    const int right_limit = 1371;
    const int bottom_limit = 743;
    const int upper_limit = 94;
    
    std::vector<cv::Mat> cloned_video_frames;
    cloned_video_frames.reserve(video_frames.size());

    for (const auto& frame : video_frames) {
        cloned_video_frames.push_back(frame.clone());
    }

     
    std::vector<cv::Mat> final_video;
    final_video.reserve(video_frames.size());

    HSV_preprocessing(video_frames, final_video);  


    // HOMOGRAPHY PART
    cv::Mat const table_scheme = cv::imread("../data/eight_ball_table/Table.jpg"); // Scheme of the table
    std::vector<cv::Point2f>  scheme_corners = {cv::Point2f(82, 81), cv::Point2f(82, 756),      // 1,2
                                                     cv::Point2f(1384, 756), cv::Point2f(1384, 81)}; // 3,4

    // First frame used to find the homography
    cv::Mat footage_homography = final_video[0]; // Camera does not change   
    
    std::string filename = TEST+"corners.txt"; // Change this to your file path
    // std::string filename = TEST+"corners_aaron.txt"; // Change this to your file path
    std::vector<cv::Point2f> footage_table_corners;

    // Read data from file
    TABLEreadPointsFromFile(filename, footage_table_corners);

    // Find Homography transformation
    cv::Mat Homog = best_homog(footage_table_corners,scheme_corners);
    std::cout<<"\nThe best Homography matrix found is: \n"<<std::endl;
    std::cout<<Homog<<std::endl;

    // TRACKING
    std::string BALLfilename_first  = TEST+"bounding_boxes/frame_first_bbox.txt";
    std::string BALLfilename_last  = TEST+"bounding_boxes/frame_last_bbox.txt";
    std::vector<cv::Rect> balls_footage_first;
    std::vector<cv::Rect> balls_footage_last;
    std::vector<cv::Rect> balls_footage;
    std::vector<cv::Rect> balls_just_draw;
    std::vector<int> color_first;
    std::vector<int> color_last;
    std::vector<int> color;
    std::vector<int> color_just_draw;

    std::vector<cv::Point2f> centers_just_draw; 

    // Read data from file
    BALLreadDataFromFile(BALLfilename_first, balls_footage_first, color_first);
    BALLreadDataFromFile(BALLfilename_last, balls_footage_last, color_last);

    classifyBalls(balls_footage_first, balls_footage_last, color_first, balls_just_draw,
                         color_just_draw, centers_just_draw, balls_footage, color);


    std::vector<std::vector<cv::Point2f>> trajectories(balls_footage.size());
    std::vector<std::vector<cv::Point2f>> trajectories_scheme(balls_footage.size());

        
    // Create a new tracker for each ball
    std::vector<cv::Ptr<cv::Tracker>> trackers_ball;

    for (const auto& bbox : balls_footage) {
        cv::Ptr<cv::TrackerCSRT> tracker = cv::TrackerCSRT::create();
        tracker->init(final_video[0], bbox);
        trackers_ball.push_back(tracker);
    }


    std::vector<bool> ball_in(balls_footage.size(), false);

    for (int j = 0; j < final_video.size(); ++j) {
        cv::Mat table_scheme_mod = table_scheme.clone();

        for (int i = 0; i < balls_footage.size(); ++i) {
            if (!ball_in[i]) {
                bool check_tracker = trackers_ball[i]->update(final_video[j], balls_footage[i]);

                if (check_tracker) {
                    cv::Point2f center_of_ball(balls_footage[i].x + balls_footage[i].width / 2, balls_footage[i].y + balls_footage[i].height / 2);
                    trajectories[i].push_back(center_of_ball);

                    cv::perspectiveTransform(trajectories[i], trajectories_scheme[i], Homog);

                    if (trajectories_scheme[i][j].x > left_limit && trajectories_scheme[i][j].x < right_limit &&
                        trajectories_scheme[i][j].y > upper_limit && trajectories_scheme[i][j].y < bottom_limit) {
                        drawMovingBall(table_scheme_mod, trajectories_scheme, color, i, j);
                    } else {
                        ball_in[i] = true;
                        std::cout << "\nDuring the video one ball went into the pocket." << std::endl;
                    }
                } else {
                    std::cerr << "Tracking failure in frame " << i << "  " << j << std::endl;
                }
            }

            drawTrajectory(table_scheme_mod, trajectories_scheme[i], color[i]);

            std::vector<cv::Point2f> centers_scheme;
            cv::perspectiveTransform(centers_just_draw, centers_scheme, Homog);

            drawStaticBalls(table_scheme_mod, centers_scheme, color_just_draw);

            cv::Mat table_scheme_mod_small;
            
            resizeAndCopyToFrame(table_scheme_mod, footage_homography, cloned_video_frames, j);
        }
    }


    return cloned_video_frames;
}


