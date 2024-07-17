# CV_Project

The goal of this project is to develop a computer vision system for analyzing video footage of various “Eight Ball” billiard game events.

The required computer vision system must provide high-level information about the status of the match (e.g., ball position) for each frame of the video; this high level information should be displayed as a 2D top-view minimap.

In more detail, for each frame of the input video the system to be developed should be able to:
1. Recognize and localize all the balls inside the playing field, distinguishing them based on their category (1-the white “cue ball”, 2-the black “8-ball”, 3-balls with solid colors, 4-balls with stripes);
2. Detect all the main lines (boundaries) of the playing field;
3. Segment the area inside the playing field boundaries detected in point 2 into the following categories: 1-the white “cue ball”, 2-the black “8-ball”, 3-balls with solid colors, 4-balls with stripes, 5-playing field;
4. Represent the current state of the game in a 2D top-view visualization map, to be updated at each new frame with the current ball positions and the trajectory of each ball that is moving.

# Table of working Hours

|          |   Hours  |                                                                      |
|----------|----------|----------------------------------------------------------------------|
| Matteo   |    03    | Set up for the project / import the dataset                          |
|          |    02    | Planning                                                             |
|          |    02    | Set Up                                                               |
|          |   01:30  | Hough Cirlce and bounding boxes                                      |
|          |    02    | Discussions with the prof and creation of the ROI                    |
|          |    04    | Haar study                                                           |
|          |    03    | Call                                                                 |
|          |    02    | TEST HOUGH                                                           |
|          |    04    | kmeans + tresholding                                                 |
|          |
|          |
|          |
|          |
|          |