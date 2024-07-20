# CV_Project

How to run the code:

1- Navigate to the empty 'build' folder.

2- Execute the command 'cmake ..'

3- Execute the command 'make' 

4a- Execute the command './main' if you want to iterate through all the provided video clips. This will compute the tracking process, save the output video, and display the output images for each game. To proceed to the next video clip, press any key when prompted.

4b- Execute the command './main ../data/game1_clip1/game1_clip1.mp4' (or any other video file path) if you want to process a single video clip. You need to pass the path to one video file as argument.

4c- Execute the command './performance' if you want to see the system's performance on the entire dataset without visualizing the resulting images and videos.


In cases 4a and 4b, the processed videos will be saved in the 'Results_from_processing' folder. The 'Results' folder already contain all the resulting videos and images of an offline processing of each clip done by us.


