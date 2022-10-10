# myYoloV5

What is one of the biggest problems people face when using AI algorithms? LACK OF DATA. Some image augmentation procedures have been selected, which allow to obtain a large quantity of output images to be supplied to the object detection algorithm, in this case YOLO v5 was used.

The system was designed to be adopted in the ERC (European Rover Challenge) race, in particular it will be used to improve the rover's localization within the race track, which is essential for autonomous driving.

The system identifies the billboards placed inside the race track and returns an estimate of the relative distance between the rover and the recognized object.

The map of the race track will be known a few weeks before the start of the race; exploiting this aspect, it is possible to cross the data obtained through the algorithm (which are an relative estimate), those of the map (known and precise) and those obtained through the odometry system already designed by the team, to obtain a precise and effective location.

That estimated relative distance will be improve by ProjectRED team, crossing the information of the normal camera with that of the depth camera; through this operation it will be possible to obtain a much more precise information.

HOW TO USE IN GOOGLE COLAB:
- System needs a csv file with coordinates of boxes, so you can use "makesense.ai" --> object detection --> upload images --> label name: from 0 to 15, using label 0 for box, where number box not identifiable --> export annotation --> csv file --> saving name = coord.csv;
- Every images used in previous step must be insert in a zipfile --> input_data.zip;
- Run "SETUP" section;
- Upload out of folders csv file, zip file, custom_data.yaml (in this repository);
- Upload in yolov5, my_detect.py;
- Initialize the parameters;
- Run every section.





