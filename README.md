# myYoloV5

What is one of the biggest problems people face when using AI algorithms? LACK OF DATA. Some image augmentation procedures have been selected, which allow to obtain a large quantity of output images to be supplied to the object detection algorithm, in this case YOLO v5 was used.

The system was designed to be adopted in the ERC (European Rover Challenge) race, in particular it will be used to improve the rover's location within the race track, which is essential for autonomous driving.

The system identifies the billboards placed inside the race track and returns an estimate of the relative distance between the rover and the recognized object.

The map of the race track will be known a few weeks before the start of the race; exploiting this aspect, it is possible to cross the data obtained through the algorithm (which are a good estimate), those of the map (known and precise) and those obtained through the odometry system already designed by the team, to obtain a precise and effective location .



