#!/bin/bash
GRAPHICS_PLATFORM="${GRAPHICS_PLATFORM:-cpu}"

# Download ROS ML container and clone required github repos
if [ ! -d  "ros-ml-container" ]; then
    git clone https://github.com/SimonSchwaiger/ros-ml-container
    # Initialer Download der benötigten Software Pakete
    mkdir ros-ml-container/src
    cd ros-ml-container/src
    git clone https://github.com/TW-Robotics/RL-with-3DOF-Robots
    git clone https://github.com/HebiRobotics/hebi_description
    cd ../..
fi

# Create src dir if it's not already present
if [ ! -d "src" ]; then
    mkdir src
fi

# Copy application
rm -rf ros-ml-container/app
cp -r app ros-ml-container
cp -r src/. ros-ml-container/src/
cp requirements.txt ros-ml-container/requirements.txt

# Start container
cd ros-ml-container
GRAPHICS_PLATFORM=$GRAPHICS_PLATFORM ./buildandrun.sh
