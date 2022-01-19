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
    git clone https://github.com/ros-infrastructure/rosdoc_lite
    cd ../..
    cd app
    git clone https://github.com/HebiRobotics/hebi_description
    cd ..
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
# Port 6006 is forwarded to allow for the tensorboard gui to be displayed on the host
cd ros-ml-container
GRAPHICS_PLATFORM=$GRAPHICS_PLATFORM PYTHONVER=3.7 DOCKER_RUN_ARGS="-p 6006:6006" ./buildandrun.sh
