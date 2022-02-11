#!/bin/bash
GRAPHICS_PLATFORM="${GRAPHICS_PLATFORM:-cpu}"

# For documentation of the container, please refer to https://github.com/SimonSchwaiger/ros-ml-container
# The container is included in this repo, rather than being downloaded automatically due to some parts of the build being customised

# Create src and app folders for building the application in the ros ml container
# If src is set up the first time, hebi-description and rosdoc lite are also downloaded
if [ ! -d  "ros-ml-container/src" ]; then
    mkdir ros-ml-container/src
    cd ros-ml-container/src
    #git clone https://github.com/TW-Robotics/RL-with-3DOF-Robots
    git clone https://github.com/HebiRobotics/hebi_description
    git clone https://github.com/ros-infrastructure/rosdoc_lite
    cd ../..
fi

if [ ! -d  "ros-ml-container/app" ]; then
    mkdir ros-ml-container/app
fi

# Create src dir if it's not already present
if [ ! -d "src" ]; then
    mkdir src  
fi

# Copy application into build structure
rm -rf ros-ml-container/app
cp -r app ros-ml-container
cp -r src/. ros-ml-container/src/
cp requirements.txt ros-ml-container/requirements.txt

# Build and run container
# Port 6006 is forwarded to allow for the tensorboard gui to be displayed on the host
# Port 8888 is forwarded for jupyter lab
cd ros-ml-container
GRAPHICS_PLATFORM=$GRAPHICS_PLATFORM PYTHONVER=3.7 DOCKER_RUN_ARGS="-p 6006:6006 -p 8888:8888" ./buildandrun.sh
