#!/bin/bash

# Download des ROS ML Containers falls dieser noch nicht vorhanden ist
if [ ! -d  "ros-ml-container" ]; then
    git clone https://github.com/SimonSchwaiger/ros-ml-container
    # Initialer Download der benötigten Software Pakete
    mkdir ros-ml-container/src
    cd ros-ml-container/src
    git clone https://github.com/TW-Robotics/RL-with-3DOF-Robots
    git clone https://github.com/HebiRobotics/hebi_description
    cd ../..
fi

# Anlegen des src Ordners falls dieser noch nicht vorhanden ist
if [ ! -d "src" ]; then
    mkdir src
fi

# Kopieren der Applikation
rm -rf ros-ml-container/app
cp -r app ros-ml-container
cp -r src/. ros-ml-container/src/
cp requirements.txt ros-ml-container/requirements.txt

# Ausführen des Containers
cd ros-ml-container
GRAPHICS_PLATFORM=intel ./buildandrun.sh
