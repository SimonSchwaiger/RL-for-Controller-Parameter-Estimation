#!/bin/bash
# source ros and python3 components
source ~/myenv/bin/activate
source /opt/ros/noetic/setup.bash
source /catkin_ws/devel/setup.bash

## Install custom gym env
pip3 install -e /catkin_ws/src/jointcontrol/gym-jointcontrol

## Start roscore
roscore &

## Load robot xacro and convert it to urdf for bullet sim
cd /catkin_ws/src/jointcontrol/urdf 
rosrun xacro xacro -o SAImon.urdf SAImon.xacro
rosrun xacro xacro -o TrainingTestbench.urdf TrainingTestbench.xacro
cd /app

## Load controller configuration as ROS parameters
python /catkin_ws/src/jointcontrol/config/parseConfig.py $(rospack find jointcontrol)/config/testbenchConfig.json

## Start shared memory physics server
# Headless
#/bullet/bullet3-3.21/build_cmake/examples/SharedMemory/App_PhysicsServer_SharedMemory &
# With GUI
/bullet/bullet3-3.21/build_cmake/examples/SharedMemory/App_PhysicsServer_SharedMemory_GUI &
sleep 1

## Connect to the server, load robots and set up synchronisation between envs
python /catkin_ws/src/jointcontrol/scripts/physicsServer.py &
# Wait until everything is loaded
sleep 15

# Output all set params
rosparam list

## Set environment variables and start jupyter lab vor html-based interactive model training
export JUPYTER_ENABLE_LAB=yes
export JUPYTER_TOKEN=docker
jupyter-lab --ip 0.0.0.0 -NotebookApp.token='smart_control' --no-browser --allow-root &

# Start interactive shell to sit idle and keep everything running
#bash
python
