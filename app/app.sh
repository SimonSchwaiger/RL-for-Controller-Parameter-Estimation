#!/bin/bash
# source ros and python3 components
source ~/myenv/bin/activate
source /opt/ros/noetic/setup.bash
source /catkin_ws/devel/setup.bash

## Install custom gym env
#pip3 install -e /catkin_ws/src/RL-with-3DOF-Robots/fhtw3dof/gym-fhtw3dof
pip3 install -e /catkin_ws/src/jointcontrol/gym-jointcontrol

## Start roscore
roscore &

## Load robot xacro and convert it to urdf for bullet sim
cd /catkin_ws/src/jointcontrol/urdf 
rosrun xacro xacro -o SAImon.urdf SAImon.xacro
rosrun xacro xacro -o TrainingTestbench.urdf TrainingTestbench.xacro
#mv SAImon.urdf /app
cd /app

## Load controller configuration as ROS parameters
python /catkin_ws/src/jointcontrol/config/parseConfig.py $(rospack find jointcontrol)/config/testbenchConfig.json

## Start shared memory physics server
# Headless
#/bullet/bullet3-3.21/build_cmake/examples/SharedMemory/App_PhysicsServer_SharedMemory &
# With GUI
/bullet/bullet3-3.21/build_cmake/examples/SharedMemory/App_PhysicsServer_SharedMemory_GUI &

## Connect to the server, load robots and set up synchronisation between envs
python physicsServer.py &
# Wait until everything is loaded
sleep 15

rosparam list
#bash

# Start simulated Robot
#python jointControllerRefactor.py &

python

# start ros nodes and put them to the background
#roslaunch --wait saimon SAImon.launch coll_map:=usecase.yaml run_on_real_robot:=false &

# run interactive marker
#python interactive_marker.py &

# start interactive shell for whatever you want to do
# if you want to automatically launch something, you can do that here
#sleep 5
#python #pyBulletTest.py




