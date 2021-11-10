#!/bin/bash
# source ros and python3 components
source ~/myenv/bin/activate
source /opt/ros/noetic/setup.bash
source /catkin_ws/devel/setup.bash

## GYM &  PlaidML Setup

# install gym envs
pip3 install -e /catkin_ws/src/RL-with-3DOF-Robots/fhtw3dof/gym-fhtw3dof
pip3 install -e /app/jointcontrol/gym-jointcontrol
# initiate plaidml
plaidml-setup

## ROS Setup

# start roscore
roscore &

# load SAImon xacro and convert it to urdf for pybullet
cd /catkin_ws/src/RL-with-3DOF-Robots/saimon/urdf 
rosrun xacro xacro -o SAImon.urdf SAImon.xacro
mv SAImon.urdf /app
cd /app

# start ros nodes and put them to the background
#roslaunch --wait saimon SAImon.launch coll_map:=usecase.yaml run_on_real_robot:=false &

# run interactive marker
#python interactive_marker.py &

# start interactive shell for whatever you want to do
# if you want to automatically launch something, you can do that here
#sleep 5
#python #pyBulletTest.py

bash


