"""
docker exec -it ros_ml_container bash

source ~/myenv/bin/activate
source /opt/ros/noetic/setup.bash
source /catkin_ws/devel/setup.bash
python
"""

import rospy
from jointcontrol.msg import jointMetric

import os

import subprocess

rospy.init_node("test2")

pub = rospy.Publisher("jointcontrol/globalEnvSync", jointMetric, queue_size = 0)

pub.publish(
    jointMetric(
        [False]
    )
)
