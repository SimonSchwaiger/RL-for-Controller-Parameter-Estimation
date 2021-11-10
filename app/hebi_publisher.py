#!/usr/bin/env python3
import rospy
from trajectory_msgs.msg import JointTrajectoryPoint, JointTrajectory
from sensor_msgs.msg import JointState
import time
import math

from serial import Serial
import serial

import hebi
import numpy as np


gripper_enable = True

portn = "/dev/ttyACM0"
baudr = 9600

pwm_open = 10
pwm_closed = 80

debug = False

dt = 0.1


class hebi_publisher():
    """!@brief Helper Class for controlling the SAImon Robot. """

    def jointstate_callback(self, js):
        """ Saves latest jointstate data. """
        self.latest_js = js

    def publish_trajectory_to_robot(self):
        """ Creates and publishes a trajectory to the real robot based on the stored waypoints. """
        #TODO: compensate for gravity in effort
        # solve trajectory using hebi solver
        num_joints = 3
        num_waypoints = len(self.waypoints)

        if debug: print("Number of waypoints: {}".format(num_waypoints))
        if debug: print(self.waypoints)

        pos = np.empty((num_joints, num_waypoints))
        vel = np.empty((num_joints, num_waypoints))
        acc = np.empty((num_joints, num_waypoints))
        # Set first and last waypoint values to 0.0
        vel[:,0] = acc[:,0] = 0.0
        vel[:,-1] = acc[:,-1] = 0.0
        # Set all other values to NaN
        vel[:,1:-1] = acc[:,1:-1] = np.nan
        # Set positions
        for point in range(num_waypoints):
            for joint in range(num_joints):
                pos[joint,point] = self.waypoints[point].position[joint]

        # The times to reach each waypoint (in seconds)
        times = np.linspace(0.0, num_waypoints*dt, num_waypoints)
        # Define trajectory
        trajectory = hebi.trajectory.create_trajectory(times, pos, vel, acc)

        if debug: print("Number of Hebi waypoints: {}".format(trajectory.number_of_waypoints))
        if debug: print(trajectory)

        # Convert trajectory to ROS JointTrajectory and publish it to group_node
        jointTrajectory = JointTrajectory()
        #jointTrajectory.joint_names = ["Arm/J1", "Arm/J2", "Arm/J3"]
        #jointTrajectory.header.stamp = rospy.Time()

        # Add waypoints of hebi trajectory to a ROS JointTrajectory
        for t in range(trajectory.number_of_waypoints):
            pos_cmd, vel_cmd, acc_cmd = trajectory.get_state(t*dt)
            tmp = JointTrajectoryPoint()
            tmp.positions = pos_cmd
            tmp.velocities = vel_cmd
            tmp.accelerations = acc_cmd
            tmp.time_from_start = rospy.Duration.from_sec(t*dt) 
            jointTrajectory.points.append(tmp)

        # Add last waypoint manually
        pos_cmd, vel_cmd, acc_cmd = trajectory.get_state(trajectory.end_time)
        tmp = JointTrajectoryPoint()
        tmp.positions = pos_cmd
        tmp.velocities = vel_cmd
        tmp.accelerations = acc_cmd
        tmp.time_from_start = rospy.Duration.from_sec(trajectory.end_time) 
        jointTrajectory.points.append(tmp)

        
        if debug: print(jointTrajectory)
        self.waypoint_pub.publish(jointTrajectory)
        
        self.clear_waypoints()

    def add_waypoint(self):
        """ Adds current JointState to the stored trajectory. """
        self.waypoints.append(self.latest_js)

    def clear_waypoints(self):
        self.waypoints = []

    def publish_pose_to_robot(self):
        #tmp = JointTrajectoryPoint()
        #tmp.positions = self.latest_js.position
        #tmp.velocities = [0, 0, 0]
        #tmp.accelerations = [0, 0, 0]
        #self.pose_pub.publish(tmp)
        pass

    def open_gripper(self):
        """ Sends command for opening gripper using the python hebi interface. """
        if self.gripper_enable:
            cmd = str(pwm_open) + " \n"
            self.ser.write(cmd.encode('utf-8'))

    def close_gripper(self):
        """ Sends command for closing gripper using the python hebi interface. """
        if self.gripper_enable:
            cmd = str(pwm_closed) + " \n"
            self.ser.write(cmd.encode('utf-8'))

    def __init__(self):
        """ Class constructor, that initialises the jointstate subscriber and publisher. """
        
        self.gripper_enable = gripper_enable

        if self.gripper_enable:
            # setup serial connection to gripper
            try:
                ##Serial connection to the Arduino
                self.ser = serial.Serial(
                    port=portn ,\
                    baudrate=baudr,\
                    parity=serial.PARITY_NONE,\
                    stopbits=serial.STOPBITS_ONE,\
                    bytesize=serial.EIGHTBITS,\
                    timeout=0)
                print("Connected to: " + self.ser.portstr)
            except IOError:
                print ("Serial error" + "\n Exiting Serial Connection \n")
            except OSError:
                print ("Serial error" + "\n Exiting Serial Connection \n")

        # wait for initialisation to be done
        time.sleep(2)

        # setup msg to cpp-based hebi arm for joint control
        # try to init ros node - no new node will be created if init_node has been already called
        try:
            rospy.init_node("hebi_publisher")
        except rospy.exceptions.ROSException:
            pass

        self.latest_js = JointState()
        rospy.Subscriber("joint_states", JointState, self.jointstate_callback)
        #self.pose_pub = rospy.Publisher("/hebi/joint_target", JointTrajectoryPoint, queue_size=1)
        self.waypoint_pub = rospy.Publisher("/hebi/joint_waypoints", JointTrajectory, queue_size=1)
        time.sleep(1)
        
        self.waypoints = []

    def __del__(self):
        if self.gripper_enable:
            self.ser.close()





"""
# Position, velocity, and acceleration waypoints.
# Each column is a separate waypoint.
# Each row is a different joint.

num_joints = 3
num_waypoints = len(self.waypoints)


pos = np.empty((num_joints, num_waypoints))
vel = np.empty((num_joints, num_waypoints))
acc = np.empty((num_joints, num_waypoints))

# Set first and last waypoint values to 0.0
vel[:,0] = acc[:,0] = 0.0
vel[:,-1] = acc[:,-1] = 0.0
# Set all other values to NaN
vel[:,1:-1] = acc[:,1:-1] = np.nan

# Set positions
for i in range(num_waypoints):
    pos[]


pos[:,0] = current_position[:]
pos[:,1] = 0.0
pos[:,2] = [math.pi*2.0, 0.0]
pos[:,3] = [0.0, -math.pi*2.0]
pos[:,4] = 0.0

# The times to reach each waypoint (in seconds)
time = np.linspace(0.0, num_waypoints*dt, dt)

# Define trajectory
trajectory = hebi.trajectory.create_trajectory(time, pos, vel, acc)

"""
