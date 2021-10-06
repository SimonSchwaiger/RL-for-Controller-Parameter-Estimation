import pybullet as p
import pybullet_data

import rospy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point, PointStamped
import tf

from inaccuracy_compensation import getJointstate

import time

import numpy as np



class SimRobot:
    """ Class that instantiates a simulated robot using Pybullet """
    def __init__(self):
        """ Class Constructor """
        #
        ## Pybullet Setup
        # Init pybullet client and GUI
        self.physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
        # Spawn gravity and groundplane
        p.setGravity(0,0,-9.8)
        planeId = p.loadURDF("plane.urdf")
        startPos = [0,0,1]
        startOrientation = p.getQuaternionFromEuler([0,0,0])
        # Load robot model
        #TODO fix Endeffector not loading in
        self.robotID = p.loadURDF("SAImon.urdf")
        # Print joint info
        print('Printing Joint Summary')
        for joint in range(p.getNumJoints(self.robotID)):
            print('Joint {}: {}\n'.format(joint, p.getJointInfo(self.robotID, joint)))
        #
        ## Virtual Control Setup
        #TODO


    def __del__(self):
        """ Class Destructor """
        # Close Pybullet simulation
        p.disconnect()

    def controlUpdate(self, cmdForce=[60,60,60], cmdPos=[0,0,0]):
        mode = p.POSITION_CONTROL
        p.setJointMotorControl2(self.robotID, 1, controlMode=mode, force=cmdForce[0], targetPosition=cmdPos[0])
        p.setJointMotorControl2(self.robotID, 3, controlMode=mode, force=cmdForce[1], targetPosition=cmdPos[1])
        p.setJointMotorControl2(self.robotID, 5, controlMode=mode, force=cmdForce[2], targetPosition=cmdPos[2])
        p.stepSimulation()






class JointController:
    
    def __init__(self):
        """ Class Constructor """
        # ROS Side Class Members
        self.joints = 3
        self.plannedTrajectory = []

        # Motor Side Class Members
        pass

    def __del__(self):
        pass

