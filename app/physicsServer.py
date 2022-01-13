#!/usr/bin/env python3

# Pybullet components
import pybullet as p
import pybullet_data

# ROS Components
import rospy

# Misc
import numpy as np
import time

# Params are controlled using rosparam. They are here as an example and to describe each value
jointSimConfig = {
    "JointLinearDamping": 0.04,     # Linear damping for simulated joints
    "JointAangularDamping": 70,     # Angular damping for simulated joints
    "Ts":                1/100,     # Time discretisation (Set to match Hebi motors)
    "NumRobots":             1,     # Number of simulated stations
    "JointsPerRobot":        3,     # Number of joints per robot
    "IdsWithinRobot":[1, 3, 5],     # IDs of movable joints within each robot. Must be of length JointsPerRobot
    "Realtime":          False,     # Activates/Deactivates sleep statements to perform simulation in real time
    "StationSize":      [3, 3],     # Spacing between multiple instances of robots in bullet
    "StationsPerCol":        3,     # Determines the amount of robots per column in simulation (purely visual)
    "URDFName":    "test.urdf",     # Robot URDF name
    "URDFPath":         "/tmp"      # Path to URDF file
}

def from_s(idx, a):
    """ Returns x and y coordinates of the indexed item in a square with length a """
    x = int(idx/a)
    y = idx%a
    return x, y

class bulletInstance:
    """ Manages workers and instantiates bullet simulation """
    def __init__(self, realtime=False, ts=None) -> None:
        # Register physics client
        self.physicsClient = p.connect(p.SHARED_MEMORY)
        #
        # Set up search path for urdf's to be found
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        #
        # Set gravity and load groundplane
        p.setGravity(0,0,-9.8)
        planeId = p.loadURDF("plane.urdf")
        #
        p.setTimestep(ts)
        #
        # Keep track of spawned robots and their id's
        self.activeModels = []
    #
    def __del__(self):
        """ Class Destructor """
        p.disconnect()
    #
    def addUrdf(self, name, pos=[0,0,0], path=None, note=None):
        """ Loads URDF into pybullet sim. If path is set, it will be used as the search path """
        if path != None: p.setAdditionalSearchPath(path)
        robotId = p.loadURDF("plane.urdf", basePosition=pos)
        self.activeModels.append([note, name, robotId])
        return robotId
    #
    def createJoints(self, simParams):
        joints = []
        # Iterate over robots
        for r in range(simParams["NumRobots"]):
            # Get base position
            x, y = from_s(r, simParams["StationsPerCol"])
            # Add URDF
            robotId = self.addUrdf(
                simParams["URDFName"],
                pos = [
                    x*simParams["StationSize"][0],
                    y*simParams["StationSize"][1],
                    0
                ],
                path = simParams["URDFPath"]
            )
            # Iterate over joints of each robot
            for j in range(simParams["JointsPerRobot"]):
                segmentId = simParams["IdsWithinRobot"][j]
                # Store joint info
                joints.append(
                    [
                        robotId,
                        segmentId
                    ]
                )
                # Set physics properties
                p.changeDynamics(
                    robotId, 
                    segmentId, 
                    linearDamping = simParams["JointLinearDamping"], 
                    angularDamping = simParams["JointAngularDamping"]
                )
        # When the loop is done, return joint info to be stored in rosparam
        return joints
    #
    def stepSimulation(self):
        p.stepSimulation()

class ROSWrapper:
    def __init__(self, config) -> None:
        # Load jointcontrol params from rospy
        assert rospy.has_param("/jointcontrol")
        params = rospy.get_param("/jointcontrol")
        #
        # Instantiate bullet instance
        self.sim = bulletInstance(
            realtime = params["SimParams"]["Realtime"],
            ts = params["SimParams"]["ts"]
        )

