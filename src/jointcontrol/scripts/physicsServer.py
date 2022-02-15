#!/usr/bin/env python3

""" The classes in this file instantiate, configure and synchronise the shared Bullet simulation.  """

# Pybullet components
from ntpath import join
import pybullet as p
import pybullet_data

# ROS Components
import rospy
from jointcontrol.msg import jointMetric

# Misc
import numpy as np
import time

def from_s(idx, a):
    """ Returns x and y coordinates of the indexed item in a square with length a """
    x = int(idx/a)
    y = idx%a
    return x, y

def bringToSameLength(list1, list2):
    """ Matches length of two lists. If one is shorter than the other, it will be extended with the contents of the other array """
    #https://stackoverflow.com/questions/29972836/numpy-how-to-resize-an-random-array-using-another-array-as-template
    if len(list1) < len(list2):
        list1[len(list1):len(list2)]=list2[len(list1):]
    elif len(list2) < len(list1):
        list2[len(list2):len(list1)]=list1[len(list2):]
    return list1, list2

class bulletInstance:
    """!@brief Manages workers and instantiates bullet simulation """
    def __init__(self, realtime=False, ts=None, spawnGroundplane=False) -> None:
        # Register physics client
        self.physicsClient = p.connect(p.SHARED_MEMORY)
        #
        # Set up search path for urdf's to be found
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        #
        # Set gravity and load groundplane
        p.setGravity(0,0,-9.8)
        if spawnGroundplane == True:
            planeId = p.loadURDF("plane.urdf")
        #
        p.setTimeStep(ts)
        #
        # Keep track of spawned robots and their id's
        self.activeModels = []
        #
        # Keep track of which envs are ready for a sim step
        self.readyEnvs = None
    #
    def __del__(self):
        """ Class Destructor """
        p.disconnect()
    #
    def addUrdf(self, name, pos=[0,0,0], path=None, note=None):
        """ Loads URDF into pybullet sim. If path is set, it will be used as the search path """
        if path != None: p.setAdditionalSearchPath(path)
        robotId = p.loadURDF(name, basePosition=pos)
        self.activeModels.append([note, name, robotId])
        # Wait for model to be loaded
        time.sleep(1)
        return robotId
    #
    def createJoints(self, simParams):
        """ Creates simulation environment based on simParams """
        joints = []
        # Iterate over robots
        for r in range(simParams["NumRobots"]):
            # Get base position
            x, y = from_s(r, simParams["StationsPerCol"])
            # Add URDF
            robotId = self.addUrdf(
                simParams["URDFName"][r],
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
                    linearDamping = simParams["JointLinearDamping"][j], 
                    angularDamping = simParams["JointAngularDamping"][j]
                )
        # When the loop is done, return joint info to be stored in rosparam
        return joints
    #
    def update(self, readyArr):
        """ Checks which envs are ready for a sim step. If all are ready, a step is called """
        # Update which envs were marked as ready in message
        if self.readyEnvs != None:
            readyArr, self.readyEnvs = bringToSameLength(readyArr, self.readyEnvs)
            readyEnvs = [
                a or b
                for a, b in zip(readyArr, self.readyEnvs)
            ]
        else:
            readyEnvs = readyArr
        # Check if all envs are ready 
        if all(readyEnvs):
            # If they are, step simulation and reset the variable
            p.stepSimulation()
            self.readyEnvs = [ False for _ in readyEnvs ]
            return 1
        else:
            # Otherwise, update class member
            self.readyEnvs = readyEnvs
            return 0

class ROSWrapper:
    """!@brief Wraps physics simulation helper class to the ROS param and msg services """
    def __init__(self) -> None:
        # Load jointcontrol params from rospy
        assert rospy.has_param("/jointcontrol")
        params = rospy.get_param("/jointcontrol")
        self.NumJoints = params["NumJoints"]
        #
        # Instantiate bullet instance
        self.sim = bulletInstance(
            realtime = params["SimParams"]["Realtime"],
            ts = params["SimParams"]["Ts"]
        )
        # Create joints and store joint info in rosparam server
        joints = self.sim.createJoints(params["SimParams"])
        for idx, j in enumerate(joints):
            rospy.set_param(
                "/jointcontrol/J{}/RobotID".format(idx), j[0]
            )
            rospy.set_param(
                "/jointcontrol/J{}/SegmentID".format(idx), j[1]
            )
        # Subscribe to synchronisation message and check in callback whether or not to perform a simulation step
        rospy.init_node("BulletSimServer")
        self.syncSub = rospy.Subscriber("jointcontrol/envSync", jointMetric, self.syncCallback)
        # Register publisher to the synchronisation message
        self.syncPub = rospy.Publisher("jointcontrol/globalEnvSync", jointMetric, queue_size = 1)
        # Wait for everything to register
        time.sleep(2)
    #
    def syncCallback(self, data):
        """ Callback for synchronisation messages """
        # Check return value of the simulation update
        if self.sim.update(data.ready):
            # If update was perfored, publish the synchronisation message in order to signal to the envs that the sim step is ready
            self.syncPub.publish(
                jointMetric( [ False for _ in data.ready ] )
            )

if __name__ == "__main__":
    wrap = ROSWrapper()
    rospy.spin()
