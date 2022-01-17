#!/usr/bin/env python3

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
                    linearDamping = simParams["JointLinearDamping"][j], 
                    angularDamping = simParams["JointAngularDamping"][j]
                )
        # Create a list of bools of len(joints) in order to track which environments are ready for a sim step
        self.readyEnvs = [ False for _ in joints ]
        # When the loop is done, return joint info to be stored in rosparam
        return joints
    #
    def update(self, readyArr):
        """ Checks which envs are ready for a sim step. If all are ready, a step is called """
        # Update which envs were marked as ready in message
        readyEnvs = [
            a or b
            for a, b in zip(readyArr, self.readyEnvs)
        ]
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
        self.syncPub = rospy.Publisher("jointcontrol/envSync", jointMetric, queue_size = params["NumJoints"])
        # Wait for everything to register
        time.sleep(2)
    #
    def syncCallback(self, data):
        # Check return value of the simulation update
        if self.sim.update(data.ready):
            # If update was perfored, publish the synchronisation message in order to signal to the envs that the sim step is ready
            self.syncPub.publish(
                jointMetric( [ False for _ in range(self.NumJoints) ] )
            )

if __name__ == "__main__":
    wrap = ROSWrapper()
    rospy.spin()
    

#import rospy
#from jointcontrol.msg import jointMetric
#rospy.init_node("test")
#pub = rospy.Publisher("jointcontrol/envSync", jointMetric, queue_size = 3)
#msg = jointMetric()
#pub.publish(msg)

