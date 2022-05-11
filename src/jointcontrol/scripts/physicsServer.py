#!/usr/bin/env python3

""" The classes in this file instantiate, configure and synchronise the shared Bullet simulation.  """

# Pybullet components
from asyncore import read
from atexit import register
from ntpath import join
from turtle import update
import pybullet as p
import pybullet_data

# ROS Components
import rospy
from jointcontrol.msg import jointMetric

# Misc
import numpy as np
import time
import copy
import parse

import sys
sys.path.append("/catkin_ws/src/jointcontrol/scripts")
from sharedMemJointMetric import *

performanceDebug = False

class performanceTimer:
    """ Tracks exection time of code segments """
    def __init__(self) -> None:
        self.stamps = []
        self.names = []
        self.reset()
    #
    def reset(self):
        self.stamps = []
        self.names = []
        self.addTimestamp()
    #
    def addTimestamp(self, name=None):
        self.stamps.append(time.time())
        if name == None: self.names.append("{}".format(len(self.names)))
        else: self.names.append(name)
    #
    def printSummary(self):
        self.stamps[0]
        print("Relative time [sec] from first timestamp: \n")
        for stamp, stampName in zip(self.stamps[1:], self.names[1:]):
            print("{:2.6f}    {}".format(stamp-self.stamps[0], stampName))
        print(" ")

def from_s(idx, a):
    """ Returns x and y coordinates of the indexed item in a square with length a """
    x = int(idx/a)
    y = idx%a
    return x, y

def evaluateControlUpdate(cmdString, torqueParser=None, posParser=None):
    """ Evaluates pybulled control update (replaces eval statement for performance) """
    # Return if command is empty
    if cmdString == "": return
    # Distinguish between position and force control
    if "POSITION_CONTROL" in cmdString:
        # Parse and perform position control
        if posParser != None: res = posParser.parse(cmdString)
        else: res = parse.parse("p.setJointMotorControl2({}, {}, controlMode=p.POSITION_CONTROL, force={}, targetPosition={})", cmdString)
        # Unpack return value to required variables
        robotID, jointID, force, targetPos = int(res[0]), int(res[1]), float(res[2]), float(res[3])
        # Perform update
        p.setJointMotorControl2(robotID, jointID, controlMode=p.POSITION_CONTROL, force=force, targetPosition=targetPos)
    elif "TORQUE_CONTROL" in cmdString:
        # Parse and perform torque control
        if posParser != None: res = torqueParser.parse(cmdString)
        else: res = parse.parse("p.setJointMotorControl2({}, {}, controlMode=p.TORQUE_CONTROL, force={})", cmdString)
        # Unpack return value to required variables
        robotID, jointID, force = int(res[0]), int(res[1]), float(res[2])
        # Perform update
        p.setJointMotorControl2(robotID, jointID, controlMode=p.TORQUE_CONTROL, force=force)

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
        # Store compiled bullet commands for performance
        self.posControlCmd = parse.compile("p.setJointMotorControl2({}, {}, controlMode=p.POSITION_CONTROL, force={}, targetPosition={})")
        self.torqueControlCmd = parse.compile("p.setJointMotorControl2({}, {}, controlMode=p.TORQUE_CONTROL, force={})")
        #
        # Store feedback blueprint for performance
        self.feedbackBlueprint = []
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
                    angularDamping = simParams["JointAngularDamping"][j],
                    maxJointVelocity = simParams["JointMaxVelocity"][j]
                )
        # When the loop is done, return joint info to be stored in rosparam
        return joints
    #
    def update(self, updateCommands, feedbackCommands):
        """ Checks which envs are ready for a sim step. If all are ready, a step is called """
        # Create feedback based on blueprint
        if len(self.feedbackBlueprint) < len(feedbackCommands):
            self.feedbackBlueprint = ["(0.0, 0.0, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 0.0)" for _ in feedbackCommands]
        #
        feedback = copy.deepcopy(self.feedbackBlueprint[:len(feedbackCommands)])
        #
        # Iterate over joints and apply update commands
        updateCommandsArr = np.array(updateCommands)
        for cmd in updateCommandsArr[updateCommandsArr != None]:
            evaluateControlUpdate(cmd, torqueParser=self.torqueControlCmd, posParser=self.posControlCmd)
        #
        # Perform sim step
        p.stepSimulation()
        #
        # Get active joint feedback
        feedbackCommandsArr = np.array(feedbackCommands)
        for i, cmd in enumerate( feedbackCommandsArr[feedbackCommandsArr != None] ):
            feedback[i] = str(eval(cmd))
        #
        return feedback

class sharedMemWrapper:
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
        # Create joints and wait for them to be loaded in bullet
        joints = self.sim.createJoints(params["SimParams"])
        time.sleep(3)
        #
        # Shared mem servers
        self.jointMetrics = [ sharedMemJointMetric(i, server=True) for i in range(self.NumJoints) ]
        #
        # Write joint and sharedmem info into rosparam server
        for idx, j in enumerate(joints):
            rospy.set_param(
                "/jointcontrol/J{}/RobotID".format(idx), j[0]
            )
            rospy.set_param(
                "/jointcontrol/J{}/SegmentID".format(idx), j[1]
            )
            rospy.set_param(
                "/jointcontrol/J{}/SharedMemID".format(idx), self.jointMetrics[idx].shm.name
            )
        #
        # Command placeholders
        self.registeredEnvs = [False for _ in range(self.NumJoints)]
        self.updateCommands = [None for _ in range(self.NumJoints)]
        self.updateCommandBlueprint = [None for _ in range(self.NumJoints)]
        self.feedbackCommands = [None for _ in range(self.NumJoints)]
        self.emptyUpdateCount = 0
    #
    def updateCall(self):
        # Check which envs are registered and ready
        ready = [ entry.checkReady() for entry in self.jointMetrics ]
        registered = [ entry.checkRegistered() for entry in self.jointMetrics ]
        #
        # Update everything, if all registered envs are ready
        # If no env is ready, three update cycles are performed in order to allow the server to recover cleaned shared mem instances
        if (np.all(ready == registered) and np.any(registered)) or (not np.any(registered) and self.emptyUpdateCount < 3):
            if np.any(registered): self.emptyUpdateCount = 0
            else: self.emptyUpdateCount += 1
            # Iterate over all envs and work on ready ones
            for idx, (regnew, regold, ready) in enumerate(zip(registered, self.registeredEnvs, ready)):
                if not ready:
                    # If the env has recently been unregistered, reset the server instance
                    if regnew != regold and regnew == False:
                        #try:
                        #    self.jointMetrics[idx].unregister()
                        #except FileNotFoundError:
                        #    pass
                        #self.jointMetrics[idx] = None
                        #self.jointMetrics[idx] = sharedMemJointMetric(idx, server=True)
                        pass
                    # If not ready, do nothing
                    self.feedbackCommands[idx] = None
                    self.jointMetrics[idx].registered = False
                    continue
                else:
                    # Update registered envs
                    self.jointMetrics[idx].loadState()
                    # If the env is newly registered, compile feedback command
                    if regnew != regold and regnew == True:
                        self.feedbackCommands[idx] = compile(self.jointMetrics[idx].feedbackCmd, "<string>", "eval")
                    # update reference command
                    self.updateCommands[idx] = self.jointMetrics[idx].updateCmd
            #
            # Update physics sim
            jointFeedback  = self.sim.update(self.updateCommands, self.feedbackCommands)
            # Reset update commands
            self.updateCommands = copy.deepcopy(self.updateCommandBlueprint)
            self.registeredEnvs = copy.deepcopy(registered)
            # Write joint feedback and flush to shared mem
            for i, entry in enumerate(jointFeedback):
                if registered[i]:
                    self.jointMetrics[i].jointFeedback = entry
                    self.jointMetrics[i].ready = False
                    self.jointMetrics[i].flushState()

if __name__ == "__main__":
    wrap = sharedMemWrapper()
    time.sleep(1)
    try:
        while True:
            wrap.updateCall()
            #time.sleep(1)
    except KeyboardInterrupt:
        [ entry.unregister for entry in wrap.jointMetrics ]
