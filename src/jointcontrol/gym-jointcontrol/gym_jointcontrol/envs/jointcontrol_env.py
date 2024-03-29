import gym
from gym import spaces, logger
from gym.utils import seeding

import time
import copy
import sys
import re
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

import rospy
from jointcontrol.msg import jointMetric

import pybullet as p

sys.path.append("/catkin_ws/src/jointcontrol/scripts")
from controllers import *
from sharedMemJointMetric import *

from multiprocessing import resource_tracker

sys.path.append("/app/MTExperiments/TrainingScripts")
from HebiUtils import *

def remove_shm_from_resource_tracker():
    """Monkey-patch multiprocessing.resource_tracker so SharedMemory won't be tracked

    More details at: https://bugs.python.org/issue38119

    https://bugs.python.org/issue38119#msg388287
    """
    #
    def fix_register(name, rtype):
        if rtype == "shared_memory":
            return
        return resource_tracker._resource_tracker.register(self, name, rtype)
    resource_tracker.register = fix_register
    #
    def fix_unregister(name, rtype):
        if rtype == "shared_memory":
            return
        return resource_tracker._resource_tracker.unregister(self, name, rtype)
    resource_tracker.unregister = fix_unregister
    #
    if "shared_memory" in resource_tracker._CLEANUP_FUNCS:
        del resource_tracker._CLEANUP_FUNCS["shared_memory"]


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

class jointcontrol_env(gym.Env):
    """!@brief Class implementing the control problem as a Gym Environment

    Observation Space:
        List of currently set params

    Action Space:
        List of how much to change each controller param

    Reward:
        Negative mean squared error of control offset

    """
    def __init__(self, **kwargs):
        """ Class constructor """
        # Get current joint index from keyword arguments
        self.jointidx = kwargs.get('jointidx', 0)
        
        # Load joint controller configuration from ros parameter server
        assert rospy.has_param("/jointcontrol")
        params = rospy.get_param("/jointcontrol")
        assert params["NumJoints"]>self.jointidx
        self.jointParams = params["J{}".format(self.jointidx)]

        # Define action space as box[-1, 1] with the same length as the params
        # Actions are scaled to the different gains, in order for the action noise distribution to be the same for every entry
        self.action_space = spaces.Box(
            np.array( [ -1 for _ in self.jointParams["MaxChange"] ], dtype=np.float32 ),
            np.array( [ 1 for _ in self.jointParams["MaxChange"] ], dtype=np.float32 )
        )

        # Define observation space as box with minimum and maximum controller params
        obsBoxMin = np.concatenate(
            (
                np.array([-10 for _ in range(15)], dtype=np.float32),
                np.array(self.jointParams["Minimums"], dtype=np.float32)
            )
        )
        obsBoxMax = np.concatenate(
            (
                np.array([10 for _ in range(15)], dtype=np.float32),
                np.array(self.jointParams["Maximums"], dtype=np.float32)
            )
        )

        self.observation_space = spaces.Box(obsBoxMin, obsBoxMax)

        # Instantiate controller as a blueprint to store ts and set params
        self.controllerBlueprint = strategy4Controller(ts=params["SimParams"]["Ts"])
        self.controllerBlueprint.updateConstants(self.jointParams["Defaults"])
        
        self.controllerInstance = None          # Current controller instance
        self.currentParams = None               # Currently set control params
        self.controlSignal = None               # Control signal that is applied each step
        self.numSteps = 0                       # Number of performed steps this episode
        self.maxSteps = 0                       # Maximum number of steps per episode
        self.latestTrajectory = None            # Stores the trajectory resulting from the latest sim step
        self.latestControllerOutput = None      # Stores the controller output [Nm] from the latest sim step
        self.ts = params["SimParams"]["Ts"]     # Stores time discretisation in order to properly visualise trajectories

        # Store episode trajectory configuration
        self.episodeType = None
        self.episodeConfig = None
        self.setTrajectoryConfig()

        # Init ROS node
        #try:
        #    rospy.init_node("j{}GymEnv".format(self.jointidx))
        #except rospy.exceptions.ROSException:
        #    pass

        # Connect to the bullet server
        #self.physicsClient = p.connect(p.SHARED_MEMORY)
        # Store update and feedback commands

        # Create ROS publisher & subscriber to synchronise with physics server
        #self.ready = False
        #self.syncSub = rospy.Subscriber("jointcontrol/globalEnvSync", jointMetric, self.syncCallback)
        #self.syncPub = rospy.Publisher("jointcontrol/envSync", jointMetric, queue_size = 1)


        #TODO: have shared mem names be autogenerated and upload them to rosparam
        # Maybe the weird behaviour comes from names not properly setting up

        # Register env with physics server
        self.physicsCommand = sharedMemJointMetric(self.jointidx, server=False)

        self.physicsCommand.setState(
            False,
            True,
            self.jointidx,
            "p.setJointMotorControl2({}, {}, controlMode=p.POSITION_CONTROL, force=0, targetPosition=0)".format(self.jointParams["RobotID"], self.jointParams["SegmentID"]),
            "p.getJointState({}, {})".format(self.jointParams["RobotID"], self.jointParams["SegmentID"]),
            "(0.0, 0.0, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 0.0)"
        )

        self.physicsCommand.flushState()

    def __del__(self):
        """ Class Deconstructor """
        pass
        #self.physicsCommand.unregister()
        # Unregister env at physics server
        # cmd = jointMetric()
        # cmd.ready = False
        # for i in range(3): self.syncPub.publish(cmd)
        #p.disconnect()

    # Gym Env methods
    # ----------------------------
    def step(self, action):
        """
        Performs one simulation (as defined with episodeType and config params in env.reset()).

        If action is set to None, a step is performed without changing internal params of the controller.
        """
        # An action of none corresponds to changes of 0 for each param
        if np.any(action == None): action = [ 0 for _ in self.jointParams["Defaults"] ]

        # If action is not within action space, terminate the episode and return
        if False in [ -1 <= a <= 1 for a in action ]:
            return self.formatObs(), -100, True, {} 

        action = self.actionToParams(action)

        # Limit action to a precision of 3 decimal points
        action = np.round( np.array(action) , decimals=3)
        
        # Calculate current controller params after action
        self.currentParams += np.array(action)

        # Clip controller params to Min/Max
        np.clip(
            self.currentParams,
            self.jointParams["Minimums"],
            self.jointParams["Maximums"],
            out = self.currentParams
        )

        # Deactivate torque control
        #self.torqueControlUpdate(0)
        #self.waitForPhysicsUpdate()
        # Set joint to initial position
        while abs( self.getJointState()[0] - self.controlSignal[0]) > 0.01:
            self.positionControlUpdate(cmdForce=100, cmdPos=self.controlSignal[0])
            self.waitForPhysicsUpdate()
        # Deactivate position control before proceeding
        self.positionControlUpdate()
        self.waitForPhysicsUpdate()

        # Update controller instance
        self.controllerInstance.updateConstants(self.currentParams)

        resultingPos = []                   # Variable to track resulting motor position
        actuatorTorque = []                 # Variable to track applied actuator torque
        feedback = self.getJointState()     # Variable to track bullet joint feedback

        # Track step performance
        debugPerformance = False
        tracker = performanceTimer()

        # Iterate over controlsignal
        for entry in self.controlSignal:
            if debugPerformance: tracker.reset()

            # Calculate output torque and apply it to the joint
            # Since we only test postion control, the vel and effort commands are set to 0
            torque = PWM2Torque(
                self.controllerInstance.update([entry, 0, 0], feedback),
                maxMotorTorque=self.jointParams["MaxTorque"]
            )

            if debugPerformance: tracker.addTimestamp(name="Torque calculated")

            # Track applied torque
            actuatorTorque.append(torque)

            if debugPerformance: tracker.addTimestamp(name="Actuator torque appended")

            # Perform control update
            self.torqueControlUpdate(torque)

            if debugPerformance: tracker.addTimestamp(name="Torque update command sent")

            # Wait for synchronisation
            self.waitForPhysicsUpdate()

            if debugPerformance: tracker.addTimestamp(name="Physics update done")

            # Get joint feedback & correct torque value (it is not reported by bullet in torque mode)
            feedback = self.getJointState()
            feedback[2] = torque
            # Log feedback for reward calculation
            resultingPos.append(feedback[0])

            if debugPerformance: 
                tracker.addTimestamp(name="Step done")
                tracker.printSummary()

        # Calculate reward
        reward = self.compute_reard(resultingPos, self.controlSignal, {})

        # Keep track of resulting trajectory and applied torque signals
        self.latestTrajectory = resultingPos
        self.latestControllerOutput = actuatorTorque

        # Check if end of episode is reached
        if self.numSteps < self.maxSteps:
            self.numSteps += 1
            done = False
        else:
            done = True

        # Return
        return self.formatObs(), reward, done, {}

    def reset(self, episodeType=None, config=None):
        """
        Resets environment and returns env observation 

        The control signal for an episode can be set using the episodeType param and configured using config, a dictionary defining the episode configuration.
        Implemented episode types are 'step' and 'custom'. They are configured using these params in the config dict:

        step (default):
        performs a step response from initialPos to stepPos
            - initialPos [float]:       Initial position before step
            - stepPos [float]:          Position after step
            - samplesPerStep [int]:     Number of discrete controller updates per performed step response
            - maxSteps [int]:           Number of performed step responses per episode

        customSignal:
        performs a custom response (raw input signal is provided in config)
            - inSignal [list of floats] Applied control signal for each step
            - maxSteps [int]:           Number of performed step responses per episode

        square:
        performs response to a square input signal
            - lowerSignal:              Input signal when wave is low
            - higherSignal:             Input signal when wave is high
            - pulseLength:              Length of each low and high pulse
            - numPulses:                Number of performed pulses
            - maxSteps [int]:           Number of performed step responses per episode

        generator:
        performs a custom response generated by the hebi trajectory generator
            - maxSteps [int]:           Number of performed step responses per episode
            - samplesPerStep [int]:     Number of discrete controller updates per performed trajectory
            - sampleRandom [bool]       Determines, whether or not trajectory will be sampled from provided configs
            - timePosSeries [list]      List of dicts describing trajectories [ {"times": [0, 1, 3], "positions": [0, 1, 2]} ]

        Switching between types of control signals is implemented using a dict and each episodeType is implemented as its own method.
        In order to use any response other than 'step', the reset method must be called as env.env.reset(episodeType=.., config=..) instead of env.reset() due to how Gym's inheritance is implemented.
        """
        # If no epsidoeType and config are provided, use the ones stored as class members
        if episodeType == None and config == None:
            episodeType = self.episodeType
            config = self.episodeConfig

        def step(self, config):
            """ Creates and configures step response """
            # Create step response as a simple list and set number of performed tests per episode
            lenInitialSignal = 10
            self.controlSignal = [config["initialPos"] for _ in range(lenInitialSignal) ] + [ config["stepPos"] for _ in range(config["samplesPerStep"]-lenInitialSignal) ]
            self.maxSteps = config["maxSteps"]

        def customSignal(self, config):
            """ Configures custom signal """
            # Set custom signal and number of performed tests per episode
            self.controlSignal = config["inSignal"]
            self.maxSteps = config["maxSteps"]

        def square(self, config):
            """ Creates and configures square signal response """
            # Create single pulse as simple list
            pulse = [ config["lowerSignal"] for _ in range(config["pulseLength"]) ] + [ config["higherSignal"] for _ in range(config["pulseLength"]) ]
            # Append single pulse multiple times to pulseSignal in order to create the whole signal
            pulseSignal = []
            for _ in range(config["numPulses"]): pulseSignal += pulse
            # Set signal and number of performed tests per episode
            self.controlSignal = pulseSignal
            self.maxSteps = config["maxSteps"]

        def generator(self, config):
            """ Creates and configures an input signal based on the hebi trajectory generator """
            # Create trajectory using hebi utils and store position
            res = createTrajectory(config, ts=self.ts)
            self.controlSignal = res[:,0].flatten()
            self.maxSteps = config["maxSteps"]

        # Create dict to mimick switch statement between different modes
        episodeTypes = {
            "step": step,
            "customSignal": customSignal,
            "square": square,
            "generator": generator
        }

        # Instantiate controller with default params
        self.controllerInstance = copy.deepcopy(self.controllerBlueprint)
        self.currentParams = np.array(self.jointParams["Defaults"], dtype=np.float64)

        # Create control signal based on set mode
        try:
            episodeTypes[episodeType](self, config)
        except KeyError:
            # Handle KeyError by setting up an empty episode
            self.controlSignal = [0]
            self.maxSteps = 0

        # Set perfomed steps for this episode to 0 and return
        self.numSteps = 0
        
        # Perform step to populate features required for obs
        self.step(None)

        return self.formatObs()

    def closeSharedMem(self):
        # Remove control over joint
        #self.positionControlUpdate()
        #self.waitForPhysicsUpdate()
        #self.torqueControlUpdate()
        #self.waitForPhysicsUpdate()
        # Remove shared memory from resource tracker in order to prevent resource tracker warnings due shared mem outliving the environment
        # https://bugs.python.org/issue38119#msg388287
        remove_shm_from_resource_tracker()
        # Unregister client
        self.physicsCommand.unregister()

    def render(self, mode="human"):
        # No need to render, as it happens in ROS
        pass
    
    # Gym GoalEnv methods
    # ----------------------------
    def compute_reard(self, achieved_goal, goal, info):
        """
        Computes reward of currently achieved vs. the global goal 
        
        In this context, the achieved goal is the actual plant position, while the actual goal is to follow the control signal as closely as possible

        """
        squaredError = np.square(
            np.array(achieved_goal) - np.array(goal)
        )
        return -1*squaredError.mean()

    # Environment-specific methods
    # ----------------------------

    def setTrajectoryConfig(self, episodeType='step', config={ "initialPos":0, "stepPos":-1.57, "samplesPerStep":150, "maxSteps":40 }):
        """ Permanently stores trajectory type and configuration for each episode """
        self.episodeType = episodeType
        self.episodeConfig = config

    def formatObs(self):
        """ Formats observation """
        def getSignalFeatures(arr):
            return np.array([
                min(arr),
                max(arr),
                np.mean(arr),
                stats.skew(arr),
                stats.kurtosis(arr)
            ])

        obs = []
        obs.extend(self.currentParams)
        obs.extend(getSignalFeatures(self.controlSignal))
        obs.extend(getSignalFeatures(self.latestControllerOutput))
        obs.extend(getSignalFeatures(self.latestTrajectory))

        return np.array(obs, dtype=np.float64)

    def actionToParams(self, action):
        """ Scales action from [-1, 1] to physical controller params """
        return np.multiply(
            np.array(action), 
            np.array(self.jointParams["MaxChange"], dtype=np.float64)
        )

    def paramsToAction(self, params):
        """ Scales controller params to action space of [-1, 1] """
        return np.divide(
            np.array(self.jointParams["MaxChange"], dtype=np.float64),
            np.array(params)
        )

    def getJointState(self):
        """ Returns Jointstate in form [pos, vel, effort] """
        #js = p.getJointState(self.jointParams["RobotID"], self.jointParams["SegmentID"])
        # Read feedback from latest physics update
        fbString = self.physicsCommand.jointFeedback
        # Remove brackets and split into individual values
        i = re.compile('\(')
        fbString = re.sub(i, '', fbString)
        i = re.compile('\)')
        fbString = re.sub(i, '', fbString)
        js = [ float(entry) for entry in fbString.split(", ") ]
        # Return as floats without information about reaction forces
        return [js[0], js[1], js[8]]

    def positionControlUpdate(self, cmdForce=0, cmdPos=0):
        """ Sets up position control for the next physics update """
        # Set target for next update
        self.physicsCommand.updateCmd = "p.setJointMotorControl2({}, {}, controlMode=p.POSITION_CONTROL, force={}, targetPosition={})".format(
            self.jointParams["RobotID"], 
            self.jointParams["SegmentID"], 
            cmdForce, 
            cmdPos
        )
        #mode = p.POSITION_CONTROL
        #p.setJointMotorControl2(self.jointParams["RobotID"], self.jointParams["SegmentID"], controlMode=mode, force=cmdForce, targetPosition=cmdPos)

    def torqueControlUpdate(self, torque):
        """ Sets up torque control for the next physics update """
        self.physicsCommand.updateCmd = "p.setJointMotorControl2({}, {}, controlMode=p.TORQUE_CONTROL, force={})".format(
            self.jointParams["RobotID"],
            self.jointParams["SegmentID"],
            torque
        )
        #mode = p.TORQUE_CONTROL
        #p.setJointMotorControl2(self.jointParams["RobotID"], self.jointParams["SegmentID"], controlMode=mode, force=torque)

    def loadBulletTarget(self):
        """ Reloads jointparams from ROS in order to enable changing of the testscene without terminating the env """
        params = rospy.get_param("/jointcontrol")
        self.jointParams = params["J{}".format(self.jointidx)]

    #def syncCallback(self, data):
        #""" Callback for synchronisation messages """
        # Wait for server response. TODO: if this hangs sometimes, send everything multiple times and synchronise using id's
        #self.ready = False
        #self.physicsCommand.jointFeedback = data.jointFeedback

    def waitForPhysicsUpdate(self):
        """ Synchronises env with physics server using ROS messages """
        
        debugPerformance = False
        if debugPerformance: tracker = performanceTimer()

        # Flush state
        self.physicsCommand.ready = True
        self.physicsCommand.flushState()

        if debugPerformance: tracker.addTimestamp(name="Cmd Published")

        # Wait for update to be done
        while self.physicsCommand.checkReady():
            pass

        if debugPerformance: tracker.addTimestamp(name="Physics response received")

        self.physicsCommand.loadState()

        if debugPerformance: 
            tracker.addTimestamp(name="Update Done")
            tracker.printSummary()

    def visualiseTrajectory(self):
        """ Plots control signal vs. resulting trajectory using matplotlib """
        plt.plot(
            self.ts*np.arange(len(self.controlSignal)),
            self.controlSignal,
            label = "Control Signal"
        )
        if self.latestTrajectory != None:
            plt.plot(
                self.ts*np.arange(len(self.latestTrajectory)),
                self.latestTrajectory,
                label = "Resulting Position"
            )
        plt.xlabel("Time [s]")
        plt.ylabel("Joint Position [rad]")
        plt.legend()
        plt.show()
