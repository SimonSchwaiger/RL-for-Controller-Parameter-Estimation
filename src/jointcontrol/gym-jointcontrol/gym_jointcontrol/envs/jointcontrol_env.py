import gym
from gym import spaces, logger
from gym.utils import seeding

import time
import copy
import sys
import numpy as np
import matplotlib.pyplot as plt

import rospy
from jointcontrol.msg import jointMetric

import pybullet as p

sys.path.append("/catkin_ws/src/jointcontrol/scripts")
from controllers import *

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
        self.observation_space = spaces.Box (
            np.array(self.jointParams["Minimums"], dtype=np.float32),
            np.array(self.jointParams["Maximums"], dtype=np.float32)
        )

        # Instantiate controller as a blueprint to store ts and set params
        self.controllerBlueprint = strategy4Controller(ts=params["SimParams"]["Ts"])
        self.controllerBlueprint.updateConstants(self.jointParams["Defaults"])
        
        self.controllerInstance = None          # Current controller instance
        self.currentParams = None               # Currently set control params
        self.controlSignal = None               # Control signal that is applied each step
        self.numSteps = 0                       # Number of performed steps this episode
        self.maxSteps = 0                       # Maximum number of steps per episode
        self.latestTrajectory = None            # Stores the trajectory resulting from the latest sim step
        self.ts = params["SimParams"]["Ts"]     # Stores time discretisation in order to properly visualise trajectories

        # Connect to the bullet server
        self.physicsClient = p.connect(p.SHARED_MEMORY)

        # Init ROS node
        try:
            rospy.init_node("j{}GymEnv".format(self.jointidx))
        except rospy.exceptions.ROSException:
            pass

        # Create ROS publisher & subscriber to synchronise with physics server
        self.ready = False
        self.syncSub = rospy.Subscriber("jointcontrol/globalEnvSync", jointMetric, self.syncCallback)
        self.syncPub = rospy.Publisher("jointcontrol/envSync", jointMetric, queue_size = 1)

    def __del__(self):
        """ Class Deconstructor """
        p.disconnect()

    # Gym Env methods
    # ----------------------------
    def step(self, action):
        """ Performs one simulation (as defined with episodeType and config params in env.reset()) """

        # If action is not within action space, terminate the episode and return
        if False in [ -1 <= a <= 1 for a in action ]:
            return self.formatObs(), -10, True, {} 

        # Scale action from [-1, 1] back to params
        #action = np.multiply(
        #    np.array(action), 
        #    np.array(self.jointParams["MaxChange"], dtype=np.float64)
        #)
        action = self.actionToParams(action)

        # Calculate current controller params after action
        self.currentParams += np.array(action)

        # If action puts the state outside of the observation space, terminate episode and return
        #if False in [ a <= b <= c for a, b, c in zip(self.jointParams["Minimums"], self.currentParams, self.jointParams["Maximums"]) ]:
        #    return self.formatObs(), -10, True, {} 

        # Clip controller params to Min/Max
        np.clip(
            self.currentParams,
            self.jointParams["Minimums"],
            self.jointParams["Maximums"],
            out = self.currentParams
        )

        # Set joint to initial position
        while abs( self.getJointState()[0] - self.controlSignal[0]) > 0.05:
            self.positionControlUpdate(cmdForce=100, cmdPos=self.controlSignal[0])
            self.waitForPhysicsUpdate()
        # Deactivate position control before proceeding
        self.positionControlUpdate()
        self.waitForPhysicsUpdate()

        # Update controller instance
        self.controllerInstance.updateConstants(self.currentParams)

        resultingPos = []                   # Variable to track resulting motor position
        feedback = self.getJointState()     # Variable to track bullet joint feedback

        # Iterate over controlsignal
        for entry in self.controlSignal:
            # Calculate output torque and apply it to the joint
            # Since we only test postion control, the vel and effort commands are set to 0
            torque = PWM2Torque(
                self.controllerInstance.update([entry, 0, 0], feedback),
                maxMotorTorque=self.jointParams["MaxTorque"]
            )
            self.torqueControlUpdate(torque)
            # Wait for synchronisation
            self.waitForPhysicsUpdate()
            # Get joint feedback & correct torque value (it is not reported by bullet in torque mode)
            feedback = self.getJointState()
            feedback[2] = torque
            # Log feedback for reward calculation
            resultingPos.append(feedback[0])

        # Calculate reward
        reward = self.compute_reard(resultingPos, self.controlSignal, {})

        # Keep track of resulting trajectory
        self.latestTrajectory = resultingPos

        # Check if end of episode is reached
        if self.numSteps < self.maxSteps:
            self.numSteps += 1
            done = False
        else:
            done = True

        # Return
        return self.formatObs(), reward, done, {}

    def reset(self, episodeType='step', config={ "initialPos":0, "stepPos":-1.57, "samplesPerStep":80, "maxSteps":40 }):
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

        generator (TODO):
        performs a custom response (a generator object is provided in config)
            - inSignal [generator]      Generator object for generaring control signal
            - maxSteps [int]:           Number of performed step responses per episode
        
        Switching between types of control signals is implemented using a dict and each episodeType is implemented as its own method.
        In order to use any response other than 'step', the reset method must be called as env.env.reset(episodeType=.., config=..) instead of env.reset() due to how Gym's inheritance is implemented.
        """
        def step(self, config):
            """ Creates and configures step response """
            # Create step response as a simple list and set number of performed tests per episode
            self.controlSignal = [config["initialPos"]] + [ config["stepPos"] for _ in range(config["samplesPerStep"]-1) ]
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
            """ Creates and configures an input signal based on a generator """
            pass

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
        return self.formatObs()

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
    def formatObs(self):
        """ Formats observation """
        obs = self.currentParams
        return obs

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
        js = p.getJointState(self.jointParams["RobotID"], self.jointParams["SegmentID"])
        # Return without information about reaction forces
        return [js[0], js[1], js[3]]

    def positionControlUpdate(self, cmdForce=0, cmdPos=0):
        """ Sets up position control for the next physics update """
        # Set target for next update
        mode = p.POSITION_CONTROL
        p.setJointMotorControl2(self.jointParams["RobotID"], self.jointParams["SegmentID"], controlMode=mode, force=cmdForce, targetPosition=cmdPos)

    def torqueControlUpdate(self, torque):
        """ Sets up torque control for the next physics update """
        mode = p.TORQUE_CONTROL
        p.setJointMotorControl2(self.jointParams["RobotID"], self.jointParams["SegmentID"], controlMode=mode, force=torque)

    def loadBulletTarget(self):
        """ Reloads jointparams from ROS in order to enable changing of the testscene without terminating the env """
        params = rospy.get_param("/jointcontrol")
        self.jointParams = params["J{}".format(self.jointidx)]

    def syncCallback(self, data):
        """ Callback for synchronisation messages """
        if not True in data.ready:
            self.ready = False

    def waitForPhysicsUpdate(self):
        """ Synchronises env with physics server using ROS messages """
        # Publish message indicating that env is ready for sim step
        self.ready = True
        self.syncPub.publish(
            jointMetric( [ False for _ in range(self.jointidx) ] + [ True ] )
        )
        while self.ready:
            time.sleep(1/10000)

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
