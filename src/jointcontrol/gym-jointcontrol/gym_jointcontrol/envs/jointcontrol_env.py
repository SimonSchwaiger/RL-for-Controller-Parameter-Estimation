import gym
from gym import spaces, logger
from gym.utils import seeding

import time
import copy
import numpy as np

import rospy
from jointcontrol.msg import jointMetric

import pybullet as p

import matplotlib.pyplot as plt

#from sensor_msgs.msg import JointState

import numpy as np

def clampValue(val, valMax):
    """ Makes sure that a value is within [-valMax, valMax] """
    if valMax == None: return val
    if val > valMax: return valMax
    elif val < -1*valMax: return -1*valMax
    else: return val

class PIDController:
    """!@brief Discrete PID controller approximated using the Tustin (trapezoid) approximation """
    def __init__(self, kp=0, ki=0, kd=0, ts=0, feedforward=0, bufferLength=3) -> None:
        #self.bufferLength = bufferLength
        self.k1 = 0
        self.k2 = 0
        self.k3 = 0
        self.e = [0 for i in range(bufferLength)]
        self.y = [0 for i in range(bufferLength)]
        self.feedforward = feedforward
        if ts != 0:  self.setConstants(kp, ki, kd, ts)
    #
    def setConstants(self, kp, ki, kd, ts):
        """ Updates controller constants """
        self.k1 = kp+((ki*ts)/2)+((2*kd)/ts)
        self.k2 = ki*ts-((4*kd)/ts)
        self.k3 = (-1*kp)+((ki*ts)/2)+((2*kd)/ts)   
    #
    def update(self, e):
        """
        Performs one discrete controller update 
        
        Receives: Input signal [float]
        Returns: Output signal [float]
        """
        # Update buffered input and output signals
        self.e = [e]+self.e[:len(self.e)-1]
        self.y = [0]+self.y[:len(self.y)-1]
        # Shorten variable names for better readability
        e = self.e
        y = self.y
        # Calculate output signal and return output
        y[0] = e[0]*self.k1 + e[1]*self.k2 + e[2]*self.k3 + y[2]
        return y[0] + e[0]*self.feedforward

class PT1Block:
    """!@brief Discrete PT1 block approximated using the Tustin (trapezoid) approximation """
    def __init__(self, kp=1, T1=0, ts=0, bufferLength=2) -> None:
        self.k1 = 0
        self.k2 = 0
        self.e = [0 for i in range(bufferLength)]
        self.y = [0 for i in range(bufferLength)]
        if ts != 0: self.setConstants(kp, T1, ts)
    #
    def setConstants(self, kp, T1, ts):
        """ Updates controller constants """
        t = 2*(T1/ts)
        self.k1 = kp/(1+t)
        self.k2 = (1-t)/(1+t)
    #
    def update(self, e):
        """
        Performs one discrete controller update 
        
        Receives: Input signal [float]
        Returns: Output signal [float]
        """        
        # Update buffered input and output signals
        self.e = [e]+self.e[:len(self.e)-1]
        self.y = [0]+self.y[:len(self.y)-1]
        # Shorten variable names for better readability
        e = self.e
        y = self.y
        # Calculate output signal and return output
        y[0] = (e[0] + e[1])*self.k1 - y[1]*self.k2
        return y[0]

class DBlock:
    """!@brief Discrete D Block approximated using the Tustin approximation """
    def __init__(self, kd=0, ts=0, bufferLength=2) -> None:
        self.k = 0
        self.e = [0 for i in range(bufferLength)]
        self.y = [0 for i in range(bufferLength)]
        if ts != 0: self.setConstants(kd, ts)
    #
    def setConstants(self, kd, ts):
        """ Updates controller constants """
        self.k = (2*kd)/ts
    #
    def update(self, e):
        """
        Performs one discrete controller update 
        
        Receives: Input signal [float]
        Returns: Output signal [float]
        """        
        if self.k == 0: return 0
        # Update buffered input and output signals
        self.e = [e]+self.e[:len(self.e)-1]
        self.y = [0]+self.y[:len(self.y)-1]
        # Shorten variable names for better readability
        e = self.e
        y = self.y
        # Calculate output signal and return output
        y[0] = e[0]*self.k - e[1]*self.k - y[1]
        return y[0]

class PT2Block:
    """!@brief Discrete PT2 Block approximated using the Tustin approximation """
    def __init__(self, T=0, D=0, kp=1, ts=0, bufferLength=3) -> None:
        self.k1 = 0
        self.k2 = 0
        self.k3 = 0
        self.k4 = 0
        self.k5 = 0
        self.k6 = 0
        self.e = [0 for i in range(bufferLength)]
        self.y = [0 for i in range(bufferLength)]
        if ts != 0:  self.setConstants(T, D, kp, ts)
    #
    def setConstants(self, T, D, kp, ts):
        """ Updates controller constants """
        self.k1 = 4*T**2 + 4*D*T*ts + ts**2
        self.k2 = 2*ts**2 - 8*T**2
        self.k3 = 4*T**2 - 4*D*T*ts + ts**2
        self.k4 = kp*ts**2
        self.k5 = 2*kp*ts**2
        self.k6 = kp*ts**2
    #
    def update(self, e):
        """
        Performs one discrete controller update 
        
        Receives: Input signal [float]
        Returns: Output signal [float]
        """        
        # Update buffered input and output signals
        self.e = [e]+self.e[:len(self.e)-1]
        self.y = [0]+self.y[:len(self.y)-1]
        # Shorten variable names for better readability
        e = self.e
        y = self.y
        # Calculate output signal and return output
        y[0] = ( e[0]*self.k4 + e[1]*self.k5 + e[2]*self.k6 - y[1]*self.k2 - y[2]*self.k3 )/self.k1
        return y[0]

class smartPID:
    """!@brief Implementation to mimick HEBI PID controller behaviour. Variable names are consistent with HEBI Gains format """
    def __init__(self, kp=0, ki=0, kd=0, targetLP=0, outputLP=0, ts=0, feedforward=0, d_on_error=True, targetMax=None, outputMax=None) -> None:
        self.d_on_error = d_on_error
        if d_on_error:
            self.PID = PIDController(kp=kp, ki=ki, kd=kd, ts=ts)
            self.D = DBlock(kd=0, ts=ts)
        else:
            self.PID = PIDController(kp=kp, ki=ki, kd=0, ts=ts)
            self.D = DBlock(kd=kd, ts=ts)
        #
        self.targetLP = targetLP
        self.outputLP = outputLP
        self.inputFilter  = PT1Block(kp=1, T1=targetLP, ts=ts)
        self.outputFilter = PT1Block(kp=1, T1=outputLP, ts=ts)
        #
        self.targetMax = targetMax
        self.outputMax = outputMax
        self.feedforward = feedforward
    #
    def setConstants(self, kp, ki, kd, ts):
        """ Updates controller constants """
        if self.d_on_error:
            self.PID.setConstants(kp, ki, kd, ts)
        else:
            self.PID.setConstants(kp, ki, 0, ts)
            self.D.setConstants(kd, ts)
        #
        self.inputFilter  = PT1Block(kp=1, T1=self.targetLP, ts=ts)
        self.outputFilter = PT1Block(kp=1, T1=self.outputLP, ts=ts)
    #
    def update(self, target, feedback):
        """
        Performs one discrete controller update 
        
        Receives: Input signal [float]
        Returns: Output signal [float]
        """        
        # Clamp and low pass input
        filteredInput = self.inputFilter.update(
            clampValue(target, self.targetMax)
        )
        # Update internal control blocks
        output = self.D.update(filteredInput) + self.PID.update(filteredInput - feedback)
        # Clamp and low pass output
        output = self.outputFilter.update(
            clampValue(output + self.feedforward*filteredInput, self.outputMax)
        )
        return output
        
class strategy4Controller:
    """!@brief Models HEBI control strategy 4 using 3 PID controllers discretised using the Tustin approximation """
    def __init__(self, ts=0, targetConstraints=[None, 3.43, 20], outputConstraints=[10, 1, 1], feedfowards=[0, 1, 1], d_on_errors=[True, True, False], constants=None) -> None:
        """ Class constructor """
        self.ts = ts
        #
        self.PositionPID = smartPID(
            targetMax=targetConstraints[0],
            outputMax=outputConstraints[0],
            feedforward=feedfowards[0],
            d_on_error=d_on_errors[0]
        )
        #
        self.VelocityPID = smartPID(
            targetMax=targetConstraints[1],
            outputMax=outputConstraints[1],
            feedforward=feedfowards[1],
            d_on_error=d_on_errors[1],
            outputLP=0.01
        )
        #
        self.EffortPID = smartPID(
            targetMax=targetConstraints[2],
            outputMax=outputConstraints[2],
            feedforward=feedfowards[2],
            d_on_error=d_on_errors[2],
            outputLP=0.001
        )
        #
        self.PWMFilter = PT2Block(kp=1, T=0.0, D=10, ts=self.ts)
        if constants != None: self.updateConstants(constants)
    #
    def updateConstants(self, constants):
        """ Updates controller constants """
        self.PositionPID.setConstants(constants[0], constants[1], constants[2], self.ts)
        self.VelocityPID.setConstants(constants[3], constants[4], constants[5], self.ts)
        self.EffortPID.setConstants(  constants[6], constants[7], constants[8], self.ts)
    #
    def update(self, vecIn, feedback):
        """ 
        Takes feedback and control signal and processes output signal 
        
        Format vecIn & feedback: [pos, vel, effort]
        """
        effort = self.PositionPID.update(vecIn[0], feedback[0])
        PWM1 = self.EffortPID.update(vecIn[2] + effort, feedback[2])
        PWM2 = self.VelocityPID.update(vecIn[2], feedback[2])
        return self.PWMFilter.update(PWM1 + PWM2)

def PWM2Torque(PWM, maxMotorTorque=7.5):
    """ Converts PWM output signals of the strategy 4 controller to direct torque outputs (Nm) for Pybullet """
    # PWM range -> [-1, 1], Since we have X5-4 motors, the max torque is 7.5 Nm
    # We just assume the conversion to be linear, might be fancier if I have time to measure a more exact conversion
    # Source: https://docs.hebi.us/core_concepts.html#control-strategies
    return PWM*maxMotorTorque

def deserialiseJointstate(js):
    """ Converts Jointstate message into the format used for the strategy 4 controller """
    return [ [pos, vel, eff] for pos, vel, eff in zip(js.position, js.velocity, js.effort) ]

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
