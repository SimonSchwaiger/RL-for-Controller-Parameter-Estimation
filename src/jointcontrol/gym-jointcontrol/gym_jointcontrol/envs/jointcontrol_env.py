import gym
from gym import spaces, logger
from gym.utils import seeding

import time

import numpy as np

import rospy

from jointcontrol.msg import jointMetric

"""
Joint Environment is set up on a per joint basis

Observation:
- features of current control metrics

Reward:
- negative average deviation between command signal and resulting jointposition

Actions:
- vector of control parameters
-> shape = box of len(num(parameters))

ROS Msgs:
- /jointcontrol/feedback -> vector of jointfeedback data

ROS Parameters
- /jointcontrol/parameters -> num of joints, num control parameters for each joint, initial values for each parameter
- /jointcontrol/optimisedParameters -> vector of new parameters

DEBUG:

import gym
import gym_jointcontrol

env = gym.make('jointcontrol-v0', jointidx=1)

TODO:
- PID Discretisation:
  https://www.scilab.org/discrete-time-pid-controller-implementation

- set bullet timestep
  p.setTimeStep(DT)
  -> default = 1/60

"""

class metricsTracker():
    """
    Subscribes to the jointcontroller/jointMetric/J.. topic in order to retrieve control signal and 
    controller feedback for a certain point in time. Messages are sent in form of jointcontrol/jointMetric.msg.
    """
    """ Tracks performance metrics of a joint for use in the environment """
    def __init__(self, jointID):
        """ Class constructor """
        self.jointID = jointID      # Index to identify the controlled joint
        self.updateCount = 0        # Amount of updates since metrics were last checked
        self.runningError = 0       # Difference between control signal and jointfeedback
        
        # Set up and register joint callback
        self.metricSub = rospy.Subscriber(
            "jointcontroller/jointMetric/J{}".format(jointID),
            jointMetric, 
            self.jointMetricCallback
        )

    def jointMetricCallback(self, data):
        """ Receives jointmetric message, calculates squared error and keeps track of the amount of messages received """
        # Check if data is meant for this channel (for Debugging)
        if data.name != self.jointID: 
            rospy.logerr("J{} received a message not meant for this channel.")
            return
        # Add squared difference between control signal and joint feedback
        self.runningError += (data.controlSignal-data.feedbackSignal)**2
        # Increment update count
        self.updateCount += 1

    def getAverageError(self):
        """ Getter for the average error since last update """
        # Make sure update count is nonzero
        if self.updateCount == 0: return None
        # Calculate squared error
        avg =  round(self.runningError/self.updateCount, 8)
        # Reset running error and updatecount
        self.runningError = 0
        self.updateCount = 0
        return avg

class jointcontrol_env(gym.Env):
    """
    Observation Space (Shape Tuple):
        - Current Controller Params
          Box of length controllerParams["NumParams"]
          (Must be within Minimum and Maximum allowed Parameters)

    Action Space (Shape Box):
        - Updates for the controller params
          Box of length controllerParams["NumParams"]
          (Must be within 0 and controllerParams["MaxChange"])

    Gym Core Env Implementation:
        https://github.com/openai/gym/blob/master/gym/core.py
        
    """
    def __init__(self, **kwargs):
        """ Class constructor """
        # Get current joint index from keyword arguments
        self.jointidx = kwargs.get('jointidx', 0)

        # Init ros node
        try:
            rospy.init_node("j{}control_gym".format(self.jointidx))
        except rospy.exceptions.ROSException:
            pass

        # Load joint controller configuration from ros parameter server
        assert rospy.has_param("/jointcontrol")
        params = rospy.get_param("/jointcontrol")
        assert params["NumJoints"]>=self.jointidx
        self.controllerParams = params["J{}".format(self.jointidx)]

        # Define action and observation spaces as boxes
        self.observation_space = spaces.Box(
            np.array(self.controllerParams["Minimums"]),
            np.array(self.controllerParams["Maximums"])
        )

        self.action_space = spaces.Box(
            np.array([ 0 for i in self.controllerParams["MaxChange"] ]),
            np.array(self.controllerParams["MaxChange"])
        )

        # Store params as class member, these will be modified by self.step() and self.reset()
        # The params start out as the set defaults from ROS
        self.currentParams = self.controllerParams["Defaults"]

        # Instantiate metrics tracker
        self.tracker = metricsTracker(self.jointidx)

        # Wait for messages to come in and clear initial output
        time.sleep(5)
        if self.tracker.getAverageError() == None:
            rospy.logwarn("J{} has not received jointmetric data on startup".format(self.jointidx))

        #TODO: wait for metrics to come in before proceeding

    def reset(self):
        pass
    
    def step(self, action):
        pass

    def render(self, mode="human"):
        # Does nothing here, since ROS renders anyways
        pass

    def close(self):
        pass