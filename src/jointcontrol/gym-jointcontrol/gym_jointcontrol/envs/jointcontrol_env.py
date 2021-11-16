import math
import sys

import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

import rospy


def getJointDict(name, numParams, defaultParams, ):
    """ Dict that tracks joint configuration. Used as parameter in ROS parameter server """
    return {
        "name": name,
        "numParams": numParams,
        "defaultParams": defaultParams,
        "optimisedParams": defaultParams,
        "performance": None
    }



class metricsTracker():
    """ Tracks performance metrics of a joint for use in the environment """
    def __init__(self):
        self.latestFeedback = None
        self.runningFeedback = None
        # set up joint feedback subscriber
        # register callback
        pass

    def feedbackCallback(self, data):
        self.latestFeedback = data
        # calculate running average
        pass

    def resetRunningFeedback(self):
        self.runningFeedback = self.latestFeedback

    def getRunningFeedback(self):
        return self.runningFeedback

    def getLatestFeedback(self):
        return self.latestFeedback



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














-> black box identification of controller

set bullet timestep
p.setTimeStep(DT)
-> default = 1/60


debug
import gym
import gym_jointcontrol

env = gym.make('jointcontrol-v0')

"""





#class PID():
#    """ https://www.scilab.org/discrete-time-pid-controller-implementation """
#    def __init__(self):
#        pass







class jointcontrol_env(gym.Env):
    """
    ...
    """
    def __init__(self, **kwargs):
        print("Hello World")

        # Get current joint index from keyword arguments
        jointidx = kwargs.get('jointidx', 0)

        # Init ros node
        try:
            rospy.init_node("j{}control_gym".format(jointidx))
        except rospy.exceptions.ROSException:
            pass

        # Create callback methods
        def rosShutdownHandler():
            sys.exit("External Shutdown")

        # Load joint controller configuration from ros parameter server
        assert rospy.has_param("/jointcontrol")
        params = rospy.get_param("/jointcontrol")
        assert params["NumJoints"]>=jointidx
        self.controllerParams = params["Joints"][jointidx]

        # Get number of params of used joint in order to create state and action representations
        numParams = self.controllerParams["NumParams"]

        # Define shape of state, action and reward
        # state = Box
        # reward = float
        # action = Box