import math
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
        "optimisedParams": defaultParams
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
