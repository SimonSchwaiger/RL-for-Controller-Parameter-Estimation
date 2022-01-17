import gym
from gym import spaces, logger
from gym.utils import seeding

import time
import numpy as np

import rospy
from jointcontrol.msg import jointMetric

from controllers import *

class jointcontrol_env(gym.Env):
    """
    ...
    """
    def __init__(self, **kwargs):
        """ Class constructor """
        # Get current joint index from keyword arguments
        self.jointidx = kwargs.get('jointidx', 0)
        
        # Init ROS node
        try:
            rospy.init_node("j{}GymEnv".format(self.jointidx))
        except rospy.exceptions.ROSException:
            pass

        # Load joint controller configuration from ros parameter server
        assert rospy.has_param("/jointcontrol")
        params = rospy.get_param("/jointcontrol")
        assert params["NumJoints"]>=self.jointidx
        self.jointParams = params["J{}".format(self.jointidx)]

        # Define action space as box with minimum and maximum controller params
        self.action_space = spaces.Box(
            np.array([ -i for i in self.jointParams["MaxChange"] ]),
            np.array(self.jointParams["MaxChange"])
        )

        # Instantiate controller and set ts
        self.controller = strategy4Controller(ts=params["SimParams"]["Ts"])
