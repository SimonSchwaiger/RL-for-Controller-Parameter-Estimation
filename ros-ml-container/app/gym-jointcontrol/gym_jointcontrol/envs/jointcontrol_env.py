import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

"""
Defines the Joint Environment


Observation (shape = discrete):


Actions (shape = discrete):



debug
import gym
import gym_jointcontrol

env = gym.make('jointcontrol-v0')

"""

class jointcontrol_env(gym.Env):
    """
    ...
    """
    def __init__(self):
        print("Hello World")
