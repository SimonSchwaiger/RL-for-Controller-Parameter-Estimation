import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

"""
Joint Environment is set up on a per joint basis



Observation:
- control offset

Actions:
- vector of control parameters


-> black box identification of controller

set bullet timestep
p.setTimeStep(DT)
-> default = 1/60


debug
import gym
import gym_jointcontrol

env = gym.make('jointcontrol-v0')

"""





class PID():
    """ https://www.scilab.org/discrete-time-pid-controller-implementation """
    def __init__(self):
        pass




class jointcontrol_env(gym.Env):
    """
    ...
    """
    def __init__(self):
        print("Hello World")
