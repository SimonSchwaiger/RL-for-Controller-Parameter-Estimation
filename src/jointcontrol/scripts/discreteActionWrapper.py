#!/usr/bin/env python3

""" This class allows discretisation of the jointcontrol environment's action space. """

import gym
from gym import spaces
import gym_jointcontrol

class jointcontrolDiscrete(gym.Wrapper):
    """!@brief Class wrapping the jointcontrol gym environment to have a discrete action space, enabling usage of more RL algorithms """
    def __init__(self, env, discretisation=0.5) -> None:
        """ 
        Class constructor
        
        Action space discretisation is set using the discretisation parameter. 
        Since discretisation is applied to the scaled action-input, 
        the real level of discretisation will be different for each param, depending on the specified maximum change.
        The actual discretisation for each controller parameter can be calculated by:

        realDiscretisation = discretisation*MaxChange

        Only the step() method is implemented, because other methods of the jointcontrol env are unaffected by a changed action space.
        """
        # Inherit from original env and store discretisation level
        super().__init__(env)
        self.env = env
        self.discretisation = discretisation
        # Set up action space as discrete
        # Possible actions include incrementing and decrementing each param
        self.numParams = len(self.env.jointParams["Defaults"])
        self.action_space = spaces.Discrete(self.numParams*2)

    def step(self, action):
        """
        Performs env.step(), but with discrete actions.
        
        Actions are decoded like this:
            1 -> increment param 1
            2 -> decrement param 1
            3 -> increment param 2
            4 -> decrement param 2
            ...

        If action is set to None, a step is performed without changing internal params of the controller.
        """
        if action==None :
            # If action is none, perform step with action as None to not change params
            actionList = None
        else:
            # Make sure inputted action is valid and can be decoded
            assert action < self.numParams*2
            # Create list of param changes based on discretisation and param size
            if action%2 == 0: tmp = self.discretisation
            else: tmp = -self.discretisation
            actionList = [ 0 for _ in range(self.numParams) ]
            actionList[int(action/2)] = tmp
        # Perform step in non-discrete env and return observation
        return self.env.step(actionList)
