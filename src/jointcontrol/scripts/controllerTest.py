#!/usr/bin/env python3

import copy

import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class controllerTest:
    """!@brief Class implementing automated performance testing of a trained controller model

    This class implements automated testing of model performance using a provided model and environment.
    The tests are performed directly in the existing Bullet instance through the gym_jointcontrol environment.
    Member functions of the jointcontrol environment are accessed as part of the GymEnv instance (using env.env.* instead of env.*).
    """
    def __init__(self, episodes = 1) -> None:
        """
        Class constructor

        Params:
            - env:          gym_jointcontrol environment instance used for testing
            - model:        RL model, that uses model.predict(obs) to determine agent actions
            - modelname:    String describing the model, name will be used as the name of the generated results folder
        """
        

    def setConfiguration(self):
        """ Stores configuration params for performed tests """
        pass
        #

    def performTests(self, env, model, modelname):
        ## Perform one RL episode and track reward
        # Set initial params of MDP
        episodeRewards = []
        obs = env.reset()
        done = False
        # Loop until episode is done
        while not done:
            # While we are not done, we store final observation, since presumably it represents optimal controller params
            finalObs = obs
            action, _states = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            episodeRewards.append(reward)
        #
        ## Test normal step response
        # Set up env for step response, set optimal params and run episode
        obs = env.reset() 
        env.env.currentParams = finalObs
        env.step([ 0 for _ in action ])
        stepTrajectory = copy.deepcopy(env.env.latestTrajectory)
        #
        ## Test square signal from 1 to 10 Hz
        # Define config for wave generation
        config = {
            "lowerSignal": 0,
            "higherSignal": -1.57,
            "pulseLength": 0,
            "numPulses": 10,
            "maxSteps": 40
        }
        # Define tested frequencies and iterate over them
        freqs = [ i+1 for i in range(10) ]
        squareTrajectories = []
        for freq in freqs:
            # Set pulse length based on frequency
            config["pulseLength"] = int((1/freq)/env.env.ts)
            # Reset env in square mode
            env.env.reset(episodeType='square', config=config)
            # Set testing params and perform step
            env.env.currentParams = finalObs
            env.step([ 0 for _ in action ])
            # Store resulting trajectories
            squareTrajectories.append(env.env.latestTrajectory)
        #
        ## Gernerate testreport based on acquired data
        # Cumulative reward over episode
        

        # Plot stepTrajectory

        # Plot trajectories resulting from square signal at various frequencies


    #
    def generateReport(self, outPath):
        """ Generates a testreport containing results from the controller test """
        # Generate directory for results
        os.system("mkdir {}/{}".format(outPath, self.modelname))
        # TODO
