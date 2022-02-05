#!/usr/bin/env python3

import copy
import os
import json

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
    def __init__(self, env, model, modelname, freqs=[1, 5, 10, 20, 50]) -> None:
        """
        Class constructor

        Params:
            - env:          gym_jointcontrol environment instance used for testing
            - model:        RL model, that uses model.predict(obs) to determine agent actions
            - modelname:    String describing the model, name will be used as the name of the generated results folder
        """
        ## Perform one RL episode and track reward
        # Set initial params of MDP
        episodeRewards = []
        obs = env.reset()
        done = False
        # Loop until episode is done
        while not done:
            # While we are not done, we store final observation, since presumably it represents optimal controller params
            finalObs = obs
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            episodeRewards.append(reward)
        #
        print("testepisode done")
        ## Test normal step response
        # Set up env for step response, set optimal params and run episode
        obs = env.reset() 
        env.env.currentParams = finalObs
        env.step([ 0 for _ in action ])
        stepTrajectory = copy.deepcopy(env.env.latestTrajectory)
        stepControlSignal = copy.deepcopy(env.env.controlSignal)
        #
        ## Test square signal from 1 to 10 Hz
        # Define config for wave generation
        config = {
            "lowerSignal": 0,
            "higherSignal": -1.57,
            "pulseLength": 0,
            "numPulses": 2,
            "maxSteps": 40
        }
        # Define tested frequencies and iterate over them
        #freqs = [ i+1 for i in range(minFreq-1, maxFreq) ]
        squareTrajectories = []
        # The lowest frequency (and therefore longest) wave is used as a reference for signal length
        refLen =  int(round((1/freqs[0])/env.env.ts))
        numPulses = config["numPulses"]
        for freq in freqs:
            # Set pulse length based on frequency
            config["pulseLength"] = int(round((1/freq)/env.env.ts))
            # Set number of pulses, in order for responses to be equally long
            config["numPulses"] = int(round(refLen/config["pulseLength"]))*numPulses
            #config["numPulses"] =  TODO
            # Reset env in square mode
            env.env.reset(episodeType='square', config=config)
            # Set testing params and perform step
            env.env.currentParams = finalObs
            print(config)
            env.step([ 0 for _ in action ])
            # Store resulting trajectories
            squareTrajectories.append(env.env.latestTrajectory)
        #
        # Generate dict including all test results and store as class member
        # modelname -> name of model
        # ts -> time discretisation
        # finalObs -> assumed ideal params after final step in episode
        # episodeRewards -> reward of each step in performed episode
        # stepTrajectory -> resulting trajectory from step response
        # freqResponseFrequencies -> frequencies of input square waves for frequency response
        # freqResponseTrajectories -> resulting trajectories from square input wave tests
        self.testResults = {
            "modelname": modelname,
            "ts": env.env.ts,
            "finalObs": finalObs,
            "episodeRewards": episodeRewards,
            "stepControlSignal": stepControlSignal,
            "stepTrajectory": stepTrajectory,
            "freqResponseFrequencies": freqs,
            "freqResponseTrajectories": squareTrajectories
        }
    #
    def plotResults(self, gui=False):
        """ Plots test results stored in self.testResults """
        # Create Matplotlib figure and set title
        fig, axs = plt.subplots(3, 1, constrained_layout=True, gridspec_kw={'height_ratios': [1, 1, 2]})
        fig.suptitle("Testreport for {}".format(self.testResults["modelname"]))
        #
        # Cumulative reward over episode
        axs[0].plot(
            self.testResults["episodeRewards"]
        )
        axs[0].set_xlabel("Episode")
        axs[0].set_ylabel("Return per Episode")
        axs[0].set_title("Step Response Episode Reward")
        #
        # Resulting trajectory and control signal of step response
        axs[1].plot (
            np.arange(len(self.testResults["stepTrajectory"]))*self.testResults["ts"],
            self.testResults["stepTrajectory"],
            label = "Resulting Position"
        )
        axs[1].plot (
            np.arange(len(self.testResults["stepControlSignal"]))*self.testResults["ts"],
            self.testResults["stepControlSignal"],
            label = "Control Signal"
        )
        axs[1].set_xlabel("Time [s]")
        axs[1].set_ylabel("Joint Position [rad]")
        axs[1].set_title("Resulting Step Trajectory")      
        #
        # Plot frequency response curves
        # Each fraquency is coloured and labelled in order to plot them all in the same plot
        # One more colour than required is generated and dropped due to the first entry being barely visible on the sequential colour maps
        colours = plt.cm.Blues(np.linspace( 0, 1, len(self.testResults["freqResponseFrequencies"])+1 ))
        colours = colours[1:]
        labels = [ "{} Hz".format(freq) for freq in self.testResults["freqResponseFrequencies"] ]
        for trajectory, colour, label in zip(
            self.testResults["freqResponseTrajectories"],
            colours,
            labels
        ):
            axs[2].plot(
                self.testResults["ts"]*np.arange(len(trajectory)),
                trajectory,
                label = label,
                color = colour
            )
        axs[2].legend()
        axs[2].set_xlabel("Time [s]")
        axs[2].set_ylabel("Average Joint Position [rad]")
        axs[2].set_title("Square Control Signal Frequency Response")          
        # If GUI var is set, visualise the report, otherwise save it to a file
        plt.show()
