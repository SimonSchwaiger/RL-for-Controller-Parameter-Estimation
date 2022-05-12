
docker exec -it ros_ml_container1 bash

source ~/myenv/bin/activate
source /opt/ros/noetic/setup.bash
source /catkin_ws/devel/setup.bash

python

import subprocess
import os
import sys
import time
sys.path.append("/catkin_ws/src/jointcontrol/scripts")

import numpy as np

""" 
Control script for experiment 4 

DQN, DDPG, PPO and Random Agents trained to perform step responses from 0 to -1.57 rad

Plots include average return of training and separate testepisodes

Logs are stored in /app/4logs


CONFIG BLUEPRINT

config = {
    "jointID": 0,
    "logdir": "/app/logExperiment4",
    "modelclass": "DQN",
    "modelrun": 0,
    "modelname": "DQN",
    "learningRate": 0.001,
    "trainingTimesteps": 10000,
    "policyNetwork": [{'pi': [32, 32], 'vf': [32, 32]}],
    "optimizer": "th.nn.tanh",
    "discretisation": 0.5,
    "resetConfig": str({ "initialPos":0, "stepPos":-1.57, "samplesPerStep":150, "maxSteps":40 }),
    "trajectoryType": "step"
}
"""

runTraining = False      # Determines, whether or not training is conducted or only visualisation is performed
trainingSteps = 15000   # Determines performed training steps

# Start tensorboard to allow for visualisation of training process
logdir = "/app/MTExperiments/Data/4StepResponseTrainingDefaultParams"
os.system("tensorboard --logdir {}/tensorboard --host 0.0.0.0 --port 6006 &".format(logdir))

# Point to RL agent deployment script
agentPath = "/app/MTExperiments/TrainingScripts/RLAgent.py"
randomAgentPath = "/app/MTExperiments/TrainingScripts/RandomAgent.py"


#############################################################################
## DQN Discrete

DQNConfig = {
    "jointID": 0,
    "logdir": logdir,
    "modelclass": "DQN",
    "modelrun": 0,
    "modelname": "DQN",
    "learningRate": None, 
    "trainingTimesteps": trainingSteps,
    "policyNetwork": None,
    "optimizer": None,
    "discretisation": 0.1,
    "resetConfig": { "initialPos":-1.57, "stepPos":0, "samplesPerStep":150, "maxSteps":40 },
    "trajectoryType": "step"
}


#############################################################################
## DDPG Continuous

DDPGConfig = {
    "jointID": 0,
    "logdir": logdir,
    "modelclass": "DDPG",
    "modelrun": 0,
    "modelname": "DDPG",
    "learningRate": None, 
    "trainingTimesteps": trainingSteps,
    "policyNetwork": None,
    "optimizer": None,
    "discretisation": None,
    "resetConfig": { "initialPos":-1.57, "stepPos":0, "samplesPerStep":150, "maxSteps":40 },
    "trajectoryType": "step"
}

#############################################################################
## PPO Discrete

PPODiscreteConfig = {
    "jointID": 0,
    "logdir": logdir,
    "modelclass": "PPO",
    "modelrun": 0,
    "modelname": "PPO_Discrete",
    "learningRate": None, 
    "trainingTimesteps": trainingSteps,
    "policyNetwork": None,
    "optimizer": None,
    "discretisation": 0.1,
    "resetConfig": { "initialPos":-1.57, "stepPos":0, "samplesPerStep":150, "maxSteps":40 },
    "trajectoryType": "step"
}

#############################################################################
## PPO Continuous

PPOContinuousConfig = {
    "jointID": 0,
    "logdir": logdir,
    "modelclass": "PPO",
    "modelrun": 0,
    "modelname": "PPOContinuous",
    "learningRate": None, 
    "trainingTimesteps": trainingSteps,
    "policyNetwork": None,
    "optimizer": None,
    "discretisation": None,
    "resetConfig": { "initialPos":-1.57, "stepPos":0, "samplesPerStep":150, "maxSteps":40 },
    "trajectoryType": "step"
}

#############################################################################
## RANDOM

RandomConfig = {
    "jointID": 0,
    "logdir": logdir,
    "modelclass": "Random",
    "modelrun": 0,
    "modelname": "RandomContinuous",
    "learningRate": None, 
    "trainingTimesteps": trainingSteps,
    "policyNetwork": None,
    "optimizer": None,
    "discretisation": None,
    "resetConfig": { "initialPos":-1.57, "stepPos":0, "samplesPerStep":150, "maxSteps":40 },
    "trajectoryType": "step"
}

# Debug: for one environment
#p = subprocess.Popen([sys.executable, agentPath, str(PPOContinuousConfig)])
#streamdata = p.communicate()[0]
#print(p.returncode)

#############################################################################
## RUN Training Episodes for each Algorithm

def runAgentTests(config, testRuns=5):
    """ Performs testRuns """
    print("Running {} training sequences".format(config["modelname"]))
    processes = [None for _ in range(testRuns)]
    #
    for i in range(testRuns):
        # Start process in the background
        config["jointID"] = i
        config["modelrun"] = i
        processes[i] = subprocess.Popen([sys.executable, agentPath, str(config)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        # Wait for env to register
        time.sleep(1)
    #
    # Wait for processes to finish and record return codes
    rc = []
    for i in range(testRuns):
        streamdata = processes[i].communicate()[0]
        rc.append(processes[i].returncode)
    #
    print("{} agent tests successfully exited with return codes {}".format(config["modelname"], rc))

# Perform testruns of all configurations
if runTraining:
    runAgentTests(DQNConfig)
    runAgentTests(DDPGConfig)
    runAgentTests(PPODiscreteConfig)
    runAgentTests(PPOContinuousConfig)


#############################################################################
## Sample random agent performance

def randomAgentTests(config, testRuns = 5):
    """ Performs testruns of a random agent """
    processes = [None for _ in range(testRuns)]
    #
    for i in range(testRuns):
        # Start process in the background
        config["jointID"] = i
        config["modelrun"] = i
        processes[i] = subprocess.Popen([sys.executable, randomAgentPath, str(config)])#, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        # Wait for env to register
        time.sleep(1)
    #
    # Wait for processes to finish and record return codes
    rc = []
    for i in range(testRuns):
        streamdata = processes[i].communicate()[0]
        rc.append(processes[i].returncode)
    #
    print("{} agent tests successfully exited with return codes {}".format(config["modelname"], rc))

if runTraining:
    randomAgentTests(RandomConfig)


#############################################################################
## VISUALISE AVERAGE RETURN IN TRAINING

import scipy.stats as st
import math

# Load Tensorboard Logs
#https://gist.github.com/willwhitney/9cecd56324183ef93c2424c9aa7a31b4
from tensorboard.backend.event_processing import event_accumulator

## Matplotlib imports
import matplotlib.pyplot as plt
import matplotlib

matplotlib.style.use("default")

## Import thesis colours
sys.path.append("/app/MTExperiments/Plots")
from colourmaps import *


def loadTensorboardLog(config, testRuns=5):
    """ Loads tensorboard logs from runs and returns their entries as one big list """
    rewards = []
    #
    for run in range(testRuns):
        logPath = "{}/tensorboard/{}_{}_1".format(config["logdir"], config["modelname"], run)
        ea = event_accumulator.EventAccumulator(logPath, size_guidance={event_accumulator.SCALARS: 0})
        ea.Reload()
        rewards.append([ entry.value for entry in ea.Scalars("rollout/ep_rew_mean") ])
    #
    return rewards


def getConfidenceInterval(a):
    """ Returns mean, upper and lower bounds of a 95% confidence interval of given values """
    tmp = st.norm.interval(0.95, loc=np.mean(a), scale=np.std(a)/math.sqrt(len(a)))
    return np.mean(a), tmp[0], tmp[1]


def computeGraphArrays(meanEpRewards, runs=5, lenPlot=15000):
    """ Computes the graph lists from mean episode reward of multiple agent runs """
    meanList = []
    lowerList = []
    upperList = []
    # Get length of shortest episode
    episodes = 0
    for run in range(runs):
        if episodes == 0: episodes = len(meanEpRewards[run])
        else: episodes = min(episodes, len(meanEpRewards[run]) )
    #
    # Iterate over episodes
    for episode in range(episodes):
        mean, lower, upper = getConfidenceInterval(
            [ meanEpRewards[run][episode] for run in range(runs) ]
        )
        meanList.append(mean)
        lowerList.append(lower)
        upperList.append(upper)
    #
    # Return lists
    interval = lenPlot/episodes
    timesteps = np.arange(episodes)*(lenPlot/(episodes-1))
    timesteps = np.around(timesteps, decimals=0, out=None)
    return timesteps, meanList, lowerList, upperList


def loadRandomResults(config, testRuns=5, episode_len=40, smoothingFactor=20):
    """ Loads random agent results based on configuration """
    ret = []
    for run in range(testRuns):
        filename = "{}/tensorboard/{}_{}_1.npy".format(config["logdir"], config["modelname"], run)
        # Load reward list
        rewardlist = np.load(filename, allow_pickle=True).tolist()
        # Helper funciton for iterating over list in chunks
        # https://stackoverflow.com/questions/434287/what-is-the-most-pythonic-way-to-iterate-over-a-list-in-chunks
        def chunker(seq, size):
            return (seq[pos:pos + size] for pos in range(0, len(seq), size))
        # Scale list to total episode reward for an episode of length 40
        episodicReward = []
        for segment in chunker(rewardlist, episode_len):
            episodicReward.append(np.sum(segment))
        # Sample mean return over an interval of episodes to smooth out curve
        rewardList = []
        for segment in chunker(episodicReward, smoothingFactor):
            rewardList.append(np.mean(segment))
        # Add smoothed rewards to return list
        ret.append(rewardList)
    #
    return ret


## Set matplotlib figure size as seen here
# https://stackoverflow.com/questions/44970010/axes-class-set-explicitly-size-width-height-of-axes-in-given-units
def set_size(w,h, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)


## Iterate over config files and load tensorboard results
tensorbaordResults = []
for config in [DQNConfig, DDPGConfig, PPODiscreteConfig, PPOContinuousConfig]:
    tensorbaordResults.append(loadTensorboardLog(config))

# Add random agent to results
tensorbaordResults.append(loadRandomResults(RandomConfig))

## Configure plot
plt.rcParams.update({'font.size': 12})

data = tensorbaordResults
labels = ["DQN", "DDPG", "PPO Discrete", "PPO Continuous", "Random Agent"]
lines = ['-', '--', '-', '-.', ':']

colours = [
    colourScheme1["darkblue"],
    colourScheme2["twblue"],
    colourScheme1["lightblue"],
    colourScheme2["yellow"],
    colourScheme1["twgrey"]
]

## Plot everything
fig, ax = plt.subplots()
for i, d in enumerate(data):
    x, y, cl, cu = computeGraphArrays(d)
    ax.plot(x,y,color=colours[i], label=labels[i], linestyle=lines[i])
    ax.fill_between(x, cl, cu,color=colours[i], alpha=.1)


plt.suptitle("Mean Return during Agent Training", fontsize=16)
plt.xlabel('Training Steps')
plt.ylabel('Mean Return')
#plt.xlim([-12500, 1000])
plt.ylim([-80, -10])
plt.grid()
plt.legend()

set_size(7,4)

plt.tight_layout()
plt.subplots_adjust(top = 0.93)

plt.savefig("/app/resultsStepResponseTrainingRewardDefaultParams.pdf", bbox_inches='tight')
plt.show()








#############################################################################
## VISUALISE EVALUATION EPISODE REWARD AS BOXPLOT

## Matplotlib imports
import matplotlib.pyplot as plt
import matplotlib

matplotlib.style.use("default")

## Set matplotlib figure size as seen here
# https://stackoverflow.com/questions/44970010/axes-class-set-explicitly-size-width-height-of-axes-in-given-units
def set_size(w,h, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)

## Import thesis colours
sys.path.append("/app/MTExperiments/Plots")
from colourmaps import *

def loadTestepisodeResults(config, testRuns=5):
    """ Loads evaluation episode results based on configuration """
    ret = []
    for run in range(testRuns):
        filename = "{}/testepisodes_{}_{}.npy".format(config["logdir"], config["modelname"], run)
        ret.append(np.load(filename, allow_pickle=True).item())
    #
    return ret


## Iterate over config files and load eval episode and tensorboard results
episodeLength = 40
evalEpisodeResults = []

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

for config in [DQNConfig, DDPGConfig, PPODiscreteConfig, PPOContinuousConfig, RandomConfig]:
    # Load eval episode result dict
    rawEvalRewards = loadTestepisodeResults(config)
    # Sum episodes up in order to match graph scale to the average return
    #episodeRewards = [ np.sum(segment) for segment in chunker(rawEvalRewards[0]["evalEpisodeRewards"], episodeLength) ]
    #evalEpisodeResults.append(episodeRewards)
    evalEpisodeResults.append(rawEvalRewards[0]["evalEpisodeRewards"])


## Determine default controller parameter performance for the test
sys.path.append("/catkin_ws/src/jointcontrol/scripts")
import gym
from discreteActionWrapper import *
from multiprocessing import resource_tracker

# Make environment
env = gym.make('jointcontrol-v0', jointidx=6)
env.reset(episodeType="step", config={ "initialPos":0, "stepPos":-1.57, "samplesPerStep":150, "maxSteps":40 })
_, referenceReward, _, _ = env.step(None)
resource_tracker.unregister(env.env.physicsCommand.shm._name, 'shared_memory')
env.env.closeSharedMem()

## Set up plot (2 Plots side-by-side) both plotting per step reward with default parameter performance as a reference
# One plot also includes the random agent

labels = ["DQN", "DDPG", "PPO(D)", "PPO(C)", "Random"]

linecolour = colourScheme1["darkblue"]
meadianlinecolour = colourScheme1["twblue"]
fillcolour = colourScheme1["lightblue"]
referenceColour = colourScheme2["yellow"]

boxwidth = 0.3

plt.rcParams.update({'font.size': 12})
fig, (ax0, ax1) = plt.subplots(1,2, sharey=False)

box0 = ax0.boxplot(
    evalEpisodeResults, labels=labels, notch=False, showfliers=True, patch_artist=True, widths=boxwidth,
    boxprops=dict(facecolor=fillcolour, color=linecolour),
    capprops=dict(color=linecolour),
    whiskerprops=dict(color=linecolour),
    flierprops=dict(color=fillcolour, markeredgecolor=linecolour),
    medianprops=dict(color=meadianlinecolour),
)

box1 = ax1.boxplot(
    evalEpisodeResults[:4], labels=labels[:4], notch=False, showfliers=False, patch_artist=True, widths=boxwidth*0.8,
    boxprops=dict(facecolor=fillcolour, color=linecolour),
    capprops=dict(color=linecolour),
    whiskerprops=dict(color=linecolour),
    flierprops=dict(color=fillcolour, markeredgecolor=linecolour),
    medianprops=dict(color=meadianlinecolour),
)

# Add reference control error
ax0.axhline(y=referenceReward, xmin=0.05, xmax=0.95, linestyle="--", color=referenceColour, label="Default Controller Parameters")
ax1.axhline(y=referenceReward, xmin=0.05, xmax=0.95, linestyle="--", color=referenceColour, label="Default Controller Parameters")

# Format axes, set size and plot
ax0.grid()
ax1.grid()

ax0.legend()

ax0.set_ylabel("Negative Mean Squared Control Error")

ax0.set_title("With Random Agent") 
ax1.set_title("Without Random Agent")

fig.suptitle("Controller Evaluation Episode Results", fontsize=16)

set_size(7,3.6)
plt.tight_layout()

plt.subplots_adjust(top = 0.862)

plt.savefig("/app/resultsStepResponseDefaultParamsControlErrorBox.pdf", bbox_inches='tight')
plt.show()


