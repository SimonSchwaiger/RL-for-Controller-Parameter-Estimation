
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

import scipy.stats as st
import math

# Load Tensorboard Logs
#https://gist.github.com/willwhitney/9cecd56324183ef93c2424c9aa7a31b4
from tensorboard.backend.event_processing import event_accumulator

## Matplotlib imports
import matplotlib.pyplot as plt
import matplotlib

matplotlib.style.use("default")
plt.rcParams.update({'font.size': 12})

## Import thesis colours
sys.path.append("/app/MTExperiments/Plots")
from colourmaps import *

plotWidth = 7.2

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
    tmp = st.norm.interval(0.75, loc=np.mean(a), scale=np.std(a)/math.sqrt(len(a)))
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


# Training curves for:
# - DDPG Hyperparameter optimisation
# - PPO Hyperparameter optimisation
# - PPO Neural network Width and Depth
# - Trajectory control


colours = [
    colourScheme1["darkblue"],
    colourScheme2["twblue"],
    colourScheme1["lightblue"],
    colourScheme2["yellow"],
    colourScheme1["twgrey"],
    colourScheme1["darkblue"]
]

lines = ['-', '--', '-', '-.', ':', '-.']

def plotTrainingCurve(ax, configs, title, labels, colours, lines, testRuns=3, trainingSteps=15000):
    """  """
    # Load tensorboard log
    data = []
    for config in configs:
        data.append(loadTensorboardLog(config, testRuns=testRuns))
    # Plot mean reward and confidence interval
    for i, d in enumerate(data):
        x, y, cl, cu = computeGraphArrays(d, runs=testRuns, lenPlot=trainingSteps)
        ax.plot(x,y,color=colours[i], label=labels[i], linestyle=lines[i])
        ax.fill_between(x, cl, cu,color=colours[i], alpha=.1)
    # Set labels and title
    #ax.set_xlabel("Training Steps")
    #ax.set_ylabel("Mean Return")
    if title != None: ax.set_title(title)
    ax.grid()
    #ax.legend()

#############################################################################
## DDPG Hyperparameter optimisation

trainingSteps = 10000
logdir = "/app/MTExperiments/Data/5DDPGStepResponseTrainingOptimisation"

def getConfig(tau, lr, run):
    return {
        "jointID": 0,
        "logdir": logdir,
        "modelclass": "DDPG",
        "modelrun": run,
        "modelname": "DDPG_lr{}_tau{}".format(lr, tau),
        "learningRate": lr,
        "tau": tau, 
        "trainingTimesteps": trainingSteps,
        "policyNetwork": None,
        "optimizer": None,
        "discretisation": None,
        "resetConfig": { "initialPos":-1.57, "stepPos":0, "samplesPerStep":150, "maxSteps":40 },
        "trajectoryType": "step"
    }

# Create configurations for each lr, tau combination
learningRates = [0.0001, 0.0005, 0.001, 0.002]         # default = 0.001
taus = [0.001, 0.002, 0.005, 0.01]                     # default = 0.005
trainingRuns = 3


fig, axs = plt.subplots(2, 2, sharey=True, sharex=True)
plotindices = [[0,0], [0,1], [1,0], [1,1]]

for lr, idx in zip(learningRates, plotindices):
    # Create configs and labels
    configs = []
    labels = []
    for tau in taus:
        labels.append("tau={}".format(tau))
        configs.append(getConfig(tau, lr, 0))
    # Plot everything at index
    title = "lr = {}".format(lr)
    plotTrainingCurve(axs[idx[0]][idx[1]], configs, title, labels, colours, lines, testRuns=trainingRuns, trainingSteps=trainingSteps)


fig.suptitle("DDPG Hyperparameter Optimisation", fontsize=16)

axs[0][0].set_ylabel("Mean Return")
axs[1][0].set_ylabel("Mean Return")

axs[1][0].set_xlabel("Training Steps")
axs[1][1].set_xlabel("Training Steps")

axs[0][1].legend()

set_size(plotWidth,6)
plt.tight_layout()
plt.subplots_adjust(wspace = 0.071, hspace=0.13, top = 0.91)

plt.savefig("/app/appendixPlotsDDPGHyperparameterOptimisation.pdf", bbox_inches='tight')
plt.show()


#############################################################################
## PPO Discrete Hyperparameter optimisation

trainingSteps = 15000
logdir = "/app/MTExperiments/Data/7PPOStepResponseTrainingOptimisation"

def getConfigDisc(run, epsilon=None, horizon=None, gae_lambda=None):
    return {
        "jointID": 0,
        "logdir": logdir,
        "modelclass": "PPO",
        "modelrun": run,
        "modelname": "PPO_Discrete_eps{}_lam{}".format(epsilon, gae_lambda),
        "learningRate": 0.0003,
        "epsilon": epsilon,
        "n_steps": horizon,
        "gae_lambda": gae_lambda,
        "trainingTimesteps": trainingSteps,
        "policyNetwork": None,
        "optimizer": None,
        "discretisation": 0.1,
        "resetConfig": { "initialPos":-1.57, "stepPos":0, "samplesPerStep":150, "maxSteps":40 },
        "trajectoryType": "step"
    }

epsilons = [ 0.01, 0.1, 1.0]
gae_lambdas = [ 0.9, 0.95, 0.99]
trainingRuns = 3

fig, axs = plt.subplots(1, 3, sharey=True, sharex=True)
plotindices = [0, 1, 2]

for epsilon, idx in zip(epsilons, plotindices):
    # Create configs and labels
    configs = []
    labels = []
    for lam in gae_lambdas:
        labels.append("gae_lambdas={}".format(epsilon))
        configs.append(getConfigDisc(0, epsilon=epsilon, gae_lambda=lam))
    # Plot everything at index
    title = "epsilon = {}".format(lam)
    plotTrainingCurve(axs[idx], configs, title, labels, colours, lines, testRuns=trainingRuns, trainingSteps=trainingSteps)

fig.suptitle("Discrete PPO Hyperparameter Optimisation", fontsize=16)

axs[0].set_ylabel("Mean Return")
axs[0].set_xlabel("Training Steps")
axs[1].set_xlabel("Training Steps")
axs[2].set_xlabel("Training Steps")


axs[0].legend()

set_size(plotWidth,3)
plt.tight_layout()
plt.subplots_adjust(wspace = 0.083, top = 0.843)

plt.savefig("/app/appendixPlotsPPODiscHyperparameterOptimisation.pdf", bbox_inches='tight')
plt.show()


#############################################################################
## PPO Continuous Hyperparameter optimisation

trainingSteps = 15000
logdir = "/app/MTExperiments/Data/7PPOStepResponseTrainingOptimisation"

def getConfigCont(run, epsilon=None, horizon=None, gae_lambda=None):
    return {
        "jointID": 0,
        "logdir": logdir,
        "modelclass": "PPO",
        "modelrun": run,
        "modelname": "PPO_Continuous_eps{}_lam{}".format(epsilon, gae_lambda),
        "learningRate": 0.0003,
        "epsilon": epsilon,
        "n_steps": horizon,
        "gae_lambda": gae_lambda,
        "trainingTimesteps": trainingSteps,
        "policyNetwork": None,
        "optimizer": None,
        "discretisation": None,
        "resetConfig": { "initialPos":-1.57, "stepPos":0, "samplesPerStep":150, "maxSteps":40 },
        "trajectoryType": "step"
    }

epsilons = [ 0.01, 0.1, 1.0]
gae_lambdas = [ 0.9, 0.95, 0.99]
trainingRuns = 3

fig, axs = plt.subplots(1, 3, sharey=True, sharex=True)
plotindices = [0, 1, 2]

for epsilon, idx in zip(epsilons, plotindices):
    # Create configs and labels
    configs = []
    labels = []
    for lam in gae_lambdas:
        labels.append("gae_lambdas={}".format(epsilon))
        configs.append(getConfigCont(0, epsilon=epsilon, gae_lambda=lam))
    # Plot everything at index
    title = "epsilon = {}".format(lam)
    plotTrainingCurve(axs[idx], configs, title, labels, colours, lines, testRuns=trainingRuns, trainingSteps=trainingSteps)

fig.suptitle("Continuous PPO Hyperparameter Optimisation", fontsize=16)

axs[0].set_ylabel("Mean Return")
axs[0].set_xlabel("Training Steps")
axs[1].set_xlabel("Training Steps")
axs[2].set_xlabel("Training Steps")


axs[0].legend()

set_size(plotWidth,3)
plt.tight_layout()
plt.subplots_adjust(wspace = 0.083, top = 0.843)

plt.savefig("/app/appendixPlotsPPOContHyperparameterOptimisation.pdf", bbox_inches='tight')
plt.show()


#############################################################################
## PPO Neural network Width

trainingSteps = 15000
logdir = "/app/MTExperiments/Data/9PPOStepResponseNetworkArchitectures"

def getConfigCont(run, policyNetwork=None, activation="th.nn.Tanh", epsilon=1, horizon=512, gae_lambda=0.9):
    if policyNetwork != None:
        piconfig = "_".join([str(i) for i in policyNetwork[0]["pi"]])
        vfconfig = "_".join([str(i) for i in policyNetwork[0]["vf"]])
    else:
        piconfig = None
        vfconfig = None
    return {
        "jointID": 0,
        "logdir": logdir,
        "modelclass": "PPO",
        "modelrun": run,
        "modelname": "PPO_Continuous_pi_{}_vf_{}".format(piconfig, vfconfig),
        "learningRate": 0.0003,
        "epsilon": epsilon,
        "n_steps": horizon,
        "gae_lambda": gae_lambda,
        "trainingTimesteps": trainingSteps,
        "policyNetwork": policyNetwork,
        "activation": activation,
        "discretisation": None,
        "resetConfig": { "initialPos":-1.57, "stepPos":0, "samplesPerStep":150, "maxSteps":40 },
        "trajectoryType": "step"
    }

fig, axs = plt.subplots(2, 3, sharey=True, sharex=True)

pis = [16, 32, 64, 128, 256, 512]
vis = [16, 32, 64, 128, 256, 512]
plotindices = [[0,0], [0,1], [0,2], [1,0], [1,1], [1,2]]
trainingRuns = 3

for piWidth, idx in zip(pis, plotindices):
    configs = []
    labels = []
    i = 0
    for viWidth in vis:
        if viWidth >= piWidth:
            if piWidth >= 128 and piWidth != viWidth:
                continue
            arch = [{
                'pi': [piWidth, piWidth],
                'vf': [viWidth, viWidth]
            }]
            labels.append("{}x{} value network".format(viWidth, viWidth))
            configs.append(getConfigCont(0, policyNetwork=arch))
        else: i += 1
    # Plot everything at index
    title = "{}x{} policy network".format(piWidth, piWidth)
    plotTrainingCurve(axs[idx[0]][idx[1]], configs, title, labels, colours[i:], lines[i:], testRuns=trainingRuns, trainingSteps=trainingSteps)


fig.suptitle("PPO Neural Network Width", fontsize=16)

axs[0][0].set_ylabel("Mean Return")
axs[1][0].set_ylabel("Mean Return")

axs[1][0].set_xlabel("Training Steps")
axs[1][1].set_xlabel("Training Steps")
axs[1][2].set_xlabel("Training Steps")

#axs[0][0].legend()

axs[0][0].legend(loc='upper center', bbox_to_anchor=(1.6, -0.05),
          fancybox=True, shadow=False, ncol=3)

set_size(plotWidth,6)
plt.tight_layout()
plt.subplots_adjust(wspace = 0.067, hspace=0.443, top = 0.914)

plt.savefig("/app/appendixPlotsPPONetworkWidth.pdf", bbox_inches='tight')
plt.show()



#############################################################################
## PPO Neural network Depth

trainingSteps = 15000
logdir = "/app/MTExperiments/Data/9PPOStepResponseNetworkArchitectures"

def getConfigCont(run, policyNetwork=None, activation="th.nn.Tanh", epsilon=1, horizon=512, gae_lambda=0.9):
    if policyNetwork != None:
        piconfig = "_".join([str(i) for i in policyNetwork[0]["pi"]])
        vfconfig = "_".join([str(i) for i in policyNetwork[0]["vf"]])
    else:
        piconfig = None
        vfconfig = None
    return {
        "jointID": 0,
        "logdir": logdir,
        "modelclass": "PPO",
        "modelrun": run,
        "modelname": "PPO_Continuous_pi_{}_vf_{}".format(piconfig, vfconfig),
        "learningRate": 0.0003,
        "epsilon": epsilon,
        "n_steps": horizon,
        "gae_lambda": gae_lambda,
        "trainingTimesteps": trainingSteps,
        "policyNetwork": policyNetwork,
        "activation": activation,
        "discretisation": None,
        "resetConfig": { "initialPos":-1.57, "stepPos":0, "samplesPerStep":150, "maxSteps":40 },
        "trajectoryType": "step"
    }

fig, ax = plt.subplots()

depths = [1, 2, 3]
trainingRuns = 3

configs = []
labels = []

for depth in depths:
    arch = [{
        'pi': [64 for _ in range(depth)],
        'vf': [64 for _ in range(depth)]
    }]
    labels.append("network depth={}".format(depth))
    configs.append(getConfigCont(0, policyNetwork=arch))

plotTrainingCurve(ax, configs, None, labels, colours, lines, testRuns=trainingRuns, trainingSteps=trainingSteps)

fig.suptitle("PPO Neural Network Depth", fontsize=16)

ax.set_ylabel("Mean Return")
ax.set_xlabel("Training Steps")
ax.legend()

set_size(plotWidth,3)
plt.tight_layout()

plt.subplots_adjust(top = 0.905)

plt.savefig("/app/appendixPlotsPPONetworkDepth.pdf", bbox_inches='tight')
plt.show()

#############################################################################
## Trajectory Control

def plotTrainingCurveAlt(ax, configs, title, labels, colours, lines, testRuns=3, trainingSteps=15000):
    """  """
    # Load tensorboard log
    data = []
    for config in configs:
        data.append(loadTensorboardLog(config, testRuns=1))
    # Chunk data due to this experiment's wrong file naming
    actualData = [
        [
            data[:3][0][0],
            data[:3][1][0],
            data[:3][2][0]
        ],
        [
            data[3:][0][0],
            data[3:][1][0],
            data[3:][2][0]
        ]
    ]
    # Plot mean reward and confidence interval
    for i, d in enumerate(actualData):
        x, y, cl, cu = computeGraphArrays(d, runs=testRuns, lenPlot=trainingSteps)
        ax.plot(x,y,color=colours[i], label=labels[i], linestyle=lines[i])
        ax.fill_between(x, cl, cu,color=colours[i], alpha=.1)
    # Set labels and title
    #ax.set_xlabel("Training Steps")
    #ax.set_ylabel("Mean Return")
    if title != None: ax.set_title(title)
    ax.grid()
    #ax.legend()

trainingSteps = 15000
logdir = "/app/MTExperiments/Data/11PickAndPlaceOptimisation"

def getDDPGConfig(jointID, resetConfig, run):
    return {
        "jointID": 0,
        "logdir": logdir,
        "modelclass": "DDPG",
        "modelrun": 0,
        "modelname": "DDPG_jid{}_{}".format(jointID, run),
        "learningRate": 0.0001,
        "tau": 0.002,  
        "trainingTimesteps": trainingSteps,
        "policyNetwork": None,
        "optimizer": None,
        "discretisation": None,
        "resetConfig": resetConfig,
        "trajectoryType": "generator"
    }

def getPPOConfig(jointID, resetConfig, run):
    return {
        "jointID": 0,
        "logdir": logdir,
        "modelclass": "PPO",
        "modelrun": 0,
        "modelname": "PPOCont_jid{}_{}".format(jointID, run),
        "learningRate": 0.0003,
        "epsilon": 1,
        "n_steps": 512,
        "gae_lambda": 0.9,
        "trainingTimesteps": trainingSteps,
        "policyNetwork": None,
        "optimizer": None,
        "discretisation": None,
        "resetConfig": resetConfig,
        "trajectoryType": "generator"
    }


jointindices = [0, 1, 2]
trainingRuns = 3

fig, axs = plt.subplots(1, 3, sharey=True, sharex=True)
plotindices = [0, 1, 2]

for jid, idx in zip(jointindices, plotindices):
    # Create configs and labels
    configs = []
    for run in range(trainingRuns):
        configs.append(getDDPGConfig(jid, None, run))
    for run in range(trainingRuns):
        configs.append(getPPOConfig(jid, None, run))
    labels = ["DDPG", "Continuous PPO"]
    # Plot everything at index
    title = "Joint {}".format(jid)
    plotTrainingCurveAlt(axs[idx], configs, title, labels, colours, lines, testRuns=trainingRuns, trainingSteps=trainingSteps)

fig.suptitle("Trajectory Control Optimisation", fontsize=16)

axs[0].set_ylabel("Mean Return")
axs[0].set_xlabel("Training Steps")
axs[1].set_xlabel("Training Steps")
axs[2].set_xlabel("Training Steps")


axs[2].legend()

set_size(plotWidth,3)
plt.tight_layout()
plt.subplots_adjust(wspace = 0.06, top = 0.852)

plt.savefig("/app/appendixPlotsPickAndPlaceOptimisation.pdf", bbox_inches='tight')
plt.show()



