
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
import copy

""" 
Control script for experiment 4 

PPO agents are trained to perform step responses from 0 to -1.57 rad using different learning rates and clipping parameters

Plots include average return of training and separate testepisodes

Logs are stored in /app/MTExperiments/Data/7PPOStepResponseTrainingOptimisation


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
logdir = "/app/MTExperiments/Data/7PPOStepResponseTrainingOptimisation"
os.system("tensorboard --logdir {}/tensorboard --host 0.0.0.0 --port 6006 &".format(logdir))

# Point to RL agent deployment script
agentPath = "/app/MTExperiments/TrainingScripts/RLAgent.py"
randomAgentPath = "/app/MTExperiments/TrainingScripts/RandomAgent.py"


#############################################################################
## PPO Discrete

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


#############################################################################
## PPO Continuous

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


# Create configurations for each lr, tau combination
epsilons = [ 0.01, 0.1, 1.0]         # default = 0.1
gae_lambdas = [ 0.9, 0.95, 0.99]

discreteConfigs = []
continuousConfigs = []
configs = []

# Iterate over combinations and define what is trained where concurrently
trainingRuns = 3
envJoints = 6

for ep in epsilons:
    for lam in gae_lambdas:
        for run in range(trainingRuns):
            discreteConfigs.append(getConfigDisc(run, epsilon=ep, gae_lambda=lam, horizon=512))
            continuousConfigs.append(getConfigCont(run, epsilon=ep, gae_lambda=lam, horizon=512))

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

configs = discreteConfigs + continuousConfigs


#############################################################################
## RUN Training Episodes for each Configuration

# Iterate over all runs and apply them to the correct joint
if runTraining == True:
    for config in chunker(configs, envJoints):
        print("Starting new training batch:")
        print([ c["modelname"] for c in config ])
        # New training cycle
        processes = [None for _ in range(envJoints)]
        for i, c in enumerate(config):
            c["jointID"] = i
            # Start process and keep track of it
            processes[i] = subprocess.Popen([sys.executable, agentPath, str(c)])#, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            # Wait for env to register
            time.sleep(1)
        # Wait for processes to finish and check return code
        rc = []
        for i in range(envJoints):
            if processes[i] != None:
                streamdata = processes[i].communicate()[0]
                rc.append(processes[i].returncode)
        #
        # Print success and start with next batch
        print("Batch Agent tests successfully exited with return codes {}".format(rc))


# Debug: for one environment
#p = subprocess.Popen([sys.executable, agentPath, str(PPOContinuousConfig)])
#streamdata = p.communicate()[0]
#print(p.returncode)








#############################################################################
## VISUALISE mean reward as matrix plot
# -> plot average return plot matrix plot of mean reward

# Load Tensorboard Logs
#https://gist.github.com/willwhitney/9cecd56324183ef93c2424c9aa7a31b4
from tensorboard.backend.event_processing import event_accumulator

import scipy.stats as st
import numpy as np
import math

## Matplotlib imports
import matplotlib.pyplot as plt
import matplotlib

matplotlib.style.use("default")

import sys
sys.path.append("/app/MTExperiments/Plots")
from colourmaps import *

plt.rcParams.update({'font.size': 12})

def loadTensorboardLog(config, testRuns=3):
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


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.
    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """
    if not ax:
        ax = plt.gca()
    # Plot the heatmap
    im = ax.imshow(data, **kwargs)
    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")
    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)
    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.
    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()
    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.
    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)
    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)
    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)
    return texts


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


## Load logs and calculate results
discreteConfigs = []
continuousConfigs = []

discreteMeanPlotData = np.empty((len(epsilons), len(gae_lambdas)))
continuousMeanPlotData = np.empty((len(epsilons), len(gae_lambdas)))

continuousRewards = []
discreteRewards = []

for row, ep in enumerate(epsilons):
    for col, lam in enumerate(gae_lambdas):
        discreteConfigs.append(getConfigDisc(0, epsilon=ep, gae_lambda=lam, horizon=512))
        continuousConfigs.append(getConfigCont(0, epsilon=ep, gae_lambda=lam, horizon=512))
        # Load and store results
        discreteResults = loadTensorboardLog(discreteConfigs[-1], testRuns=trainingRuns)
        continuousResults = loadTensorboardLog(continuousConfigs[-1], testRuns=trainingRuns)
        # Get mean of last episode
        discreteMean, _, _ = getConfidenceInterval( np.array(discreteResults)[:,-1] )
        continuousMean, _, _ = getConfidenceInterval( np.array(continuousResults)[:,-1] )
        # Store everything mean reward data
        discreteRewards.append(discreteResults)
        continuousRewards.append(continuousResults)
        # Store heatmap mean data
        discreteMeanPlotData[row][col] = discreteMean
        continuousMeanPlotData[row][col] = continuousMean


## Heatmap plot
fig, (ax1, ax2) = plt.subplots(1, 2)

epislonLabels = [ "epsilon={}".format(ep) for ep in epsilons ]
lambdaLabels = [ "lambda={}".format(lam) for lam in gae_lambdas ]

# Create colourbars
CmYellowBlue = combineColourmaps(
    createColourmapToWhite( hexstring2rgb(colourScheme1["twblue"]), reverse=False ), # Top = TWBlue
    createColourmapToWhite( hexstring2rgb(colourScheme2["yellow"]), reverse=True  )  # Bottom = Yellow
)

CmBlueYellow = combineColourmaps(
    createColourmapToWhite( hexstring2rgb(colourScheme2["yellow"]), reverse=False), # Top = Yellow
    createColourmapToWhite( hexstring2rgb(colourScheme1["twblue"]), reverse=True )  # Bottom = TWBlue
)

im, cbar = heatmap(discreteMeanPlotData, epislonLabels, lambdaLabels, ax=ax1,
                   cmap=CmBlueYellow, cbarlabel=" ", cbar_kw={"shrink":0.62})
texts = annotate_heatmap(im, valfmt="{x:.0f}", textcolors=("black", "black"))


im, cbar = heatmap(continuousMeanPlotData, epislonLabels, lambdaLabels, ax=ax2,
                   cmap=CmBlueYellow, cbarlabel=" ", cbar_kw={"shrink":0.62})
texts = annotate_heatmap(im, valfmt="{x:.0f}", textcolors=("black", "black"))


#ax1.set_xlabel("Learning Rates")
#ax2.set_xlabel("Learning Rates")
#ax1.set_ylabel("Soft Target Update Parameters")
#ax2.set_ylabel("Soft Target Update Parameters")
#ax1.set_title("Width of 95% Conficence Interval")
#ax2.set_title("Average Return per Episode")

fig.suptitle("PPO Hyperparameter Optimisation", fontsize=16)

ax1.set_xlabel("Average Return (Discrete)")
ax2.set_xlabel("Average Return (Continuous)")

set_size(7,3.2)
fig.tight_layout()
plt.subplots_adjust(top = 0.836, bottom=0, wspace=0.633)

plt.savefig('/app/resultsPPOStepResponseTrainingHyperparameterHeatmap.pdf')
plt.show()













#############################################################################
## VISUALISE mean reward during training

plt.rcParams.update({'font.size': 12})

## Load logs and calculate results
discreteConfigs = []
continuousConfigs = []

continuousRewards = []
discreteRewards = []

ep = 1.0
for col, lam in enumerate(gae_lambdas):
    discreteConfigs.append(getConfigDisc(0, epsilon=ep, gae_lambda=lam, horizon=512))
    continuousConfigs.append(getConfigCont(0, epsilon=ep, gae_lambda=lam, horizon=512))
    # Load and store results
    discreteResults = loadTensorboardLog(discreteConfigs[-1], testRuns=trainingRuns)
    continuousResults = loadTensorboardLog(continuousConfigs[-1], testRuns=trainingRuns)
    # Get mean of last episode
    discreteMean, _, _ = getConfidenceInterval( np.array(discreteResults)[:,-1] )
    continuousMean, _, _ = getConfidenceInterval( np.array(continuousResults)[:,-1] )
    # Store everything mean reward data
    discreteRewards.append(discreteResults)
    continuousRewards.append(continuousResults)

lambdaLabels = [ "GAE lambda={}".format(lam) for lam in gae_lambdas ]

lines = ['-', '--', ':']

colours = [
    colourScheme1["darkblue"],
    colourScheme2["twblue"],
    colourScheme1["lightblue"],
]

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

## Plot everything
for i, d in enumerate(discreteRewards):
    x, y, cl, cu = computeGraphArrays(d, runs=3)
    ax1.plot(x,y,color=colours[i], label=lambdaLabels[i], linestyle=lines[i])
    ax1.fill_between(x, cl, cu,color=colours[i], alpha=.1)

for i, d in enumerate(continuousRewards):
    x, y, cl, cu = computeGraphArrays(d, runs=3)
    ax2.plot(x,y,color=colours[i], label=lambdaLabels[i], linestyle=lines[i])
    ax2.fill_between(x, cl, cu,color=colours[i], alpha=.1)

ax1.set_xlabel('Training Steps')
ax1.set_ylabel('Mean Return')

ax2.set_xlabel('Training Steps')
#ax2.set_ylabel('Mean Return')


plt.suptitle("PPO Advantage Estimation", fontsize=16)

#xlim = []
ylim = [-90, -10]

#ax1.set_xlim(xlim)
#ax2.set_xlim(xlim)
ax1.set_ylim(ylim)
ax2.set_ylim(ylim)

ax1.legend()
#ax2.legend()

ax1.grid()
ax2.grid()

set_size(7,3.5)
plt.tight_layout()

plt.subplots_adjust(top=0.92, wspace=0.062)

plt.savefig("/app/resultsPPOStepResponseTrainingHyperparameterMeanReward.pdf", bbox_inches='tight')
plt.show()




