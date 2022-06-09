
docker exec -it ros_ml_container1 bash

source ~/myenv/bin/activate
source /opt/ros/noetic/setup.bash
source /catkin_ws/devel/setup.bash

python

from json import load
import subprocess
import os
import sys
import time
from turtle import color, right, width
sys.path.append("/catkin_ws/src/jointcontrol/scripts")

import numpy as np
import copy

""" 
Control script for experiment 4 

PPO agents are trained to perform step responses from 0 to -1.57 rad using different learning rates and clipping parameters

Plots include average return of training and separate testepisodes

Logs are stored in /app/MTExperiments/Data/9PPOStepResponseNetworkArchitectures


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
logdir = "/app/MTExperiments/Data/9PPOStepResponseNetworkArchitectures"
os.system("tensorboard --logdir {}/tensorboard --host 0.0.0.0 --port 6006 &".format(logdir))

# Point to RL agent deployment script
agentPath = "/app/MTExperiments/TrainingScripts/RLAgent.py"
randomAgentPath = "/app/MTExperiments/TrainingScripts/RandomAgent.py"


#############################################################################
## PPO Continuous

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



## First test different widths
networkArchs = [
    [{'pi': [16, 16], 'vf': [16, 16]}], # Test various policy sizes and bigger value networks
    [{'pi': [16, 16], 'vf': [32, 32]}],
    [{'pi': [16, 16], 'vf': [64, 64]}],
    [{'pi': [16, 16], 'vf': [128, 128]}],
    [{'pi': [16, 16], 'vf': [256, 256]}],
    [{'pi': [16, 16], 'vf': [512, 512]}],
    [{'pi': [32, 32], 'vf': [32, 32]}],
    [{'pi': [32, 32], 'vf': [64, 64]}],
    [{'pi': [32, 32], 'vf': [128, 128]}],
    [{'pi': [32, 32], 'vf': [256, 256]}],
    [{'pi': [32, 32], 'vf': [512, 512]}],
    [{'pi': [64, 64], 'vf': [64, 64]}],
    [{'pi': [64, 64], 'vf': [128, 128]}],
    [{'pi': [64, 64], 'vf': [256, 256]}],
    [{'pi': [64, 64], 'vf': [512, 512]}],
    [{'pi': [128, 128], 'vf': [128, 128]}],
    [{'pi': [256, 256], 'vf': [256, 256]}],
    [{'pi': [512, 512], 'vf': [512, 512]}],
    [{'pi': [64], 'vf': [64]}], # Test less deep/deeper networks
    [{'pi': [64, 64, 64], 'vf': [64, 64, 64]}],
]


trainingRuns = 3
envJoints = 6
configs = []

for arch in networkArchs:
    for run in range(trainingRuns):
        configs.append( getConfigCont(run, policyNetwork=arch) )



#############################################################################
## RUN Training Episodes for each Configuration

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

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


#############################################################################
## Plot bar graph of agent performance per arch

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


configs = [
    [{'pi': [16, 16], 'vf': [16, 16]}],
    [{'pi': [32, 32], 'vf': [32, 32]}],
    [{'pi': [64, 64], 'vf': [64, 64]}],
    [{'pi': [128, 128], 'vf': [128, 128]}],
    [{'pi': [256, 256], 'vf': [256, 256]}],
    [{'pi': [512, 512], 'vf': [512, 512]}]
]

widthData = []

for arch in configs:
    # Load logs of current config
    config = ( getConfigCont(run, policyNetwork=arch) )
    res = loadTensorboardLog(config, testRuns=trainingRuns)
    # Get mean and 95% confidence interval of last episode's mean reward
    widthData.append( getConfidenceInterval( np.array(res)[:,-1] ) )

widthMean = np.array(widthData)[:,0]
# Get relative errors for bar graph
widthYerrs = np.array(widthData)[:,0] - np.array(widthData)[:,1]

###

configs = [
    [{'pi': [64], 'vf': [64]}],
    [{'pi': [64, 64], 'vf': [64, 64]}],
    [{'pi': [64, 64, 64], 'vf': [64, 64, 64]}]
]

depthData = []

for arch in configs:
    # Load logs of current config
    config = ( getConfigCont(run, policyNetwork=arch) )
    res = loadTensorboardLog(config, testRuns=trainingRuns)
    # Get mean and 95% confidence interval of last episode's mean reward
    depthData.append( getConfidenceInterval( np.array(res)[:,-1] ) )

depthMean = np.array(depthData)[:,0]
# Get relative errors for bar graph
depthYerrs = np.array(depthData)[:,0] - np.array(depthData)[:,1]

## Set up bar plot
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [2, 1]})

widthLabels = ["16", "32", "64", "128", "256", "512"]
depthLabels = ["1", "2", "3"]

widthColours = CmTWBlue( np.linspace(0, 0.6, len(widthLabels)) )
depthColours = CmDarkBlue( np.linspace(0.4, 0.6, len(depthLabels)) )

ax1.set_ylabel("Average Return")
ax1.set_xlabel("Neural Network Width")
ax2.set_xlabel("Neural Network Depth")

barwidth = 0.8

ax1.bar(np.arange(len(widthMean)), widthMean, width=barwidth, yerr=widthYerrs, label=widthLabels, color=widthColours)
ax2.bar([0.5, 1.0, 1.5], depthMean, width=barwidth*0.5, yerr=depthYerrs, label=depthLabels, color=depthColours)


ax1.set_xticks(np.arange(len(widthMean)), widthLabels)
ax2.set_xticks([0.5, 1.0, 1.5], depthLabels)

fig.suptitle("PPO Neural Network Width and Depth", fontsize=16)

set_size(7,3.2)
fig.tight_layout()
plt.subplots_adjust(top = 0.92, right=0.952)

plt.savefig('/app/resultsPPOStepResponseTrainingNetworkWidthDepth.pdf')
plt.show()














#############################################################################
## VISUALISE mean reward as matrix plot

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
                     threshold=None, disregardVal=None, **textkw):
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
            if data[i, j] == disregardVal:
                print(data[i, j])
                text = im.axes.text(j, i, "N/A", **kw)
            else:
                text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)
    return texts

## Set up data
pilabels = [16, 32, 64]
vflabels = [16, 32, 64, 128, 256, 512]

heatArr = np.zeros((len(pilabels), len(vflabels)))

for row, pi in enumerate(pilabels):
    for col, vf in enumerate(vflabels):
        if vf >= pi:
            arch = [{'pi': [pi, pi], 'vf': [vf, vf]}]
            config = ( getConfigCont(0, policyNetwork=arch) )
            res = loadTensorboardLog(config, testRuns=trainingRuns)
            mean,_ , _ = getConfidenceInterval( np.array(res)[:,-1] )
            heatArr[row][col] = mean

# Get array bounds apart from zeros
minval = np.amin(heatArr[heatArr<0])
maxval = np.amax(heatArr[heatArr<0])

for row, pi in enumerate(pilabels):
    for col, vf in enumerate(vflabels):
        if heatArr[row][col] == 0:
            heatArr[row][col] = minval-1

## Create modified colourmap in order to plot N/A values
CmBlueYellow = combineColourmaps(
    createColourmapToWhite( hexstring2rgb(colourScheme2["yellow"]), reverse=False), # Top = Yellow
    createColourmapToWhite( hexstring2rgb(colourScheme1["twblue"]), reverse=True )  # Bottom = TWBlue
)

newcolors = CmBlueYellow(np.linspace(0, 1, 256))
white = np.array([256/256, 256/256, 256/256, 1])
newcolors[:1, :] = white
newcmp = ListedColormap(newcolors)


fig, ax = plt.subplots()

im, cbar = heatmap(heatArr, [ "{} Neurons".format(lab) for lab in pilabels ], [ "{} Neurons".format(lab) for lab in vflabels ],
                    ax=ax, cmap=newcmp, cbarlabel="Average Return", cbar_kw={"shrink":1.0})
texts = annotate_heatmap(im, disregardVal=minval-1, valfmt="{x:.0f}", textcolors=("black", "black"))


fig.suptitle("PPO Policy and Value Network Width", fontsize=16)
ax.set_xlabel("Value Neural Network Width")
ax.set_ylabel("Policy Neural Network Width")

set_size(7,3.2)
fig.tight_layout()

plt.subplots_adjust(right=0.945, left=0.193, top=0.702, bottom=0.08)

plt.savefig('/app/resultsPPOStepResponseTrainingNetworkWidthHeatmap.pdf')
plt.show()


