
## Matplotlib imports
import matplotlib.pyplot as plt
import matplotlib

matplotlib.style.use("default")

import sys
import csv
sys.path.append("/catkin_ws/src/jointcontrol/scripts")

import gym
import gym_jointcontrol
from discreteActionWrapper import *

import numpy as np

## Import thesis colours
sys.path.append("/app/MTExperiments/Plots")
from colourmaps import *

# Helper function for step detection
def detectStep(signal, thresshold = 0.001):
    """ Detects a step greater than thresshold in input signal """
    idx = 0
    for i in range(len(signal)-1):
        if abs(signal[i] - signal[i+1]) > thresshold:
            idx = i
            break
    return idx

####################################################
## Load Step 1 real robot data from hebi logs
ts = 0.002
stepDuration=1.5
filename = "/app/MTExperiments/HebiLogs/export/Step1X59ShortTubeStepNoLoad.csv"

cmd = []
pos = []
# Load csv
with open(filename, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    for row in reader:
        cmd.append(row[3])
        pos.append(row[6])

# Convert hebi data to floats
cmd = np.array(cmd[1:], dtype=np.float64)
pos = np.array(pos[1:], dtype=np.float64)
# Detect where step happens
start = detectStep(cmd)
end = int(stepDuration/ts) + start
time = np.arange(0, len(cmd[start:end]))*ts

# Store trajectory for plotting
step1RealRobotTime = np.arange(0, len(cmd[start:end]))*ts
step1RealRobotReference = cmd[start:end]
step1RealRobotPosition = pos[start:end]


####################################################
## Simulate step 1 and track feedback
# Set up env, perform step and store response
env = gym.make('jointcontrol-v0', jointidx=0)
env.reset(config={ "initialPos":0, "stepPos":-1.57, "samplesPerStep":200, "maxSteps":40 })

start = detectStep(env.env.controlSignal)
end = int(stepDuration/0.01)+start

step1SimRobotTime = env.env.ts*np.arange(len(env.env.latestTrajectory[start:end]))
step1SimRobotPos = env.env.latestTrajectory[start:end]

####################################################
## Load Step 2 real robot data from hebi logs
filename = "/app/MTExperiments/HebiLogs/export/Step2X59ShortTubeStepNoLoad.csv"

cmd = []
pos = []
# Load csv
with open(filename, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    for row in reader:
        cmd.append(row[3])
        pos.append(row[6])

# Convert hebi data to floats
cmd = np.array(cmd[1:], dtype=np.float64)
pos = np.array(pos[1:], dtype=np.float64)
# Detect where step happens
start = detectStep(cmd)
end = int(stepDuration/ts) + start
time = np.arange(0, len(cmd[start:end]))*ts

# Store trajectory for plotting
step2RealRobotTime = np.arange(0, len(cmd[start:end]))*ts
step2RealRobotReference = cmd[start:end]
step2RealRobotPosition = pos[start:end]

####################################################
## Simulate step 2 and track feedback
# Set up env, perform step and store response
#env = gym.make('jointcontrol-v0', jointidx=0)
env.reset(config={ "initialPos":-1.57, "stepPos":0, "samplesPerStep":220, "maxSteps":40 })

start = detectStep(env.env.controlSignal)
end = int(stepDuration/0.01)+start

step2SimRobotTime = env.env.ts*np.arange(len(env.env.latestTrajectory[start:end]))
step2SimRobotPos = env.env.latestTrajectory[start:end]

####################################################
## Set up multiplot and plot everything

## Set figure size as seen here
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

## Multiplot Setup
plt.rcParams.update({'font.size': 12})

fig, (ax0, ax1) = plt.subplots(1,2, sharey=True)
#fig.suptitle("Approximated Transfer Function Step Response", fontsize=16)

# Config order: [realStep, simstep, inputSignal]
zorder = [3, 2, 1]
alphas = alphas = [1.0, 0.6, 0.3]

colours = [
    colourScheme1["darkblue"],
    colourScheme2["twblue"],
    colourScheme1["lightblue"]
]

linestyles = ["--", "-", "-"]
dashes = [(1, 2), None, None]

labels = [
    "Real Actuator",
    "Simulated Actuator",
    "Input Step Signal"
]

ax0.set_xlabel("Time [sec]")
ax1.set_xlabel("Time [sec]")

ax0.set_ylabel("Actuator Position [rad]")

ax0.set_title("Step Response 0 to -1.57 Radians") 
ax1.set_title("Step Response -1.57 to 0 Radians")

## Plot step 1
# Real robot position
ax0.plot(step1RealRobotTime, step1RealRobotPosition, label=labels[0], linestyle=linestyles[0], color=colours[0], alpha=alphas[0], zorder=zorder[0], dashes=dashes[0])
# Sim robot position
ax0.plot(step1SimRobotTime, step1SimRobotPos, label=labels[1], linestyle=linestyles[1], color=colours[1], alpha=alphas[1], zorder=zorder[1])
# Reference signal
ax0.plot(step1RealRobotTime, step1RealRobotReference, label=labels[2], linestyle=linestyles[2], color=colours[2], alpha=alphas[2], zorder=zorder[2])


## Plot step 2
# Real robot position
ax1.plot(step2RealRobotTime, step2RealRobotPosition, label=labels[0], linestyle=linestyles[0], color=colours[0], alpha=alphas[0], zorder=zorder[0], dashes=dashes[0])
# Sim robot position
ax1.plot(step2SimRobotTime, step2SimRobotPos, label=labels[1], linestyle=linestyles[1], color=colours[1], alpha=alphas[1], zorder=zorder[1])
# Reference signal
ax1.plot(step2RealRobotTime, step2RealRobotReference, label=labels[2], linestyle=linestyles[2], color=colours[2], alpha=alphas[2], zorder=zorder[2])

###################################################
# Add legends, set layout and save figure

#ax0.legend()
ax1.legend()

set_size(7,2.8) # (7, 3.2)

plt.tight_layout()

plt.savefig("/app/resultsRealVsSimulatedStep.pdf")
plt.show()



