
# Set matplotlib to inline mode and import necessary python components
#%matplotlib inline
import matplotlib.pyplot as plt
#import matplotlib_inline.backend_inline
#matplotlib_inline.backend_inline.set_matplotlib_formats('svg')


import sys
import csv
sys.path.append("/catkin_ws/src/jointcontrol/scripts")

import gym
import gym_jointcontrol
from discreteActionWrapper import *

import numpy as np


def detectStep(signal, thresshold = 0.001):
    """ Detects a step greater than thresshold in input signal """
    idx = 0
    for i in range(len(signal)-1):
        if abs(signal[i] - signal[i+1]) > thresshold:
            idx = i
            break
    return idx


## Load log from real robot step response testing
ts = 0.002
stepDuration=3
filename = "/app/MTExperiments/HebiLogs/export/Step1X59ShortTubeStepNoLoad.csv"

cmd = []
pos = []
# Load csv
with open(filename, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    for row in reader:
        cmd.append(row[3])
        pos.append(row[6])


# Convert data to floats
cmd = np.array(cmd[1:], dtype=np.float64)
pos = np.array(pos[1:], dtype=np.float64)
# Detect where step happens
start = detectStep(cmd)
end = int(stepDuration/ts) + start
time = np.arange(0, len(cmd[start:end]))*ts

realRobotTime = np.arange(0, len(cmd[start:end]))*ts
realRobotReference = cmd[start:end]
realRobotPosition = pos[start:end]


## Simulate step and track feedback
# Set up env, perform step and store response
env = gym.make('jointcontrol-v0', jointidx=0)
env.reset()

start = detectStep(env.env.controlSignal)

simRobotTime = env.env.ts*np.arange(len(env.env.latestTrajectory[start:]))
simRobotPos = env.env.latestTrajectory[start:]



## Plot everything
plt.plot(realRobotTime, realRobotReference, label="Step Input Signal")
plt.plot(realRobotTime, realRobotPosition, label="Real Actuator Position")
plt.plot(simRobotTime, simRobotPos, label="Simulated Actuator Position")

plt.xlabel = "Time [sec]"
plt.legend()
plt.show()








##################################################################################################################


# Set matplotlib to inline mode and import necessary python components
#%matplotlib inline
import matplotlib.pyplot as plt
#import matplotlib_inline.backend_inline
#matplotlib_inline.backend_inline.set_matplotlib_formats('svg')


import sys
import csv
sys.path.append("/catkin_ws/src/jointcontrol/scripts")

import gym
import gym_jointcontrol
from discreteActionWrapper import *

import numpy as np


def detectStep(signal, thresshold = 0.001):
    """ Detects a step greater than thresshold in input signal """
    idx = 0
    for i in range(len(signal)-1):
        if abs(signal[i] - signal[i+1]) > thresshold:
            idx = i
            break
    return idx


## Load log from real robot step response testing
ts = 0.002
stepDuration=3
filename = "/app/MTExperiments/HebiLogs/export/Step2X59ShortTubeStepNoLoad.csv"

cmd = []
pos = []
# Load csv
with open(filename, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    for row in reader:
        cmd.append(row[3])
        pos.append(row[6])


# Convert data to floats
cmd = np.array(cmd[1:], dtype=np.float64)
pos = np.array(pos[1:], dtype=np.float64)
# Detect where step happens
start = detectStep(cmd)
end = int(stepDuration/ts) + start
time = np.arange(0, len(cmd[start:end]))*ts

realRobotTime = np.arange(0, len(cmd[start:end]))*ts
realRobotReference = cmd[start:end]
realRobotPosition = pos[start:end]


## Simulate step and track feedback
# Set up env, perform step and store response
env = gym.make('jointcontrol-v0', jointidx=0)
env.reset(config={ "initialPos":-1.57, "stepPos":0, "samplesPerStep":150, "maxSteps":40 })

start = detectStep(env.env.controlSignal)

simRobotTime = env.env.ts*np.arange(len(env.env.latestTrajectory[start:]))
simRobotPos = env.env.latestTrajectory[start:]



## Plot everything
plt.plot(realRobotTime, realRobotReference, label="Step Input Signal")
plt.plot(realRobotTime, realRobotPosition, label="Real Actuator Position")
plt.plot(simRobotTime, simRobotPos, label="Simulated Actuator Position")

plt.xlabel = "Time [sec]"
plt.legend()
plt.show()




