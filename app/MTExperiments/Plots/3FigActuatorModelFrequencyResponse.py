
## Matplotlib imports
import matplotlib.pyplot as plt
import matplotlib

matplotlib.style.use("default")

import sys
sys.path.append("/catkin_ws/src/jointcontrol/scripts")

import gym
from discreteActionWrapper import *

import numpy as np
import math

## Import thesis colours
sys.path.append("/app/MTExperiments/Plots")
from colourmaps import *

# Make environment
env = gym.make('jointcontrol-v0', jointidx=0)
env.reset(episodeType="step", config={ "initialPos":0, "stepPos":-1.57, "samplesPerStep":200, "maxSteps":40 })

## Perform step responses to empirically determine amplification at each frequency
# Set up frequencies that will be tested
df = 0.05
startfreq = 0.1
endfreq = 20
# Ensure, that frequencies can actually be created
assert startfreq >= df 
assert endfreq > startfreq
freqs = np.array([ i for i in range(int(startfreq/df), int(endfreq/df)+1) ])*df

# Config placeholder for square step responses
lowerSignal = -1.57
higherSignal = 1.57

#lowerSignal = -1
#higherSignal = 1

config = {
    "lowerSignal": lowerSignal,
    "higherSignal": higherSignal,
    "pulseLength": 0,
    "numPulses": 5,
    "maxSteps": 40
}

# Store amplification based on min and max signals
amp = []

# Iterate over frequencies and apply square waves
for freq in freqs:
    # Set pulse length based on frequency
    config["pulseLength"] = int(round((1/freq)/env.env.ts))
    # Reset env in square mode
    env.env.reset(episodeType='square', config=config)
    # Perform step
    #env.step(None)
    # Store resulting amplification
    amp.append((max(env.env.latestTrajectory) - min(env.env.latestTrajectory)) / (higherSignal - lowerSignal))

## Create plot

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

# Helper function for step detection
def detectStep(signal, thresshold = 0.001):
    """ Detects a step greater than thresshold in input signal """
    idx = 0
    for i in range(len(signal)-1):
        if abs(signal[i] - signal[i+1]) > thresshold:
            idx = i
            break
    return idx

plt.rcParams.update({'font.size': 12})

# Convert amplification to dB
amplification = [ 20*math.log(entry) for entry in amp ]

# Plot amplification
plt.plot(freqs, amplification, color=colourScheme1["darkblue"], alpha=1.0)

# Plot cutoff frequency
cutoffFreq = freqs[detectStep(amplification, thresshold=0.0005)]
plt.vlines(x = cutoffFreq, ymin = -65, ymax = 5, colors = colourScheme2["yellow"], label = 'Cutoff Frequency')
matplotlib.pyplot.text(0.112, -57.5, '0.45 Hz Cutoff Frequency', color=colourScheme2["darkblue"])

plt.suptitle("Simulated Actuator Frequency Response", fontsize=16)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Amplification [dB]")

plt.xscale('log')
plt.yscale('linear')

plt.xlim([0.08, 12.5])
plt.ylim([-70, 10])

plt.grid()

set_size(7,3.2)

plt.tight_layout()
plt.subplots_adjust(top = 0.917)

plt.savefig("/app/resultsSimulatedActuatorFrequencyResponse.pdf")
plt.show()

##NOTE: Discretisation will be visible at higher frequencies in this plot
# run it at a lower discretisation (config.json) to avoid that.

#TODO: measure -40dB/dec

