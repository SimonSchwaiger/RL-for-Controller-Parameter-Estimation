
import csv
import os

import numpy as np
import matplotlib.pyplot as plt


"""
NOTES Hebi Step Response Tests

Sampling Rate: 500Hz

Test Conducted with long and short connector piece and with and without a second hebi X5-4 attached as a weight

Step 1 = [0, -1.57]
Step 2 = [-1.57, 0]

"""

# Get files in results directory
path = "."

files = [
    f 
    for f in os.listdir(path) 
    if os.path.isfile(os.path.join(path, f))
]

# Filter out non-csv files
files = np.array(files, dtype=str)[
    [ ".csv" in entry for entry in files ]
]

def detectStep(signal, thresshold = 0.001):
    """ Detects a step greater than thresshold in input signal """
    idx = 0
    for i in range(len(signal-1)):
        if abs(signal[i] - signal[i+1]) > thresshold:
            idx = i
            break
    return idx

def plotStep(filename, ts=0.002, stepDuration=3):
    cmd = []
    pos = []
    # Load csv
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        for row in reader:
            cmd.append(row[3])
            pos.append(row[6])
    #
    # Convert data to floats
    cmd = np.array(cmd[1:], dtype=np.float64)
    pos = np.array(pos[1:], dtype=np.float64)
    # Detect where step happens
    start = detectStep(cmd)
    end = int(stepDuration/ts) + start
    time = np.arange(0, len(cmd[start:end]))*ts
    # Plot functions
    plt.plot(time, cmd[start:end])
    plt.plot(time, pos[start:end])
    plt.suptitle("{}".format(filename))
    plt.xlabel = "Time [sec]"
    plt.show()

for filename in files:
    plotStep(filename)



