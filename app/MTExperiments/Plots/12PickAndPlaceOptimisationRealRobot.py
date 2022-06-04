import csv
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.style.use("default")

## Import thesis colours
sys.path.append("/app/MTExperiments/Plots")
from colourmaps import *


def loadTrajLog(filename, trajDuration=13.5, start=16, starttime=4, endtime=16):
    """ Loads trajectory logs from filename, cleans data and returns it with associated time """
    jid = []
    cmd = []
    pos = []
    #
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        for row in reader:
            jid.append(row[1])
            cmd.append(row[3])
            pos.append(row[6])
    #
    # Convert to arrays and omit identifier
    jid = np.array(jid[start:], dtype=int)
    cmd = np.array(cmd[start:], dtype=float)
    pos = np.array(pos[start:], dtype=float)
    #
    j0cmd = cmd[ jid == 0 ]
    j1cmd = cmd[ jid == 1 ]
    j2cmd = cmd[ jid == 2 ]
    #
    j0pos = pos[ jid == 0 ]
    j1pos = pos[ jid == 1 ]
    j2pos = pos[ jid == 2 ]
    #
    time = np.arange(0, trajDuration, trajDuration/len(j0cmd))
    #
    labels = [
        "j0ref",
        "j1ref",
        "j2ref",
        "j0pos",
        "j1pos",
        "j2pos",
    ]
    #
    data = np.array([
        j0cmd,
        j1cmd,
        j2cmd,
        j0pos,
        j1pos,
        j2pos
    ])
    #
    # Get Length of Trajectory
    idxes = np.logical_and(time >= starttime, time <= endtime)
    #
    # Return relevant part of trajectory data
    return time[idxes], data[:,idxes]

## Trajectory 0 files
traj0Filepaths = [
    "../HebiLogs/Trajectory/hebilog_Default_traj0/log_file_2022-06-03_16.40.15.csv",            # Default Parameters
    "../HebiLogs/Trajectory/hebilog_DDPG_Opt_traj0/log_file_2022-06-03_16.24.58.csv",           # DDPG Optimised
    "../HebiLogs/Trajectory/hebilog_PPOCont_Optimised_traj0/log_file_2022-06-03_16.35.47.csv"   # PPO Optimised
]

## Trajectory 1 files
traj1Filepaths = [
    "../HebiLogs/Trajectory/hebilog_Default_traj1/log_file_2022-06-03_16.42.33.csv",            # Default Parameters
    "../HebiLogs/Trajectory/hebilog_DDPG_Opt_traj1/log_file_2022-06-03_16.29.19.csv",           # DDPG Optimised
    "../HebiLogs/Trajectory/hebilog_PPOCont_Optimised_traj1/log_file_2022-06-03_16.39.29.csv"   # PPO Optimised
]

# Set up suptitles, colours and annotations for each cell
subplotTitles = [
    [ "Reference Trajectory", "Reference Trajectory" ],
    [ "Joint 0 Error", "Joint 0 Error" ],
    [ "Joint 1 Error", "Joint 1 Error" ],
    [ "Joint 2 Error", "Joint 2 Error" ]
]

errorColours = [
    colourScheme2["yellow"],    # Default
    colourScheme1["darkblue"],  # DDPG Optimised
    colourScheme1["twblue"]     # PPO Optimised
]

trajectoryColours = [
    colourScheme1["twblue"],   #J1
    colourScheme1["twgrey"],   #J2
    colourScheme2["yellow"]    #J3
]

plotColours = [
    [ trajectoryColours, trajectoryColours ],
    [ errorColours, errorColours ],
    [ errorColours, errorColours ],
    [ errorColours, errorColours ]
]

errorLinestyles = [
    "-", "-", ":"
]

trajectoryLinestyles = [
    "-", ":", "-."
]

plotLinestyles = [
    [trajectoryLinestyles, trajectoryLinestyles],
    [errorLinestyles, errorLinestyles],
    [errorLinestyles, errorLinestyles],
    [errorLinestyles, errorLinestyles]
]

errorLabels = [
    "Default", "DDPG", "PPO"
]

trajectoryLabels = [
    "J1", "J2", "J3"
]

plotLables = [
    [ trajectoryLabels, trajectoryLabels ],
    [ errorLabels, errorLabels ],
    [ errorLabels, errorLabels ],
    [ errorLabels, errorLabels ]
]

plotLineThickness = [
    [1.2, 1.2],
    [0.7, 0.7],
    [0.7, 0.7],
    [0.7, 0.7]
]

plotYLimits = [
    [ [-2.3, 1.3], [-2.3, 1.3] ],
    [ [-0.02, 0.02], [-0.02, 0.02] ],
    [ [-0.02, 0.02], [-0.02, 0.02] ],
    [ [-0.02, 0.02], [-0.02, 0.02] ]
]

plotYLimits = [
    [ [-2.3, 1.3], [-2.3, 1.3] ],
    [ [-0.01, 0.02], [-0.01, 0.02] ],
    [ [-0.02, 0.01], [-0.02, 0.01] ],
    [ [-0.01, 0.02], [-0.01, 0.02] ]
]

plotXLimits = [
    [ [4, 12], [4, 12] ],
    [ [4, 12], [4, 12] ],
    [ [4, 12], [4, 12] ],
    [ [4, 12], [4, 12] ]
]

## Load and process plot data
traj0Data = [ loadTrajLog(filename)[1] for filename in traj0Filepaths ]
traj1Data = [ loadTrajLog(filename)[1] for filename in traj1Filepaths ]

traj0Time = [ loadTrajLog(filename)[0] for filename in traj0Filepaths ]
traj1Time = [ loadTrajLog(filename)[0] for filename in traj1Filepaths ]


def getError(data, test, jid):
    return data[test][jid] - data[test][jid+3]

def getErrorBatch(data, jid):
    return np.array([ getError(data, test, jid) for test in range(3) ])


plotData = [
    [ [traj0Time, traj0Data[0][3:6]], [traj1Time, traj1Data[0][3:6]] ],
    [ [ traj0Time, getErrorBatch(traj0Data, 0) ], [ traj1Time, getErrorBatch(traj1Data, 0) ] ],
    [ [ traj0Time, getErrorBatch(traj0Data, 1) ], [ traj1Time, getErrorBatch(traj1Data, 1) ] ],
    [ [ traj0Time, getErrorBatch(traj0Data, 2) ], [ traj1Time, getErrorBatch(traj1Data, 2) ] ],
]


#for entry in plotData: print("{}  -  {}".format(entry[0][1].shape, entry[1][1].shape))
#for entry in plotData: print("{}  -  {}".format(entry[0][0].shape, entry[1][0].shape))

plt.rcParams.update({'font.size': 12})
fig, axs = plt.subplots(4, 2, sharey=False, sharex=True)


for col in range(4):
    for row in range(2):
        # Plot all shuffled data
        for i in range(3):
            axs[col][row].plot( plotData[col][row][0][i], plotData[col][row][1][i], c=plotColours[col][row][i], linewidth=plotLineThickness[col][row], ls=plotLinestyles[col][row][i], label=plotLables[col][row][i] )
        # Set suptitle
        axs[col][row].set_title(subplotTitles[col][row])
        # Set axis limits
        axs[col][row].set_ylim(plotYLimits[col][row])
        axs[col][row].set_xlim(plotXLimits[col][row])
        axs[col][row].grid()


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

fig.suptitle("Pick and Place Control Error", fontsize=16)
axs[0][0].legend()
axs[1][1].legend()

axs[0][0].set_ylabel("Position [rad]")
axs[1][0].set_ylabel("Control Error [rad]")
axs[2][0].set_ylabel("Control Error [rad]")
axs[3][0].set_ylabel("Control Error [rad]")

axs[3][0].set_xlabel("Time[sec]")
axs[3][1].set_xlabel("Time[sec]")

axs[3][0].set_xticks(np.arange(4,14,2), [0, 2, 4, 6, 8])
axs[3][1].set_xticks(np.arange(4,14,2), [0, 2, 4, 6, 8])

set_size(7,7)
plt.tight_layout()
plt.subplots_adjust(wspace = 0.21, bottom = 0.064, top = 0.914)

plt.savefig("/app/resultsRealRobotFeedbackPlot.pdf", bbox_inches='tight')
plt.show()

