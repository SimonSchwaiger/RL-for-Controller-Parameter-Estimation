import csv
import os

import numpy as np
import matplotlib.pyplot as plt



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


def plotTrajLog(filename, trajDuration=13.5, start=16):
    """  """
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
    data = [
        j0cmd,
        j1cmd,
        j2cmd,
        j0pos,
        j1pos,
        j2pos
    ]
    #
    linetypes = [
        "-",
        "-",
        "-",
        ":",
        ":",
        ":",
    ]
    #
    for i in range(len(data)):
        plt.plot(time, data[i], label=labels[i], linestyle=linetypes[i])
    #
    plt.suptitle("{}".format(filename))
    plt.grid()
    plt.legend()
    plt.xlabel = "Time [sec]"
    plt.ylabel = "Actuator Positon [rad]"
    plt.show()


for filename in files:
    plotTrajLog(filename)

