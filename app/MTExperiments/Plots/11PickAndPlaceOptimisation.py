

docker exec -it ros_ml_container1 bash

source ~/myenv/bin/activate
source /opt/ros/noetic/setup.bash
source /catkin_ws/devel/setup.bash

python

import subprocess
import os
import sys
import time
import itertools
import copy
sys.path.append("/catkin_ws/src/jointcontrol/scripts")
sys.path.append("/app/MTExperiments/TrainingScripts")

import numpy as np
from scipy import stats

from HebiUtils import *

""" 



"""

runTraining = False      # Determines, whether or not training is conducted or only visualisation is performed
trainingSteps = 15000   # Determines performed training steps

# Start tensorboard to allow for visualisation of training process
logdir = "/app/MTExperiments/Data/11PickAndPlaceOptimisation"
os.system("tensorboard --logdir {}/tensorboard --host 0.0.0.0 --port 6006 &".format(logdir))

# Point to RL agent deployment script
agentPath = "/app/MTExperiments/TrainingScripts/RLAgent.py"
randomAgentPath = "/app/MTExperiments/TrainingScripts/RandomAgent.py"


"""
pick up pose
[0.635761, -0.459321, -1.147022]


home pose
[0.068747, -1.772611, -0.514472]



put down pose
[-1.988591, -1.398532, -1.617373]


aux pose
[-0.719315, -2.459165, -2.211645]



resetConfig = { 
    "sampleRandom": True,
    "timePosSeries": [
        {"times": [0, 1.5, 3], "positions": [-1.147022, -0.514472, -1.147022]},
        {"times": [0, 1.5, 3], "positions": [-1.147022, -1.617373, -1.147022]},
        {"times": [0, 1.5, 3], "positions": [-0.514472, -1.617373, -0.514472]},
        {"times": [0, 1.5, 3], "positions": [-1.147022, -0.514472, -1.617373]}
    ],          
    "samplesPerStep": 350, 
    "maxSteps": 40 
}

"""

def createAllJointTrajectoryConfigs(poses, joint, cycleTime):
    """ Creates a configuration depicting all trajectories to move between poses. """
    arr = poses[:,joint]
    trajectories = []
    for subset in itertools.combinations(arr, 2):
        trajectories.append(list(subset))
    #
    trajectoriesrev = copy.deepcopy(trajectories)
    trajectories.reverse()
    trajectories += trajectoriesrev
    #
    # Format them with the correct cycle time as the env config
    timePosSeries = []
    for entry in trajectories:
        timePosSeries.append(
            {
                "times": np.arange(0, cycleTime*(len(entry)), cycleTime).tolist(),
                "positions": entry
            }
        )
    #
    return { 
        "sampleRandom": True,
        "timePosSeries": timePosSeries,
        "samplesPerStep": 180, 
        "maxSteps": 40 
    }

def createJointTrajectoryConfig(poses, joint, cycleTime):
    """ Creates a configuration depicting all trajectories to move between poses. """
    arr = poses[:,joint]
    trajectories = []
    for i in range(len(arr)):
        trajectories.append([arr[i-1], arr[i]])
    #
    # Format them with the correct cycle time as the env config
    timePosSeries = []
    for entry in trajectories:
        timePosSeries.append(
            {
                "times": np.arange(0, cycleTime*(len(entry)), cycleTime).tolist(),
                "positions": entry
            }
        )
    #
    return { 
        "sampleRandom": True,
        "timePosSeries": timePosSeries,
        "samplesPerStep": 180, 
        "maxSteps": 40 
    }


# CREATE PATH CONFIGURATIONS
poses = np.array([
    [0.635761, -0.459321, -1.147022],       # Pick up
    [0.068747, -1.772611, -0.514472],       # Home
    [-1.988591, -1.398532, -1.617373]       # Put down
])

#poses = np.array([
#    [0.635761, -0.459321, -1.147022],       # Pick up
#    [0.068747, -1.772611, -0.514472],       # Home
#    [-1.988591, -1.398532, -1.617373],      # Put down
#    [-0.719315, -2.459165, -2.211645]       # Aux
#])

#j0Config = createJointTrajectoryConfig(poses, 0, 1.5)
#j1Config = createJointTrajectoryConfig(poses, 1, 1.5)
#j2Config = createJointTrajectoryConfig(poses, 2, 1.5)


configs = [ createJointTrajectoryConfig(poses, j, 1.5) for j in range(3) ]


#############################################################################
## DDPG Continuous

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

#############################################################################
## PPO Continuous

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


# Debug: for one environment
#p = subprocess.Popen([sys.executable, agentPath, str(PPOContinuousConfig)])
#streamdata = p.communicate()[0]
#print(p.returncode)

#############################################################################
## RUN Training Episodes for each Algorithm

repeatRuns = 3
envJoints = 6
trainingConfigs = []

for run in range(repeatRuns):
    for idx, config in enumerate(configs):
        trainingConfigs.append(getDDPGConfig(idx, config, run))

for run in range(repeatRuns):
    for idx, config in enumerate(configs):
        trainingConfigs.append(getPPOConfig(idx, config, run))

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

# Iterate over all runs and apply them to the correct joint
if runTraining == True:
    for config in chunker(trainingConfigs, envJoints):
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
## Store joint feature, trajectory pairs for lookup during robot operation

def getSignalFeatures(arr):
    return np.array([
        min(arr),
        max(arr),
        np.mean(arr),
        stats.skew(arr),
        stats.kurtosis(arr)
    ])

cycleTime = 1.5
tmpconfigs = []

# Get single trajectory config for each joint and joint trajectory
if runTraining == True:
    for joint in range(3):
        arr = poses[:,joint]
        tmp = []
        for i in range(len(arr)):
            trajectory = [arr[i-1], arr[i]]
            #
            timePosSeries = [
                {
                    "times": np.arange(0, cycleTime*(len(trajectory)), cycleTime).tolist(),
                    "positions": trajectory
                }
            ]
            #
            tmp.append(
                { 
                    "sampleRandom": False,
                    "timePosSeries": timePosSeries,
                    "samplesPerStep": 180, 
                    "maxSteps": 40 
                }
            )
        #
        tmpconfigs.append(tmp)
    #
    # Generate feature trajectory pairs
    lookup = []
    #
    for joint, entry in enumerate(tmpconfigs):
        tmp = []
        for config in entry:
            res = createTrajectory(config, ts=0.01)
            controlSignal = res[:,0].flatten()
            features = getSignalFeatures(controlSignal)
            tmp.append( [controlSignal, features] )
        #
        lookup.append(tmp)
    #
    # Convert lookup to array and save it
    with open("{}/lookup.npy".format( logdir ), "wb") as f:
        np.save(f, np.array(
            lookup
        ))

# Load lookup
#lookup2 = np.load("{}/lookup.npy".format( logdir ), allow_pickle=True)


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

for config in trainingConfigs:
    # Load eval episode result dict
    rawEvalRewards = loadTestepisodeResults(config, testRuns=1)
    # Sum episodes up in order to match graph scale to the average return
    #episodeRewards = [ np.sum(segment) for segment in chunker(rawEvalRewards[0]["evalEpisodeRewards"], episodeLength) ]
    #evalEpisodeResults.append(episodeRewards)
    evalEpisodeResults.append(rawEvalRewards[0]["evalEpisodeRewards"])


## Determine default controller parameter performance for the test
sys.path.append("/catkin_ws/src/jointcontrol/scripts")
import gym
from discreteActionWrapper import *
from multiprocessing import resource_tracker

# Make environments for reference performance
envIDs = [6, 7, 8]
evaluationSteps = 400
defaultParamResults = []

for eID in envIDs:
    env = gym.make('jointcontrol-v0', jointidx=eID)
    envresults = []
    for ev in range(evaluationSteps):
        _ = env.reset(episodeType="generator", config=configs[eID-6])
        _, referenceReward, _, _ = env.step(None)
        envresults.append(referenceReward)
    defaultParamResults.append(envresults)
    resource_tracker.unregister(env.env.physicsCommand.shm._name, 'shared_memory')
    env.env.closeSharedMem()


## Set up plot (2 Plots side-by-side) both plotting per step reward with default parameter performance as a reference
# One plot also includes the random agent

linecolour = colourScheme1["darkblue"]
meadianlinecolour = colourScheme1["twblue"]
fillcolour = colourScheme1["lightblue"]
referenceColour = colourScheme2["yellow"]

boxwidth = 0.3

plt.rcParams.update({'font.size': 12})
fig, (ax0, ax1, ax2) = plt.subplots(1, 3, sharey=True)


# Get best ddpg and ppo performer
bestDDPGPerformer = []

for i in range(3):
    tmp = [ evalEpisodeResults[0+i], evalEpisodeResults[3+i], evalEpisodeResults[6+i] ]
    idx = np.argmax( [ np.mean(entry) for entry in tmp ] )
    bestDDPGPerformer.append(tmp[idx])

bestPPOPerformer = []

for i in range(3):
    tmp = [ evalEpisodeResults[9+i], evalEpisodeResults[12+i], evalEpisodeResults[15+i] ]
    idx = np.argmax( [ np.mean(entry) for entry in tmp ] )
    bestPPOPerformer.append(tmp[idx])


box0 = ax0.boxplot(
    defaultParamResults, notch=False, showfliers=False, patch_artist=True, widths=boxwidth*0.8,
    boxprops=dict(facecolor=fillcolour, color=linecolour),
    capprops=dict(color=linecolour),
    whiskerprops=dict(color=linecolour),
    flierprops=dict(color=fillcolour, markeredgecolor=linecolour),
    medianprops=dict(color=meadianlinecolour),
)

box1 = ax1.boxplot(
    bestDDPGPerformer, notch=False, showfliers=False, patch_artist=True, widths=boxwidth*0.8,
    boxprops=dict(facecolor=fillcolour, color=linecolour),
    capprops=dict(color=linecolour),
    whiskerprops=dict(color=linecolour),
    flierprops=dict(color=fillcolour, markeredgecolor=linecolour),
    medianprops=dict(color=meadianlinecolour),
)

box2 = ax2.boxplot(
    bestPPOPerformer, notch=False, showfliers=False, patch_artist=True, widths=boxwidth*0.8,
    boxprops=dict(facecolor=fillcolour, color=linecolour),
    capprops=dict(color=linecolour),
    whiskerprops=dict(color=linecolour),
    flierprops=dict(color=fillcolour, markeredgecolor=linecolour),
    medianprops=dict(color=meadianlinecolour),
)


# Format axes, set size and plot
ax0.grid()
ax1.grid()
ax2.grid()

ax0.set_ylabel("Negative Mean Squared Control Error")

ax0.set_title("Default Controller Parameters") 
ax1.set_title("DDPG")
ax2.set_title("PPO")

ax0.set_xticks([1, 2, 3], ['J1', 'J2', 'J3'])
ax1.set_xticks([1, 2, 3], ['J1', 'J2', 'J3'])
ax2.set_xticks([1, 2, 3], ['J1', 'J2', 'J3'])

fig.suptitle("Simulated Pick and Place Trajectory Control", fontsize=16)

set_size(7,3.6)
plt.tight_layout()

plt.subplots_adjust(top = 0.862)

plt.savefig("/app/resultsPickAndPlaceControllerOptimisation.pdf", bbox_inches='tight')
plt.show()











c2 = colourScheme1["darkblue"]
c1 = colourScheme2["twblue"]


commandsignal = [0] + [1.57 for i in range(149)]
simsignal = [ 1.57*i/150 for i in range(150) ]
time = np.array(range(0, 150))*0.01

fig, axs = plt.subplots(2, 2, sharey=True, sharex=True)
axs[0,0].plot(time, simsignal, label="Actuator Position", color=c1)
axs[0,0].plot(time, commandsignal, label="ReferenceSignal", color=c2)

axs[0,1].plot(time, simsignal, label="Actuator Position", color=c1)
axs[0,1].plot(time, commandsignal, label="ReferenceSignal", color=c2)

axs[1,0].plot(time, simsignal, label="Actuator Position", color=c1)
axs[1,0].plot(time, commandsignal, label="ReferenceSignal", color=c2)

axs[1,1].plot(time, simsignal, label="Actuator Position", color=c1)
axs[1,1].plot(time, commandsignal, label="ReferenceSignal", color=c2)

axs[1,0].set_ylabel("Time[sec]")
axs[0,0].set_ylabel("Time[sec]")

axs[1,0].set_xlabel("Position [rad]")
axs[1,1].set_xlabel("Position [rad]")

axs[1,1].legend()

plt.suptitle("Controller Parameter Estimation in Robot Operation ")

set_size(7,5)
plt.subplots_adjust(top = 0.936, wspace=0.07, hspace=0.0043)


plt.savefig("/app/dummy.pdf", bbox_inches='tight')
plt.show()









# Calculate percentual decrease in control error:
errorReductionPercent = []
for defaultEntry, entry in zip(defaultParamResults, evalEpisodeResults[3:6]):
    errorReductionPercent.append((1-(max(entry)/max(defaultEntry)))*100)

print("Best case error reduction [%]: {} ".format(errorReductionPercent))





