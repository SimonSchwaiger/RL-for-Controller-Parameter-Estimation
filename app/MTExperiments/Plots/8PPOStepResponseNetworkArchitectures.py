
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

runTraining = True      # Determines, whether or not training is conducted or only visualisation is performed
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


# Store configs
widthTestConfigs = copy.deepcopy(configs)