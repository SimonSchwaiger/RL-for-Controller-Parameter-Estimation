
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

""" 
Control script for experiment 4 

DQN, DDPG, PPO and Random Agents trained to perform step responses from 0 to -1.57 rad

Plots include average return of training and separate testepisodes

Logs are stored in /app/4logs


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
logdir = "/app/MTExperiments/Data/4StepResponseTrainingDefaultParams"
os.system("tensorboard --logdir {}/tensorboard --host 0.0.0.0 --port 6006 &".format(logdir))

# Point to RL agent deployment script
agentPath = "/app/MTExperiments/TrainingScripts/RLAgent.py"


#############################################################################
## DQN Discrete

DQNConfig = {
    "jointID": 0,
    "logdir": logdir,
    "modelclass": "DQN",
    "modelrun": 0,
    "modelname": "DQN",
    "learningRate": None, 
    "trainingTimesteps": trainingSteps,
    "policyNetwork": None,
    "optimizer": None,
    "discretisation": 0.1,
    "resetConfig": { "initialPos":-1.57, "stepPos":0, "samplesPerStep":150, "maxSteps":40 },
    "trajectoryType": "step"
}


#############################################################################
## DDPG Continuous

DDPGConfig = {
    "jointID": 0,
    "logdir": logdir,
    "modelclass": "DDPG",
    "modelrun": 0,
    "modelname": "DDPG",
    "learningRate": None, 
    "trainingTimesteps": trainingSteps,
    "policyNetwork": None,
    "optimizer": None,
    "discretisation": None,
    "resetConfig": { "initialPos":-1.57, "stepPos":0, "samplesPerStep":150, "maxSteps":40 },
    "trajectoryType": "step"
}

#############################################################################
## PPO Discrete

PPODiscreteConfig = {
    "jointID": 0,
    "logdir": logdir,
    "modelclass": "PPO",
    "modelrun": 0,
    "modelname": "PPO_Discrete",
    "learningRate": None, 
    "trainingTimesteps": trainingSteps,
    "policyNetwork": None,
    "optimizer": None,
    "discretisation": 0.1,
    "resetConfig": { "initialPos":-1.57, "stepPos":0, "samplesPerStep":150, "maxSteps":40 },
    "trajectoryType": "step"
}

#############################################################################
## PPO Continuous

PPOContinuousConfig = {
    "jointID": 0,
    "logdir": logdir,
    "modelclass": "PPO",
    "modelrun": 0,
    "modelname": "PPOContinuous",
    "learningRate": None, 
    "trainingTimesteps": trainingSteps,
    "policyNetwork": None,
    "optimizer": None,
    "discretisation": None,
    "resetConfig": { "initialPos":-1.57, "stepPos":0, "samplesPerStep":150, "maxSteps":40 },
    "trajectoryType": "step"
}


# Debug: for one environment
#p = subprocess.Popen([sys.executable, agentPath, str(PPOContinuousConfig)])
#streamdata = p.communicate()[0]
#print(p.returncode)

#############################################################################
## RUN Training Episodes for each Algorithm

def runAgentTests(config, testRuns=5):
    """ Performs testRuns """
    processes = [None for _ in range(testRuns)]
    #
    for i in range(testRuns):
        # Start process in the background
        config["jointID"] = i
        config["modelrun"] = i
        processes[i] = subprocess.Popen([sys.executable, agentPath, str(config)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        # Wait for env to register
        time.sleep(1)
    #
    # Wait for processes to finish and record return codes
    rc = []
    for i in range(testRuns):
        streamdata = processes[i].communicate()[0]
        rc.append(processes[i].returncode)
    #
    print("{} agent tests successfully exited with return codes {}".format(config["modelname"], rc))

# Perform testruns of all configurations
if runTraining:
    runAgentTests(DQNConfig)
    runAgentTests(DDPGConfig)
    runAgentTests(PPODiscreteConfig)
    runAgentTests(PPOContinuousConfig)






#############################################################################
## VISUALISE training results

# Load numpy results
def loadTestepisodeResults(config, testRuns=5):
    """ Loads evaluation episode results based on configuration """
    ret = []
    for run in range(testRuns):
        filename = "{}/testepisodes_{}_{}.npy".format(config["logdir"], config["modelname"], run)
        ret.append(np.load(filename), allow_pickle=True).item()
    #
    return ret


# 
#testepisodeResults = np.load('/app/MTExperiments/Data/4StepResponseTrainingDefaultParams/testepisodes_DQN_0.npy', allow_pickle=True).item()

