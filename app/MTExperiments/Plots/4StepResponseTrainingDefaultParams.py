
docker exec -it ros_ml_container1 bash

source ~/myenv/bin/activate
source /opt/ros/noetic/setup.bash
source /catkin_ws/devel/setup.bash

python



import subprocess
import os
import sys
sys.path.append("/catkin_ws/src/jointcontrol/scripts")

import numpy as np
from stable_baselines3.common import utils

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


# PPO & DDPG type policy args
policy_kwargs = dict(
    activation_fn = th.nn.Tanh,
    net_arch=[ dict(pi=[ 32,32 ], vf=[ 32,32 ]) ]
)

# DQN type policy args
policy_kwargs = dict(
    activation_fn = th.nn.Tanh,
    net_arch=[ 32, 32, 32 ]
)

"""

# Start tensorboard to allow for visualisation of training process
logdir = "/app/MTExperiments/Data/4StepResponseTrainingDefaultParams"
os.system("tensorboard --logdir {} --host 0.0.0.0 --port 6006 &".format(logdir))

agentPath = "/app/MTExperiments/TrainingScripts/RLAgent.py"

# Test DQN
config = {
    "jointID": 0,
    "logdir": logdir,
    "modelclass": "DQN",
    "modelrun": 0,
    "modelname": "DQN",
    "learningRate": None, 
    "trainingTimesteps": 10,
    "policyNetwork": None,
    "optimizer": None,
    "discretisation": 0.1,
    "resetConfig": { "initialPos":0, "stepPos":-1.57, "samplesPerStep":150, "maxSteps":40 },
    "trajectoryType": "step"
}


# Start process in the background
p = subprocess.Popen([sys.executable, agentPath, str(config)])


#TODO fix environment deletion resetting shared memory
# 
# https://bugs.python.org/issue38119#msg388287
# https://stackoverflow.com/questions/62748654/python-3-8-shared-memory-resource-tracker-producing-unexpected-warnings-at-appli





# Set random seed
utils.set_random_seed(np.random.randint(0, 2**32-1))


streamdata = p.communicate()[0]
rc = p.returncode

print("successfully done {}".format(rc))


