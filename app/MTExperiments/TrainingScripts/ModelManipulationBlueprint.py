import os
import sys
import time
sys.path.append("/catkin_ws/src/jointcontrol/scripts")

import gym
import gym_jointcontrol
from discreteActionWrapper import *

import numpy as np
import torch as th

from stable_baselines3 import PPO, DQN, DDPG
from stable_baselines3.common.evaluation import evaluate_policy



## Instantiate env and model
model_class = PPO

if model_class == DQN:
    env = jointcontrolDiscrete(
        gym.make('jointcontrol-v0', jointidx = 0),
        discretisation = 0.1
    )
else:
    env = gym.make('jointcontrol-v0', jointidx = 0) 



env.reset(config={ "initialPos":0, "stepPos":-1.57, "samplesPerStep":150, "maxSteps":40 })


#optimParams = {"param_groups": [{"lr":0.01}]}


kwargs = {
    "n_steps": 128
}


model = model_class("MlpPolicy", env, tensorboard_log="/training_tensorboard/", verbose=1, **kwargs)




# Close env
if model_class == DQN:
    env.env.env.closeSharedMem()
else:
    env.env.closeSharedMem()







## Set network architecture kwargs
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


## Change optimiser params
optimParams = model.policy.optimizer.state_dict()
optimParams["param_groups"][0]["lr"] = 0.0003
optimParams["param_groups"][0]["betas"] = (0.9, 0.999)
model.policy.optimizer.load_state_dict( optimParams )

























## Evaluation episodes with callback
evalEpisodeRewards = []
evalEpisodeObservations = []
def evaluationCallback(localArgs, globalArgs):
    """ Custom callback to track controller parameters during test episodes """
    evalEpisodeRewards.append(localArgs["rewards"][0])
    evalEpisodeObservations.append(localArgs["observations"][0])


mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=1, callback=evaluationCallback)


