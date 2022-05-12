import os
import sys
import time
sys.path.append("/catkin_ws/src/jointcontrol/scripts")

# Import gym, the custom gym environment and the discrete env wrapper
import gym
import gym_jointcontrol
from discreteActionWrapper import *

import numpy as np

# Import agent implementations
from stable_baselines3 import PPO, DQN, DDPG

from stable_baselines3.common.logger import Logger, configure
from stable_baselines3.common import utils
from stable_baselines3.common.evaluation import evaluate_policy

import torch as th

from multiprocessing import resource_tracker


""" RL agent and environment instantiation script """

# Load configuration from script args
assert len(sys.argv) > 1
config = eval(sys.argv[1])

# Set random seed
utils.set_random_seed(np.random.randint(0, 2**32-1))


## Instantiate gym environment
# if discretisation is set, wrap it with the discrete action wrapper
if config["discretisation"] != None:
    # Create discrete env
    env = jointcontrolDiscrete(
        gym.make('jointcontrol-v0', jointidx = config["jointID"]),
        discretisation = config["discretisation"]
    )
    # Check if trajectory type or reset config are provided and set them
    if config["resetConfig"] != None and config["trajectoryType"] != None:
        env.env.env.setTrajectoryConfig(
            episodeType = config["trajectoryType"],
            config = config["resetConfig"]
        )
else:
    # Create continuous env
    env = gym.make('jointcontrol-v0', jointidx = config["jointID"])
    # Check if trajectory type or reset config are provided and set them
    # This call is done here, because it is wrapped one layer deeper in the discrete env
    if config["resetConfig"] != None and config["trajectoryType"] != None:
        env.env.setTrajectoryConfig(
            episodeType = config["trajectoryType"],
            config = config["resetConfig"]
        )


## Instantiate model of correct class
modelclass = eval(config["modelclass"])

if config["policyNetwork"] != None and config["activation"] != None:
    # If custom network architecture is provided, initialise the model using that
    policy_kwargs = dict(
        activation_fn = eval(config["activation"]),
        net_arch = config["policyNetwork"]
    )
    model = modelclass("MlpPolicy", env, tensorboard_log="{}/tensorboard".format(config["logdir"]), policy_kwargs=policy_kwargs, verbose=1)
else:
    # Otherwise, initialise the default model
    model = modelclass("MlpPolicy", env, tensorboard_log="{}/tensorboard".format(config["logdir"]), verbose=1)

# If the modelclass is DQN, reduce the warmup phase from 50000 to 5000 steps
if modelclass == DQN: model.learning_starts = 5000

# Apply optional parameters
try:
    if modelclass == DDPG and config["tau"] != None: model.tau = config["tau"]
    if modelclass == PPO and config["epsilon"] != None: 
        def clip_range(_):
            return config["epsilon"]
        model.clip_range = clip_range
except KeyError:
    pass

# Configure learning rate if desired
if config["learningRate"] != None:
    model.learning_rate = config["learningRate"]

## Start training
model.learn(total_timesteps=config["trainingTimesteps"], tb_log_name="{}_{}".format(config["modelname"], config["modelrun"]))


## Perform Evaluation Episodes and store log
evalEpisodeRewards = []
evalEpisodeObservations = []

# The custom callback trakcs evaluation episode rewards and observations for controller parameter testing later
def evaluationCallback(localArgs, globalArgs):
    """ Custom callback to track controller parameters during test episodes """
    evalEpisodeRewards.append(localArgs["rewards"][0])
    evalEpisodeObservations.append(localArgs["observations"][0])


mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10, callback=evaluationCallback)

# Save evaluation episodes
with open("{}/testepisodes_{}_{}.npy".format(
    config["logdir"],
    config["modelname"],
    config["modelrun"]
), "wb") as f:
    np.save(f, np.array(
        {
            "model": config["modelname"],
            "run": config["modelrun"],
            "evalEpisodeRewards": evalEpisodeRewards,
            "evalEpisodeObservations": evalEpisodeObservations,
            "mean_reward": mean_reward,
            "std_reward": std_reward
        }
    ))

# Close environment to unregister and release shared memory
# Unregister from resource tracker in case we are not the server to prevent accidental cleanup
# https://bugs.python.org/issue39959#msg368770
# https://stackoverflow.com/questions/64102502/shared-memory-deleted-at-exit
if config["discretisation"] == None:
    resource_tracker.unregister(env.env.physicsCommand.shm._name, 'shared_memory')
    env.env.closeSharedMem()
else:
    resource_tracker.unregister(env.env.env.physicsCommand.shm._name, 'shared_memory')
    env.env.env.closeSharedMem()

