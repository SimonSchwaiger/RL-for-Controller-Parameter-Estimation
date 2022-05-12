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

""" Random agent and environment instantiation script """
#NOTE: Random discrete agent is not yet implemented, so config["discretisation"] must be set to none

# Load configuration from script args
assert len(sys.argv) > 1
config = eval(sys.argv[1])

# Set random seed
utils.set_random_seed(np.random.randint(0, 2**32-1))
np.random.seed()


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

## Start "Training"
stepRewards = []
while True:
    done = False
    obs = env.reset()
    while not done:
        action = []
        for lower, upper in zip(env.action_space.low, env.action_space.high):
            action.append(
                ((upper-lower)*np.random.random())+lower
            )
        obs, reward, done, _ = env.step(action)
        stepRewards.append(reward)
        # Break out when we have enough steps
        if len(stepRewards) >= config["trainingTimesteps"]:
            break
    if len(stepRewards) >= config["trainingTimesteps"]:
        break

# Save training rewards
trainingRewardPath = "{}/tensorboard/{}_{}_1.npy".format(config["logdir"], config["modelname"], config["modelrun"])

with open(trainingRewardPath, "wb") as f:
    np.save(f, np.array(
        stepRewards
    ))

## Start evaluation episodes
evalEpisodeRewards = []
evalEpisodeObservations = []
mean_reward = []
std_reward = []


n_eval_episodes=10
for i in range(n_eval_episodes):
    episodeReward = []
    done = False
    obs = env.reset()
    while not done:
        action = []
        for lower, upper in zip(env.action_space.low, env.action_space.high):
            action.append(
                ((upper-lower)*np.random.random())+lower
            )
        obs, reward, done, _ = env.step(action)
        # Track all rewards and observations
        evalEpisodeRewards.append(reward)
        evalEpisodeObservations.append(obs)
        # Track reward separately in order to calculate mean and std_dev
        episodeReward.append(reward)
        # If episode is over, break out of the loop
        if done: break
    # Calculate mean reward of episode
    mean_reward.append(np.mean(episodeReward))

# Calcualte mean and std_dev over all eval episodes
std_reward = np.std(mean_reward)
mean_reward = np.mean(mean_reward)


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