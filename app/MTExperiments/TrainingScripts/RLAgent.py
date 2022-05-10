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

""" DQN agent and instantiation script """

# Load configuration from script args
assert len(sys.argv) > 1
config = eval(sys.argv[1])


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


# Configure logging for stdout, csv (as a backup) and tensorboard
#new_logger = configure(config["logdir"], ["stdout", "csv", "tensorboard"])
#model.set_logger(new_logger)

# Configure optimiser parameters if desired
if config["learningRate"] != None:
    optimParams = model.policy.optimizer.state_dict()
    optimParams["param_groups"][0]["lr"] = config["learningRate"]
    model.policy.optimizer.load_state_dict( optimParams )


## Start training
model.learn(total_timesteps=config["trainingTimesteps"], tb_log_name="{}_{}".format(config["modelname"], config["modelrun"]))

## Perform Evaluation Episodes and store log
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

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
            "mean_reward": mean_reward,
            "std_reward": std_reward
        }
    ))

# Close environment to unregister and release shared memory
# Unregister from resource tracker in case we are not the server to prevent accidental cleanup
# https://bugs.python.org/issue39959#msg368770
# https://stackoverflow.com/questions/64102502/shared-memory-deleted-at-exit
if config["discretisation"] != None:
    resource_tracker.unregister(env.env.physicsCommand.shm._name, 'shared_memory')
    env.env.closeSharedMem()
else:
    resource_tracker.unregister(env.env.env.physicsCommand.shm._name, 'shared_memory')
    env.env.env.closeSharedMem()

