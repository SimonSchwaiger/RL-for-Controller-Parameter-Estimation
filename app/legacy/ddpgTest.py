#!/bin/bash

# https://github.com/hill-a/stable-baselines

# https://stable-baselines.readthedocs.io/en/master/guide/tensorboard.html

import gym
import os
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG

import gym_jointcontrol
import rospy

import time

env = gym.make('jointcontrol-v0', jointidx=0)
env.reset()
# env.step([0,0,0,0,0,0,0,0,0])
# env.env.visualiseTrajectory()

# Start tensorboard to allow for a webUI to track training progress
os.system("tensorboard --logdir /training_tensorboard/ --host 0.0.0.0 --port 6006 &")

# the noise objects for DDPG
n_actions = env.action_space.shape[-1]
param_noise = None
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

model = DDPG(MlpPolicy, env, tensorboard_log="/training_tensorboard/", verbose=1, param_noise=param_noise, action_noise=action_noise)
model.learn(total_timesteps=10000)



model.save("ddpg_trajectorycontrol_test")

#del model # remove to demonstrate saving and loading


# Load Model
model = DDPG.load("ddpg_trajectorycontrol_test")

rewards = []

for e in range(100):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        rewards.append(reward)
        #env.render()
        #time.sleep(1/5)

plt.plot(
    [ sum(rewards[:idx]) for idx, _ in enumerate(rewards) ]
)

plt.show()



#############################################################
# TEST MODEL BENCHMARKING

import sys
import time
sys.path.append("/catkin_ws/src/jointcontrol/scripts")
from controllerTest import *

import gym
import gym_jointcontrol

env = gym.make('jointcontrol-v0', jointidx=0)
env.reset()

class dummyModel:
    def __init__(self) -> None:
        pass    
    #
    def predict(self, obs):
        return [0,0,0,0,0,0,0,0,0], None

model = dummyModel()

time.sleep(2)

test = controllerTest(env, model, "testModel")
test.plotResults(gui=False, outpath="/app")




#############################################################
# TEST DISCRETE WRAPPER

import sys
import time
sys.path.append("/catkin_ws/src/jointcontrol/scripts")
from discreteActionWrapper import *

env = jointcontrolDiscrete(gym.make('jointcontrol-v0', jointidx=0))
env.reset()


