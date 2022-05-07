
# Set matplotlib to inline mode and import necessary python components
#%matplotlib inline
import matplotlib.pyplot as plt
#import matplotlib_inline.backend_inline
#matplotlib_inline.backend_inline.set_matplotlib_formats('svg')

import os
import sys
sys.path.append("/catkin_ws/src/jointcontrol/scripts")

# Import gym, the custom gym environment and the discrete env wrapper
import gym
import gym_jointcontrol
from discreteActionWrapper import *

# Import PPO2 implementation
from stable_baselines3 import PPO, DQN, DDPG
import torch as th

# Instantiate gym environment and wrap it with the discrete action wrapper
env = jointcontrolDiscrete(
    gym.make('jointcontrol-v0', jointidx=1),
    discretisation = 0.5
)


env.reset(config={ "initialPos":0, "stepPos":-1.57, "samplesPerStep":150000, "maxSteps":40 })


def cEnv(idx):
    return jointcontrolDiscrete(
        gym.make('jointcontrol-v0', jointidx=idx),
        discretisation = 0.5
    )


envs = [cEnv for i in range(5)]



import time
start_time = time.time()

env.step(None)

print("--- Step Time: {} Seconds ---".format((time.time() - start_time)/150))





env.env.env.visualiseTrajectory()






# Start tensorboard to allow for visualisation of training process
os.system("tensorboard --logdir /training_tensorboard/ --host 0.0.0.0 --port 6006 &")

# Instantiate model and start training
policy_kwargs = dict(
    activation_fn = th.nn.Tanh,
    net_arch=[ dict(pi=[ 32,32 ], vf=[ 32,32 ]) ]
)
model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, tensorboard_log="/training_tensorboard/", verbose=1)

# Change optimiser params
optimParams = model.policy.optimizer.state_dict()
optimParams["param_groups"][0]["lr"] = 0.0003
optimParams["param_groups"][0]["betas"] = (0.9, 0.999)
model.policy.optimizer.load_state_dict( optimParams )





model = PPO("MlpPolicy", env, tensorboard_log="/training_tensorboard/", verbose=1)



policy_kwargs = dict(
    activation_fn = th.nn.Tanh,
    net_arch=[ 32, 32, 32 ]
)
model = DQN("MlpPolicy", env, policy_kwargs=policy_kwargs, tensorboard_log="/training_tensorboard/", verbose=1)






model = DQN("MlpPolicy", env, tensorboard_log="/training_tensorboard/",verbose=1)



env = gym.make('jointcontrol-v0', jointidx=0)
model = DDPG("MlpPolicy", env, tensorboard_log="/training_tensorboard/",verbose=1)



model.learn(total_timesteps=10000)





















# Import controllerTest module from jointcontrol package
from controllerTest import *

# Reset environment, instantiate controllerTest and visualise results
env.reset()
test = controllerTest(env, model, "Discrete Proximal Policy Optimisation Agent")
test.plotResults(gui=True)