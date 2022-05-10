
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

# Import agent implementations
from stable_baselines3 import PPO, DQN, DDPG
import torch as th
from stable_baselines3.common.logger import configure

# Start tensorboard to allow for visualisation of training process
os.system("tensorboard --logdir /training_tensorboard/ --host 0.0.0.0 --port 6006 &")


"""
THE PLAN:

write masterscript that keeps track of training processes

create dir for current experiment logs

-> start tensorboard
https://stable-baselines3.readthedocs.io/en/master/common/logger.html

-> open subprocesses for each training run
https://stackoverflow.com/questions/546017/how-do-i-run-another-script-in-python-without-waiting-for-it-to-finish

Then get all logs into pandas and do visualisations from there
https://tryolabs.com/blog/2017/03/16/pandas-seaborn-a-guide-to-handle-visualize-data-elegantly

TODO: how to check if training is done.. jobsystem?

"""


jointidx = 0
discretisation = 0
logPath = "/tmp/sb3_log/"

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

modelClass = DQN

trainingTimesteps = 10000

## Instantiate gym environment and wrap it with the discrete action wrapper
env = jointcontrolDiscrete(
    gym.make('jointcontrol-v0', jointidx=jointidx),
    discretisation=discretisation
)

env.reset(config={ "initialPos":0, "stepPos":-1.57, "samplesPerStep":150, "maxSteps":40 })



## Instantiate model and overwrite default logger
new_logger = configure(logPath, ["stdout", "csv", "tensorboard"])


model.set_logger(new_logger)


model.learn(total_timesteps=trainingTimesteps, tb_log_name="first_run")

















model = DQN("MlpPolicy", env, tensorboard_log="/training_tensorboard/", verbose=1)







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