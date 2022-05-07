

###################################################################
#  TRAINING SETUP BLUEPRINT

import os
import time
import sys
sys.path.append("/catkin_ws/src/jointcontrol/scripts")

# Import controllers
from controllers import *

# Import gym, the custom gym environment and the discrete env wrapper
import gym
import gym_jointcontrol
from discreteActionWrapper import *

# Import agent implementations
from stable_baselines3 import DDPG, DQN, PPO2
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


env = jointcontrolDiscrete(
    gym.make('jointcontrol-v0', jointidx=0),
    discretisation = 0.5
)


time.sleep(4)
env.reset()




###################################################################
#  FIGURE 1: step responses of all approximated elements




## Imports
import matplotlib.pyplot as plt
import numpy as np

# Import discrete controllers
import sys
sys.path.append("/catkin_ws/src/jointcontrol/scripts")
from controllers import *

# Import scipy signal for continuous controllers
from scipy import signal


def clip(s, lower=-1, upper=1):
    if s < upper: return max(lower, s)
    return upper


# Set simulation time and discretisation
simTime = 3
ts = 1/100
samples = int(simTime*(1/ts))
# Set up time, input and feedback signals
time = [ i*ts for i in range(samples) ]
inSignal = [[0, 0, 0]] + [ [-1.57, 0, 0] for _ in range(samples-1) ]
feedback = [ [0, 0, 0] for _ in range(samples) ]
# Calculate output signal
block = strategy4Controller(
    ts=ts,
    constants = [30, 0, 0, 0.05, 0, 0, 0.25, 0, 0.001]
)
t1 = PT1Block(kp=1, T1=1, ts=ts)
PT2 = PT2Block(T=0.15, D=1, kp=1, ts=ts)
#result = [ PT2.update(block.update(s, f)) for s, f in zip(inSignal, feedback) ]
result = [ PT2.update(clip(block.update(s, f))) for s, f in zip(inSignal, feedback) ]
# Plot everything
plt.plot(time, np.array(inSignal)[:,0], label="Reference Signal", linestyle=":")
plt.plot(time, result, label="Actuator Output")
plt.xlabel("Time [s]")
plt.ylabel("Actuator Position [radians]")
plt.legend()
plt.show()




###################################################################
#  FIGURE 1: step responses of all approximated elements




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
    gym.make('jointcontrol-v0', jointidx=2),
    discretisation = 0.5
)
