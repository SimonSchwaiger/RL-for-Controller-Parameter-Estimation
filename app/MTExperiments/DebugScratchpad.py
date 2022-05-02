

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





# Set simulation time and discretisation
simTime = 2
ts = 1/100
samples = int(simTime*(1/ts))
# Set up time, input and feedback signals
time = [ i*ts for i in range(samples) ]
inSignal = [[0, 0, 0]] + [ [1, 1, 1] for _ in range(samples-1) ]
feedback = [ [0, 0, 0] for _ in range(samples) ]
# Calculate output signal
block = strategy4Controller(
    ts=ts,
    constants = [30, 0, 0, 0.05, 0, 0, 0.25, 0, 0.001]
)
PT2 = PT2Block(T=0.1, D=1, kp=0.5, ts=ts)
#result = [ PT2.update(block.update(s, f)) for s, f in zip(inSignal, feedback) ]
result = [ block.update(s, f) for s, f in zip(inSignal, feedback) ]
# Plot everything
plt.plot(time, np.array(inSignal)[:,0], label="Reference Signal", linestyle=":")
plt.plot(time, result, label="Actuator Output")
plt.xlabel("Time [s]")
plt.ylabel("Actuator Position [radians]")
plt.legend()
plt.show()









