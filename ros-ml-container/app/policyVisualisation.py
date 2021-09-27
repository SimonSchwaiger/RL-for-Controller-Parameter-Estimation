#!/usr/bin/env python3
import numpy as np
import pandas as pd
import gym
import gym_fhtw3dof

from geometry_msgs.msg import Point

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.cm as cm
from matplotlib.colors import ListedColormap

import fhtw3dof_usecase_tools
policy_iteration = fhtw3dof_usecase_tools.iterations()

#https://medium.com/@prasadostwal/multi-dimension-plots-in-python-from-2d-to-6d-9a2bf7b8cc74
#https://towardsdatascience.com/visualizing-three-dimensional-data-heatmaps-contours-and-3d-plots-with-python-bd718d1b42b4
#https://stackoverflow.com/questions/17682216/scatter-plot-and-color-mapping-in-python
#https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/scatter_with_legend.html

env = gym.make('fhtw3dof-v0', Stop_Early=False, Constrain_Workspace=False,
                GoalX=0.179, GoalY=0.0026, GoalZ=0.335,
                J1Limit=15, J2Limit=25, J3Limit=31, joint_res = 0.1,
                J1Scale=1, J2Scale=-1, J3Scale=-1, J3Offset=0.5,
                Debug=True, Publish_Frequency=500)

# define goal state
goal_state = 2000

env.s = goal_state
env.render_state(env.s)

# set initial state distribution to all states
env.isd.fill(0)
for s in range(env.nS):
    if env.Colliding_States[s] == False: env.isd[s] = 1
    else: env.isd[s] = 0


env.isd /= env.isd.sum()
print("{} initial states set".format(np.count_nonzero(env.isd)))

# set state rewards
env.P = env.reset_rewards(env.P, 0, env.nS)
env.P, idx = env.set_state_reward(env.P, goal_state)
assert(idx > 1)

policy = policy_iteration.policy_iteration(env)




# loop through states
# look up tcp position of state
# look up tcp position of resulting state
# distance new - distance old
# end loop

# fit range of distances to color gradient
# plot tcp position of each point with color based on distance

# create array to store distances
TCP_X = []
TCP_Y = []
TCP_Z = []
distances = []

for s in range(env.nS):
    # get action for current state from policy
    action = policy[s]
    # get jointstates of current state
    j3, j2, j1 = env.from_index(s)
    # apply policy
    j3, j2, j1 = env.increment(j3, j2, j1, action)
    # get resulting state
    newstate = env.to_index(j3, j2, j1)
    # check if the state is stored (required, when constrain workspace or stop early are used)
    try:
        # store the difference in distance to goal from one state to the next
        distances.append(env.Distance[s] - env.Distance[newstate])
        # convert the tcp position to regular lists
        TCP_X.append(env.TCP_Positions[s].x)
        TCP_Y.append(env.TCP_Positions[s].y)
        TCP_Z.append(env.TCP_Positions[s].z)
    except TypeError:
        continue
    except KeyError:
        break

# Get color map normalisation for the range of distances
norm = plt.Normalize(vmin=min(distances), vmax=max(distances))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(TCP_X, TCP_Y, TCP_Z, c=distances, cmap='hsv')

plt.show()








def track_states(policy, target_state, env):
    """ Performs a learned policy and counts which states were encountered. """
    obs = env.s
    step_idx = 0
    step_tracker = np.zeros_like(policy)
    while True:
        #break if goal or max_step size are reached
        if obs == target_state: break
        elif step_idx >= 250: break
        #track how often the robot was in each state
        step_tracker[obs] += 1
        #get action
        action = int(policy[obs])
        #take action
        obs, _, _, _ = env.step(action)
        step_idx += 1
        #env.render_state(obs)
    return step_tracker

states = np.zeros_like(policy)
TCP_X = np.zeros_like(policy, dtype=float)
TCP_Y = np.zeros_like(policy, dtype=float)
TCP_Z = np.zeros_like(policy, dtype=float)

# start from any valid state, apply policy and track how often the robot was in each state
for s in range(env.nS):
    try:
        if (env.Colliding_States[s] == False):
            env.reset()
            env.s = s
            #env.render_state(env.s)
            states += track_states(policy, goal_state, env)
    except TypeError:
        continue
    except KeyError:
        break

# convert TCP positions to numpy arrays
for s in range(env.nS):
    try:
        TCP_X[s] = env.TCP_Positions[s].x
        TCP_Y[s] = env.TCP_Positions[s].y
        TCP_Z[s] = env.TCP_Positions[s].z
    except AttributeError:
        continue

newstates = states/np.std(states)
newstates = np.clip(states, 0, 1000)

matrix = np.array([TCP_X, TCP_Y, TCP_Z, newstates])
dataframe = pd.DataFrame(matrix.T)
dataframe.columns = ['TCP_X', 'TCP_Y', 'TCP_Z', 'Num_Encounters']

# replace 0 encounters with nan
dataframe['Num_Encounters'].replace(0, np.nan, inplace=True)
# drop all entries containing nan
dataframe.dropna(subset=['Num_Encounters'], inplace=True)



# create custom color map
cmap = plt.cm.PiYG
# Get the colormap colors
my_cmap = cmap(np.arange(cmap.N))
# Set alpha
my_cmap[:,-1] = np.linspace(0.4, 1, cmap.N)
# Create new colormap
my_cmap = ListedColormap(my_cmap)




fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

axis_min = -0.4
axis_max = 0.4

ax.set_xlim([axis_min,axis_max])
ax.set_ylim([axis_min,axis_max])
ax.set_zlim([axis_min,axis_max])

ax.set_xlabel("x-position[m]")
ax.set_ylabel("y-position[m]")
ax.set_zlabel("z-position[m]")

img = ax.scatter(dataframe['TCP_X'], dataframe['TCP_Y'], dataframe['TCP_Z'], c=dataframe['Num_Encounters'], cmap=my_cmap)
fig.colorbar(img)
plt.show()



