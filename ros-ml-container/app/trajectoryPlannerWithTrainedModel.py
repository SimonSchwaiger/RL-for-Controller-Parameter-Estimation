

plaidml = True
tensorflow = False

if plaidml:
    # setup plaidml as keras backend
    import os 
    os.environ['KERAS_BACKEND'] = "plaidml.keras.backend"
    # import keras
    from keras.models import Sequential, load_model
    from keras.layers import Dense, Dropout, Flatten, Activation
    from keras.optimizers import Adam, RMSprop

if tensorflow:
    # import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
    from tensorflow.keras.optimizers import Adam, RMSprop

from collections import deque

import gym
import gym_fhtw3dof

import numpy as np
import random
import math

import matplotlib
import matplotlib.pyplot as plt

from random import randrange

# set random internal numpy seed
np.random.seed()

# wrapped fhtw3dof environment
class fhtw3dof_episodic(gym.Wrapper):
    def __init__(self, env):
        """ Class constructor, inherits fhtw3dof environment """
        super().__init__(env)
        self.env = env
        self.max_goalstate_distance = 0.08                      # maximum distance between goal and robot tcp for the goal to count as reached
        self.render=True                                      # render to rviz
        self.env.P = self.env.reset_rewards(env.P, 0, env.nS)
        self.goalstate = 0
        self.goalstate = self.generate_goalstate()
    def generate_goalstate(self):
        """ Returns a valid goalstate, that can be reached by the robot """
        for i in range(1000):
            try:
                rand = np.random.randint(0, self.env.nS)
                if self.env.Colliding_States[rand]: continue
                else: 
                    # get goalstate position
                    point = self.env.TCP_Positions[rand]
                    # discard goalstate if x<0, to avoid edgecases during training
                    if point.x < 0.04: continue
                    # discard goalstate if it is at the edge of the workspace
                    j3, j2, j1 = self.env.from_index(rand)
                    if (j3 >= self.env.joint_state_size-1) or (j2 >= self.env.joint_state_size-1) or (j1 >= self.env.joint_state_size-1): continue
                    if (j3 <= 1) or (j2 <= 1) or (j1 <= 1): continue
                    # discard goalstate if the it would already be reached
                    if self.get_goalstate_distance(rand) <= self.max_goalstate_distance: continue
                    #visualise new goal
                    self.env.publish_point(point)
                    # get states near the goalstate and set env.P accordingly
                    states = self.env.find_near_states(point, self.max_goalstate_distance)
                    self.env.P = self.env.reset_rewards( self.env.P, 0, self.env.nS)
                    self.env.P, _ = self.env.set_state_rewards(self.env.P, states)
                    return rand
            except (ValueError, AttributeError, KeyError):
                continue
        return -1
    def set_goalstate(self, rand):
        """ Sets the goalstate to rand, returns -1 if it's not reachable, returns rand on success """
        try:
            if self.env.Colliding_States[rand]: return -1
            else: 
                # get goalstate position
                point = self.env.TCP_Positions[rand]
                # visualise new goal
                self.env.publish_point(point)
                # get states near the goalstate and set env.P accordingly
                states = self.env.find_near_states(point, self.max_goalstate_distance)
                self.env.P = self.env.reset_rewards( self.env.P, 0, self.env.nS)
                self.env.P, _ = self.env.set_state_rewards(self.env.P, states)
                return rand
        except (ValueError, AttributeError):
            pass
        return -1
    def get_goalstate_distance(self, state):
        """ Returns the distance between the goalstate and robot tcp """
        try:
            x = self.env.TCP_Positions[self.goalstate].x - self.env.TCP_Positions[state].x
            y = self.env.TCP_Positions[self.goalstate].y - self.env.TCP_Positions[state].y
            z = self.env.TCP_Positions[self.goalstate].z - self.env.TCP_Positions[state].z
        except AttributeError:
            return -1
        else:
            return round(math.sqrt(x**2 + y**2 + z**2), 6)
    def get_goal_vector(self, state):
        """ Returns the vector pointing from robot tcp to goal """
        try:
            x = self.env.TCP_Positions[self.goalstate].x - self.env.TCP_Positions[state].x
            y = self.env.TCP_Positions[self.goalstate].y - self.env.TCP_Positions[state].y
            z = self.env.TCP_Positions[self.goalstate].z - self.env.TCP_Positions[state].z
        except AttributeError:
            return np.array([0,0,0])
        else: 
            return np.array([x, y, z])
    def normalise_vector(self, vector):
        """ Returns the input vector divided by its norm """
        norm = np.linalg.norm(vector)
        if norm == 0: return vector
        else: return vector / norm
    def concat_state(self, state):
        """ Returns tuple of state and distance to goalstate, reshaped to (1,7) for proper processing in keras """
        # get data for state and convert everything to the same range
        j3, j2, j1 = self.env.from_index(state)                         # jointsate as indicies
        arr1 = np.array([j1, j2, j3], dtype=float)/env.joint_state_size # convert to range 0 - 1
        arr2 = self.normalise_vector(self.get_goal_vector(state))       # normalised vector pointing from tcp to goal, converted to range 0 - 1
        distance = self.get_goalstate_distance(state)
        arr3 = np.array([distance])                                     # distance to goal in m
        new_state = np.concatenate((arr1, arr2, arr3), axis=None)
        return new_state.reshape(1,7)
    def clip_reward(self, reward):
        """ Clips the reward to minimum and maximum values """
        if reward > 5: return 5
        elif reward < -1: return -1
        else: return reward
    def step(self, action):
        # take action
        new_state, reward, done, prob = self.env.step(action)
        # render to rviz if render flag is true
        if self.render: self.env.render_state(new_state)
        # check if step was successful and generate a new goal if required
        # if there are 5 successes in an episode, the episode is done
        if reward == 1: done = True
        # create state vector for q-network
        new_state = self.concat_state(new_state)
        # return in gym formatting
        return new_state, self.clip_reward(reward), done, prob
    def reset(self):
        new_state = self.env.reset()
        if self.render: self.env.render_state(new_state) 
        return self.concat_state(new_state)

# wrap fhtw3dof environment and set it up for use with saimon
env = fhtw3dof_episodic(gym.make('fhtw3dof-v0', Stop_Early=False, Constrain_Workspace=False,
                    GoalX=0.0, GoalY=0.0, GoalZ=0.48,
                    J1Limit=31, J2Limit=31, J3Limit=31, joint_res = 0.1,
                    J1Scale=1, J2Scale=-1, J3Scale=-1, J3Offset=0.5,
                    Debug=True, Publish_Frequency=500, Goal_Max_Distance=0.1))


# load trajectory planning model
model = load_model('targetModel')

# set movement parameters
goalState = 18885
maxSteps = 500

# set goal and reset robot to random position
goal = env.set_goalstate(goalState)
state = env.reset()

# track states and actions
states = []
states.append(state)
actions = []

for step in range(maxSteps):
    # let model predict the most beneficial action to take
    action = np.argmax(model.predict(state.reshape(1,7)))
    # take action
    state, reward, done, _ = env.step(action)
    # track states and actions
    states.append(state)
    actions.append(action)
    # check if goal has been reached
    if done: break


