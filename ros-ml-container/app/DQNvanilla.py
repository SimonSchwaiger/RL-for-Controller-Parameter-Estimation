"""
Experiment 1 of Open Loop Robot Control using Deep Q-Learning

Test vanilla, her and reward shaping and compare successful episodes and total culmulative reward.

Save culmulative reward and average maximal q-value for visualisation.
"""

plaidml = True
tensorflow = False

if plaidml:
    # setup plaidml as keras backend
    import os 
    os.environ['KERAS_BACKEND'] = "plaidml.keras.backend"
    # import keras
    from keras.models import Sequential
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
        self.goal_reach_count = 0                               # track how often a goal has been reached during an episode
        self.max_goalstate_distance = 0.08                      # maximum distance between goal and robot tcp for the goal to count as reached
        self.render=False                                       # render to rviz
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
        if reward == 1:
            if self.goal_reach_count >= 5: 
                done = True
            else: 
                self.goal_reach_count += 1
                done = False
            self.goalstate = self.generate_goalstate()
        # create state vector for q-network
        new_state = self.concat_state(new_state)
        # return in gym formatting
        return new_state, self.clip_reward(reward), done, prob
    def reset(self):
        self.goal_reach_count = 0
        new_state = self.env.reset()
        if self.render: self.env.render_state(new_state) 
        return self.concat_state(new_state)

# wrap fhtw3dof environment and set it up for use with the morobot
env = fhtw3dof_episodic(gym.make('fhtw3dof-v0', Stop_Early=False, Constrain_Workspace=False,
                    GoalX=0.0, GoalY=0.0, GoalZ=0.48,
                    J1Limit=31, J2Limit=31, J3Limit=31, joint_res = 0.1,
                    J1Scale=1, J2Scale=-1, J3Scale=-1, J3Offset=0.5,
                    Debug=True, Publish_Frequency=500, Goal_Max_Distance=0.1))

assert (env.reachable)

# agent
class DQN_Agent:
    def __init__(self, env):
        """ Class constructor """
        self.env     = env
        self.replay_buffer  = deque(maxlen=10000000)
        self.batch_size = 64                    # random sample size during replay
        self.gamma = 0.99                       # discount factor
        self.epsilon = 1.0                      # inital value of exploration factor epsilon
        self.epsilon_min = 0.01                 # minimum epsilon
        self.epsilon_decay = 0.9993             # rate at which epsilon is discounted
        self.learning_rate = 0.00025            # learning rate
        self.target_update = 5000               # target updates are only performed after len(replay_buffer>target_update)                 
        self.local_update = 1000                # local updates are only performed after len(replay_buffer>local_update)
        self.tau = .3                           # soft target network update parameter    
        self.nA = 6
        self.nS = 7
        self.local_model  = self.create_model() # create local and target models of similar architecture
        self.target_model = self.create_model()
        self.hard_target_train()                # synchronise weights of the two models
    def create_model(self):
        """ Returns compiled neural network used as the local and target models """
        kernel_init = 'random_uniform'
        bias_init = 'random_uniform'
        model = Sequential()
        model.add(Dense(
            64, 
            kernel_initializer=kernel_init, 
            bias_initializer=bias_init,
            input_shape=(self.nS,), 
            activation='relu'))
        model.add(Dense(
            64, 
            kernel_initializer=kernel_init, 
            bias_initializer=bias_init,
            activation='relu'))
        model.add(Dense(
            self.nA,
            kernel_initializer=kernel_init, 
            bias_initializer=bias_init,
            activation='linear'))
        model.compile(
            loss='mean_squared_error',
            optimizer=Adam(lr=self.learning_rate), 
            metrics=['accuracy']
        )
        print(model.summary())
        return model
    def update_epsilon(self):
        """ Decays epsilon until epsilon_min is reached """
        self.epsilon = max(self.epsilon*self.epsilon_decay, self.epsilon_min)
    def get_action(self, state):
        """ Returns an action to be performed. A random action is returned with epsilon probability, otherwise the local model is used for prediction """
        if np.random.random() < self.epsilon: return self.env.action_space.sample()         # decide whether or not to perform exploration
        return np.argmax(self.local_model.predict(state.reshape(1,self.nS)))                # get index of max q value for state
    def store(self, state, action, reward, new_state, done):
        """ Stores a transition in the agent's memory. """
        self.replay_buffer.append([state, action, reward, new_state, done])
    def replay(self):
        if len(self.replay_buffer) < self.batch_size: return                        # check if replay buffer is long enough
        if len(self.replay_buffer) < self.local_update: return                      # check if replay buffer is longer than the local update start parameter
        samples = random.sample(self.replay_buffer, self.batch_size)                # sample batch from replay memory
        current_states = np.array([transition[0] for transition in samples])        # get lists of current and future states from samples
        future_states = np.array([transition[3] for transition in samples])
        current_states = current_states.reshape(self.batch_size,self.nS)            # correctly reshape states to (batch_size, nS)
        future_states = future_states.reshape(self.batch_size,self.nS)
        current_q_list = self.local_model.predict(current_states)                   # let the local model predict the q values from current states
        future_q_list = self.target_model.predict(future_states)                    # let the target model predict the q values from future states
        arr1 = []
        arr2 = []
        # loop through samples and get index and individual entries
        for idx, (state, action, reward, new_state, done) in enumerate(samples):
            if not done: future_q_max = self.gamma * np.max(future_q_list[idx]) + reward # calculate future q value using bellman equation, if episode was not over
            else: future_q_max = reward                                                  # reward = future q value, if episode was over
            # get current q value and update data on the taken action
            current_q = current_q_list[idx]
            current_q[action] = (1 - self.learning_rate) * current_q[action] + self.learning_rate * future_q_max
            arr1.append(state.reshape(self.nS))          # store state and updated q values for training to be done in one go
            arr2.append(current_q)
        # fit the local model using the updated q values
        self.local_model.fit(np.array(arr1), np.array(arr2), batch_size=self.batch_size, verbose=0, shuffle=True)
    def soft_target_train(self):
        """ Softly changes the target model's weights towards those from the local model """
        if len(self.replay_buffer) < self.target_update: return
        weights = self.local_model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)  # perform soft update of the target network
        self.target_model.set_weights(target_weights)
        print("Target network updated.")
    def hard_target_train(self):
        """ Sets the target model's weights to those from the local (main) model """
        if len(self.replay_buffer) < self.target_update: return
        self.target_model.set_weights(self.local_model.get_weights())
        print("Target network updated.")

# instantiate agent
agent = DQN_Agent(env=env)

# set parameters and train agent
debug = False

max_episodes = 2000
max_steps = 500
local_model_update_rate = 4
target_model_update_rate = 550

total_reward = []
total_q = []

steps_since_update = 0

for episode in range(max_episodes):
    # track total reward and average q_value for testing
    episode_reward = 0
    episode_q = 0
    # reset environment
    state = env.reset()
    for step in range(max_steps):
        # get action from dqn
        action = agent.get_action(state)
        # perform action
        new_state, reward, done, _ = env.step(action)
        # store step
        agent.store(state, action, reward, new_state, done)
        # update local model every n steps
        if step % local_model_update_rate == 0 or done: agent.replay()
        # set new state as state and break if the episode is done
        state = new_state
        # track reward and q value 
        episode_reward += reward
        episode_q += max(agent.local_model.predict(state.reshape(1,agent.nS))[0])
        # if done, then end episode
        if done:
            break
    print("Episode {} finished after {} steps with a reward of {}".format(episode, step, episode_reward))
    # update the target model periodically at a slower pace compared to the local model
    steps_since_update += step
    if steps_since_update >= target_model_update_rate: 
        agent.soft_target_train()
        steps_since_update = 0
    # decay epsilon
    agent.update_epsilon()
    # track total reward and max q values
    total_reward.append(episode_reward)
    total_q.append(episode_q)


# plot training results
total_reward = np.array(total_reward)
total_q = np.array(total_q)

print("Number of successful episodes: {} out of {} ({}%)".format(np.count_nonzero(total_reward>=0), (episode+1), 100*np.count_nonzero(total_reward>=0)/(episode+1)))
print("Average reward per episode: {}, average q value per episode: {}".format(np.sum(total_reward)/(episode+1), np.sum(total_q)/(episode+1)))

cul_reward = []
for i in range(len(total_reward)):
    cul_reward.append(np.sum(total_reward[:i]))

cul_reward = np.array(cul_reward)

print("Culmulative reward on {}th episode is {}".format(episode+1, cul_reward[len(cul_reward)-1]))

fig, axs = plt.subplots(2)
axs[0].plot(cul_reward)
axs[0].set_xlabel('Number of Episodes')
axs[0].set_ylabel('Culmulative Reward per Episode')

axs[1].plot(total_q)
axs[1].set_xlabel('Number of Episodes')
axs[1].set_ylabel('Average maximum Q-Value per Episode')
plt.show()

# save models for later use
agent.local_model.save('localModel')
agent.target_model.save('targetModel')

