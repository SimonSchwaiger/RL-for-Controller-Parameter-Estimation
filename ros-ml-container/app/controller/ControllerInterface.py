# Pybullet components
import pybullet as p
import pybullet_data

# ROS components
import rospy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point, PointStamped
import tf

# Robot specific Hebi lib
import hebi

# Gym for trajectory planning
import gym
import gym_fhtw3dof

# Misc
import os
import time
import numpy as np
import matplotlib.pyplot as plt

# Import policy iteration trajectory planner
import fhtw3dof_usecase_tools
policy_iteration = fhtw3dof_usecase_tools.iterations()

def createDir(directory):
    """ Creates directory if it does not exist yet """
    if not os.path.exists(directory):
        os.makedirs(directory)

class TrajectoryGenerator:
    #
    def jointstate_callback(self, js):
        """ Saves latest jointstate data """
        self.latestJs = js
    #
    def addWaypoint(self):
        """ Adds current JointState to the stored trajectory """
        self.waypoints.append(self.latestJs)
    #
    def clear_waypoints(self):
        """ Clears buffered Waypoints """
        self.waypoints = []        
    #
    def computeTrajectory(self, dt=0.1):
        numJoints = 3
        numWaypoints = len(self.waypoints)
        # set up numpy arrays
        pos = np.empty((numJoints, numWaypoints))
        vel = np.empty((numJoints, numWaypoints))
        acc = np.empty((numJoints, numWaypoints))
        # Set first waypoint to 0.0
        vel[:,0] = acc[:,0] = 0.0
        vel[:,-1] = acc[:,-1] = 0.0
        # Set all other waypoints to undefined
        vel[:,1:-1] = acc[:,1:-1] = np.nan
        # Set all positions
        for point in range(numWaypoints):
            for joint in range(numJoints):
                pos[joint,point] = self.waypoints[point].position[joint]
        # Define required time to reach each waypoint
        times = np.linspace(0.0, numWaypoints*dt, numWaypoints)
        # Plan trejectory using Hebi python utility
        trajectory = hebi.trajectory.create_trajectory(times, pos, vel, acc)
        # Clear waypoints and add current position
        #self.clear_waypoints()
        #self.addWaypoint()
        return trajectory
    #
    def computeWaypoints(self, policy, targetState, env, maxSteps=100):
            """ Performs a learned policy on the simulated robot in order to obtain all waypoints. """
            # Check if robot is already at goal
            obs = env.env.s
            if obs == targetState: return True, []
            # Step through policy to determine trajectory
            for step in range(maxSteps):
                action = int(policy[obs])
                obs, _, _, _ = env.step(action)
                env.render_state(obs)
                self.addWaypoint()
                if obs == targetState: 
                    return True, self.waypoints
            return False, []
    #
    def __init__(self):
        """ Class Constructor """
        # Setup ROS subscriber so we always have the latest jointstate information
        self.latestJs = JointState()
        self.waypoints = []
        rospy.Subscriber("joint_states", JointState, self.jointstate_callback)
    #
    def __del__(self):
        """ Class Deconstructor """
        pass

## Init Environment, home robot and set start and goal states
rospy.init_node("SmartControl")

save_model = False
load_pretrained = True

env = gym.make('fhtw3dof-v0', Stop_Early=False, Constrain_Workspace=False,
                GoalX=0.0, GoalY=0.0, GoalZ=0.48,
                J1Limit=31, J2Limit=31, J3Limit=31, joint_res = 0.1,
                J1Scale=1, J2Scale=-1, J3Scale=-1, J3Offset=0.5,
                Debug=True, Publish_Frequency=500, Goal_Max_Distance=0.15)

assert (env.reachable)
env.env.Publish_Frequency = 20

if load_pretrained:
    known_goalstates = np.loadtxt('./trajectoryModel/known_goalstates.txt', dtype=int).tolist()
    policies = np.loadtxt('./trajectoryModel/policies.txt', dtype=int).tolist()
    if (not isinstance(known_goalstates, list)): 
        known_goalstates = [known_goalstates]
        policies = [policies]
else:
    known_goalstates = []
    policies = []

## Set start and goal states, plan and execute trajectory
home = env.env.to_index(16, 16, 0)
start = env.env.to_index(24, 3, 21)
goal = env.env.to_index(7, 27, 8)

_ = env.reset()
env.s = start
env.env.s = start
env.render_state(env.env.s)

## Set and check goal state
if goal < 0 or goal >= env.nS:
    print("Goalstate is not within the Robot's State-Size")

if env.Colliding_States[goal]:
    print("Goalstate is Colliding with Cbstacles.")

# Check if the goalstate has already been trained
try:
    idx = known_goalstates.index(goal)
except ValueError as err:
    # Reset env goal
    env.P = env.reset_rewards(env.P, 0, env.nS)
    # Set goalpose
    env.P, idx = env.set_state_reward(env.P, goal)
    if idx < 1:
        print("Goalstate not reachable.")
    # Compute policy to goal
    tmp_policy = policy_iteration.policy_iteration(env)
    # Append to lists of known goalstates and policies
    known_goalstates.append(goal)
    policies.append(tmp_policy)
    idx = len(known_goalstates)-1

# Instantiate trajectory planner and compute trajectory
planner = TrajectoryGenerator()
_ = env.reset()
env.env.s = start
env.s = start
env.render_state(env.env.s)
success, _ = planner.computeWaypoints(policies[idx], goal, env)
if success: tmpTrajectory = planner.computeTrajectory(dt=0.04)

# Get discrete command signal
dt = 0.001
t = np.linspace(0, int(tmpTrajectory.duration), num=int(tmpTrajectory.duration/dt))
pos_cmd = np.array([ tmpTrajectory.get_state(i)[0] for i in t ])

# Save trajectory models
if save_model:
    createDir('./trajectoryModel')
    np.savetxt('./trajectoryModel/policies.txt', policies,fmt='%d')
    np.savetxt('./trajectoryModel/known_goalstates.txt', known_goalstates,fmt='%d')
    print("Policies saved successfully.")

def synchroniseRobots(state, env, robot):
    """ Synchronises physical and ROS Robots at state """
    _ = env.reset()
    env.env.s = state
    env.s = state
    env.render_state(env.env.s)
    j1 = env.env.js.position[0]
    j2 = env.env.js.position[1]
    j3 = env.env.js.position[2]
    #TODO Loop only required until i find out how simulation timing works
    for i in range(100):
        robot.controlUpdate(cmdPos=[j1, j2, j3])

class SimRobot:
    """ Class that instantiates a simulated robot using Pybullet """
    def __init__(self):
        """ Class Constructor """
        #
        ## Pybullet Setup
        # Init pybullet client and GUI
        self.physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
        # Spawn gravity and groundplane
        p.setGravity(0,0,-9.8)
        planeId = p.loadURDF("plane.urdf")
        startPos = [0,0,1]
        startOrientation = p.getQuaternionFromEuler([0,0,0])
        # Load robot model
        #TODO fix Endeffector not loading in
        self.robotID = p.loadURDF("SAImon.urdf")
        # Print joint info
        print('Printing Joint Summary')
        for joint in range(p.getNumJoints(self.robotID)):
            print('Joint {}: {}\n'.format(joint, p.getJointInfo(self.robotID, joint)))
        #
        ## Virtual Control Setup
        #TODO
        #
        ## Start realtime simulation
        #p.setRealTimeSimulation(1)
    def __del__(self):
        """ Class Destructor """
        # Close Pybullet simulation
        p.disconnect()
        #
    def controlUpdate(self, cmdForce=[60,60,60], cmdPos=[0,0,0]):
        mode = p.POSITION_CONTROL
        p.setJointMotorControl2(self.robotID, 1, controlMode=mode, force=cmdForce[0], targetPosition=cmdPos[0])
        p.setJointMotorControl2(self.robotID, 3, controlMode=mode, force=cmdForce[1], targetPosition=cmdPos[1])
        p.setJointMotorControl2(self.robotID, 5, controlMode=mode, force=cmdForce[2], targetPosition=cmdPos[2])
        p.stepSimulation()
        jointState = [
            p.getJointState(self.robotID, 1),
            p.getJointState(self.robotID, 3),
            p.getJointState(self.robotID, 5)
        ]
        return jointState

## Run simulation and record joint feedback
sim = SimRobot()

# Synchronise simulated robot
synchroniseRobots(start, env, sim)

# Execute Trajectory in and store joint feedback
pos_sim = []

for cmd in pos_cmd:
    pos_sim.append(sim.controlUpdate(cmdPos=cmd))
    time.sleep(dt)

pos_sim = np.array(pos_sim)

# Plot comparison between cmd signals and motor feedback for each joint

# Plot command signals
fig, axs = plt.subplots(3)
fig.suptitle('Controller Command Signal vs. Motor Feedback')
axs[0].set_title('J1')
axs[0].plot(t, pos_cmd[:,0])
axs[0].plot(t, pos_sim[:,0,0])

axs[1].set_title('J2')
axs[1].plot(t, pos_cmd[:,1])
axs[1].plot(t, pos_sim[:,1,0])

axs[2].set_title('J3')
axs[2].plot(t, pos_cmd[:,2])
axs[2].plot(t, pos_sim[:,2,0])

plt.show()

# Hebi Motor Control Docs
# https://docs.hebi.us/core_concepts.html#motor_control


# Plot command signals
#fig, axs = plt.subplots(3)
#fig.suptitle('Hebi Controller internal control signal')
#axs[0].set_title('J1')
#axs[0].plot(t, pos_cmd[:,0])
#axs[1].set_title('J2')
#axs[1].plot(t, pos_cmd[:,1])
#axs[2].set_title('J3')
#axs[2].plot(t, pos_cmd[:,2])
#plt.show()






"""
Good Sources:

- Variational Inference Review
https://ieeexplore.ieee.org/abstract/document/8588399


- Variational Inference Tutorial Python
https://zhiyzuo.github.io/VI/

- Variational Inference Tutorial Python (das gleiche wie oben?)
https://www.ritchievink.com/blog/2019/09/16/variational-inference-from-scratch/

"""